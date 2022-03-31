import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools
import pickle

from common.set_seed import *
from common.bert_model.bert import *
set_seed()
head_joint_idx_17 = [0, 1, 4, 7, 8, 11, 14, 9, 10]
hand_joint_left_idx_17 = [0, 1, 4, 7, 8, 11, 14, 12, 13]
hand_joint_right_idx_17 = [0, 1, 4, 7, 8, 11, 14, 15, 16]
foot_joint_right_idx_17 = [0, 1, 4, 7, 8, 11, 14,  2, 3, ]
foot_joint_left_idx_17 = [0, 1, 4, 7, 8, 11, 14,  5, 6, ]

head_joint_idx_16 = [0, 1, 4, 7, 8, 11, 14, 9]
hand_joint_left_idx_16 = [0, 1, 4, 7, 8, 10, 13, 11, 12]
hand_joint_right_idx_16= [0, 1, 4, 7, 8, 10, 13, 14, 15]
foot_joint_right_idx_16= [0, 1, 4, 7, 8, 10 ,13 ,2, 3, ]
foot_joint_left_idx_16= [0, 1, 4, 7, 8, 10 ,13 , 5, 6, ]
DIM = 3
NUM_VIEW =5
CAT_CONF = False

CONF_MOD = True
USE_ATT = True
USE_AG = True
MASK_RATE = 0.4

class AgRefine(nn.Module):
    def __init__(self, dropout = 0.25, channels = 1024, T = 9):
        super().__init__()
        in_channels = DIM *DIM* T

        channels = channels#1024
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = 1

        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1), bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum = 0.1)
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, (1, 1),bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum = 0.1))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum = 0.1))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, in_channels, 1)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, ag):
        #(B, C, T, N)
        B, C, T, N = ag.shape
        
        ag = ag.view(B, C*T, 1, N).contiguous()
        res_ag = ag
 
        ag = self.dropout(self.relu(self.expand_bn(self.expand_conv(ag))))
        for i in range(self.num_layers):
            res = ag
            ag = self.dropout(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](ag))))
            ag = self.dropout(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](ag))))
            ag = ag + res
        
        ag = self.shrink(ag)
        
        ag = res_ag + ag
        ag = ag.view(B, C, T, N).contiguous()
        return ag
    
class AttRefine(nn.Module):
    def __init__(self, dropout = 0.25, channels = 1024, T = 9):
        super().__init__()
        in_channels = DIM* T
        channels = channels#1024
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = 1
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1), bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum = 0.1)
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, (1, 1), bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum = 0.1))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum = 0.1))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, in_channels, 1)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, att):
        #(B, C, T, N)
        B, C, T, N = att.shape
        
        att = att.view(-1, DIM, T, N).contiguous()
        att = att.view(-1, DIM * T,1, N).contiguous()
        res_att = att
        att = self.dropout(self.relu(self.expand_bn(self.expand_conv(att))))
        for i in range(self.num_layers):
            res = att
            att = self.dropout(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](att))))
            att = self.dropout(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](att))))
            att = att + res
        
        att = self.shrink(att)
        
        att = res_att + att
        att = att.view(B, C*T, 1, N).contiguous()
        att = att.view(B, C, T, N).contiguous()
        return att
    

class AgShrink(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.1, T = 9):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.T = T
        self.shrink = nn.Conv2d(in_channels, DIM * DIM, 1)
        self.Flag = False
        if self.T > 1 and self.Flag:
            self.rot_refine = AgRefine(dropout = dropout, channels = DIM * DIM * self.T * 3, T = self.T)
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        if self.T > 1 and self.Flag:
            self.rot_refine.set_bn_momentum(momentum)
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        p1  =self.shrink(x)
        if self.T > 1 and self.Flag:
            p1 = self.rot_refine(p1)
        
        p1 = p1.view(p1.shape[0], DIM, DIM, T, -1)
        return p1
    
class AttShrink(nn.Module):
    def __init__(self, in_channels, channels, out_channels, dropout = 0.1, T = 9):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.T = T
        self.shrink = nn.Conv2d(in_channels, out_channels, 1)
        self.Flag = False
        if self.T > 1 and self.Flag:
            self.att_refine = AttRefine(dropout = 0, channels = DIM * self.T * 2, T = self.T)
        
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        if self.T > 1 and self.Flag:
            self.att_refine.set_bn_momentum(momentum)
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        x  =self.shrink(x)
        if self.T > 1 and self.Flag:
            x = self.att_refine(x)
        x = x.view(x.shape[0], -1, DIM, T, x.shape[-1])
        return x
class VAL(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25, momentum = 0.1,T = 9):
        super().__init__()
        h_channels = channels #// 2
        self.expand_conv = nn.Conv2d(in_channels, h_channels, (1, 1), bias = False)
        self.expand_bn = nn.BatchNorm2d(h_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.num_layers = 1
        
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(h_channels, h_channels, 1, bias = False, ))
            bn_layers.append(nn.BatchNorm2d(h_channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(h_channels, h_channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(h_channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        if USE_AG:
            self.p_shrink = AgShrink(in_channels = h_channels, channels = channels, dropout = dropout, T = T)
        if USE_ATT:
            self.att_shrink = AttShrink(in_channels = h_channels, channels = channels, out_channels = channels, dropout = dropout, T = T)
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        if USE_AG:
            self.p_shrink.set_bn_momentum(momentum)
        if USE_ATT:
            self.att_shrink.set_bn_momentum(momentum)
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        B, _, _, T, N = x.shape
        x = x.view(x.shape[0], -1, T, N)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        outs = []
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x
        if USE_AG:
            ag= self.p_shrink(x)
        else:
            ag = None
        if USE_ATT:
            att = self.att_shrink(x)
        else:
            att = None
            
        return ag,  att
class MyConv(nn.Module):
    def __init__(self,  V,channels):
        super().__init__()
        self.expand_conv = nn.Conv2d(V, V*2*channels, (1, 1),stride = (1, 1), bias = False)

    def forward(self,pos_2d, vis_score):
        conv_p = self.expand_conv(vis_score)#(B, C_1*C_2, T, N)
        B, _,T, N = conv_p.shape
        conv_p = conv_p.view(B, pos_2d.shape[1],-1, T, N)
        x = torch.einsum('bcktn, bctn -> bktn', conv_p, pos_2d).contiguous()
        return x    

class PartFeatureModel(nn.Module):
    def __init__(self, in_N, h_N, dropout = 0.25,momentum = 0.1,is_train = False):
        super().__init__()
        D = 3 if CAT_CONF else 2
        in_channels = in_N * D
        h_D = DIM
        channels = h_N * h_D
        
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1),stride = (1, 1), bias = False)
        
        self.expand_bn = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 2
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        if CONF_MOD:
            self.vis_conv = MyConv(V = in_N, channels = h_N * DIM)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d):
        
        vis_score = pos_2d[:,:,:,-1:]
        pos_2d = pos_2d[:,:,:,:2]
        B, T,V1, C1, N = pos_2d.shape
        pos_2d = pos_2d.permute(0, 2, 3, 1, 4).contiguous()
        pos_2d = pos_2d.view(B, V1 * C1, T, N).contiguous()
        
        vis_score = vis_score.permute(0, 2, 3, 1, 4).contiguous()
        vis_score = vis_score.view(B, V1, T, N).contiguous()
        if CONF_MOD:
            vis_x = self.vis_conv(pos_2d, vis_score)
        
        if not CAT_CONF:
            x = pos_2d
        else:
            x =  torch.cat((pos_2d, vis_score), dim = 1)
        
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        if CONF_MOD:
            x[:,:vis_x.shape[1]] = x[:,:vis_x.shape[1]] + vis_x
        x = x.contiguous()
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, DIM, x.shape[-2], x.shape[-1])

class MultiPartFeatureModel(nn.Module):
    def __init__(self, channels, oup_channels, dropout = 0.25,momentum = 0.1,is_train = False, num_joints = 17):
        super().__init__()
        DIM_joint = 2
        
        if num_joints == 17:
            self.head_joint_idx = head_joint_idx_17
            self.hand_joint_left_idx = hand_joint_left_idx_17
            self.hand_joint_right_idx = hand_joint_right_idx_17
            self.foot_joint_left_idx = foot_joint_left_idx_17
            self.foot_joint_right_idx = foot_joint_right_idx_17
        elif num_joints == 16:
            self.head_joint_idx = head_joint_idx_16
            self.hand_joint_left_idx = hand_joint_left_idx_16
            self.hand_joint_right_idx = hand_joint_right_idx_16
            self.foot_joint_left_idx = foot_joint_left_idx_16
            self.foot_joint_right_idx = foot_joint_right_idx_16
        
        in_channels = len(self.head_joint_idx)
        self.head_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM ,dropout = dropout,momentum = momentum,is_train = is_train)
        in_channels = len(self.hand_joint_left_idx)
        self.hand_left_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum, is_train = is_train)
        in_channels = len(self.foot_joint_left_idx)
        self.foot_left_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum,is_train = is_train)
        
        in_channels = len(self.hand_joint_right_idx) 
        self.hand_right_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum, is_train = is_train)
        in_channels = len(self.foot_joint_right_idx)
        self.foot_right_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum,is_train = is_train)
        c =  DIM
        self.shrink_conv =nn.Conv2d(channels // DIM * 5 * c, oup_channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm2d(oup_channels, momentum = momentum)
    
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
          
    def set_bn_momentum(self, momentum):
        self.head_model.set_bn_momentum(momentum)
        self.hand_left_model.set_bn_momentum(momentum)
        self.foot_left_model.set_bn_momentum(momentum)
        self.hand_right_model.set_bn_momentum(momentum)
        self.foot_right_model.set_bn_momentum(momentum)
        self.shrink_bn.momentum = momentum

    def forward(self, pos_2d):
        head = self.head_model(pos_2d[:,:,self.head_joint_idx])
        hand_left = self.hand_left_model(pos_2d[:,:,self.hand_joint_left_idx])
        foot_left = self.foot_left_model(pos_2d[:,:,self.foot_joint_left_idx])
        hand_right = self.hand_right_model(pos_2d[:,:,self.hand_joint_right_idx])
        foot_right = self.foot_right_model(pos_2d[:,:,self.foot_joint_right_idx])

        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        
        f = f.view(f.shape[0], -1, f.shape[3], f.shape[4])

        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        return f.view(f.shape[0], -1, DIM, f.shape[-2], f.shape[-1])
class Pose3dShrink(nn.Module):
    def __init__(self, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
        super().__init__()
        self.conv_1 = nn.Conv2d(N * DIM, channels, 1,stride = (1, 1), bias = False)
        
        self.bn_1 = nn.BatchNorm2d(channels, momentum = momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 0
        
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, (1,1),stride = (1, 1), dilation = (3**(i+1), 1),bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, num_joints * dim_joint, 1)

        self.dim_joint = dim_joint
        self.num_joints = num_joints
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):

        B, _, _, T, N = x.shape
        x = x.view(x.shape[0], -1, T, N)
        
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
     
            x = res + x

        x = self.shrink(x)
       
        x = x.view(x.shape[0], self.num_joints, self.dim_joint, -1,N)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        return x
def cal_dist(f):
    f = f[:,:,:,0]
    B, K, C, N, N = f.shape
    f = f.view(B, -1, N, N)
    fl = torch.norm(f, dim = 1, keepdim = True)
    f = f / (fl + 1e-6)
    for i in range(4):
        for view_list in itertools.combinations(list(range(f.shape[-1])), 2):
            tmp = torch.sum(f[:,:,view_list[0], i]* f[:,:,view_list[1], i], dim = 1)
            print(view_list, torch.mean(tmp))
        print('*******************')
    print('#####################')
    for i in range(4):
        for j in range(4):
            if i != j:
                tmp = torch.sum(f[:,:,i, i]* f[:,:,i, j], dim = 1)
                print(i, j, torch.mean(tmp))
        print('*******************')

class AverageMeter_(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += torch.sum(val, dim = 0)
        self.count += val.shape[0]
        self.avg = self.sum / self.count
        
class FuseView(nn.Module):
    def __init__(self, N, dropout = 0.25, is_train = False, T = 9):
        super().__init__()
        D =  DIM
        if USE_AG or USE_ATT:
            self.pose_model = VAL(in_channels = D * N * 2, dropout = dropout, channels = N * DIM, T = T)
        self.drop = nn.Dropout(dropout)
        self.m = torch.zeros(1, 1, 1, 1, NUM_VIEW, NUM_VIEW)
        for i in range(NUM_VIEW):
            self.m[:,:,:,:,i, i] = 1
        
        self.m = self.m.float()
        self.Thr = MASK_RATE
        
        if is_train:
            print(self.m[0,0,0,0])
            print('mask rate:{}'.format(self.Thr))
        self.meter = AverageMeter()
        
    def set_bn_momentum(self, momentum):
        if USE_ATT or USE_AG:
            self.pose_model.set_bn_momentum(momentum)
        else:
            pass
    def forward(self, x):
        B, C1, C2, T, N = x.shape
        f = x[:,:,:DIM].contiguous()
        x1 = x.view(B, C1, C2, T, N, 1).repeat(1, 1, 1, 1, 1, N)
        x2 = x.view(B, C1, C2, T, 1, N).repeat(1, 1, 1, 1, N, 1)
        
        x = torch.cat((x1, x2), dim = 1)
        
        x = x.view(B, C1 * 2, C2, T, N*N)
        if USE_ATT or USE_AG:
            ag, att = self.pose_model(x) #ag:(B, D, D, T, N*N) att: (B, K, D, T, N*N)
        tran = None
        if USE_AG:
            ag = ag.view(ag.shape[0], ag.shape[1], ag.shape[2], ag.shape[3], N, N) #(B, 3, 3, T, N, N)

            tran = ag

            f_conv = torch.einsum('bnctm, bqctsm -> bnqtsm', f, ag)
        else:
            f_conv =x2
            
        if USE_ATT:
            att = att.view(att.shape[0], att.shape[1], att.shape[2], att.shape[3], N, N)
        else:
            att = torch.zeros(1, 1, 1, 1, N, N) + (1 / N)
            att = att.to(x.device)
            att = att.expand(B, *att.shape[1:])
            
        if self.training and self.Thr > 0:
            mask = torch.rand(B, 1, 1, 1, NUM_VIEW, NUM_VIEW).to(x.device) + self.m.to(x.device)

            mask = mask < self.Thr
            att = att.masked_fill(mask, -1e9)

        att = F.softmax(att, dim=-1) #(B, K, C, T, N, N)
        
        f_fuse = f_conv * att
        
        f_rcpe = f_fuse
        f_fuse = torch.sum(f_fuse, dim  =-1)

        f_fuse = f_fuse + f
        
        return f_fuse, tran, att, f, f_rcpe[...,0,:]
    

class Pose3dShrinkOther(nn.Module):
    def __init__(self, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
        super().__init__()
        self.conv_1 = nn.Conv2d(N * DIM, channels, (1, 1),stride = (1, 1),dilation = (3**0, 1), bias = False)
        
        self.bn_1 = nn.BatchNorm2d(channels, momentum = momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 0
        
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            if is_train:
                conv_layers.append(nn.Conv2d(channels, channels, (3,1),stride = (3, 1), bias = False))
            else:
                conv_layers.append(nn.Conv2d(channels, channels, (3,1),stride = (1, 1), dilation = (3**(i+1), 1),bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, num_joints * dim_joint, 1)
        
        self.dim_joint = dim_joint
        self.num_joints = num_joints
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):

        B, _, _, T, N = x.shape

        x = x.view(x.shape[0], -1, T, N)
        
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        
        K = 2
        for i in range(self.num_layers): 
            if self.training:
                res = x[:,:,1::3]
            else:
                res = x[:,:,3**(i+1):-3**(i+1)]
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
     
            x = res + x
        
        x = self.shrink(x)
    
        x = x.view(x.shape[0], self.num_joints, self.dim_joint, -1,N)
        
        return x
    
class VideoMultiViewModel(nn.Module):
    def __init__(self, args, num_view,momentum = 0.1,is_train = False, use_inter_loss = True, num_joints = 17, ):
        super().__init__() 
        T = args.t_length
        dropout = args.dropout
        channels = args.channels
        
        global DIM
        global NUM_VIEW
        global CAT_CONF, CONF_MOD, USE_AG, MASK_RATE, USE_ATT
        MASK_RATE = args.mask_rate
        USE_ATT = args.attention
        DIM = args.dim
        
        assert args.conf in ['no', 'concat', 'modulate']
        if args.conf == 'no':
            CAT_CONF = False
            CONF_MOD = False
        elif args.conf == 'concat':
            CAT_CONF = True
            CONF_MOD = False
        elif args.conf == 'modulate':
            CAT_CONF = False
            CONF_MOD = True
        
        if args.feature_alignment:
            USE_AG = True
        else:
            USE_AG = False
        
        self.fuse = args.multiview_fuse
   
        NUM_VIEW = num_view
        if is_train:
            print('dim: {}'.format(DIM))
        oup_channels = channels
        self.T = T
        
        self.f_model = MultiPartFeatureModel( channels = channels // 2,oup_channels = oup_channels, dropout = dropout,momentum = momentum, is_train = is_train, num_joints = num_joints)
        N = channels // DIM
        
        if self.fuse:
            self.fuse_model = FuseView(N,dropout = dropout, is_train = is_train, T = self.T)

        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        SHRINK_CHANNEL = channels
        self.shrink = BERT(T = self.T, inp_channels= channels, hidden=512, n_layers=2, attn_heads=8, dropout=0.1, num_joints = num_joints)
        if is_train and use_inter_loss:
            self.shrink_1 = Pose3dShrinkOther(N = N, channels = SHRINK_CHANNEL,dropout = dropout, dim_joint = 3, is_train = is_train, num_joints = num_joints)
            if self.fuse:
                self.shrink_2 = Pose3dShrinkOther(N = N, channels = SHRINK_CHANNEL, dropout = dropout, dim_joint = 3, is_train = is_train, num_joints = num_joints)
        self.use_inter_loss = use_inter_loss
    def set_bn_momentum(self, momentum):
        self.f_model.set_bn_momentum(momentum)
        if self.fuse:
            self.fuse_model.set_bn_momentum(momentum)
        
        self.shrink.set_bn_momentum(momentum)
        if self.training and self.use_inter_loss:
            self.shrink_1.set_bn_momentum(momentum)
            if self.fuse:
                self.shrink_2.set_bn_momentum(momentum)
    
    def forward(self, pos_2d, bone_angle = None):
        B, T,V, C, N = pos_2d.shape
            
        pos_2d =pos_2d.contiguous()

        f = self.f_model(pos_2d)

        if self.training and self.use_inter_loss:
            out_1 = self.shrink_1(f[:,:,:DIM].contiguous())
            out_1 = out_1.permute(0, 3, 1, 2, 4)
        tran = None
        if self.fuse:
            f, tran, att, f_tmp, f_tmp_rcpe   = self.fuse_model(f)
            if self.training and self.use_inter_loss:
                out_2 = self.shrink_2(f)
                out_2 = out_2.permute(0, 3, 1, 2, 4)
        out = self.shrink(f)

        if self.training and self.use_inter_loss:
            return out, [out_1, out_2] if self.fuse else [out_1], tran 
        else:
            return out, tran, att, f_tmp, f_tmp_rcpe

        