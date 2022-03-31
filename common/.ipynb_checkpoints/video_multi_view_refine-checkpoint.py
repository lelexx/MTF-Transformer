import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from common.set_seed import *
from common.bert_model.bert import *
set_seed()

head_joint_idx_17 = [0, 1, 4, 7, 8, 11, 14, 9, 10]
hand_joint_left_idx_17 = [0, 1, 4, 7, 8, 11, 14, 12, 13]
hand_joint_right_idx_17 = [0, 1, 4, 7, 8, 11, 14, 15, 16]
foot_joint_right_idx_17 = [0, 1, 4, 7, 8, 11, 14,  2, 3, ]
foot_joint_left_idx_17 = [0, 1, 4, 7, 8, 11, 14,  5, 6, ]

head_joint_idx_16 = [0, 1, 4, 7, 8, 11, 14, 9]
hand_joint_left_idx_16 = [0, 1, 4, 7, 8, 10 13 11 12
hand_joint_right_idx_16= [0, 1, 4, 7, 8, 10 13 14 15
foot_joint_right_idx_16= [0, 1, 4, 7, 8, 10 13  2, 3, ]
foot_joint_left_idx_16= [0, 1, 4, 7, 8, 10 13  5, 6, ]

BN_MOMENTUM = 0.1
DIM = 3
N_K = 1
GROUP = 1
NUM_VIEW =5

TLEN = 9
USE_VIS = False
class RotRefine(nn.Module):
    def __init__(self, dropout = 0.25, channels = 1024):
        super().__init__()
        in_channels = DIM *DIM* TLEN
        channels = channels#1024
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = 1
        self.group = 1
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1), groups = self.group,bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum = 0.1)
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, (1, 1), groups = self.group,bias = False))
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
    def forward(self, rot):
        #(B, C, T, N)
        B, C, T, N = rot.shape
        
        rot = rot.view(B, C*T, 1, N).contiguous()
        res_rot = rot
        rot = self.dropout(self.relu(self.expand_bn(self.expand_conv(rot))))
        for i in range(self.num_layers):
            res = rot
            rot = self.dropout(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](rot))))
            rot = self.dropout(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](rot))))
            rot = rot + res
        
        rot = self.shrink(rot)
        
        rot = res_rot + rot
        rot = rot.view(B, C, T, N).contiguous()
        return rot
    
class AttRefine(nn.Module):
    def __init__(self, dropout = 0.25, channels = 1024):
        super().__init__()
        in_channels = DIM* TLEN
        channels = channels#1024
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = 1
        self.group = 1
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1), groups = self.group,bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum = 0.1)
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, (1, 1), groups = self.group,bias = False))
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
    
class PoseShrink_other(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25):
        super().__init__()
        self.shrink = RotBERT(filter_widths = [3,3], inp_channels= in_channels, hidden=512, n_layers=2, attn_heads=8, dropout=0, dim  =DIM)
        
    def set_bn_momentum(self, momentum):
        self.shrink.set_bn_momentum(momentum)
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.shrink(x)
        
        return x
class PoseShrink(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv2d(in_channels, DIM * DIM, 1)
        self.rot_refine = RotRefine(dropout = dropout, channels = DIM * DIM * TLEN * 3)
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        self.rot_refine.set_bn_momentum(momentum)
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        p1  =self.shrink(x)
        
        p1 = self.rot_refine(p1)
        
        p1 = p1.view(p1.shape[0], DIM, DIM, T, -1)
        return p1
    
class AttnShrink(nn.Module):
    def __init__(self, in_channels, channels, out_channels, dropout = 0.25):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        assert out_channels % (N_K * DIM) == 0
        self.shrink = nn.Conv2d(in_channels, out_channels // N_K, 1)
        self.att_refine = AttRefine(dropout = 0, channels = DIM * TLEN * 2)
        
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        self.att_refine.set_bn_momentum(momentum)
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        x  =self.shrink(x)
        x = self.att_refine(x)
        x = x.view(x.shape[0], -1, DIM, T, x.shape[-1])
        return x
class Pose(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25, momentum = 0.1):
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
        self.p_shrink = PoseShrink(in_channels = h_channels, channels = channels, dropout = dropout)
        self.att_shrink = AttnShrink(in_channels = h_channels, channels = channels, out_channels = channels, dropout = dropout)
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        self.p_shrink.set_bn_momentum(momentum)
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
        p1= self.p_shrink(x)
        att = self.att_shrink(x)

        return p1,  att
    
class MyConv(nn.Module):
    def __init__(self,  V,channels, dropout):
        super().__init__()
        self.expand_conv = nn.Conv2d(V, V*2*channels, (1, 1),stride = (1, 1), bias = False)
        
#         self.expand_bn = nn.BatchNorm2d(V*2*channels, momentum=momentum)
#         self.relu = nn.ReLU(inplace = True)
#         self.drop = nn.Dropout(dropout)
        
    def forward(self,pos_2d, vis_score):
        conv_p = self.expand_conv(vis_score)#(B, C_1*C_2, T, N)
        B, _,T, N = conv_p.shape
        conv_p = conv_p.view(B, pos_2d.shape[1],-1, T, N)
        x = torch.einsum('bcktn, bctn -> bktn', conv_p, pos_2d).contiguous()
        return x
class PartFeatureModel(nn.Module):
    def __init__(self, in_N, h_N, dropout = 0.25,momentum = 0.1,num_layers = 2,is_train = False):
        super().__init__()
        print('group:{}'.format(GROUP))
        D = 3 if USE_VIS else 2
        in_channels = in_N * D
        h_D =(DIM + 1) if USE_VIS else DIM
        channels = h_N * h_D
        
        self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1),stride = (1, 1), bias = False)
        
        self.expand_bn = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = num_layers
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False, groups = GROUP))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.vis_conv = MyConv(V = in_N, channels = h_N * DIM, dropout = dropout)


    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        
        vis_score = pos_2d[:,:,:,-1:]
        pos_2d = pos_2d[:,:,:,:2]
        B, T,V1, C1, N = pos_2d.shape
        pos_2d = pos_2d.permute(0, 2, 3, 1, 4).contiguous()
        pos_2d = pos_2d.view(B, V1 * C1, T, N).contiguous()
        
        vis_score = vis_score.permute(0, 2, 3, 1, 4).contiguous()
        vis_score = vis_score.view(B, V1, T, N).contiguous()
        
        vis_x = self.vis_conv(pos_2d, vis_score)
        
        if bone_angle is not None:
            B, T,V2, C2, N = bone_angle.shape
            bone_angle = bone_angle.permute(0, 2, 3, 1, 4).contiguous()
            bone_angle = bone_angle.view(B, V2 * C2, T, N).contiguous()
            x = torch.cat((pos_2d, bone_angle), dim = 1)
        else:
            if not USE_VIS:
                x = pos_2d
            else:
                x =  torch.cat((pos_2d, vis_score), dim = 1)
        
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        x[:,:vis_x.shape[1]] = x[:,:vis_x.shape[1]] + vis_x
        x = x.contiguous()
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, DIM if not USE_VIS else (DIM + 1), x.shape[-2], x.shape[-1])

class MultiPartFeatureModel(nn.Module):
    def __init__(self, in_channels, channels, oup_channels, dropout = 0.25,momentum = 0.1,num_layers=2, is_train = False, num_joints =17):
        super().__init__()
        print(channels, oup_channels)
        N_bone = 0
        DIM_joint = 2
        in_channels = len(head_joint_idx)
        
        self.head_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM ,dropout = dropout,momentum = momentum,num_layers = num_layers, is_train = is_train)
        in_channels = len(hand_joint_left_idx)
        self.hand_left_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum, num_layers = num_layers,is_train = is_train)
        in_channels = len(foot_joint_left_idx)
        self.foot_left_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum,num_layers = num_layers,is_train = is_train)
        
        in_channels = len(hand_joint_right_idx) 
        self.hand_right_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum, num_layers = num_layers,is_train = is_train)
        in_channels = len(foot_joint_right_idx)
        self.foot_right_model = PartFeatureModel(in_N = in_channels, h_N = channels // DIM , dropout = dropout,momentum = momentum,num_layers = num_layers,is_train = is_train)
        c = (DIM +1) if USE_VIS else DIM
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

    def forward(self, pos_2d, bone_angle):
        head = self.head_model(pos_2d[:,:,head_joint_idx], bone_angle[:,:,head_bone_idx] if bone_angle is not None else None)
        hand_left = self.hand_left_model(pos_2d[:,:,hand_joint_left_idx], bone_angle[:,:, hand_bone_left_idx] if bone_angle is not None else None)
        foot_left = self.foot_left_model(pos_2d[:,:,foot_joint_left_idx], bone_angle[:, :,foot_bone_left_idx] if bone_angle is not None else None)
        hand_right = self.hand_right_model(pos_2d[:,:,hand_joint_right_idx], bone_angle[:,:, hand_bone_right_idx] if bone_angle is not None else None)
        foot_right = self.foot_right_model(pos_2d[:,:,foot_joint_right_idx], bone_angle[:,:, foot_bone_right_idx] if bone_angle is not None else None)

        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        
        f = f.view(f.shape[0], -1, f.shape[3], f.shape[4])

        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        return f.view(f.shape[0], -1, (DIM + 1) if USE_VIS else DIM, f.shape[-2], f.shape[-1])
class Pose3dShrink(nn.Module):
    def __init__(self, in_channels,channels, dropout = 0.25, momentum = 0.1,dim_joint = 3, is_train = False):
        super().__init__()
        if is_train:
            self.conv_1 = nn.Conv2d(in_channels, channels, (3, 1),stride = (3, 1), bias = False)
        else:
            self.conv_1 = nn.Conv2d(in_channels, channels, (3, 1),stride = (1, 1),dilation = (3**0, 1), bias = False)
        
        self.bn_1 = nn.BatchNorm2d(channels, momentum = momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 1
        
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
        self.shrink = nn.Conv2d(channels, 17 * dim_joint, 1)

        self.dim_joint = dim_joint
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
       
        x = x.view(x.shape[0], 17, self.dim_joint, -1,N)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        return x
class FuseView(nn.Module):
    def __init__(self, N, dropout = 0.25, is_train = False):
        super().__init__()
        D = (DIM + 1) if USE_VIS else DIM
        self.pose_model = Pose(in_channels = D * N * 2, dropout = dropout, channels = N * DIM)
        self.drop = nn.Dropout(dropout)
        self.m = torch.zeros(1, 1, 1, 1, NUM_VIEW, NUM_VIEW)
        print(self.m.shape)
        for i in range(NUM_VIEW):
            self.m[:,:,:,:,i, i] = 1
        print(self.m[0,0,0,0])
        self.m = self.m.float()
        self.Thr = 0.4
        print('thr:{}'.format(self.Thr))
    def set_bn_momentum(self, momentum):
        self.pose_model.set_bn_momentum(momentum)
    def forward(self, x):
        B, C1, C2, T, N = x.shape
        f = x[:,:,:DIM].contiguous()
        x1 = x.view(B, C1, C2, T, N, 1).repeat(1, 1, 1, 1, 1, N)
        x2 = x.view(B, C1, C2, T, 1, N).repeat(1, 1, 1, 1, N, 1)
        
        x = torch.cat((x1, x2), dim = 1)
        
        x = x.view(B, C1 * 2, C2, T, N*N)
 
        p, att = self.pose_model(x)
        
        p = p.view(p.shape[0], p.shape[1], p.shape[2], p.shape[3], N, N)
        att = att.view(att.shape[0], att.shape[1], att.shape[2], att.shape[3], N, N)

        f_conv = torch.einsum('bnctm, bqctsm -> bnqtsm', f, p)
        if self.training and NUM_VIEW > 4:
            mask = torch.rand(1, 1, 1, 1, NUM_VIEW, NUM_VIEW).to(x.device) + self.m.to(x.device)

            mask = mask < self.Thr
            att = att.masked_fill(mask, -1e9)
            #print('lele')
      
        else:
            pass

        att = F.softmax(att, dim=-1)
        f_conv = f_conv.view(f_conv.shape[0], N_K,f_conv.shape[1] // N_K, f_conv.shape[2], f_conv.shape[3], f_conv.shape[4], f_conv.shape[5])
        f_fuse = torch.einsum('benctsm, bnctsm -> bencts', f_conv, att).contiguous()
        f_fuse = f_fuse.view(f_fuse.shape[0], -1, f_fuse.shape[3], f_fuse.shape[4], f_fuse.shape[5])

        f_fuse = f_fuse + f
        
        return f_fuse
class Pose3dShrinkOther(nn.Module):
    def __init__(self, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False):
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
        self.shrink = nn.Conv2d(channels, 17 * dim_joint, 1)
        
        self.dim_joint = dim_joint
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
    
        x = x.view(x.shape[0], 17, self.dim_joint, -1,N)
        
        return x
    
class VideoMultiViewModelRefine(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,is_train = False, use_inter_loss = False):
        super().__init__() 
        global DIM
        print('dim: {}'.format(DIM))
        oup_channels = (channels//DIM * (DIM + 1)) if USE_VIS else channels
        self.f_model = MultiPartFeatureModel(in_channels = in_channels, channels = channels // 2,oup_channels = oup_channels, dropout = dropout,momentum = momentum, n_layers = 2, is_train = is_train, dilation = 1)
        self.to2d_model = MultiPartFeatureModel(in_channels = in_channels, channels = channels // 2,oup_channels = oup_channels * 17 * 2, dropout = dropout,momentum = momentum, n_layers = 0, is_train = is_train, dilation = 1)
        N = channels // DIM
        self.fuse_model = FuseView(N,dropout = dropout, is_train = is_train)

        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        SHRINK_CHANNEL = channels
        
        self.shrink = BERT(filter_widths = [3,3], inp_channels= channels, hidden=512, n_layers=2, attn_heads=8, dropout=dropout)
        if is_train and use_inter_loss:
            self.shrink_1 = Pose3dShrinkOther(N = N, channels = SHRINK_CHANNEL,dropout = dropout, dim_joint = 3, is_train = is_train)
            self.shrink_2 = Pose3dShrinkOther(N = N, channels = SHRINK_CHANNEL, dropout = dropout, dim_joint = 3, is_train = is_train)
        self.use_inter_loss = use_inter_loss
    def set_bn_momentum(self, momentum):
        self.f_model.set_bn_momentum(momentum)
        self.fuse_model.set_bn_momentum(momentum)
        
        self.shrink.set_bn_momentum(momentum)
        if self.training and self.use_inter_loss:
            self.shrink_1.set_bn_momentum(momentum)
            self.shrink_2.set_bn_momentum(momentum)
    
    def propocess_data(self, pos_2d, Choose = 1):
        B, T,V, C, N = pos_2d.shape
    
        if Choose == 1:
            K = 10
            scale = torch.rand(B, 1, 1, 1, N) * K
            scale = torch.clamp(scale, 1 / K, K).float().to(pos_2d.device)
            pos_2d[:,:,:,:2] = pos_2d[:,:,:,:2] * scale
        elif Choose == 2:
            x_temp = pos_2d.cpu().numpy()
            x_temp[:,:,:,:2] = x_temp[:,:,:,:2] - x_temp[:,:,:1,:2]
            scale = np.max(np.abs(x_temp[:,:,:,:2]), axis = (1, 2, 3), keepdims = True)
            x_temp[:,:,:,:2] = x_temp[:,:,:,:2] / scale
            x_temp = torch.from_numpy(x_temp).float().to(pos_2d.device)
            pos_2d = x_temp
        elif Choose == 3:
            x_temp = pos_2d.cpu().numpy()
            scale = np.max(np.abs(x_temp[:,:,:,:2]), axis = (1, 2, 3), keepdims = True)
            x_temp[:,:,:,:2]  =x_temp[:,:,:,:2] / scale
            
            x_temp = torch.from_numpy(x_temp).float().to(pos_2d.device)
            pos_2d = x_temp
        return pos_2d
    
    def forward(self, pos_2d, bone_angle = None):
        B, T,V, C, N = pos_2d.shape
        if self.training and 0:
            limb_idx = [2, 3, 5, 6, 12, 13, 15, 16]
            vis = pos_2d[:,:,:,-1:]
            p = torch.rand(*vis[:,:,limb_idx].shape).float().to(vis.device)
            pos_2d[:,:,limb_idx] = pos_2d[:,:,limb_idx] * (p < vis[:,:,limb_idx]).float().to(vis.device)
            pos_2d = pos_2d.contiguous()

        Choose = 3
        if Choose == 1:
            if self.training:
                pos_2d = self.propocess_data(pos_2d, Choose)
        elif Choose == 2:
            pos_2d = self.propocess_data(pos_2d, Choose)
        elif Choose == 3:
            pos_2d = self.propocess_data(pos_2d, Choose)
            
        pos_2d =pos_2d.contiguous()

        f = self.f_model(pos_2d, bone_angle)

        if self.training and self.use_inter_loss:
            out_1 = self.shrink_1(f[:,:,:DIM].contiguous())
            out_1 = out_1.permute(0, 3, 1, 2, 4)
        
        f= self.fuse_model(f)
        if self.training and self.use_inter_loss:
            out_2 = self.shrink_2(f)
            out_2 = out_2.permute(0, 3, 1, 2, 4)

        out = self.shrink(f)
        

        if self.training and self.use_inter_loss:
            return out, out_1, out_2
        else:
            return out
