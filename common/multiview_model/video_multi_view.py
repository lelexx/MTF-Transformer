from pydoc import visiblename
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
from .point_transformer_pytorch import *
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


class AgShrink(nn.Module):
    def __init__(self, cfg, in_channels, channels, dropout = 0.1, T = 9, is_train = False):
        super().__init__()
        self.cfg = cfg
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.T = T
        if is_train and self.cfg.TRAIN.USE_ROT_LOSS:
            print('use_rot_loss')
            self.shrink_1 = nn.Conv2d(in_channels, cfg.NETWORK.TRANSFORM_DIM**2, 1)
        self.shrink = nn.Conv2d(in_channels, cfg.NETWORK.TRANSFORM_DIM**2, 1)
        
    def set_bn_momentum(self, momentum):
        print('use_rot_loss:{}'.format(self.cfg.TRAIN.USE_ROT_LOSS))
        self.bn_1.momentum = momentum

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        if self.training and self.cfg.TRAIN.USE_ROT_LOSS:
            p1  = self.shrink(x)
            p1 = p1.view(p1.shape[0], self.cfg.NETWORK.TRANSFORM_DIM, self.cfg.NETWORK.TRANSFORM_DIM,  T, -1)
        else:
            p1 = None
        p = self.shrink(x)
        
        p = p.view(p.shape[0], self.cfg.NETWORK.TRANSFORM_DIM, self.cfg.NETWORK.TRANSFORM_DIM,  T, -1)

        return p1 , p
    
class AttShrink(nn.Module):
    def __init__(self, cfg, in_channels, channels, out_channels, dropout = 0.1, T = 9, is_train = False):
        super().__init__()
        self.cfg = cfg
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.T = T
        self.shrink = nn.Conv2d(in_channels, out_channels, 1)
        
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        
    def forward(self, x):
        B, C, T, V = x.shape
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        x  =self.shrink(x)

        x = x.view(x.shape[0], -1, self.cfg.NETWORK.TRANSFORM_DIM, T, x.shape[-1])
        return x
class VAL(nn.Module):
    def __init__(self, cfg, in_channels, channels, dropout = 0.25, momentum = 0.1,T = 9, is_train = False, num_joints = 17):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        h_channels = channels
        self.expand_conv = nn.Conv2d(in_channels, h_channels, (1, 1), bias = False)
        self.expand_bn = nn.BatchNorm2d(h_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.FLAG = self.cfg.NETWORK.M_FORMER.USE_POSE2D
        self.DIM = self.cfg.NETWORK.TRANSFORM_DIM
        self.TRAN_FLAG = True
        self.num_view = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        if self.TRAN_FLAG and is_train and self.cfg.TRAIN.USE_ROT_LOSS:
            self.self_tran= nn.Parameter(torch.zeros(1,self.DIM, self.DIM, 1, 1)) #(B, dim, dim, T, 1)
            num_view = 10
            self.m = torch.zeros(1, 1, 1, 1, num_view, num_view)
            for i in range(num_view):
                self.m[:,:,:,:,i, i] = 1
            self.m = self.m.float()

        #use_vis 和 cat都有提升，但提升不大为了不增加参数量，未采用
        self.USE_VIS = False
        self.CAT = False
        if self.FLAG:
            if not self.CAT:
                self.pos_emb_conv = nn.Conv2d(int(num_joints *2), h_channels, 1, bias = True)
            else:
                self.pos_emb_conv = nn.Conv2d(int(num_joints *2 * 2), h_channels, 1, bias = True)
                
            if self.USE_VIS:
                if not self.CAT:
                    self.vis_conv = nn.Conv2d(num_joints, num_joints * 2 * h_channels, 1, bias = False)
                else:
                    self.vis_conv = nn.Conv2d(num_joints , num_joints * 2 * 2 * h_channels, 1, bias = False)

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
        if cfg.NETWORK.USE_FEATURE_TRAN:
            if not cfg.NETWORK.USE_GT_TRANSFORM:
                self.p_shrink = AgShrink(cfg, in_channels = h_channels, channels = channels, dropout = dropout, T = T, is_train=is_train)
            if cfg.NETWORK.USE_GT_TRANSFORM:
                self.p_shrink_rot = RotationModel(cfg, dropout = dropout, momentum = momentum)
        self.att_shrink = AttShrink(cfg, in_channels = h_channels, channels = channels, out_channels = channels, dropout = dropout, T = T, is_train=is_train)
  
    def set_bn_momentum(self, momentum):
        print('tran_flag:{}'.format(self.TRAN_FLAG))
        self.expand_bn.momentum = momentum
        if self.cfg.NETWORK.USE_FEATURE_TRAN:
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                self.p_shrink.set_bn_momentum(momentum)
            if self.cfg.NETWORK.USE_GT_TRANSFORM:
                self.p_shrink_rot.set_bn_momentum(momentum)
        self.att_shrink.set_bn_momentum(momentum)
        
        for bn in self.bn_layers:
            bn.momentum = momentum

    def forward(self, x, pos_2d, rotation = None):
        '''
        Args:
            pos_2d:(B, T, J, C, N)
        '''
        device = x.device
        B, _, _, T, N = x.shape
        N_view = int(N**(0.5))
        if self.FLAG:
            if self.USE_VIS:
                vis = pos_2d[:,:,:,-1:,:].permute(0, 2, 3, 1, 4).contiguous() #(B, J, 1, T, N_view)
                vis = vis.view(B,-1, T, N_view)
                
                vis_emb = self.vis_conv(vis) #(B, C, T, N_view)
                vis_emb = vis_emb.view(B, 17 * 2, -1, T, N_view)
            
            
            pos_2d = pos_2d[:,:,:,:2,:].permute(0, 2, 3, 1, 4).contiguous() #(B, J, 2, T, N_view)
            pos_2d = pos_2d.view(B,-1, T, N_view)
            rel_pos = pos_2d[:, :, :,:,None] - pos_2d[:, :,:,None,:] #(B, J*C, T, N_view, N_view)
            rel_pos = rel_pos.view(B, -1, T, N)
            
            pos_emb = self.pos_emb_conv(rel_pos) #(B,C, T, N_view**2)
            if self.USE_VIS:
                vis_emb = torch.einsum('bkqtn,bktn->bqtn', vis_emb, pos_2d)
                vis_emb = vis_emb[:,:,:,:,None] - vis_emb[:,:,:,None,:]
                vis_emb = vis_emb.view(B, -1, T, N)
                pos_emb = pos_emb + vis_emb #(B,C, T, N_view)

                
        x = x.view(x.shape[0], -1, T, N)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        if self.FLAG:
            x = x + pos_emb

        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x
        
        if self.cfg.NETWORK.USE_FEATURE_TRAN:
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                ag1, ag2 = self.p_shrink(x)#(B, dim, dim, T, N_view**2)

                if self.TRAN_FLAG and self.training and self.cfg.TRAIN.USE_ROT_LOSS:
                    self_tran = self.self_tran.repeat(1, 1, 1, 1, N_view**2)
                    mask = self.m[...,:N_view, :N_view].contiguous().view(1, 1, 1, 1, -1).to(device)
                    ag1 = self_tran * mask + ag1 * (1 - mask)
                ag_rot = None

            if self.cfg.NETWORK.USE_GT_TRANSFORM:
                ag1 = None
                ag2 = None
                ag_rot = self.p_shrink_rot(rotation)
        else:
            ag1 = None
            ag2 = None
            ag_rot = None
            
        att = self.att_shrink(x)

        return ag1, ag2, ag_rot, att
class MyConv(nn.Module):
    def __init__(self,cfg, V,channels):
        super().__init__()
        
        self.expand_conv = nn.Conv2d(V, V*cfg.NETWORK.INPUT_DIM *channels, (1, 1),stride = (1, 1), bias = False)

    def forward(self,pos_2d, vis_score):
        conv_p = self.expand_conv(vis_score)#(B, C_1*C_2, T, N)
        B, _,T, N = conv_p.shape
        
        conv_p = conv_p.view(B, pos_2d.shape[1],-1, T, N)
        x = torch.einsum('bcktn, bctn -> bktn', conv_p, pos_2d).contiguous()
        return x 
       
class RotationModel(nn.Module):
    def __init__(self,cfg,  dropout, momentum):
        super().__init__()

        self.cfg = cfg 
        channels = self.cfg.NETWORK.ROT_MODEL.NUM_CHANNELS
        if cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
            self.expand_conv = nn.Conv2d(12, channels, (1, 1),stride = (1, 1), bias = False)
        else:
            self.expand_conv = nn.Conv2d(9, channels, (1, 1),stride = (1, 1), bias = False)

        self.expand_bn = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = self.cfg.NETWORK.ROT_MODEL.NUM_LAYERS
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, self.cfg.NETWORK.TRANSFORM_DIM**2, 1, 1, bias = True)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        B, _,_, T, _, _ = x.shape
        x = x.view(B, x.shape[1] * x.shape[2], T, x.shape[4] * x.shape[5])
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        K = 2
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x
        x = self.shrink(x).view(B, self.cfg.NETWORK.TRANSFORM_DIM, self.cfg.NETWORK.TRANSFORM_DIM, T, -1)

        return x

class PartFeatureModel(nn.Module):
    def __init__(self, cfg, in_N, h_N, dropout = 0.25,momentum = 0.1,is_train = False):
        super().__init__()
        self.cfg = cfg 
        if cfg.NETWORK.CONFIDENCE_METHOD == 'no':
            self.CAT_CONF = False
            self.CONF_MOD = False
        elif cfg.NETWORK.CONFIDENCE_METHOD == 'concat':
            self.CAT_CONF = True
            self.CONF_MOD = False
        elif cfg.NETWORK.CONFIDENCE_METHOD == 'modulate':
            self.CAT_CONF = False
            self.CONF_MOD = True
            
        D = self.cfg.NETWORK.INPUT_DIM + 1 if self.CAT_CONF else self.cfg.NETWORK.INPUT_DIM

        in_channels = in_N * D
        h_D = cfg.NETWORK.TRANSFORM_DIM
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
        if self.CONF_MOD:
            self.vis_conv = MyConv(cfg, V = in_N, channels = h_N * cfg.NETWORK.TRANSFORM_DIM)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d):

        vis_score = pos_2d[:,:,:,-1:] #(B, T, J, C, N)
        pos_2d = pos_2d[:,:,:,:-1]
        
        B, T,V1, C1, N = pos_2d.shape

        pos_2d = pos_2d.permute(0, 2, 3, 1, 4).contiguous() #(B, J, C, T, N)
        pos_2d = pos_2d.view(B, V1 * C1, T, N).contiguous()
        
        vis_score = vis_score.permute(0, 2, 3, 1, 4).contiguous()
        vis_score = vis_score.view(B, V1, T, N).contiguous()
        
        if self.CONF_MOD:
            vis_x = self.vis_conv(pos_2d, vis_score)
        
        if not self.CAT_CONF:
            x = pos_2d
        else:
            x =  torch.cat((pos_2d, vis_score), dim = 1)
        
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        if self.CONF_MOD:
            x = x+ vis_x
        x = x.contiguous()
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, self.cfg.NETWORK.TRANSFORM_DIM, x.shape[-2], x.shape[-1])

class MultiPartFeatureModel(nn.Module):
    def __init__(self, cfg, channels, oup_channels, dropout = 0.25,momentum = 0.1,is_train = False, num_joints = 17):
        super().__init__()
        self.cfg = cfg
        
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
        self.head_model = PartFeatureModel(cfg, in_N = in_channels, h_N = channels // cfg.NETWORK.TRANSFORM_DIM ,dropout = dropout,momentum = momentum,is_train = is_train)
        in_channels = len(self.hand_joint_left_idx)
        self.hand_left_model = PartFeatureModel(cfg, in_N = in_channels, h_N = channels // cfg.NETWORK.TRANSFORM_DIM , dropout = dropout,momentum = momentum, is_train = is_train)
        in_channels = len(self.foot_joint_left_idx)
        self.foot_left_model = PartFeatureModel(cfg, in_N = in_channels, h_N = channels // cfg.NETWORK.TRANSFORM_DIM , dropout = dropout,momentum = momentum,is_train = is_train)
        
        in_channels = len(self.hand_joint_right_idx) 
        self.hand_right_model = PartFeatureModel(cfg, in_N = in_channels, h_N = channels // cfg.NETWORK.TRANSFORM_DIM , dropout = dropout,momentum = momentum, is_train = is_train)
        in_channels = len(self.foot_joint_right_idx)
        self.foot_right_model = PartFeatureModel(cfg, in_N = in_channels, h_N = channels // cfg.NETWORK.TRANSFORM_DIM , dropout = dropout,momentum = momentum,is_train = is_train)
        c = cfg.NETWORK.TRANSFORM_DIM
        self.shrink_conv =nn.Conv2d(channels // cfg.NETWORK.TRANSFORM_DIM * 5 * c, oup_channels, 1, bias = False)
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
        return f.view(f.shape[0], -1, self.cfg.NETWORK.TRANSFORM_DIM, f.shape[-2], f.shape[-1])
class Pose3dShrink(nn.Module):
    def __init__(self, cfg, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
        super().__init__()
        self.conv_1 = nn.Conv2d(N * cfg.NETWORK.TRANSFORM_DIM, channels, 1,stride = (1, 1), bias = False)
        
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
        
class FuseView(nn.Module):
    def __init__(self, cfg, N, dropout = 0.25, is_train = False, T = 9, num_joints = 17):
        super().__init__()
        self.cfg = cfg
        dropout = cfg.NETWORK.M_FORMER.DROPOUT
        D = cfg.NETWORK.TRANSFORM_DIM
        if self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            self.pose_model = VAL(cfg, in_channels = D * N * 2, dropout = dropout, channels = N * cfg.NETWORK.TRANSFORM_DIM, T = T, is_train = is_train, num_joints = num_joints)
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU(inplace = True)
        elif self.cfg.NETWORK.M_FORMER.MODE == 'origin':
            self.pose_model = MultiViewBert(cfg)
        elif self.cfg.NETWORK.M_FORMER.MODE == 'point':
            self.pose_model = PointTransformerLayer(cfg, dim = D * N,pos_mlp_hidden_dim = int(D * N / 2), attn_mlp_hidden_mult = 4)
        self.num_view = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        self.m = torch.zeros(1, 1, 1, 1, self.num_view, self.num_view)
        for i in range(self.num_view):
            self.m[:,:,:,:,i, i] = 1
        self.m = self.m.float()
        
        self.Thr = cfg.NETWORK.MASK_RATE
        if is_train:
            print(self.m[0,0,0,0])
            print('mask rate:{}'.format(self.Thr))
        
    def set_bn_momentum(self, momentum):
        self.pose_model.set_bn_momentum(momentum)
        
    def forward(self, x, pos_2d, rotation = None):
        if self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            return self.mtf_forward(x, pos_2d, rotation)
        elif self.cfg.NETWORK.M_FORMER.MODE == 'origin':
            return self.origin_forward(x, rotation)
        elif self.cfg.NETWORK.M_FORMER.MODE == 'point':
            return self.point_forward(x, pos_2d)
    def point_forward(self, x, pos_2d):
        B, C1, C2, T, N = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous() #(B, T, N, C1, C2)
        x = x.view(B*T, N, C1*C2)
        if self.training and self.Thr > 0:
            mask = torch.rand(B, 1, 1, 1, self.num_view, self.num_view).to(x.device) + self.m.to(x.device)
            mask = mask < self.Thr
            mask = mask[:,0,0,:,:,:,] #(B, 1, N, N)
            mask = mask.repeat(1, T, 1, 1)
            mask = mask.view(B*T, N, N, 1)
        else:
            mask = None
        f = self.pose_model(x, pos_2d, mask) #(B*T, N, C1 * C3)
        f = f.view(B, T, N, C1, C2).permute(0, 3, 4, 1, 2)
        if self.training and self.Thr > 0:
            return f, None, None, None, x, mask.view(B, T, 1, 1, N, N)
        else:
            return f, None, None, None, x, self.m.to(x.device)
    def origin_forward(self, x, rotation = None):
        B, C1, C2, T, N = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous() #(B, T, N, C1, C2)
        x = x.view(B*T, N, C1*C2)
        if self.training and self.Thr > 0:
            mask = torch.rand(B, 1, 1, 1, self.num_view, self.num_view).to(x.device) + self.m.to(x.device)
            mask = mask < self.Thr
            mask = mask[:,0,0,:,:,:,] #(B, 1, N, N)
            mask = mask.repeat(1, T, 1, 1)
            mask = mask.view(B*T, 1, N, N)
        else:
            mask = None
        f = self.pose_model(x, mask) #(B*T, N, C1 * C3)
        f = f.view(B, T, N, C1, C2).permute(0, 3, 4, 1, 2)
        if self.training and self.Thr > 0:
            return f, None, None, None, None, mask.view(B, T, 1, 1, N, N)
        else:
            return f, None, None, None, None, self.m.to(x.device)
        
        
    def mtf_forward(self, x, pos_2d, rotation = None):
        B, C1, C2, T, N = x.shape
        f = x[:,:,:self.cfg.NETWORK.TRANSFORM_DIM].contiguous()
        x1 = x.view(B, C1, C2, T, N, 1).repeat(1, 1, 1, 1, 1, N)
        x2 = x.view(B, C1, C2, T, 1, N).repeat(1, 1, 1, 1, N, 1)
        
        x = torch.cat((x1, x2), dim = 1)

        x = x.view(B, C1 * 2, C2, T, N*N)
        ag1, ag2, ag_rot, att = self.pose_model(x, pos_2d, rotation) #ag:(B, D, D, T, N*N) att: (B, K, D, T, N*N)
        
        tran = None
        tran_gt = None
        if self.cfg.NETWORK.USE_FEATURE_TRAN:
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.training and self.cfg.TRAIN.USE_ROT_LOSS:
                    ag1 = ag1.view(ag1.shape[0], ag1.shape[1], ag1.shape[2], ag1.shape[3], N, N) #(B, 3, 3, T, N, N)
                ag2 = ag2.view(ag2.shape[0], ag2.shape[1], ag2.shape[2], ag2.shape[3], N, N) #(B, 3, 3, T, N, N)
                tran = ag1
                ag = ag2
                
                f_conv = torch.einsum('bnctm, bqctsm -> bnqtsm', f, ag)
                tran_gt = None
            else:
                ag_rot = ag_rot.view(ag_rot.shape[0], ag_rot.shape[1], ag_rot.shape[2], ag_rot.shape[3], N, N) #(B, 3, 3, T, N, N)

                tran = None
                tran_gt = ag_rot
                f_conv = torch.einsum('bnctm, bqcsm -> bnqtsm', f, ag_rot[:,:,:,0])
        else:
            f_conv = x2
            
        att = att.view(att.shape[0], att.shape[1], att.shape[2], att.shape[3], N, N)
            
        if self.training and self.Thr > 0:
            mask = torch.rand(B, 1, 1, 1, self.num_view, self.num_view).to(x.device) + self.m.to(x.device) #(B, C1, C1, T, N, N)
            mask = mask < self.Thr
            att = att.masked_fill(mask, -1e9)
        elif self.cfg.NETWORK.M_FORMER.MASK_SELF == True:
            mask = self.m.to(x.device)[..., :N, :N]
            mask = mask < -0.1
            att = att.masked_fill(mask, -1e9)

        att = F.softmax(att, dim=-1) #(B, K, C, T, N, N)
        
        f_fuse = f_conv * att
        
        f_rcpe = f_fuse
        f_fuse = torch.sum(f_fuse, dim  =-1)

        f_fuse = f_fuse + f
        if self.training and self.Thr > 0:
            return f_fuse, tran, att, tran_gt, f, mask
        else:
            return f_fuse, tran, att, tran_gt, f, self.m.to(x.device)
    

class Pose3dShrinkOther(nn.Module):
    def __init__(self, cfg, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
        super().__init__()
        self.conv_1 = nn.Conv2d(N * cfg.NETWORK.TRANSFORM_DIM, channels, (1, 1),stride = (1, 1),dilation = (3**0, 1), bias = False)
        
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
        x = x.permute(0, 3, 1, 2, 4) #(B, T, J, C, N)
        return x
class RotShrink(nn.Module):
    def __init__(self, cfg, N,channels, dropout = 0.25, momentum = 0.1, dim_joint = 3,is_train = False, num_joints = 17):
        super().__init__()
        self.cfg = cfg 
        self.conv_1 = nn.Conv2d(cfg.NETWORK.TRANSFORM_DIM**2, channels, (1, 1),stride = (1, 1),dilation = (3**0, 1), bias = False)
        
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
        if cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
            self.shrink = nn.Conv2d(channels, 4*3, 1)
        else:
            self.shrink = nn.Conv2d(channels, 3*3, 1)
        
        self.dim_joint = dim_joint
        self.num_joints = num_joints
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        B, _, _, T, N, _ = x.shape
        x = x.view(x.shape[0], -1, T, N**2)
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
        if self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_MODE == 'rt':
            x = x.view(x.shape[0], 4, 3, T,N, N)
        else:
            x = x.view(x.shape[0], 3, 3, T,N, N)
        
        return x
class VideoMultiViewModel(nn.Module):
    def __init__(self, cfg, momentum = 0.1,is_train = False,  num_joints = 17, ):
        super().__init__() 
        self.cfg = cfg
        use_inter_loss = cfg.TRAIN.USE_INTER_LOSS
        assert cfg.NETWORK.CONFIDENCE_METHOD in ['no', 'concat', 'modulate']
       
        if is_train:
            print('dim: {}'.format(cfg.NETWORK.TRANSFORM_DIM))
        self.T = cfg.NETWORK.TEMPORAL_LENGTH
        
        self.f_model = MultiPartFeatureModel(cfg, channels = cfg.NETWORK.NUM_CHANNELS//2 ,oup_channels = cfg.NETWORK.NUM_CHANNELS, dropout = cfg.NETWORK.DROPOUT,momentum = momentum, is_train = is_train, num_joints = num_joints)
        N = cfg.NETWORK.NUM_CHANNELS // cfg.NETWORK.TRANSFORM_DIM
        if self.cfg.NETWORK.USE_MFT:
            self.fuse_model = FuseView(cfg, N,dropout = cfg.NETWORK.DROPOUT, is_train = is_train, T = self.T, num_joints = num_joints)
        

        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(cfg.NETWORK.DROPOUT)
        self.shrink = BERT(cfg, T = cfg.NETWORK.TEMPORAL_LENGTH, dropout=0.1, num_joints = num_joints)
        if is_train and use_inter_loss:
            self.shrink_1 = Pose3dShrinkOther(cfg, N = N, channels = cfg.NETWORK.NUM_CHANNELS,dropout = cfg.NETWORK.DROPOUT, dim_joint = 3, is_train = is_train, num_joints = num_joints)
            if self.cfg.NETWORK.USE_MFT:
                self.shrink_2 = Pose3dShrinkOther(cfg, N = N, channels = cfg.NETWORK.NUM_CHANNELS, dropout = cfg.NETWORK.DROPOUT, dim_joint = 3, is_train = is_train, num_joints = num_joints)
        self.use_inter_loss = use_inter_loss
        
        if self.cfg.NETWORK.USE_FEATURE_TRAN and is_train and self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.training:
                    self.tran_shrink = RotShrink(cfg, N = N, channels = cfg.NETWORK.NUM_CHANNELS, dropout = cfg.NETWORK.DROPOUT, dim_joint = 3, is_train = is_train, num_joints = num_joints)
            else:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                    self.tran_shrink = RotShrink(cfg, N = N, channels = cfg.NETWORK.NUM_CHANNELS, dropout = cfg.NETWORK.DROPOUT, dim_joint = 3, is_train = is_train, num_joints = num_joints)
   
    def set_bn_momentum(self, momentum):
        ####f_model
        self.f_model.set_bn_momentum(momentum)
        ####fuse_model
        if self.cfg.NETWORK.USE_MFT:
            self.fuse_model.set_bn_momentum(momentum)
        ####shrink, shrink_1, shrink_2
        self.shrink.set_bn_momentum(momentum)
        if self.training and self.use_inter_loss:
            self.shrink_1.set_bn_momentum(momentum)
            if self.cfg.NETWORK.USE_MFT:
                self.shrink_2.set_bn_momentum(momentum)
        ####tran_shrink  
        if self.cfg.NETWORK.USE_FEATURE_TRAN and self.training and self.cfg.NETWORK.M_FORMER.MODE == 'mtf':
            if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.training:
                    self.tran_shrink.set_bn_momentum(momentum=momentum)
            else:
                if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                    self.tran_shrink.set_bn_momentum(momentum=momentum)
   
    def forward(self, pos_2d, rotation = None):
        B, T,V, C, N = pos_2d.shape
         
        pos_2d = pos_2d.contiguous()

        f = self.f_model(pos_2d) #(B, K, D, T, N)   

        if self.training and self.use_inter_loss:
            out_1 = self.shrink_1(f[:,:,:self.cfg.NETWORK.TRANSFORM_DIM].contiguous())
        tran = None
        rot = None
        f_fuse_before = f
        
        if self.cfg.NETWORK.USE_MFT:
            f, tran, att, tran_rot, f_tmp_rcpe, mask = self.fuse_model(f, pos_2d, rotation)
            if self.training and self.use_inter_loss:
                out_2 = self.shrink_2(f)
            if self.cfg.NETWORK.USE_FEATURE_TRAN and self.cfg.NETWORK.M_FORMER.MODE == 'mtf' and self.training:
                if not self.cfg.NETWORK.USE_GT_TRANSFORM:
                    if self.cfg.TRAIN.USE_ROT_LOSS:
                        rot = self.tran_shrink(tran)
                    else:
                        rot = None
                else:
                    if self.cfg.TRAIN.USE_ROT_LOSS and self.cfg.NETWORK.M_FORMER.GT_TRANSFORM_RES:
                        rot = self.tran_shrink(tran)
                    else:
                        rot = None  
            else:
                rot = None
        out = self.shrink(f)

        if self.training and self.use_inter_loss:
            return out, [out_1, out_2] if self.cfg.NETWORK.USE_MFT else [out_1], tran, rot 
        else:
            return out, [f_fuse_before]

        
