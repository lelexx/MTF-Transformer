import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

head_joint_idx = [0, 1, 4, 7, 8, 11, 14, 9, 10]
hand_joint_idx = [0, 1, 4, 7, 8, 11, 14, 12, 13, 15, 16]
foot_joint_idx = [0, 1, 4, 7, 8, 11, 14,  2, 3, 5, 6, ]
hand_joint_left_idx = [0, 1, 4, 7, 8, 11, 14, 12, 13]
hand_joint_right_idx = [0, 1, 4, 7, 8, 11, 14, 15, 16]
foot_joint_right_idx = [0, 1, 4, 7, 8, 11, 14,  2, 3, ]
foot_joint_left_idx = [0, 1, 4, 7, 8, 11, 14,  5, 6, ]
common_joint_idx = [0, 1, 4, 7, 8, 11, 14]
head_bone_idx = [0, 3, 7, 10, 6]
hand_bone_idx = [0, 3, 7, 10, 8, 9, 11, 12, 17, 18, 19, 20]
hand_bone_left_idx  = [0, 3, 7, 10, 8, 9,  17, 18, ]
hand_bone_right_idx  = [0, 3, 7, 10,11, 12,  19, 20]

foot_bone_idx = [0, 3, 7, 10, 1, 2, 4, 5, 13, 14, 15, 16]
foot_bone_left_idx = [0, 3, 7, 10,  4, 5, 15, 16]
foot_bone_right_idx = [0, 3, 7, 10, 1, 2, 13, 14,]
BN_MOMENTUM = 0.1
DIM = 5
N_K = 1
GROUP = 1
NUM_VIEW =5
class PoseShrink(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(in_channels, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv2d(in_channels, DIM * DIM, 1)
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
        
    def forward(self, x):
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        p1  =self.shrink(x)
        p1 = p1.view(p1.shape[0], DIM, DIM, 1, -1)
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
        self.sigmoid = nn.Sigmoid()
    def set_bn_momentum(self, momentum):
        self.bn_1.momentum = momentum
    def forward(self, x):
        x = self.drop(self.relu(self.bn_1(self.conv_1(x))))
        x  =self.shrink(x)
        x = x.view(x.shape[0], -1, DIM, 1, x.shape[-1])
        return x
class Pose(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25, momentum = 0.1):
        super().__init__()
        h_channels = channels // 2
        self.expand_conv = nn.Conv2d(in_channels, h_channels, 1, bias = False)
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
        self.p_shrink = PoseShrink(in_channels = h_channels, channels = channels)
        self.att_shrink = AttnShrink(in_channels = h_channels, channels = channels, out_channels = channels)
        
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

class FeatureModel(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,):
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels, channels, 1, bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 2
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False,))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
          
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        B, V1, C1, N = pos_2d.shape
        B, V2, C2, N = bone_angle.shape
        
        pos_2d = pos_2d.view(B, V1 * C1, 1, N).contiguous()
        bone_angle = bone_angle.view(B, V2 * C2, 1, N).contiguous()
        x = torch.cat((pos_2d, bone_angle), dim = 1)

        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, DIM, x.shape[-2], x.shape[-1])
class PartFeatureModel(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,):
        super().__init__()
        print('group:{}'.format(GROUP))
        self.expand_conv = nn.Conv2d(in_channels, channels, 1, bias = False)
        self.expand_bn = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 2
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False, groups = GROUP))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
          
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        B, V1, C1, N = pos_2d.shape
        pos_2d = pos_2d.view(B, V1 * C1, 1, N).contiguous()
        if bone_angle is not None:
            B, V2, C2, N = bone_angle.shape
            bone_angle = bone_angle.view(B, V2 * C2, 1, N).contiguous()
            x = torch.cat((pos_2d, bone_angle), dim = 1)
        else:
            x = pos_2d
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x

        return x.view(x.shape[0], -1, DIM, x.shape[-2], x.shape[-1])
class MultiPartFeatureModel(nn.Module):
    def __init__(self, in_channels, channels, oup_channels, dropout = 0.25,momentum = 0.1,):
        super().__init__()
        print(channels, oup_channels)
        N_bone = 0
        in_channels = len(head_joint_idx) * 2 + len(head_bone_idx) * N_bone
        
        self.head_model = PartFeatureModel(in_channels = in_channels, channels = channels,dropout = dropout,momentum = momentum,)
        in_channels = len(hand_joint_left_idx) * 2 + len(hand_bone_left_idx) * N_bone
        self.hand_left_model = PartFeatureModel(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum, )
        in_channels = len(foot_joint_left_idx) * 2 + len(foot_bone_left_idx) * N_bone
        self.foot_left_model = PartFeatureModel(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum,)
        
        in_channels = len(hand_joint_right_idx) * 2 + len(hand_bone_right_idx) * N_bone
        self.hand_right_model = PartFeatureModel(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum, )
        in_channels = len(foot_joint_right_idx) * 2 + len(foot_bone_right_idx) * N_bone
        self.foot_right_model = PartFeatureModel(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum,)
        
        self.shrink_conv =nn.Conv2d(channels * 5, oup_channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm2d(oup_channels, momentum = momentum)
    
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
#         self.fuse_head = FuseView(channels = channels)
#         self.fuse_hand_left = FuseView(channels = channels)
#         self.fuse_foot_left = FuseView(channels = channels)
#         self.fuse_hand_right = FuseView(channels = channels)
#         self.fuse_foot_right = FuseView(channels = channels)
        
          
    def set_bn_momentum(self, momentum):
        self.head_model.set_bn_momentum(momentum)
        self.hand_left_model.set_bn_momentum(momentum)
        self.foot_left_model.set_bn_momentum(momentum)
        self.hand_right_model.set_bn_momentum(momentum)
        self.foot_right_model.set_bn_momentum(momentum)
        self.shrink_bn.momentum = momentum
#         self.fuse_head.set_bn_momentum(momentum)
#         self.fuse_hand_left.set_bn_momentum(momentum)
#         self.fuse_hand_right.set_bn_momentum(momentum)
#         self.fuse_foot_left.set_bn_momentum(momentum)
#         self.fuse_foot_right.set_bn_momentum(momentum)
    def forward(self, pos_2d, bone_angle):
        head = self.head_model(pos_2d[:,head_joint_idx], bone_angle[:,head_bone_idx] if bone_angle is not None else None)
        hand_left = self.hand_left_model(pos_2d[:,hand_joint_left_idx], bone_angle[:, hand_bone_left_idx] if bone_angle is not None else None)
        foot_left = self.foot_left_model(pos_2d[:,foot_joint_left_idx], bone_angle[:, foot_bone_left_idx] if bone_angle is not None else None)
        hand_right = self.hand_right_model(pos_2d[:,hand_joint_right_idx], bone_angle[:, hand_bone_right_idx] if bone_angle is not None else None)
        foot_right = self.foot_right_model(pos_2d[:,foot_joint_right_idx], bone_angle[:, foot_bone_right_idx] if bone_angle is not None else None)
        ##########
#         head = self.fuse_head(head)
#         hand_left = self.fuse_hand_left(hand_left)
#         hand_right = self.fuse_hand_right(hand_right)
#         foot_left = self.fuse_foot_left(foot_left)
#         foot_right = self.fuse_foot_right(foot_right)
        ##########
        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        
        f = f.view(f.shape[0], -1, f.shape[3], f.shape[4])

        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        return f.view(f.shape[0], -1, DIM, f.shape[-2], f.shape[-1])
class Pose3dShrink(nn.Module):
    def __init__(self, channels, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.conv_1 = nn.Conv2d(channels, channels, 1, bias = False)
        self.bn_1 = nn.BatchNorm2d(channels, momentum = momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.num_layers = 0
        
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
            conv_layers.append(nn.Conv2d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm2d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Conv2d(channels, 17 * 3, 1)
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
        x  =self.shrink(x)
        x = x.view(x.shape[0], -1, 3, N)
        
        return x
class FuseView(nn.Module):
    def __init__(self, channels, dropout = 0.25):
        super().__init__()
        self.pose_model = Pose(in_channels = channels* 2,  channels = channels)
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
        f = x
        x1 = x.view(B, C1, C2, T, N, 1).repeat(1, 1, 1, 1, 1, N)
        x2 = x.view(B, C1, C2, T, 1, N).repeat(1, 1, 1, 1, N, 1)
        x = torch.cat((x1, x2), dim = 1).view(B, C1 * 2, C2, T, N*N)
        p, att = self.pose_model(x)
        p = p.view(p.shape[0], p.shape[1], p.shape[2], p.shape[3], N, N)
        att = att.view(att.shape[0], att.shape[1], att.shape[2], att.shape[3], N, N)

        f_conv = torch.einsum('bnctm, bqctsm -> bnqtsm', f, p)
        if self.training:
            #mask = torch.rand(att.shape[0], 1, 1, 1, NUM_VIEW, NUM_VIEW).to(x.device) + self.m.to(x.device)
            mask = torch.rand(1, 1, 1, 1, NUM_VIEW, NUM_VIEW).to(x.device) + self.m.to(x.device)
            #mask = torch.rand(1, 1, 1, 1, NUM_VIEW, NUM_VIEW).to(x.device)
            mask = mask < self.Thr
            att = att.masked_fill(mask, -1e9)
        else:
            pass

        att = F.softmax(att, dim=-1)
        f_conv = f_conv.view(f_conv.shape[0], N_K,f_conv.shape[1] // N_K, f_conv.shape[2], f_conv.shape[3], f_conv.shape[4], f_conv.shape[5])
        f_fuse = torch.einsum('benctsm, bnctsm -> bencts', f_conv, att).contiguous()
        f_fuse = f_fuse.view(f_fuse.shape[0], -1, f_fuse.shape[3], f_fuse.shape[4], f_fuse.shape[5])

        f_fuse = f_fuse + f
        
        return f_fuse
class SingleMultiViewModel(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,):
        super().__init__() 
        global DIM
        print('dim: {}'.format(DIM))
        #self.f_model = FeatureModel(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum)
        self.f_model = MultiPartFeatureModel(in_channels = in_channels, channels = channels // 2,oup_channels = channels, dropout = dropout,momentum = momentum)
        
        self.fuse_model = FuseView(channels = channels)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = Pose3dShrink(channels = channels)
          
    def set_bn_momentum(self, momentum):
        self.f_model.set_bn_momentum(momentum)
        self.fuse_model.set_bn_momentum(momentum)
        
        self.shrink.set_bn_momentum(momentum)
        
    def forward(self, pos_2d, bone_angle):
        f = self.f_model(pos_2d, bone_angle)
        f = self.fuse_model(f)
        out = self.shrink(f)

        return out
