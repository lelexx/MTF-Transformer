# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
BN_MOMENTUM = 0.1

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
def get_joint(head_p, hand_p, foot_p):
    B, _, _ = head_p.shape
    pos = torch.zeros(B, 17, 3).float().cuda()
    common = (head_p[:,:len(common_joint_idx)] + hand_p[:, :len(common_joint_idx)] + foot_p[:, :len(common_joint_idx)]) / 3
    pos[:,common_joint_idx] = common
    head = head_p[:, -2:]
    hand = hand_p[:, -4:]
    foot = foot_p[:,-4:]
    pos[:,hand_joint_idx[-4:]] = hand
    pos[:,head_joint_idx[-2:]] = head
    pos[:,foot_joint_idx[-4:]] = foot
    return pos
def single_to_video(pre, N = 50, K = 7):
    NF = N
    
    out = copy.deepcopy(pre)#(F, C1, C2)
    B, C1, C2 = out.shape
    try:
        for i in range(int(out.shape[0] / NF + 1)):
            st = i*NF
            end = (i+1)*NF if out.shape[0] > (i+1) * NF else out.shape[0]
            if (st == end or st == out.shape[0]):
                break
            F = NF if out.shape[0] > (i+1) * NF else out.shape[0] - i*NF
            x = np.arange(F)
            fixed_bases = [np.ones([F]) * np.sqrt(0.5)]
            for i in range(1, K):
                fixed_bases.append(np.cos(i * np.pi * ((x + 0.5) / F)))
            fixed_bases = np.array(fixed_bases)
            bases_tmp = torch.from_numpy(fixed_bases).float().cuda()
            out_tmp = copy.deepcopy(out[st:end].view(F, -1))#(F, C1, C2)

            out_tmp = out_tmp.permute(1, 0)

            out_tmp = torch.matmul(out_tmp, torch.transpose(bases_tmp, 0, 1)) / F * 2

            out_tmp = torch.matmul(out_tmp, bases_tmp)

            out_tmp = out_tmp.view(C1, C2, F)
            out_tmp = out_tmp.permute(2, 0, 1)
            out[st:end] = out_tmp
            if out.shape[0] <= (i + 1) * NF:
                break
    except:
        print(st, end, out.shape[0])
        sys.exit()
    return out

class LeftRightModel(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,):
        super().__init__()
        self.expand_conv_left = nn.Conv1d(in_channels, channels, 1, bias = False)
        self.expand_bn_left = nn.BatchNorm1d(channels, momentum=momentum)
        self.expand_conv_right = nn.Conv1d(in_channels, channels, 1, bias = False)
        self.expand_bn_right = nn.BatchNorm1d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.num_layers = 1
        self.N = 2
        
        conv_layers_left = []
        bn_layers_left = []
        for i in range(self.num_layers * self.N):
            conv_layers_left.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers_left.append(nn.BatchNorm1d(channels, momentum=momentum))
            conv_layers_left.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers_left.append(nn.BatchNorm1d(channels, momentum=momentum))
        self.conv_layers_left = nn.ModuleList(conv_layers_left)
        self.bn_layers_left = nn.ModuleList(bn_layers_left)
        conv_layers_right = []
        bn_layers_right = []
        for i in range(self.num_layers * self.N):
            conv_layers_right.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers_right.append(nn.BatchNorm1d(channels, momentum=momentum))
            conv_layers_right.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers_right.append(nn.BatchNorm1d(channels, momentum=momentum))
        self.conv_layers_right = nn.ModuleList(conv_layers_right)
        self.bn_layers_right = nn.ModuleList(bn_layers_right)
        self.shrink_conv =nn.Conv1d(channels * 2, channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm1d(channels, momentum = momentum) 
          
    def set_bn_momentum(self, momentum):
        self.expand_bn_left.momentum = momentum
        for bn in self.bn_layers_left:
            bn.momentum = momentum
        self.expand_bn_right.momentum = momentum
        for bn in self.bn_layers_right:
            bn.momentum = momentum
        self.shrink_bn.momentum = momentum
    def forward(self, pos_2d_left,pos_2d_right, bone_angle_left, bone_angle_right):
        B, V1, C1 = pos_2d_left.shape
        B, V2, C2 = bone_angle_right.shape
        
        pos_2d_left = pos_2d_left.view(B, V1 * C1, 1).contiguous()
        bone_angle_left = bone_angle_left.view(B, V2 * C2, 1).contiguous()
        x_left = torch.cat((pos_2d_left, bone_angle_left), dim = 1)
        x_left = self.drop(self.relu(self.expand_bn_left(self.expand_conv_left(x_left))))
 
        for i in range(self.num_layers * self.N): 
            res_left = x_left
            f_left = self.drop(self.relu(self.bn_layers_left[2 * i](self.conv_layers_left[2 * i](x_left))))
            x_left = self.drop(self.relu(self.bn_layers_left[2 * i + 1](self.conv_layers_left[2 * i + 1](f_left))))
            
            x_left = res_left + x_left
        
        pos_2d_right = pos_2d_right.view(B, V1 * C1, 1).contiguous()
        bone_angle_right = bone_angle_right.view(B, V2 * C2, 1).contiguous()
        x_right = torch.cat((pos_2d_right, bone_angle_right), dim = 1)
        x_right = self.drop(self.relu(self.expand_bn_right(self.expand_conv_right(x_right))))

 
        for i in range(self.num_layers * self.N): 
            res_right = x_right
            f_right = self.drop(self.relu(self.bn_layers_right[2 * i](self.conv_layers_right[2 * i](x_right))))
            x_right = self.drop(self.relu(self.bn_layers_right[2 * i + 1](self.conv_layers_right[2 * i + 1](f_right))))             
            x_right = res_right + x_right
        x = torch.cat((x_left, x_right), dim = 1)
        x = self.drop(self.relu(self.shrink_bn(self.shrink_conv(x))))
                            
        return x
class Model(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channels, channels, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.num_layers = 1
        self.N = 2
        
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers * self.N):
            conv_layers.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum=momentum))
            conv_layers.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
          
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        B, V1, C1 = pos_2d.shape
        B, V2, C2 = bone_angle.shape
        
        pos_2d = pos_2d.view(B, V1 * C1, 1).contiguous()
        bone_angle = bone_angle.view(B, V2 * C2, 1).contiguous()
        x = torch.cat((pos_2d, bone_angle), dim = 1)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        outs = []
 
        for i in range(self.num_layers * self.N): 
            res = x
            f = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2 * i + 1](self.conv_layers[2 * i + 1](f))))
            x = res + x
        return x
class SingleGroupModel(nn.Module):
    def __init__(self, in_channels, channels,oup_channels = 3, dropout = 0.25,momentum = 0.1):
        super().__init__()
        in_channels = len(head_joint_idx) * 2 + len(head_bone_idx) * 2
        
        self.head_model = Model(in_channels = in_channels, channels = channels,dropout = dropout,momentum = momentum,)
        in_channels = len(hand_joint_left_idx) * 2 + len(hand_bone_left_idx) * 2
        self.hand_left_model = Model(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum, )
        in_channels = len(foot_joint_left_idx) * 2 + len(foot_bone_left_idx) * 2
        self.foot_left_model = Model(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum,)
        
        in_channels = len(hand_joint_right_idx) * 2 + len(hand_bone_right_idx) * 2
        self.hand_right_model = Model(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum, )
        in_channels = len(foot_joint_right_idx) * 2 + len(foot_bone_right_idx) * 2
        self.foot_right_model = Model(in_channels = in_channels, channels = channels, dropout = dropout,momentum = momentum,)
        
        self.shrink_conv =nn.Conv1d(channels * 5, channels, 1, bias = False)
        self.shrink_bn = nn.BatchNorm1d(channels, momentum = momentum)
        self.num_layers = 0
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channels,channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = momentum))
            conv_layers.append(nn.Conv1d(channels, channels, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channels, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
                             
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv1d(channels, 17 * oup_channels, 1)
    def set_bn_momentum(self, momentum):
        self.head_model.set_bn_momentum(momentum)
        self.hand_left_model.set_bn_momentum(momentum)
        self.foot_left_model.set_bn_momentum(momentum)
        self.hand_right_model.set_bn_momentum(momentum)
        self.foot_right_model.set_bn_momentum(momentum)
        self.shrink_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_2d, bone_angle):
        head = self.head_model(pos_2d[:,head_joint_idx], bone_angle[:,head_bone_idx])
        hand_left = self.hand_left_model(pos_2d[:,hand_joint_left_idx], bone_angle[:, hand_bone_left_idx])
        foot_left = self.foot_left_model(pos_2d[:,foot_joint_left_idx], bone_angle[:, foot_bone_left_idx])
        hand_right = self.hand_right_model(pos_2d[:,hand_joint_right_idx], bone_angle[:, hand_bone_right_idx])
        foot_right = self.foot_right_model(pos_2d[:,foot_joint_right_idx], bone_angle[:, foot_bone_right_idx])
        f =torch.cat((head, hand_left,hand_right, foot_left, foot_right), dim = 1)
        f = self.drop(self.relu(self.shrink_bn(self.shrink_conv(f))))
        for i in range(self.num_layers):
            res = f
            f = self.drop(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](f))))
            f = self.drop(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](f))))
            f = f + res
        
        out = self.shrink(f)
        out = out.view(-1, 17, 3)
        return out

    
 
        
        
        
