import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .video_multi_view_refine import VideoMultiViewModelRefine
from common.set_seed import *
from common.bert_model.bert import *

from common.smpl.SMPL import *
from common.smpl.lbs import *
set_seed()
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
DIM = 3
N_K = 1
GROUP = 1
NUM_VIEW =5

TLEN = 9

class Convert(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25,momentum = 0.1,is_train = False):
        super().__init__()
        if is_train:
            self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1),stride = (1, 1), bias = False)
        else:
            self.expand_conv = nn.Conv2d(in_channels, channels, (1, 1),stride = (1, 1),dilation = (1, 1), bias = False)
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
        self.shrink = nn.Conv2d(channels, 24 * 3, 1, bias = True)
        
        
        
        h36m_jregressor = np.load('./data/model_files/J_regressor_h36m.npy')

        self.learn_h36m_jregressor = torch.from_numpy(h36m_jregressor).float()#nn.Parameter(torch.from_numpy(h36m_jregressor).float())
        self.smpl = SMPL_layer(
                    './data/model_files/basicModel_f_lbs_10_207_0_v1.0.0.pkl',
                    dtype=torch.float32
                )
        init_shape = np.load('./data/model_files/h36m_mean_beta.npy')
        self.betas = torch.from_numpy(init_shape).view(1, -1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos_3d):

        B, T,V1, C1, N = pos_3d.shape
        pos_3d = pos_3d.permute(0, 2, 3, 1, 4).contiguous()
        pos_3d = pos_3d.view(B, V1 * C1, T, N).contiguous()
        
       
        x = pos_3d
  
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        K = 2
        for i in range(self.num_layers): 
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i + 1](x))))
            x = res + x
        out = self.shrink(x)#(B, C, T, N)
        out = out.view(B, 24, 3, T, N).permute(0, 3, 4, 1, 2).contiguous()
        out = out.view(-1, 24, 3).contiguous()
        Thr = 3.14
        pred_angle = torch.norm(out, dim = -1, keepdim = True)
        arg = (pred_angle > Thr).detach()
        arg = torch.cat((arg, arg, arg), dim = -1)
        max = Thr
        out[arg] = out[arg] / max * Thr
        output, smpl_out,vertices = self.smpl(
            pose_skeleton=out,
            betas=self.betas.repeat(out.shape[0],1).float().to(out.device),
            J_regressor_h36m = self.learn_h36m_jregressor.to(out.device),
            return_verts=True
        )

        output = output.view(B, T, N, 17, 3).permute(0, 1, 3, 4, 2).contiguous()
        smpl_out = smpl_out.view(B, T, N, 24, 3).permute(0, 1, 3, 4, 2).contiguous()
        vertices = vertices.view(B, T, N, -1, 3).permute(0, 1, 3, 4, 2).contiguous()
#         angle = torch.norm(out, dim  =-1)
#         print(torch.min(angle), torch.max(angle))
#         angle = angle.view(B, T, N, 24)[-1, 0, 0]
#         print(angle)
#         save_vertices(vertices[-1,0,:,:,0])
       
        if self.training:
            return output, out
        else:
            return output, smpl_out,vertices
