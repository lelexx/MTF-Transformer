import torch.nn as nn
import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

BN_MOMENTUM = 0.1
DIM = 8
class SingleMultiViewModel(nn.Module):
    def __init__(self, in_channels, channels, dropout = 0.25, momentum = 0.1):
        super().__init__()
        view_channel = channels
        pose_channel = channels
        view_oup_channel = 100
        pose_oup_channel = channels
        print(view_channel, pose_channel, view_oup_channel,pose_oup_channel)
        
        self.backbone = BackBone(in_channel = in_channels, channel = channels)
        self.viewbranch = ViewBranch(in_channel = channels, channel = view_channel, oup_channel = view_oup_channel)
        self.pose3dbranch = Pose3DBranch(in_channel = channels, channel = pose_channel, oup_channel = pose_oup_channel)
        self.multiviewattntion = MultiViewAttention(in_channel = view_oup_channel*2, channel =channels, out_channel = pose_channel)
        self.shrink = Shrink(in_channel = view_oup_channel + pose_oup_channel, channel =channels, out_channel = 17 * 3)
        self.fusemultiview  = FuseMultiView()
        
    def set_bn_momentum(self, momentum):
        self.backbone.set_bn_momentum(momentum)
        self.viewbranch.set_bn_momentum(momentum)
        self.pose3dbranch.set_bn_momentum(momentum)
        self.multiviewattntion.set_bn_momentum(momentum)
        self.shrink.set_bn_momentum(momentum)
        self.fusemultiview.set_bn_momentum(momentum)
    def forward(self, pos_2d, bone_angle, pos_2d_other_view, bone_angle_other_view):
        B, V1, C1 = pos_2d.shape
        B, V2, C2 = bone_angle.shape
        
        pos_2d = pos_2d.view(B, V1 * C1, 1).contiguous()
        bone_angle = bone_angle.view(B, V2 * C2, 1).contiguous()
        
        pos_2d_other_view = pos_2d_other_view.view(B, V1 * C1, 1).contiguous()
        bone_angle_other_view = bone_angle_other_view.view(B, V2 * C2, 1).contiguous()
        inp_view1 = torch.cat((pos_2d, bone_angle), dim = 1)
        inp_view2 = torch.cat((pos_2d_other_view, bone_angle_other_view), dim = 1)
        
        f_view1 = self.backbone(inp_view1)
        f_view2 = self.backbone(inp_view2)
        
        x_view1 = self.viewbranch(f_view1)
        pos3d_view1 = self.pose3dbranch(f_view1)
        x_view2 = self.viewbranch(f_view2)
        pos3d_view2 = self.pose3dbranch(f_view2)
        
        att = self.multiviewattntion(x_view1, x_view2)
        pose = self.fusemultiview(pos3d_view1, pos3d_view2, att)
        oup_view1 = self.shrink(pose, x_view1)
        oup_view2 = self.shrink(pose, x_view2)
        
        oup_view1 = oup_view1.view(oup_view1.shape[0], 17, 3)
        oup_view2 = oup_view2.view(oup_view2.shape[0], 17, 3)

        return oup_view1, oup_view2
        
        
class BackBone(nn.Module):
    def __init__(self, in_channel, channel, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channel, channel, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channel, momentum = momentum)
        self.num_layers = 1
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        '''
        x:(B, C, T)
        '''
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2*i+1](self.conv_layers[2 * i + 1](x))))
            x = res + x
        return x
class ViewBranch(nn.Module):
    def __init__(self, in_channel, channel,oup_channel, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channel, channel, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channel, momentum = momentum)
        self.num_layers = 1
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.shrink = nn.Conv1d(channel, oup_channel, 1)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        '''
        x:(B, C, T)
        '''
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2*i+1](self.conv_layers[2 * i + 1](x))))
            x = res + x
        x = self.shrink(x)
        return x

class Pose3DBranch(nn.Module):
    def __init__(self, in_channel, channel,oup_channel, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channel, channel, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channel, momentum = momentum)
        self.num_layers = 1
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.shrink = nn.Conv1d(channel, oup_channel, 1)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, x):
        '''
        x:(B, C, T)
        '''
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2*i+1](self.conv_layers[2 * i + 1](x))))
            x = res + x
        x = self.shrink(x)
        return x

class MultiViewAttention(nn.Module):
    def __init__(self, in_channel, channel, out_channel, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channel, channel, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channel, momentum = momentum)
        self.num_layers = 1
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv1d(channel, out_channel, 1)
        self.sigmoid = nn.Sigmoid()
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, view_1, view_2):
        '''
        view_i:(B, C, T)
        '''
        x = torch.cat((view_1, view_2), dim = 1)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2*i+1](self.conv_layers[2 * i + 1](x))))
            x = res + x
        x = self.sigmoid(self.shrink(x))
        return x

class Shrink(nn.Module):
    def __init__(self, in_channel, channel, out_channel, dropout = 0.25, momentum = 0.1):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channel, channel, 1, bias = False)
        self.expand_bn = nn.BatchNorm1d(channel, momentum = momentum)
        self.num_layers = 0
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
            conv_layers.append(nn.Conv1d(channel, channel, 1, bias = False))
            bn_layers.append(nn.BatchNorm1d(channel, momentum = momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv1d(channel, out_channel, 1)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, pos3d, view):
        '''
        (B,C, T)
        '''
        x = torch.cat((pos3d, view), dim= 1)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = self.drop(self.relu(self.bn_layers[2*i+1](self.conv_layers[2 * i + 1](x))))
            x = res + x
        x = self.shrink(x)
        return x
        

class FuseMultiView(nn.Module):
    def __init__(self):
        super().__init__()
    def set_bn_momentum(self, momentum):
        pass
    def forward(self, view_1, view_2, att):
        '''
        view_i:(B,C, T)
        '''
        view = view_1 * att +view_2 * (1 - att)
        return view
        
