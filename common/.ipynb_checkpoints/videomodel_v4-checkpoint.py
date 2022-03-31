import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import sys
import copy
#import torch.cuda.amp as amp
BN_MOMENTUM = 0.1
link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])
par = link[:,0]
child = link[:,1]
selected_bone = [1,2,3, 4,5,6, 7,8,9,10, 11, 12,13, 14,15,16]
def get_angle(cam_3d_tmp):
    cam_3d_tmp = cam_3d_tmp.cpu().numpy()
    bone_direction_gt = cam_3d_tmp[:,:,child] - cam_3d_tmp[:,:,par]
                
    bone_direction_gt = bone_direction_gt / (np.linalg.norm(bone_direction_gt, axis = -1, keepdims= True) + 1e-6)
    return torch.from_numpy(bone_direction_gt[:,:,1:,:]).float().cuda()


class TemporalModel(nn.Module):
    def __init__(self, in_channels, filter_widths, dropout=0.25, channels=1024,):

        super().__init__()
        other_channel = channels // 2
        self.expand_conv = nn.Conv1d(in_channels, channels, filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        
        self.expand_conv_angle = nn.Conv1d(in_channels, other_channel, filter_widths[0], bias=False)
        self.expand_bn_angle = nn.BatchNorm1d(other_channel, momentum = 0.1)
        self.pad = [ filter_widths[0] // 2 ]
        layers_conv = []
        layers_bn = []
        layers_conv_angle = []
        layers_bn_angle = []
        
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.causal_shift = [0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append( 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] ,
                                         dilation=next_dilation,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels + other_channel, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            layers_conv_angle.append(nn.Conv1d(other_channel, other_channel,
                                         filter_widths[i] ,
                                         dilation=next_dilation,
                                         bias=False))
            layers_bn_angle.append(nn.BatchNorm1d(other_channel, momentum=0.1))
            layers_conv_angle.append(nn.Conv1d(other_channel, other_channel, 1, dilation=1, bias=False))
            layers_bn_angle.append(nn.BatchNorm1d(other_channel, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
        self.layers_conv_angle = nn.ModuleList(layers_conv_angle)
        self.layers_bn_angle = nn.ModuleList(layers_bn_angle)
        self.shrink_angle = nn.Conv1d(other_channel, 16 * 3, 1, bias = True)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        self.expand_bn_angle.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
        for bn in self.layers_bn_angle:
            bn.momentum = momentum
    def forward(self, pos_3d, pos_2d, bone_angle):
        B,T, V1, C1 = pos_2d.shape
        B,T, V2, C2 = bone_angle.shape
        B,T, V3, C3 = pos_3d.shape
        
        pos_3d = pos_3d.view(B, T, V3 * C3).contiguous()
        pos_2d = pos_2d.view(B, T, V1 * C1).contiguous()
        bone_angle = bone_angle.view(B,T, V2 * C2,).contiguous()
        x = torch.cat((pos_3d,pos_2d, bone_angle), dim = -1)
        x = x.permute(0, 2, 1).contiguous()
        x_angle = self.drop(self.relu(self.expand_bn_angle(self.expand_conv_angle(x))))
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        angles = []
        angle_tmp = self.shrink_angle(x_angle)#(B, 16 * 3, T)
        B, C, T = angle_tmp.shape
        angle_tmp = angle_tmp.permute(0, 2, 1).contiguous().view(B,T, 16, 3)
        angles.append(angle_tmp)
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res_angle = x_angle[:, :, pad + shift : x_angle.shape[2] - pad + shift]
            
            x_angle = self.drop(self.relu(self.layers_bn_angle[2*i](self.layers_conv_angle[2*i](x_angle))))
            x_angle = res_angle + self.drop(self.relu(self.layers_bn_angle[2*i + 1](self.layers_conv_angle[2*i + 1](x_angle))))
            angle_tmp = self.shrink_angle(x_angle)
            B, C, T = angle_tmp.shape
            angle_tmp = angle_tmp.permute(0, 2, 1).contiguous().view(B, T,16, 3)
            angles.append(angle_tmp)
            
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = torch.cat((x, x_angle), dim = 1)
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        return x, angles
    
class TemporalModelOptimized1f(nn.Module):
    def __init__(self,in_channels,filter_widths, dropout=0.25, channels=1024):
        super().__init__()
        other_channel = channels // 2
        self.expand_conv = nn.Conv1d(in_channels, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum = 0.1)
        
        self.expand_conv_angle = nn.Conv1d(in_channels, other_channel, filter_widths[0], stride=filter_widths[0], bias=False)
        self.expand_bn_angle = nn.BatchNorm1d(other_channel, momentum = 0.1)
        self.pad = [ filter_widths[0] // 2 ]
        layers_conv = []
        layers_bn = []
        
        layers_conv_angle = []
        layers_bn_angle = []
        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout(dropout)
        self.causal_shift = [ 0 ]
        self.filter_widths = filter_widths
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append( 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels + other_channel, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            layers_conv_angle.append(nn.Conv1d(other_channel, other_channel, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn_angle.append(nn.BatchNorm1d(other_channel, momentum=0.1))
            layers_conv_angle.append(nn.Conv1d(other_channel, other_channel, 1, dilation=1, bias=False))
            layers_bn_angle.append(nn.BatchNorm1d(other_channel, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
        self.layers_conv_angle = nn.ModuleList(layers_conv_angle)
        self.layers_bn_angle = nn.ModuleList(layers_bn_angle)
        self.shrink_angle = nn.Conv1d(other_channel, 16 * 3, 1, bias = True)
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        self.expand_bn_angle.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
        for bn in self.layers_bn_angle:
            bn.momentum = momentum
    def forward(self,pos_3d,  pos_2d, bone_angle):
        B,T, V1, C1 = pos_2d.shape
        B,T, V2, C2 = bone_angle.shape
        B,T, V3, C3 = pos_3d.shape
        
        pos_3d = pos_3d.view(B, T, V3 * C3).contiguous()
        pos_2d = pos_2d.view(B, T, V1 * C1).contiguous()
        bone_angle = bone_angle.view(B,T, V2 * C2,).contiguous()
        x = torch.cat((pos_3d, pos_2d, bone_angle), dim = -1)
        x = x.permute(0, 2, 1).contiguous()
        x_angle = self.drop(self.relu(self.expand_bn_angle(self.expand_conv_angle(x))))
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        angles = []
        angle_tmp = self.shrink_angle(x_angle)
        B, C, T = angle_tmp.shape
        angle_tmp = angle_tmp.permute(0, 2, 1).contiguous().view(B,T, 16, 3)
        angles.append(angle_tmp)
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            res_angle = x_angle[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x_angle = self.drop(self.relu(self.layers_bn_angle[2*i](self.layers_conv_angle[2*i](x_angle))))
            x_angle = res_angle + self.drop(self.relu(self.layers_bn_angle[2*i + 1](self.layers_conv_angle[2*i + 1](x_angle))))
            angle_tmp = self.shrink_angle(x_angle)
            B, C, T = angle_tmp.shape
            angle_tmp = angle_tmp.permute(0, 2, 1).contiguous().view(B, T,16, 3)
            angles.append(angle_tmp)
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = torch.cat((x, x_angle), dim = 1)
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
    
        print(x.shape)
        exit()
        return x, angles


class MyVideoTrainModel(nn.Module):
    def __init__(self,filter_widths,  dropout=0.25, channels=1024):
        super().__init__()
        in_channels = 17 * 5 + 21 * 2
        
        self.model = TemporalModelOptimized1f(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        
        self.shrink = nn.Conv1d(channels, 17 * 3, 1)
    def set_bn_momentum(self, momentum):
        self.model.set_bn_momentum(momentum)
    def forward(self, pos_3d, pos_2d, bone_angle):
        f, angles = self.model(pos_3d,pos_2d, bone_angle)
        out = self.shrink(f)
        out = out.view(-1, 17, 3)
        return out, angles
class MyVideoTestModel(nn.Module):
    def __init__(self,filter_widths,  dropout=0.25, channels=1024):
        super().__init__()
        in_channels = 17 * 5 + 21 * 2
        
        self.model = TemporalModel(in_channels = in_channels, channels = channels,dropout = dropout, filter_widths = filter_widths)
        
        self.shrink = nn.Conv1d(channels, 17 * 3, 1)
    def set_bn_momentum(self, momentum):
        self.model.set_bn_momentum(momentum)
    def forward(self, pos_3d, pos_2d, bone_angle):
        f, angles = self.model(pos_3d,pos_2d, bone_angle)
        out = self.shrink(f)
        B,C, T = out.shape
        out = out.view(-1, 17, 3, T)
        out = out.permute(0, 3, 1, 2)
        return out, angles

