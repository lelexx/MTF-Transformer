import numpy as np
import torch
import errno
from common.camera import *
from common.loss import *
from common.arguments import parse_args
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import copy
import math
from common.model import *
from common.loss import *

link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])
par = link[:,0]
child = link[:,1]
selected_bone = [1,2,3, 4,5,6, 10, 11, 12,13, 14,15,16]
selected_bone_2 = [1,2, 4,5, 11,12, 14,15]
import matplotlib.pyplot as plt

from common.h36m_dataset import Human36mDataset

ClassNum = 12
K = 3
JointGroup = [[0,1,4,7,8,9,10,11,14],[2,3,5,6],  [12,13,15,16]]

JointIndex = []
for i in JointGroup:
    JointIndex.extend(i) 
print(JointIndex)
JointID = list(range(len(JointIndex)))
for i, j in enumerate(JointIndex):
    JointID[j] = i

Refine = 0
data = {}

left_joint = [4, 5, 6, 11, 12, 13]
right_joint = [1, 2, 3, 14, 15, 16]
left_bone = [3, 4, 5, 7, 8, 9]
right_bone = [0, 1, 2, 10, 11, 12]
#上，下，左，右，左对称，右对称
skeleton_graph = [  
    [0,7, 0, 4, 1, 0, 0], 
    [1, 1, 2, 0, 1, 4, 1],
    [2, 1, 3, 2, 2, 5, 2], 
    [3, 2, 3, 3, 3, 6, 3],
    [4, 4, 5, 4, 0, 4, 1],
    [5, 4, 6, 5, 5, 5, 2],
    [6, 5, 6, 6, 6, 6, 3],
    [7, 8, 0, 7, 7, 7, 7],
    [8, 9, 7, 11, 14, 8, 8],
    [9, 10, 8, 9, 9, 9, 9],
    [10, 10, 9, 10, 10, 10, 10],
    [11, 11, 11, 12, 8, 11, 14],
    [12, 12, 12, 13, 11, 12, 15],
    [13, 13, 13, 13, 12, 13, 16],
    [14, 14, 14, 8, 15, 11, 14],
    [15, 15, 15, 14, 16, 12, 15],
    [16, 16, 16, 15, 16, 13, 16],
]
#本身， 上， 下， 左， 右， 左对称， 右对称
id_label = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
idx_em = torch.zeros(1, 1, 17, 17)
for i in range(17):
    k = skeleton_graph[i]

    for j in range(len(k)):
        idx_em[0,0,i,k[j]] = id_label[j]
    idx_em[0,0,i,k[0]] = id_label[0]

def getKCS(pos3d):
    flag = 0
    if pos3d.__class__ == np.ndarray:
        pos3d = torch.from_numpy(pos3d)
    if len(pos3d.shape) == 3:
        pos3d = pos3d[np.newaxis]
        flag = 1
    B, T, V, C = pos3d.shape
    assert V == 17 and C == 3
    par_joint = pos3d[:,:,par[1:]]#(B, T, V -1, C)
    child_joint = pos3d[:,:,child[1:]]
    bone = child_joint - par_joint
    kcs = torch.matmul(bone, bone.permute(0, 1, 3, 2)) #(B, T, V -1, V -1)
    if flag == 1 and B == 1:
        kcs = kcs[0]
    return kcs

def getTKCS(pos3d):
    flag = 0
    if pos3d.__class__ == np.ndarray:
        pos3d = torch.from_numpy(pos3d)
    if len(pos3d.shape) == 3:
        pos3d = pos3d[np.newaxis]
        flag = 1
    
    B, T, V, C = pos3d.shape
    pos3d_tmp = torch.zeros(B,T + 2, V, C)
    pos3d_tmp[:,1:-1] = pos3d
    pos3d_tmp[:,:1] = pos3d[:,:1]
    pos3d_tmp[:,-1] = pos3d[:,-1]
    assert V == 17 and C == 3
    kcs = getKCS(pos3d_tmp)
    t1kcs = kcs[:,1:-1] - kcs[:,:-2]
    t2kcs = kcs[:,1:-1] - kcs[:,2:]
    if flag == 1 and B == 1:
        t1kcs = t1kcs[0]
        t2kcs = t2kcs[0]
    return t1kcs, t2kcs
    
def get_skeleton(pos3d, pos2d):
    flag = 0
    if pos3d.__class__ == np.ndarray:
        pos3d = torch.from_numpy(pos3d)
    if pos2d.__class__ == np.ndarray:
        pos2d = torch.from_numpy(pos2d)  
    if len(pos3d.shape) == 3:
        pos3d = pos3d.unsqueeze(0)
        pos2d = pos2d.unsqueeze(0)
        flag = 1
    
    B, T, V, C = pos3d.shape
    #
    CS = 7 * 3 + 4 * (1 + 3) + 17
    
    graph = copy.deepcopy(skeleton_graph)
    graph = torch.tensor(graph).long()
   
    joint = pos3d[:,:,graph.view(-1), :]
    joint = joint.view(B, T, V, -1, C)
    self_joint = joint[:,:,:,:1]
    link_joint = joint[:,:,:,1:5]
    link_bone = link_joint - self_joint
    link_bone_len = torch.norm(link_bone, dim = -1, keepdim = True) 
    link_bone_dir = link_bone / (link_bone_len + 1e-6)
    position_em = torch.zeros(B, T, V, V)
    position_em[:,:,] = idx_em[0,0]
    
    all = []
    all.append(pos2d)
    all.append(self_joint)
    all.append(link_joint)
    all.append(link_bone)
    all.append(link_bone_len)
    all.append(link_bone_dir)
    all.append(position_em)
    for i in range(len(all)):
        assert all[i].shape[0] == B and all[i].shape[1] == T and all[i].shape[2] == V
        all[i] = all[i].view(B, T, V, -1)
    all = torch.cat(all, dim = -1)
    if B == 1 and flag == 1:
        all = all[0]
    
    return all
    
v_b = [] 
def get_k(pre, N = 51, K = 7):
    B, V, C = pre.shape
    assert N % 2 == 1 and V == 17
    out = torch.zeros(B, V, C *K)
    out_err = torch.zeros(B, V, C)
    pre_tmp = torch.zeros(B + N -1, V, C)
    pad = (N - 1) // 2
    pre_tmp[:pad] = pre[:1]
    pre_tmp[pad:-pad] = pre
    pre_tmp[-pad:] = pre[-1:]
    fixed_bases = [np.ones([N]) * np.sqrt(0.5)]
    x = np.arange(N)
    for i in range(1, K):
        fixed_bases.append(np.cos(i * np.pi * ((x + 0.5) / N)))
    fixed_bases = np.array(fixed_bases)
    bases_tmp = torch.from_numpy(fixed_bases).float()
    bases_tmp = bases_tmp.permute(1,0)#(N,K)
    for i in range(B):
        pos = pre_tmp[i:i+N]#(N, V, C)
        pos = pos.view(N, -1).permute(1, 0)#(V*C, N)
        
        tmp = torch.matmul(pos, bases_tmp) / N * 2#(V * C, K), 
        
        sys.exit()
        r_pos = torch.matmul(tmp, bases_tmp.permute(1, 0))
        r_pos = r_pos.view(V, C, N)
        r_pos = r_pos[:,:,pad]
        out_err[i] = pos.view(V, C, N)[:,:,pad] - r_pos
        tmp = tmp.view(1, V, C*K)
        out[i] = tmp
       
    return out, out_err
if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,'
    model = torch.load('/home/wulele/Angle3dPose/checkpoint/single_model_g3_b6_att_aug_21angle.pkl')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    keypoints = {}
    for sub in [7, 8, 9, 11,1,5,6]:
        keypoint = np.load('data/h36m_bone_16_sub{}_class67_cpn_21angle-Copy2.npz'.format(sub), allow_pickle=True)
        metadata = keypoint['metadata'].item()
        keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]
    for sub in keypoints.keys():
        print(sub)
        sub_id = sub.split('S')[-1]
        sub_id = int(sub_id)
        for act in keypoints[sub].keys():
            for cam in range(4):
                data = torch.from_numpy(keypoints[sub][act][cam]).float()
                inputs_2d_pre = data[...,2:4]
                bone_angle_pre = data[:,selected_bone, ][:,:,[11, 13]]
                bone_angle_pre2 = data[:,selected_bone_2, ][:,:,[12, 14]]

                inp = inputs_2d_pre
                bone_inp = torch.cat((bone_angle_pre, bone_angle_pre2), dim = 1)
                out = model(inp, bone_inp)
                out = out[-1][:,JointID].detach().cpu()
                #r = get_skeleton(out, inputs_2d_pre)
                pos2d_gt = data[..., :2]
                pos3d_gt = data[...,4:7]
                
                keypoints[sub][act][cam] = torch.cat((pos2d_gt, inp.cpu(),pos3d_gt, out), dim = -1).numpy()
        pth = 'data/h36m_sub{}_agcn.npz'.format(sub_id)
        np.savez(pth, positions_2d = keypoints[sub],metadata = metadata)       
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
