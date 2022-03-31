import numpy as np
import torch
import copy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sys.path.append('/home/wulele/code/bone_pos_3d/Angle3dPose')
sys.path.append('/home/wulele/code/bone_pos_3d/Angle3dPose/common')
from human_aug import *
import random
from set_seed import *


class Camera:
    def __init__(self):
        super().__init__()
        self.angle_y = [-np.pi, np.pi]
        self.angle_x = [-0.2 * np.pi, 0.2 * np.pi]
        self.angle_z = [-0.2 * np.pi, 0.2 * np.pi]
    def getT(self,size):
        angle_y = np.random.uniform(self.angle_y[0], self.angle_y[1], size)
        angle_x = np.random.uniform(self.angle_x[0], self.angle_x[1], size)
        angle_z = np.random.uniform(self.angle_z[0], self.angle_z[1], size)
        
        sin_y = np.sin(angle_y)
        cos_y = np.cos(angle_y)
        
        sin_x = np.sin(angle_x)
        cos_x = np.cos(angle_x)
        
        sin_z = np.sin(angle_z)
        cos_z = np.cos(angle_z)
        T_x = np.zeros((size, 3, 3))
        T_x[:,0, 0] = 1
        T_x[:,1, 1] = cos_x
        T_x[:,2, 1] = sin_x
        T_x[:,1, 2] = -sin_x
        T_x[:,2, 2] = cos_x
        
        T_y = np.zeros((size, 3, 3))
        T_y[:,1, 1] = 1
        T_y[:,0, 0] = cos_y
        T_y[:,2, 0] = -sin_y
        T_y[:,0, 2] = sin_y
        T_y[:,2, 2] = cos_y
        
        T_z = np.zeros((size, 3, 3))
        T_z[:,2, 2] = 1
        T_z[:,0, 0] = cos_z
        T_z[:,1, 0] = sin_z
        T_z[:,0, 1] = -sin_z
        T_z[:,1, 1] = cos_z
        
        T = np.matmul(np.matmul(T_x, T_z), T_y)
        
        return T
    
class DataAug():
    def __init__(self, add_view):
        self.common_mask_p = 0.15
        self.hand_mask_p = 0.3
        self.foot_mask_p = 0.2
        self.cam_aug = Camera()
        self.link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])
        self.par = self.link[:,0]
        self.child = self.link[:,1]
        self.angle_split_6 = np.array([[-90,-60], [-60,-30],[-30,0],[0, 30] ,[30, 60], [60,90]])
        self.angle_class_6 = np.array([-0.6, -0.4, -0.2, 0.2, 0.4, 0.6])

        self.angle_split_7 = np.array([[-90, -75], [-75, -45], [-45, -15], [-15, 15], [15, 45], [45, 75],[75, 90]])
        self.angle_class_7 = np.array([-0.6, -0.4, -0.2, 0.1, 0.2,0.4, 0.6 ])
        self.selected_bone = [1,2,3, 4,5,6, 10, 11, 12,13, 14,15,16]
        self.selected_bone_2 = [1,2, 4,5, 11,12, 14,15,]
        self.add_view = add_view
 
    def change(self,pos_gt_3d, pos_gt_2d,pos_tmp, R):       
        pos_gt_3d[:,:1] = pos_gt_3d[:,:1] + pos_tmp
        pos_root = pos_gt_3d[:,:1]
        cam = copy.deepcopy(pos_gt_3d)
        cam[:,1:] += pos_root

        cam_r = cam - pos_root
        cam_r_R = np.matmul(cam_r, R)
        
        cam_r_R = cam_r_R + pos_root
        
        f =1000
        pos_r_gt = cam_r_R[:,:,:2] / cam_r_R[:,:,-1:]* f / 400

        pos_r_gt = torch.from_numpy(pos_r_gt).float()
        cam_3d = copy.deepcopy(human_0.cam_3d)

        cam_3d = cam_3d - cam_3d[:,:1]

        cam_3d = torch.from_numpy(cam_3d).float()

        return cam_3d, pos_r_gt

    
    def __call__(self, pos_gt_3d = None, pos_gt_2d = None):
        flag = 0
        if len(pos_gt_2d.shape) == 5:
            flag = 1
            B, T, V, _, NUM_VIEW = pos_gt_2d.shape
            pos_gt_2d = pos_gt_2d.view(B*T, V, -1, NUM_VIEW)
            pos_gt_3d = pos_gt_3d.view(B*T, V, -1, NUM_VIEW)
        elif len(pos_gt_2d.shape) == 4:
            flag = 0
            T = 1
        
        pos_gt_3d_copy = copy.deepcopy(pos_gt_3d)
        pos_gt_2d_copy = copy.deepcopy(pos_gt_2d)
        
        view_idx = np.random.randint(0, NUM_VIEW)
        N = self.add_view

        for view_idx in range(N):
            view_idx = np.random.randint(NUM_VIEW)
            pos_gt_3d_tmp = pos_gt_3d[:,:,:,view_idx].cpu().numpy()
            pos_gt_2d_tmp = pos_gt_2d[:,:,:,view_idx].cpu().numpy()
            pos_root = pos_gt_3d_tmp[:,:1]
            
            pos_tmp =  (np.random.rand(B,1, 3) - 0.5) * 2
            pos_tmp = pos_tmp.repeat(T, axis = 0)
            pos_tmp[:,:,:2] = pos_tmp[:,:,:2] / 10
            R = self.cam_aug.getT(B)
            R = R.repeat(T, axis = 0)
      
            cam_3d, pos_r_gt = self.change(pos_gt_3d_tmp, pos_gt_2d_tmp,pos_tmp,R)

            cam_3d = cam_3d.unsqueeze(-1)
            pos_r_gt = pos_r_gt.unsqueeze(-1)
            
            pos_gt_2d_copy = torch.cat((pos_gt_2d_copy, pos_r_gt), dim = -1)
            pos_gt_3d_copy = torch.cat((pos_gt_3d_copy, cam_3d.cuda()), dim = -1)
        if flag:
            pos_gt_2d_copy = pos_gt_2d_copy.reshape(B, T, V, 2, -1)
            pos_gt_3d_copy = pos_gt_3d_copy.reshape(B, T, V, 3, -1)
        return pos_gt_2d_copy, pos_gt_3d_copy
    

        
