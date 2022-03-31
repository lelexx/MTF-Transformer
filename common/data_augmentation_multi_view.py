import numpy as np
import torch
import copy
import sys
import random
from set_seed import *
set_seed()

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
    def __init__(self,cfg, add_view):
        self.cam_aug = Camera()
        self.add_view = add_view
        self.cfg = cfg
 
    def change(self,pos_gt_3d,pos_tmp, R):       
        pos_gt_3d[:,:1] = pos_gt_3d[:,:1] + pos_tmp
        pos_root = pos_gt_3d[:,:1]

        pos_gt_3d[:,1:] += pos_root

        cam_r = pos_gt_3d - pos_root
        cam_r_R = np.matmul(cam_r, R)


        cam_r_R = cam_r_R + pos_root
        cam_r_R = torch.from_numpy(cam_r_R).float()

        if self.cfg.DATA.DATASET_NAME == 'h36m':
            f = 1000.0
            w_2 = 400
        elif self.cfg.DATA.DATASET_NAME == 'total_cap':
            f = 1000.0
            w_2 = 800.0 
        
        pos_r_gt = cam_r_R[:,:,:2] / cam_r_R[:,:,-1:]* f / w_2

        cam_r_R[:,1:] = cam_r_R[:,1:] - cam_r_R[:,:1]

        return cam_r_R, pos_r_gt

    
    def __call__(self, pos_gt_3d = None, pos_gt_2d = None):
        B, T, V, _, NUM_VIEW = pos_gt_2d.shape
        pos_gt_2d = pos_gt_2d.view(B*T, V, -1, NUM_VIEW)
        pos_gt_3d = pos_gt_3d.view(B*T, V, -1, NUM_VIEW)
        
        
        pos_gt_3d_copy = copy.deepcopy(pos_gt_3d)
        pos_gt_2d_copy = copy.deepcopy(pos_gt_2d)
        
        view_idx = np.random.randint(0, NUM_VIEW)
        N = self.add_view

        for view_idx in range(N):
            view_idx = np.random.randint(NUM_VIEW)
            pos_gt_3d_tmp = pos_gt_3d[:,:,:,view_idx].cpu().numpy()
            
            pos_tmp =  (np.random.rand(B,1, 3) - 0.5) * 2
            pos_tmp = pos_tmp.repeat(T, axis = 0)
            pos_tmp[:,:,:2] = pos_tmp[:,:,:2] / 10
            R = self.cam_aug.getT(B)
            R = R.repeat(T, axis = 0)
      
            cam_3d, pos_r_gt = self.change(pos_gt_3d_tmp,pos_tmp,R)

            cam_3d = cam_3d.unsqueeze(-1)
            pos_r_gt = pos_r_gt.unsqueeze(-1)
            
            pos_gt_2d_copy = torch.cat((pos_gt_2d_copy, pos_r_gt), dim = -1)
            pos_gt_3d_copy = torch.cat((pos_gt_3d_copy, cam_3d.cuda()), dim = -1)
        pos_gt_2d_copy = pos_gt_2d_copy.reshape(B, T, V, 2, -1)
        pos_gt_3d_copy = pos_gt_3d_copy.reshape(B, T, V, 3, -1)
        return pos_gt_2d_copy, pos_gt_3d_copy
    


        
