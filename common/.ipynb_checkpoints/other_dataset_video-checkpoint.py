import torch
import torch.nn as nn
import sys, os
import numpy as np
import pickle
import copy
from torch.utils.data import Dataset
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '../')
sys.path.insert(0, lib_path)
from human_aug import *
from get_angle import get_angle
from h36m_dataset import Human36mDataset
from common.camera import *
class BoneLength:
    def __init__(self, Num = 0):
        self.human_bone_sub1 = np.array([[0.,0.13294859,0.44289458,0.45420644,0.13294883,0.4428944,0.4542066,0.23347826,0.2570777,0.12113493,0.11500223,0.15103422,0.27888277,0.25173345,0.15103145,0.27889293,0.25172868]])
        self.human_bone_sub5 = np.array([[0,0.11931363,0.4282862,0.4424442,0.11931279,0.4282858,0.4424442,0.22430354,0.2540556,0.11711778,0.11499855,0.14309624,0.26458478,0.24862035,0.14309771,0.26458365,0.24862036]])
        self.human_bone_sub6 = np.array([[0,0.14261378,0.48657095,0.46149364,0.14261167,0.4865638,0.461494,0.26221436,0.26000926,0.11939894,0.11500003,0.14937478,0.3010083,0.25791448,0.1493742,0.30100805,0.2579143]])
        self.human_bone_sub7 = np.array([[0,0.13587892,0.44861463,0.43801025,0.1358788, 0.44861427,0.43800855,0.22624777,0.25540757,0.10714753,0.11500315,0.13971765,0.27556357,0.24729861,0.13971417,0.27557185,0.2472981]])
        self.human_bone_sub8 = np.array([[0,0.14653784,0.45214748,0.43863332,0.14653659,0.45214623,0.43863323,0.261215,0.25102425,0.12045857,0.11500029,0.16931632,0.2899085,0.24417701,0.16931592,0.2899073,0.24417748]])
        self.human_bone = np.concatenate((self.human_bone_sub1, self.human_bone_sub5,self.human_bone_sub6, self.human_bone_sub7, self.human_bone_sub8), axis = 0)
        self.Num = Num
        self.init_db()
        self.bones_id = np.array([[0,0],[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]])
        self.child_id = self.bones_id[:,1]
        self.par_id = self.bones_id[:,0]
        self.joint_bone_chain = np.array([
		[0, 0, 0, 0, 0,],
		[0, 0, 0, 0, 1,],
		[0, 0, 0, 1, 2,],
		[0, 0, 1, 2, 3,],
		[0, 0, 0, 0, 4,],
		[0, 0, 0, 4, 5,],
		[0, 0, 4, 5, 6,],
		[0, 0, 0, 0, 7,],
		[0, 0, 0, 7, 8,],
		[0, 0, 7, 8, 9,],
		[0, 7, 8, 9, 10],
		[0, 0, 7, 8, 11,],
		[0, 7, 8, 11, 12],
		[7, 8, 11, 12, 13],
		[0, 0, 7, 8, 14],
		[0, 7, 8, 14, 15],
		[7, 8, 14, 15, 16]
])
    def init_db(self):
        self.db = []
        for i in range(self.human_bone.shape[0]):
            self.db.append(self.human_bone[i][np.newaxis])
        for i in range(self.Num):
            idx1 = np.random.randint(0, len(self.db))
            while True:
                idx2 = np.random.randint(0, len(self.db))
                if idx2 != idx1:
                    break
            bone1 = self.db[idx1]
            bone2 = self.db[idx2]
            bone = (bone1 + bone2) / 2
            self.db.append(bone)
        self.db = np.concatenate(self.db, axis = 0)
        print('db:', self.db.shape)
    def get_bone_length(self, batch_size = 1,num = 1):
        n = batch_size * num
        idx = np.random.randint(0, self.db.shape[0], (n))
        bone = self.db[idx]
        bone = bone.reshape(batch_size, num, -1)
        return bone
    def update_pose(self, pose, bone_length = None):
        pose = np.array(pose)
        flag = 0
        if bone_length is not None:
            bone_length = np.array(bone_length)
            
        if len(pose.shape) == 3:
            pose  = pose[np.newaxis]
            flag = 1
            if bone_length is not None:
                bone_length = bone_length[np.newaxis]
        
        pose_root = copy.deepcopy(pose[:,:,:1])
        pose -= pose_root
        
        B, T, V, C = pose.shape
        if bone_length == None:
            bone_length = self.get_bone_length(batch_size = B, num = T)
        if bone_length.shape[-1] != 1:
            bone_length = bone_length[...,np.newaxis]
        bone = pose[:,:,self.child_id] - pose[:,:,self.par_id]
        bone_dir = bone / (np.linalg.norm(bone, axis = -1, keepdims = True) + 1e-6)
        joint_dir = bone_dir[:,:,self.joint_bone_chain]
        joint_len = bone_length[:,:,self.joint_bone_chain]
        joint = joint_dir * joint_len
        joint = np.sum(joint, axis = 3)
        joint += pose_root
        if flag and joint.shape[0] == 1:
            joint = joint[0]
        return joint  
    
class OtherDataSetVideo(Dataset):
    def __init__(self, T = 1):
        pths  =  ['../data/train_bone_direction.pth',
                 ]
        self.pths = pths
        self.ratio = [1]
        self.T = T
        self.selected_bone = [1,2,3, 4,5,6, 10, 11, 12,13, 14,15,16]
        self.selected_bone_2 = [1,2, 4,5, 11,12, 14,15, 7]
        
        dataset_path = '../data/data_3d_h36m.npz'
        
        dataset = Human36mDataset(dataset_path)
        
        self.pose_root_db = []
        for subject in dataset.subjects():
            if subject == 'S9' or subject == 'S11':
                continue
            print(subject)
            
            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                if 'positions' in anim:
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d[:, 1:] -= pos_3d[:, :1]
                
                        self.pose_root_db.append(pos_3d[:,:1])
        self.pose_root_db = np.concatenate(self.pose_root_db, axis = 0)
        
        self.human_bone_length = BoneLength(Num = 0)
        print(self.ratio)
#         if self.T > 1:
#             model_pth = './checkpoint/single_model_baseline.pkl'
#             self.model = torch.load(model_pth)
#             self.model.eval()
        self.init_db()
        self.N_mpi_lsp = 0
        
    def init_db(self):
        print('init_db')
        self.datas = [None] * len(self.pths)
        self.length = [0] * len(self.pths)
        self.pairs = [None] * len(self.pths)#[[]]*n这个慎用
        for n_d, pth in enumerate(self.pths):
            if self.ratio[n_d] < 0.001:
                continue
            with open(pth, 'rb') as f:
                data = pickle.load(f)
            self.datas[n_d] = data
            self.pairs[n_d] = []

           
            for i in range(len(data)):
                length = data[i].shape[0] -self.T + 1
                self.length[n_d] += length
                for j in range(length):
                    self.pairs[n_d].append([i, j, j + self.T])
            
            for i in range(len(data)):
                pose = copy.deepcopy(data[i])

                pose -= pose[:,:1]
                idx_root = np.random.randint(0, self.pose_root_db.shape[0], pose.shape[0])
                pose_root = self.pose_root_db[idx_root]
                pose += pose_root

                pose = self.human_bone_length.update_pose(pose)
                human = VideoHuman(pose)
                pos_2d = human.get_cam_2d()
                bone_pre, bone_pre2 = get_angle(pose)
                cam3d = copy.deepcopy(pose)
                cam3d[:,1:] -= cam3d[:,:1]

                if self.T > 1:
#                     inp = torch.from_numpy(pos_2d).float()
#                     bone_inp = np.concatenate((bone_pre[:,self.selected_bone], bone_pre2[:,self.selected_bone_2]), axis = 1)
#                     bone_inp = torch.from_numpy(bone_inp).float()
#                     out = self.model(inp, bone_inp)
#                     if isinstance(out, list):
#                         out = out[-1]
#                     out = out.detach().cpu().numpy()
                    out = copy.deepcopy(cam3d)
                    out[:,:1] = 0

                    p = np.concatenate((cam3d, pos_2d, bone_pre, bone_pre2, out), axis = -1)
                else:
                    p = np.concatenate((cam3d, pos_2d, bone_pre, bone_pre2), axis = -1)
                self.datas[n_d][i] = p

    def __len__(self):
        return np.max(self.length)
    def __getitem__(self, idx):
        p = np.random.rand()
        tmp = 0
        for n_d in range(len(self.ratio)):
            tmp += self.ratio[n_d]
            if p <= tmp:
                pairs = self.pairs[n_d]
                data = self.datas[n_d]
                idx = idx % self.length[n_d]
                break
        pair = pairs[idx]
        
        seq, s_id, e_id = pair
        data = copy.deepcopy(data[seq][s_id:e_id])

        cam3d = data[:,:,:3]
        pos2d = data[:,:,3:5]
        bone_pre = data[:,:,5:7][:,self.selected_bone]
        bone_pre2 = data[:,:,7:9][:,self.selected_bone_2]
  
        if self.T>1:
            pose_3d = data[:,:,9:]
            return cam3d, pos2d, bone_pre, bone_pre2, pose_3d
        return cam3d, pos2d, bone_pre, bone_pre2
    

if __name__ == '__main__':
    dataset = OtherDataSetVideo(T = 10)
    print(len(dataset))
    for i in dataset:
        pass
    
    #cam_3d, pos2d, bone_pre, bone_pre2 = dataset[10000]

    
    
