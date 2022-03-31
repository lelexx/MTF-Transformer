import os, sys
import numpy as np
import torch
import pickle
import copy


h36m_cameras_intrinsic_params = [
    {
        'res_w': 1000,
        'res_h': 1002,
    },
    {
        'res_w': 1000,
        'res_h': 1000,
    },
    {
        'res_w': 1000,
        'res_h': 1000,
    },
    {
        'res_w': 1000,
        'res_h': 1002,
    },
]
       
keypoints = {}

vis_score = pickle.load(open('../data/score.pkl', 'rb'))
for sub in [1, 5, 6, 7, 8, 9, 11]:
    keypoint = np.load('../data/h36m_sub{}.npz'.format(sub), allow_pickle=True)
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoint = keypoint['positions_2d'].item()['S{}'.format(sub)]
    keypoint_ada_path = '../data/h36m_adafuse/sub_{}.pkl'.format(sub)
    with open(keypoint_ada_path, 'rb') as f:
        keypoint_adafuse = pickle.load(f)[sub]
    keypoint_ada_before = copy.deepcopy(keypoint)
    keypoint_ada_fuse = copy.deepcopy(keypoint)
    vis_ada = copy.deepcopy(vis_score)

    for act, act_data in keypoint.items():
        n_frames = len(act_data[0]) #(T, J, 7) (p2d_gt, p2d_pre, p3d)
        print('sub:{} act:{} n_frames:{}'.format(sub, act, n_frames))
        ada_act_data = keypoint_adafuse[act] #(T, N, 5, J) (p2d_fuse, p2d_before, conf)
        ada_act_data = ada_act_data.permute(0, 1, 3, 2) #(T, N, J, 5)
        N_view  = len(act_data)
        for view_id in range(N_view):
            vis_name = 'S{}_{}.{}'.format(sub, act, view_id)
            vis_data = vis_score[vis_name] #(T, J)
            n_frames = min(n_frames, vis_data.shape[0])
        for view_id in range(N_view):
            vis_name = 'S{}_{}.{}'.format(sub, act, view_id)
            
            vis_data = vis_score[vis_name] #(T, J)
            
            res_h = h36m_cameras_intrinsic_params[view_id]['res_h']
            res_w = h36m_cameras_intrinsic_params[view_id]['res_w']
            ada_act_data[:,view_id, :, 2:4] = ada_act_data[:,view_id, :, 2:4] /res_w*2 - torch.FloatTensor([1, res_h/res_w])
            ada_act_data[:,view_id, :, 0:2] = ada_act_data[:,view_id, :, 0:2] /res_w*2 - torch.FloatTensor([1, res_h/res_w])
            vis_data_ada = ada_act_data[:,view_id, :, -1][:n_frames]
            
            vis_ada[vis_name] = vis_data_ada

            keypoint_ada_before[act][view_id][:,:, 2:4] = ada_act_data[:,view_id, :, 2:4]
            keypoint_ada_fuse[act][view_id][:,:,2:4] = ada_act_data[:,view_id, :, 0:2]
            keypoint_ada_before[act][view_id] = keypoint_ada_before[act][view_id][:n_frames]
            keypoint_ada_fuse[act][view_id] = keypoint_ada_fuse[act][view_id][:n_frames]
        
    
    np.savez('../data/h36m_sub{}_ada_before.npz'.format(sub), positions_2d = {'S{}'.format(sub):keypoint_ada_before},metadata = keypoints_metadata)
    np.savez('../data/h36m_sub{}_ada_fuse.npz'.format(sub), positions_2d = {'S{}'.format(sub):keypoint_ada_fuse},metadata = keypoints_metadata)
    with open('../data/vis_ada.pkl', 'wb') as f:
        pickle.dump(vis_ada, f)
            
            
        