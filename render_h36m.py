import numpy as np
import itertools
from common.arguments import parse_args
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import copy
from common.camera import *
from common.video_multi_view import *
from common.loss import *
from common.generators_copy import *
from common.data_augmentation_multi_view import *
from time import time
from common.utils import deterministic_random
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from common.other_dataset import OtherDataSet
from torch.utils.data import Dataset,DataLoader
from common.h36m_dataset import Human36mDataset
from common.set_seed import *
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
set_seed()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = parse_args()
link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])
par = link[:,0]
child = link[:,1]
color_left = 'B'
color_right = 'G'
color_other = 'R'
bone_color = [color_right, color_right, color_right, color_left, color_left, color_left, color_other, color_other, color_other, color_other, color_left, color_left, color_left, color_right, color_right, color_right]

RGB_color= {'B':(0, 0, 255), 'G':(0, 255, 0), 'R':(255, 0, 0)}

POS = 0
selected_bone = [1,2,3, 4,5,6, 10, 11,12,13, 14,15,16]
selected_bone_2 = [1,2, 4,5, 11,12, 14,15]
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
dataset = Human36mDataset(dataset_path)
Dataset = 'cpn'
print('Loading 2D detections...')
print('You are using Dataset {}!'.format(Dataset))
keypoints_metadata = None
keypoints_symmetry = None
keypoints = {}

for sub in [1, 5, 6, 7, 8, 9, 11]:
    keypoint = np.load('data/h36m_bone_16_sub{}_class67_cpn_21angle-Copy2_change_2d.npz'.format(sub), allow_pickle=True)
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]

kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
subjects_train = args.subjects_train.split(',')

if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]
N_frame_action_dict = {
2699:'Directions',2356:'Directions',1552:'Directions',
5873:'Discussion', 5306:'Discussion',2684:'Discussion',2198:'Discussion',
2686:'Eating', 2663:'Eating',2203:'Eating',2275 :'Eating',
1447:'Greeting', 2711:'Greeting',1808:'Greeting', 1695:'Greeting',
3319:'Phoning',3821:'Phoning',3492:'Phoning',3390:'Phoning',
2346:'Photo',1449:'Photo',1990:'Photo',1545:'Photo',
1964:'Posing', 1968:'Posing',1407:'Posing',1481 :'Posing',
1529:'Purchases', 1226:'Purchases',1040:'Purchases', 1026:'Purchases',
2962:'Sitting', 3071:'Sitting',2179:'Sitting', 1857:'Sitting',
2932:'SittingDown', 1554:'SittingDown',1841:'SittingDown', 2004:'SittingDown',
4334:'Smoking',4377:'Smoking',2410:'Smoking',2767:'Smoking',
3312:'Waiting', 1612:'Waiting',2262:'Waiting', 2280:'Waiting',
2237:'WalkDog', 2217:'WalkDog',1435:'WalkDog', 1187:'WalkDog',
1703:'WalkTogether',1685:'WalkTogether',1360:'WalkTogether',1793:'WalkTogether',
1611:'Walking', 2446:'Walking',1621:'Walking', 1637:'Walking',
}

actions = ['Sitting','SittingDown']
test_actions = actions
vis_actions = actions


vis_score = pickle.load(open('./data/score.pkl', 'rb'))

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True, is_test = False):
    #print(action_filter)
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a) and len(action.split(a)[1]) <3:
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            
            n_frames = poses_2d[0].shape[0]
            vis_name_1 = '{}_{}.{}'.format(subject, action, 0)
            vis_name_2 = '{}_{}.{}'.format(subject, action, 1)
            vis_name_3 = '{}_{}.{}'.format(subject, action, 2)
            vis_name_4 = '{}_{}.{}'.format(subject, action, 3)
            vis_score_cam0 = vis_score[vis_name_1][:n_frames][...,np.newaxis]
            vis_score_cam1 = vis_score[vis_name_2][:n_frames][...,np.newaxis]
            vis_score_cam2 = vis_score[vis_name_3][:n_frames][...,np.newaxis]
            vis_score_cam3 = vis_score[vis_name_4][:n_frames][...,np.newaxis]
            if vis_score_cam3.shape[0] != vis_score_cam2.shape[0]:
                vis_score_cam2 = vis_score_cam2[:-1]
                vis_score_cam1 = vis_score_cam1[:-1]
                vis_score_cam0 = vis_score_cam0[:-1]
                for i in range(4):
                    poses_2d[i] = poses_2d[i][:-1]
                    
            try:
                if is_test == True and action == 'Walking' and poses_2d[0].shape[0] == 1612:
                    out_poses_2d_view1.append(np.concatenate((poses_2d[0][1:][...,:7], vis_score_cam0[1:]), axis =-1))
                    out_poses_2d_view2.append(np.concatenate((poses_2d[1][1:][...,:7], vis_score_cam1[1:]), axis =-1))
                    out_poses_2d_view3.append(np.concatenate((poses_2d[2][1:][...,:7], vis_score_cam2[1:]), axis =-1))
                    out_poses_2d_view4.append(np.concatenate((poses_2d[3][1:][...,:7], vis_score_cam3[1:]), axis =-1))
                else:
                    out_poses_2d_view1.append(np.concatenate((poses_2d[0][...,:7], vis_score_cam0), axis =-1))
                    out_poses_2d_view2.append(np.concatenate((poses_2d[1][...,:7], vis_score_cam1), axis =-1))
                    out_poses_2d_view3.append(np.concatenate((poses_2d[2][...,:7], vis_score_cam2), axis =-1))
                    out_poses_2d_view4.append(np.concatenate((poses_2d[3][...,:7], vis_score_cam3), axis =-1))
            except:
                print(poses_2d[0].shape, poses_2d[1].shape, poses_2d[2].shape, poses_2d[3].shape)
                print(vis_score_cam0.shape, vis_score_cam1.shape, vis_score_cam2.shape, vis_score_cam3.shape, vis_score[vis_name_4].shape, subject)
                
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    

    return out_camera_params, out_poses_3d, [out_poses_2d_view1, out_poses_2d_view2, out_poses_2d_view3, out_poses_2d_view4] 

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)
    

filter_widths = [int(x) for x in args.architecture.split(',')]


use_angle = False
use_2d_gt = False
N_aug = True
use_otherdataset = False
use_inter_loss = True
channels = 3 * 200
ADD_VIEW = 1
in_channels = 17 * 3 + (21 * 2) if use_angle else 0
dropout = 0.1
model = VideoMultiViewModel( num_view = 4 + ADD_VIEW, in_channels = in_channels, channels = channels, dropout = dropout, is_train = True, use_inter_loss = use_inter_loss)
model_test = VideoMultiViewModel(num_view = 4 + ADD_VIEW,in_channels = in_channels, channels = channels, dropout = dropout, is_train = False)
model_params = 0
for parameter in model_test.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

use_mpjpe = True
if use_mpjpe:
    loss_fun = mpjpe
    test_loss_fun = mpjpe
else:
    loss_fun = n_mpjpe
    test_loss_fun = n_mpjpe
receptive_field =9
pad = receptive_field // 2
causal_shift = 0
EVAL =True
if EVAL:
    plt.ion()
    fig = plt.figure(figsize = (39, 5))
    ax_views = []
    t = '15{}'.format(1)
    ax = fig.add_subplot(t)
    ax_views.append(ax)
    for i in range(4):
        t = '15{}'.format(i + 2)
        ax = fig.add_subplot(t, projection='3d')
        ax_views.append(ax)
  
def load_state(model_train, model_test):
    train_state = model_train.state_dict()
    test_state = model_test.state_dict()
    pretrained_dict = {k:v for k, v in train_state.items() if k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)
def show_3d(pose_2d, predicted,gt):
    pose_2d = pose_2d.numpy()
    pose_2d[:,:,1] *=-1
    gt = gt.cpu().numpy()
    step = 50
    for i in range(predicted[0].shape[0] // step):
        radius = 1.6
        for view_id in range(1):
            ax_views[view_id * 5].clear()
            #ax_views[view_id * 5].set_xlim([-radius/2, radius/2])
            #ax_views[view_id * 5].set_ylim([-radius/2, radius*2])
            ax_views[view_id * 5].set_xticklabels([])
            ax_views[view_id * 5].set_yticklabels([])
            
            ax_views[view_id * 5 + 1].clear()
            ax_views[view_id * 5 + 1].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 1].set_ylim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 1].set_zlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 1].set_xticklabels([])
            ax_views[view_id * 5 + 1].set_yticklabels([])
            ax_views[view_id * 5 + 1].set_zticklabels([])
            
            ax_views[view_id * 5 + 2].clear()
            ax_views[view_id * 5 + 2].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 2].set_ylim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 2].set_zlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 2].set_xticklabels([])
            ax_views[view_id * 5 + 2].set_yticklabels([])
            ax_views[view_id * 5 + 2].set_zticklabels([])
            
            ax_views[view_id * 5 + 3].clear()
            ax_views[view_id * 5 + 3].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 3].set_ylim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 3].set_zlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 3].set_xticklabels([])
            ax_views[view_id * 5 + 3].set_yticklabels([])
            ax_views[view_id * 5 + 3].set_zticklabels([])
            
            ax_views[view_id * 5 + 4].clear()
            ax_views[view_id * 5 + 4].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 4].set_ylim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 4].set_zlim3d([-radius/2, radius/2])
            ax_views[view_id * 5 + 4].set_xticklabels([])
            ax_views[view_id * 5 + 4].set_yticklabels([])
            ax_views[view_id * 5 + 4].set_zticklabels([])

            ax_views[view_id * 5 + 1].view_init(10, -45)
            ax_views[view_id * 5 + 2].view_init(10, -45)
            ax_views[view_id * 5 + 3].view_init(10, -45)
            ax_views[view_id * 5 + 4].view_init(10, -45)
            
            for k in range(4):
                for bone_id, l in enumerate(link[1:]):
                    x = list(gt[i * step, l, 0, view_id])
                    y = list(gt[i * step, l, 2, view_id])
                    z = list(-1 * gt[i * step, l, 1, view_id])
                    ax_views[view_id * 5 + k + 1].plot(x, y, z, C = 'R', linewidth=4, linestyle=':')
                    
                    x = list(predicted[k][i * step, l, 0, view_id])
                    y = list(predicted[k][i * step, l, 2, view_id])
                    z = list(-1 * predicted[k][i * step, l, 1, view_id])

                    ax_views[view_id * 5 + k + 1].plot(x, y, z, C = 'B', linewidth=2)
                    

  
            for bone_id, l in enumerate(link[1:]):
                x = list(pose_2d[i * step, l, 0, view_id])
                y = list(pose_2d[i * step, l, 1, view_id])
                
                ax_views[view_id * 5 + 0].plot(x, y, C = 'G', linewidth=4)


        plt.pause(0.1)
        input()   

if EVAL:
    chk_filename = './checkpoint/multi_view_4_mpjpe_att_tran_conf.bin'

    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['model'])
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    model_test = torch.nn.DataParallel(model_test).cuda()

if not args.evaluate:
    
        with torch.no_grad(): 
            load_state(model, model_test)
            model_test.eval()

            for act in vis_actions:
                cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, [act], is_test =True)
                test_generator = ChunkedGenerator(args.batch_size//2, None,  poses_valid_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                for batch_2d in test_generator.next_epoch():
                    inputs = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d_gt = inputs[...,:2,:]
                    inputs_2d_pre = inputs[...,2:4,:]
                    cam_3d = inputs[..., 4:7,:]
                    vis = inputs[...,7:8,:]
                    inputs_3d_gt = cam_3d[:,pad:-pad]
                    inputs_3d_gt[:,:,0] = 0
                    if use_2d_gt:
                        inp = inputs_2d_gt
                        vis = torch.ones(*vis.shape)
                    else:
                        inp = inputs_2d_pre
                    B = inp.shape[0]
                    result = []
                    for i in range(4):
                        view_list = list(range(i + 1))
                        inp_tmp = inp[...,view_list]
                        inp_tmp = torch.cat((inp_tmp, vis[..., view_list]), dim = -2)
                         
                        inp_tmp_flip = copy.deepcopy(inp_tmp)
                        inp_tmp_flip[:,:,:,0] *= -1
                        inp_tmp_flip[:,:,joints_left + joints_right] = inp_tmp_flip[:,:,joints_right + joints_left]
                        out = model_test(torch.cat((inp_tmp, inp_tmp_flip), dim = 0))
                                
                        out[B:,:,:,0] *= -1
                        out[B:,:,joints_left + joints_right] = out[B:,:,joints_right + joints_left]
                             
                        out = (out[:B] + out[B:]) / 2
                        out[:,:,0] = 0
                   
                        result.append(out[:,0,:,:,:1].detach().cpu().numpy())
                        
                    show_3d(inputs_2d_pre[:,pad, :,:,:1], result, inputs_3d_gt[:,0,:,:,:1])

        