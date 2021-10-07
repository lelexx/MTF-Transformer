from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import pickle
import torch
from mpl_toolkits.mplot3d import Axes3D
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import errno
import copy
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from thop import profile
from thop import clever_format

from common.arguments import parse_args
from common.utils import deterministic_random
from common.camera import *
from common.video_multi_view import *
from common.loss import *
from common.generators import *
from common.data_augmentation_multi_view import *
from common.h36m_dataset import Human36mDataset
from common.set_seed import *

set_seed()

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)
keypoints = {}
for sub in [9, 11]:
    keypoint = np.load('data/h36m_sub{}.npz'.format(sub), allow_pickle=True)
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]

kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = [kps_left, kps_right]

subjects_test = args.subjects_test.split(',')
actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases','Sitting','SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
vis_actions = ['Directions', 'Sitting', 'Walking']
test_cameras = [int(v) for v in args.test_camera.split(',')]

link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])

color_left = 'B'
color_right = 'G'
color_other = 'R'
bone_color = [color_right, color_right, color_right, color_left, color_left, color_left, color_other, color_other, color_other, color_other, color_left, color_left, color_left, color_right, color_right, color_right]
RGB_color= {'B':(0, 0, 255), 'G':(0, 255, 0), 'R':(255, 0, 0)}

vis_score = pickle.load(open('./data/score.pkl', 'rb'))
def change_rot(f):
    #B, T, V, C, N
    B, T, V, C, N = f.shape
    root = copy.deepcopy(f[:,:,:1])
    f[:,:,:1] = 0
    c_1 = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]).float().view(1, 1, 3, 3, 1).repeat(B, T, 1, 1, 1)
    c_2 = torch.Tensor([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ]).float().view(1, 1, 3, 3, 1).repeat(B, T, 1, 1, 1)
    c_3 = torch.Tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]).float().view(1, 1, 3, 3, 1).repeat(B, T, 1, 1, 1)
    c_4 = torch.Tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]).float().view(1, 1, 3, 3, 1).repeat(B, T, 1, 1, 1)
    c_cam = torch.cat((c_1, c_2, c_3, c_4), dim = -1)
    z = f[:, :,7:8,:,] - f[:, :,:1,:,] #(B, T, 1, C, N)
    y = f[:, :,1:2,:,] - f[:, :,:1,:,]
    z = z / torch.norm(z, dim = -2, keepdim = True)
    y = y / torch.norm(y, dim = -2, keepdim = True)
    x = torch.cross(y, z, dim  = -2)
    c = torch.cat((y, -z, -x), dim = -3)#(B, T, 3, C, N)
    r = torch.einsum('btqkn, bthkn -> btqhn',c, c_cam )
#     print(r[0,0,:,:,0])
#     exit()
    f_conv = torch.einsum('btqcn, btvcn -> btvqn',r, f )
    f_conv = f_conv + root
    cam_3d = f_conv - f_conv[:,:,:1]
#     print(f_conv[0, 0, :,:,0])
#     print(f[0, 0, :,:,0])
#     print(cam_3d[0, 0, :,:,0])
#     exit()
    ff = 1000
    p_2d = f_conv[:,:,:,:2] / f_conv[:,:,:,-1:]* ff / 400

    return p_2d, cam_3d
def fetch(subjects, action_filter=None,  parse_3d_poses=True, is_test = False):
    out_poses_3d = []
    out_poses_2d_view1 = []
    out_poses_2d_view2 = []
    out_poses_2d_view3 = []
    out_poses_2d_view4 = []
    out_camera_params = []
    used_cameras = test_cameras if is_test else train_cameras
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
                    
            if is_test == True and action == 'Walking' and poses_2d[0].shape[0] == 1612:
                out_poses_2d_view1.append(np.concatenate((poses_2d[0][1:], vis_score_cam0[1:]), axis =-1))
                out_poses_2d_view2.append(np.concatenate((poses_2d[1][1:], vis_score_cam1[1:]), axis =-1))
                out_poses_2d_view3.append(np.concatenate((poses_2d[2][1:], vis_score_cam2[1:]), axis =-1))
                out_poses_2d_view4.append(np.concatenate((poses_2d[3][1:], vis_score_cam3[1:]), axis =-1))
            else:
                out_poses_2d_view1.append(np.concatenate((poses_2d[0], vis_score_cam0), axis =-1))
                out_poses_2d_view2.append(np.concatenate((poses_2d[1], vis_score_cam1), axis =-1))
                out_poses_2d_view3.append(np.concatenate((poses_2d[2], vis_score_cam2), axis =-1))
                out_poses_2d_view4.append(np.concatenate((poses_2d[3], vis_score_cam3), axis =-1))

    
    final_pose = []
    if 0 in used_cameras:
        final_pose.append(out_poses_2d_view1)
    if 1 in used_cameras:
        final_pose.append(out_poses_2d_view2)
    if 2 in used_cameras:
        final_pose.append(out_poses_2d_view3)
    if 3 in used_cameras:
        final_pose.append(out_poses_2d_view4)
        
        
    return final_pose

use_2d_gt = False
dropout = args.dropout
receptive_field = args.t_length
pad = receptive_field // 2
model_test = VideoMultiViewModel(args, num_view = len(test_cameras) + args.add_view, is_train = False)
if 1:
    plt.ion()
    fig = plt.figure()
    ax_views = []
    for i in range(12):
        t = '13{}'.format(i + 1)
        ax = fig.add_subplot(3, 4, i + 1)
        ax_views.append(ax)
 
def show_3d(pose_2d, predicted,gt):
    gt[:,:1] = 0
    pose_2d = pose_2d.numpy()
    predicted = predicted.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(predicted.shape[0]):
        radius = 1.7
        for view_id in range(4):
            ax_views[view_id].clear()
            ax_views[view_id+4].clear()
            ax_views[view_id].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id].set_ylim3d([-radius/2, radius*2])
            ax_views[view_id].set_zlim3d([-radius/2, radius/2])

            ax_views[view_id].view_init(30, -45)
        
#             for bone_id, l in enumerate(link[1:]):
#                 x = list(predicted[i, l, 0, view_id])
#                 y = list(predicted[i, l, 2, view_id])
#                 z = list(-1 * predicted[i, l, 1, view_id])

#                 ax_views[view_id].plot(x, y, z, C = 'B')

            for bone_id, l in enumerate(link[1:]):
                x = list(gt[i, l, 0, view_id])
                y = list(gt[i, l, 2, view_id])
                z = list(-1 * gt[i, l, 1, view_id])
                ax_views[view_id].plot(x, y, z, C = bone_color[bone_id])
                
            for bone_id, l in enumerate(link[1:]):
                x = list(pose_2d[i, l, 0, view_id])
                y = list(-pose_2d[i, l, 1, view_id])
                
                ax_views[view_id + 4].plot(x, y, C = bone_color[bone_id])


        plt.pause(0.1)
        x = input()
        
def load_state(model_test, args):
    chk_filename = args.checkpoint
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    test_state = model_test.state_dict()
    pretrained_dict = {'module.'+k:v for k, v in checkpoint['model'].items() if 'module.'+k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)

if torch.cuda.is_available():
    model_test = torch.nn.DataParallel(model_test).cuda()

all_grad  = np.zeros((4, 17, 4, 3))
if True:
    load_state(model_test, args)
    model_test.eval()
    NUM_VIEW = len(test_cameras)
    TEST_VIEW = [4]#args.eval_n_views if isinstance(args.eval_n_views, list) else [args.eval_n_views]
            
    for num_view in TEST_VIEW:
        for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
            view_list = list(view_list)

            poses_valid_2d = fetch(subjects_test, vis_actions, is_test =True)
                            
            test_generator = ChunkedGenerator(args.batch_size, poses_valid_2d, 1,pad=pad, causal_shift=0, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            for batch_2d in test_generator.next_epoch():
                inputs = torch.from_numpy(batch_2d.astype('float32'))
                inputs_2d_gt = inputs[...,:2,:]
                inputs_2d_pre = inputs[...,2:4,:]

                cam_3d = inputs[..., 4:7,:]
                vis = inputs[...,7:8,:]
                inputs_3d_gt = cam_3d[:,pad:pad+1]
                #inputs_2d_pre, inputs_3d_gt = change_rot(inputs_3d_gt)
                #vis = torch.ones(*vis.shape)
                #inputs_2d_gt = inputs_2d_pre
                #show_3d(inputs_2d_pre[150:151,pad], inputs_3d_gt[150:151,0].detach(), inputs_3d_gt[50:51,0])
                if torch.cuda.is_available():  
                    inputs_3d_gt = inputs_3d_gt.cuda()

                inputs_3d_gt[:,:,0] = 0
                inputs_3d_gt = inputs_3d_gt[...,view_list]
                if use_2d_gt:
                    inp = inputs_2d_gt
                    vis = torch.ones(*vis.shape)
                else:
                    inp = inputs_2d_pre

                inp = inp[...,view_list]
                inp = torch.cat((inp, vis[..., view_list]), dim = -2)
                B = inp.shape[0]
                                
                inp_flip = copy.deepcopy(inp)
                inp_flip[:,:,:,0] *= -1
                inp_flip[:,:,joints_left + joints_right] = inp_flip[:,:,joints_right + joints_left]
                
                out, ag, att, f = model_test(torch.cat((inp, inp_flip), dim = 0))
                for b_id in range(B):
                    for h in range(17):
                        print('***************************joint{}'.format(h))
                        for i in range(4):
                            print('********cam{}'.format(i))
                            for j in range(3):
                                grad = torch.autograd.grad(out[b_id, 0, h, j, i],f, create_graph = True, retain_graph = True)[0]
                                #(B, V, C, T, N)
                                grad = grad.detach().cpu().numpy()
                                grad = np.max(grad[b_id, :,:,0,:], axis = (0, 1)) * 100
#                                 print(grad)
#                                 continue
                                grad = np.exp(grad)/np.sum(np.exp(grad),axis=-1)
                                all_grad[i, h, :,j] = grad
                                print(grad)

                    print(all_grad.shape)

                    for i in range(12):
                        im = ax_views[i].imshow(all_grad[i % 4, :,:,i // 4], cmap=plt.get_cmap('hot'), interpolation='nearest',
                   vmin=0, vmax=1)
                        ax_views[i].set_xticks([0, 1, 2, 3])
                        ax_views[i].set_yticks(list(range(17)))


                    #plt.colorbar(im, ax=ax_views, fraction=1e-2, pad=0.05)
                    plt.pause(0.0001)

                continue
                x =input()

                exit()
#                     print(view_list)
#                     print(mpjpe(out[50:51,:,:,:,-1], inputs_3d_gt[50:51,:,:,:,-1]))
                    
#                     #f = torch.cat((feature, feature_fuse, f[...,0,:]), dim = -1).contiguous()
#                 B1 = f.shape[0] 
#                 f = f.detach().cpu().view(B1, 600, -1).contiguous().permute(0, 2, 1).contiguous()[:B]
                #all_f = torch.cat((all_f, f), dim  = 0)
                break
