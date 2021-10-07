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
#plt.rc('font', family='Times New Roman')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from thop import profile
from thop import clever_format
import random

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
receptive_field = 1
pad = receptive_field // 2
model_test = VideoMultiViewModel(args, num_view = len(test_cameras) + args.add_view, is_train = False)

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

all_f = torch.zeros(0,4, 600).float()
all_f_rcpe = torch.zeros(0,4, 600).float()
if True:
    load_state(model_test, args)
    model_test.eval()
    NUM_VIEW = len(test_cameras)
    TEST_VIEW = [4]
            
    for num_view in TEST_VIEW:
        for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
            view_list = list(view_list)

            poses_valid_2d = fetch(subjects_test, vis_actions, is_test =True)
                            
            test_generator = ChunkedGenerator(args.batch_size, poses_valid_2d, 1,pad=pad, causal_shift=0, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            for idx, batch_2d in enumerate(test_generator.next_epoch()):
                inputs = torch.from_numpy(batch_2d.astype('float32'))
                inputs_2d_gt = inputs[...,:2,:]
                inputs_2d_pre = inputs[...,2:4,:]

                cam_3d = inputs[..., 4:7,:]
                vis = inputs[...,7:8,:]
                inputs_3d_gt = cam_3d[:,pad:pad+1]
                
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

                out, ag, att, f, f_rcpe = model_test(torch.cat((inp, inp_flip), dim = 0))
                print(mpjpe(out[:inputs_3d_gt.shape[0], ...,0], inputs_3d_gt[...,0]))
                f = f[:B,:,:,0, :]
                f = f.detach().cpu().view(B, 600, -1).contiguous().permute(0, 2, 1).contiguous()
                f_rcpe = f_rcpe[:B,:,:,0, :]
                f_rcpe = f_rcpe.detach().cpu().view(B, 600, -1).contiguous().permute(0, 2, 1).contiguous()
                if idx == 0:
                    f_seq = f
                    f_seq_rcpe = f_rcpe
                all_f = torch.cat((all_f, f), dim  = 0)
                all_f_rcpe = torch.cat((all_f_rcpe, f_rcpe), dim  = 0)

def get_data(): 
    _, N, C = f_seq.shape
    s = N * 50
    e = N * 150
    B = all_f.shape[0]
    Nl = list(range(B))
    random.shuffle(Nl)
    datas = [f_seq, f_seq_rcpe, all_f[Nl], all_f_rcpe[Nl]]
    
    datas_out = []
    for idx, data in enumerate(datas):
        label = torch.Tensor(list(range(N))).view(1, N)
        if idx < 2:
            data = data.view(-1, C)[s:e]
            label = label.repeat(e-s, 1)
        else:
            n = 12000
            data = data.view(-1, C)[:n]
            label = label.repeat(n, 1)
        label = label.view(-1)
        data = np.array(data)
        label= np.array(label)
        datas_out.append([data, label])
    return datas_out

def plot_embedding(datas, labels, titles):
    global N
    fig = plt.figure(figsize=(13,9))
    N = len(datas)
    color = ['r', 'g', 'b', 'y']
    for idx, (data, label, title) in enumerate(zip(datas, labels, titles)):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min) / 1.05
    
        ax = plt.subplot(int(2*100 + (N//2) * 10 + idx + 1))
        for i in range(data.shape[0]):
            ax.scatter(data[i, 0], data[i, 1], c=color[int(label[i])],label='right', edgecolors=color[int(label[i])],s = 5)

        plt.xticks([])
        plt.yticks([])
        if len(title):
            plt.title(title, fontsize = 20)
    plt.savefig('./images/vis_feature.pdf')
    plt.show()
    x = input()

def main():
    datas = get_data()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    results = []
    labels = []
    for data, label in datas:
        print(data.shape)
        result = tsne.fit_transform(data)
        results.append(result)
        labels.append(label)
    plot_embedding(results, labels, ['seq_no_rcpe', 'seq_rcpe', 'data_no_rcpe', 'data_rcpe'] if 0 else ['w/o RCPE', 'w RCPE', '', ''])
    
    
    

if __name__ == '__main__':
    main()
