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
vis_actions = ['Directions', 'Eating', 'Photo','Sitting','Smoking', 'Waiting',]
test_cameras = [int(v) for v in args.test_camera.split(',')]

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
all_ag_1 = torch.zeros(0, 4, 9).float()
all_ag_flip_1 = torch.zeros(0, 4, 9).float()
all_ag_2 = torch.zeros(0, 4, 9).float()
all_ag_flip_2 = torch.zeros(0, 4, 9).float()
if True:
    with torch.no_grad(): 
        load_state(model_test, args)
        model_test.eval()
        NUM_VIEW = len(test_cameras)
        TEST_VIEW = [4]
            
        for num_view in TEST_VIEW:
            for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                view_list = list(view_list)
                poses_valid_2d = fetch(subjects_test, vis_actions, is_test =True)
                            
                test_generator = ChunkedGenerator(args.batch_size, poses_valid_2d, 1,pad=pad, causal_shift=0, shuffle=True, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                for batch_2d in test_generator.next_epoch():
                    inputs = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d_gt = inputs[...,:2,:]
                    inputs_2d_pre = inputs[...,2:4,:]

                    cam_3d = inputs[..., 4:7,:]
                    vis = inputs[...,7:8,:]
                    inputs_3d_gt = cam_3d[:,pad:pad+1]
                                
                    if torch.cuda.is_available():  
                        inputs_3d_gt = inputs_3d_gt.cuda()

                    inputs_3d_gt[:,:,0] = 0

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

                    out, ag, _, _, _ = model_test(torch.cat((inp, inp_flip), dim = 0))
                    print(mpjpe(out[:inputs_3d_gt.shape[0], ...,0], inputs_3d_gt[...,0]))
                    

                    B1, D, _, T, _, _ = ag.shape
                    ag1 = ag[:,:,:,T//2, 0,:].view(B1, D * D, -1).contiguous().permute(0, 2, 1).detach().cpu()
                    ag2 = ag[:,:,:,T//2,:,0].view(B1, D * D, -1).contiguous().permute(0, 2, 1).detach().cpu()
                    
                    all_ag_1 = torch.cat((all_ag_1, ag1[:B]), dim  =0)
                    all_ag_flip_1 = torch.cat((all_ag_flip_1, ag1), dim  =0)
                    all_ag_2 = torch.cat((all_ag_2, ag2[:B]), dim  =0)
                    all_ag_flip_2 = torch.cat((all_ag_flip_2, ag2), dim  =0)

N = 4

def get_data(): 
    global N, all_ag_1, all_ag_flip_1, all_ag_2, all_ag_flip_2
    
    _, N, C = all_ag_1.shape
    s = N * 0
    e = N * 1000
    B = all_ag_1.shape[0]
    Nl = list(range(B))
    random.shuffle(Nl)
    datas = [all_ag_1[Nl], all_ag_flip_1[Nl], all_ag_2[Nl], all_ag_flip_2[Nl]]
    
    datas_out = []
    for idx, data in enumerate(datas):
        label = torch.Tensor(list(range(N))).view(1, N)
        data = data.view(-1, C)[s:e]
        label = label.repeat(e-s, 1)
        label = label.view(-1)
        data = np.array(data)
        label= np.array(label)
        datas_out.append([data, label])
    return datas_out

def plot_embedding(datas, labels, titles):
    global N
    fig = plt.figure(figsize=(13,9))
    N = len(datas)
    for idx, (data, label, title) in enumerate(zip(datas, labels, titles)):
        if idx < 2:
            color = ['r', 'g', 'b', 'y']
        else:
            color = ['brown', 'gold', 'purple', 'orange']
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min) / 1.05
    
        ax = plt.subplot(int(2*100 + (N//2) * 10 + idx + 1))
        for i in range(data.shape[0]):
            #ax.scatter(data[i, 0], data[i, 1], c=color[int(label[i])],label='right', edgecolors=color[int(label[i])])
            ax.scatter(data[i, 0], data[i, 1], c=color[int(label[i])],label='right', edgecolors=color[int(label[i])],s = 5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./images/vis_ag.pdf')
        if len(title):
            plt.title(title, fontsize = 20)
        #plt.title(title)

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
    plot_embedding(results, labels, ['seq_no_rcpe', 'seq_rcpe', 'data_no_rcpe', 'data_rcpe'] if 0 else ['w/o flip', 'w flip', '', ''])
    plt.show()
    x = input()

if __name__ == '__main__':
    main()