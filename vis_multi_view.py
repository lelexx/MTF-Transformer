import numpy as np
import itertools
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import errno
import copy
from time import time
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = parse_args()
link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])


joints_left, joints_right = [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
use_2d_gt = True
channels = 3 * 200
in_channels = 17 * 3
model = VideoMultiViewModel( num_view = 4 + 1, in_channels = in_channels, channels = channels, dropout = 0.1, is_train = False, use_inter_loss = False)
receptive_field = 9
pad = receptive_field // 2
EVAL = True
color_left = 'B'
color_right = 'G'
color_other = 'R'
bone_color = [color_right, color_right, color_right, color_left, color_left, color_left, color_other, color_other, color_other, color_other, color_left, color_left, color_left, color_right, color_right, color_right]

RGB_color= {'R':(0, 0, 255), 'G':(0, 255, 0), 'B':(255, 0, 0)}
dataset_name = 'h36m'

if dataset_name == 'kth':
    num_view_dataset = 3
    final_data = [[], [], []]
    final_file = [[], [], []]
    select_videos = [0]
    with open('/home/data/lele/DataSet/KTH_MV_Football_2/Code/h36m_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
        for video_id, video_data in enumerate(data):
            print(video_id)
            if video_id not in select_videos:
                continue
            print(video_id)
            for cam_id, cam_data in video_data.items():
                final_cam_pose2d = np.zeros((len(cam_data), 17, 2))
                
                for i, frame_data in enumerate(cam_data):
                    w = frame_data['w']
                    h = frame_data['h']
                    frame_pose2d = frame_data['pos_2d']
                    file_name = frame_data['image_name']
                    img = cv.imread(file_name)
                    for bone_id, l in enumerate(link[1:]):
                        c = bone_color[bone_id]
                        c = RGB_color[c]
                        
                        x = list(frame_pose2d[l, 0])
                        y = list(frame_pose2d[l, 1])
                        s = (int(x[0]), int(y[0]))
                        e = (int(x[1]), int(y[1]))
                        cv.line(img,s, e,c, 10, 1)
                    final_file[cam_id].append(img[:,:,::-1])
                    frame_pose2d =  normalize_screen_coordinates(frame_pose2d, w, h)
                    final_cam_pose2d[i, :,:] = frame_pose2d
                final_data[cam_id].append(final_cam_pose2d)
elif dataset_name == 'h36m':
    ##load_2dpose
    with open('./data/h36m_pose2d.pkl', 'rb') as f:
        data_2dpose = pickle.load(f)
    print(data_2dpose.__class__)
    exit()

    
                 
test_generator = ChunkedGenerator(140, None,  final_data, 1,pad=pad, causal_shift=0, shuffle=False, augment=False,kps_left=joints_left, kps_right=joints_right, joints_left=joints_left, joints_right=joints_right)

if EVAL:
    plt.ion()
    fig = plt.figure(figsize = (16, 8))
    ax_views = []
    for i in range(num_view_dataset):
        t = '2{}{}'.format(num_view_dataset,i + 1)
        ax = fig.add_subplot(t, projection='3d')
        ax_views.append(ax)
    for i in range(num_view_dataset):
        t = '2{}{}'.format(num_view_dataset,i + 1 + num_view_dataset)
        ax = fig.add_subplot(t)
        ax_views.append(ax)
        
def show_3d(predicted, file_names):
    predicted = predicted.cpu().numpy()
    #rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    #predicted = predicted.transpose(0, 1, 3, 2)
    #predicted = camera_to_world(predicted, R=rot, t=0)
    #predicted = predicted.transpose(0, 1, 3, 2)

    imgs = file_names
    
    for i in range(predicted.shape[0]):
        radius = 1.7
        for view_id in range(predicted.shape[-1]):
            ax_views[view_id].clear()
            ax_views[view_id].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id].set_ylim3d([-radius / 2, radius*2])
            ax_views[view_id].set_zlim3d([-radius/2, radius/2])
            ax_views[view_id].set_xticklabels([])
            ax_views[view_id].set_yticklabels([])
            ax_views[view_id].set_zticklabels([])
            
            ax_views[view_id + num_view_dataset].set_xticklabels([])
            ax_views[view_id + num_view_dataset].set_yticklabels([])

            ax_views[view_id].view_init(30 - 20, -43)
        
            for bone_id, l in enumerate(link[1:]):
                c = bone_color[bone_id]
                x = list(predicted[i, l, 0, view_id])
                y = list(predicted[i, l, 2, view_id])
                z = list(-1 * predicted[i, l, 1, view_id])

                ax_views[view_id].plot(x, y, z, C = c, linewidth=4)
            ax_views[num_view_dataset + view_id].imshow(imgs[view_id][i])

        plt.pause(0.00001)
        #input()    
if EVAL:
    chk_filename = './checkpoint/multi_view_4_mpjpe_att_tran_conf.bin'

    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in checkpoint['model'].items() if k in model_dict}
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()


def eval(model, dataset):
    with torch.no_grad(): 
        model.eval()           
        epoch_loss_valid = 0
        N_frames = 0
        VIEWS = dataset.VIEWS
        NUM_VIEW = len(VIEWS)
        SHOW_VIEW = NUM_VIEW

        action_mpjpe = [0] * NUM_VIEW
        for i in range(NUM_VIEW):
            action_mpjpe[i] = [0] * (NUM_VIEW + 1)
        N = [0] * NUM_VIEW
        for i in range(NUM_VIEW):
            N[i] = [0] * (NUM_VIEW + 1)
        for num_view in range(NUM_VIEW, NUM_VIEW + 1):
            if num_view > SHOW_VIEW:
                break
            for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                view_list = list(view_list)
                print(view_list)
                N[num_view - 1][-1] += 1
                for i in view_list:
                    N[num_view - 1][i] += 1
                frame_idx = 0
                for batch_2d in test_generator.next_epoch():
                    inputs = torch.from_numpy(batch_2d.astype('float32'))
                   
                    inputs_2d_gt = inputs[...,:2,:]

                    B, T, V, C, N_V = inputs_2d_gt.shape
                    vis = torch.ones(B, T, V, 1, N_V).float()
                    inp = torch.cat((inputs_2d_gt, vis), dim = 3)

                    inp_flip = copy.deepcopy(inp)
                    inp_flip[:,:,joints_left + joints_right] = inp_flip[:,:,joints_right+joints_left]
                    inp_flip[:,:,:,0] *= -1
             
                    inp = inp[...,view_list]
                    inp_flip = inp_flip[..., view_list]
                    B, T,V, C, N_V = inp.shape
                    inp = torch.cat((inp, inp_flip), dim = 0).contiguous()
                    out = model(inp)
                    
                    out[B:,:,:,0] *= -1
                    out[B:,:,joints_left + joints_right] = out[B:,:,joints_right + joints_left]
                    out = (out[:B] + out[B:]) / 2
                    out[:,:,0] = 0
                    N = out.shape[0]
                     
                    if EVAL:
                        show_3d(out[:,0].detach(), [final_file[0][frame_idx:frame_idx + N], final_file[1][frame_idx:frame_idx+N], final_file[2][frame_idx:frame_idx+N]])
                        frame_idx += N



        
if __name__ == '__main__':
    eval(model, test_generator)