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

IMG_H =1080
IMG_W = 1920
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)
link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[8,10],[10,11],[11, 12], [8, 13],[13, 14], [14, 15]])
color_left = 'B'
color_right = 'G'
color_other = 'R'
bone_color = [color_right, color_right, color_right, color_left, color_left, color_left, color_other, color_other, color_other, color_other, color_left, color_left, color_left, color_right, color_right, color_right]

RGB_color= {'B':(0, 0, 255), 'G':(0, 255, 0), 'R':(255, 0, 0)}

#load train dataset
train_dataset = np.load('/home/data/lele/DataSet/TotalCapture/TotalCapture-Toolbox-master/data/images/totalcapture_train.pkl', allow_pickle = True)

#load valid dataset
valid_dataset = np.load('/home/data/lele/DataSet/TotalCapture/TotalCapture-Toolbox-master/data/images/totalcapture_validation.pkl', allow_pickle = True)

print('num frame of train:{}'.format(len(train_dataset)))
print('num frame of valid:{}'.format(len(valid_dataset)))

EVAL = args.eval
if EVAL and args.vis_3d:
    plt.ion()
    fig = plt.figure()
    ax_views = []
    for i in range(4):
        t = '22{}'.format(i)
        ax = fig.add_subplot(t, projection='3d')
        ax_views.append(ax)
def show_3d(predicted,gt):
    for i in range(predicted.shape[0]):
        radius = 1.7
        for view_id in range(1):
            ax_views[view_id].clear()
            ax_views[view_id].set_xlim3d([-radius/2, radius/2])
            ax_views[view_id].set_ylim3d([-radius/2, radius*2])
            ax_views[view_id].set_zlim3d([-radius/2, radius/2])

            ax_views[view_id].view_init(30, -45)
        
            for bone_id, l in enumerate(link[1:]):
                x = list(predicted[i, l, 0, view_id])
                y = list(predicted[i, l, 2, view_id])
                z = list(-1 * predicted[i, l, 1, view_id])

                ax_views[view_id].plot(x, y, z, C = 'B')

            for bone_id, l in enumerate(link[1:]):
                x = list(gt[i, l, 0, view_id])
                y = list(gt[i, l, 2, view_id])
                z = list(-1 * gt[i, l, 1, view_id])
                ax_views[view_id].plot(x, y, z, C = 'R')


        plt.pause(0.1)
        print('lele')
train_cams = [1,3,5,7]
test_cams = [2, 4, 6, 8]
def fetch(dataset,cameras = [1, 3, 5, 7], actions = ['rom', 'walking', 'acting', 'freestyle'], subjects = ['s1', 's2', 's3']):
    cams = cameras
    num_cam = len(cams)
    out_poses = []
    for i in range(num_cam):
        out_poses.append([])

    for subject in dataset.keys():
        if subject not in subjects:
            print('skip sub:{}'.format(subject))
            continue
        for action in dataset[subject].keys():
            found = False
            for act in actions:
                if act in action:
                    found = True
            if not found:
                print('skip act:{}'.format(action))
                continue
            poses = dataset[subject][action]
            
            n_frames = len(poses[0])
            assert len(poses) == 8
            poses_tmp = []
            for i in cams:
                poses_tmp.append(poses[i - 1])
            poses = poses_tmp
            for cam_id in range(len(poses)):
                poses_cam = np.concatenate(poses[cam_id], axis = 0)
                out_poses[cam_id].append(poses_cam)


    return out_poses

action_map={'rom':1, 'walking':2, 'acting':3, 'running':4, 'freestyle':5}
action_reverse_map= {1: 'rom', 2: 'walking', 3: 'acting', 4: 'running', 5: 'freestyle'}
train_seqs = {'rom':[1,2,3], 'walking':[1,3], 'acting':[1,2], 'running':[], 'freestyle':[1,2]}
test_seqs = {'rom':[], 'walking':[2], 'acting':[3], 'running':[], 'freestyle':[3]}

def get_action(act, subact, is_train):
    global action_map, action_reverse_map, train_seqs, test_seqs
    if is_train:
        seqs= train_seqs
    else:
        seqs = test_seqs
    action = action_reverse_map[act]
    try:
        action_id = subact
    except:
        print(action, subact, is_train)
    action = action+str(action_id)
    return action

def handle_dataset(dataset, is_train):
    N = 0
    final_data = {}
    for i in range(len(dataset)):
        item_data = dataset[i]
        joint_2d = item_data['joints_2d'][np.newaxis]
        
        cam_3d = item_data['joints_3d'][np.newaxis]
        joint_2d = normalize_screen_coordinates(joint_2d, IMG_W, IMG_H)
 
        cam_3d[:,1:] -= cam_3d[:,:1,]
        cam_3d = cam_3d / 1000
        #show_3d(cam_3d[..., np.newaxis], cam_3d[..., np.newaxis])
        cam_id = item_data['camera_id']
        
        image_id = item_data['image_id']
        joints_vis = item_data['joints_vis']
        
        joints_vis = np.sum(joints_vis, axis = -1, keepdims = True)
        joints_vis = joints_vis > 2.5
        
        if np.sum(joints_vis) < joints_vis.shape[0]:
#             if np.sum(joints_vis) == 0:
#                 print(joint_2d)
#                 print(cam_3d)
#                 exit()
            N += 1
        subject = 's' + str(item_data['subject'])
        act = item_data['action']
        subaction = item_data['subaction']
        action = get_action(act, subaction, is_train)
        if subject not in final_data:
            final_data[subject] = {}
        if action not in final_data[subject]:
            final_data[subject][action] = []
        assert cam_id <= len(final_data[subject][action])
        if cam_id == len(final_data[subject][action]):
            final_data[subject][action].append([])
        assert image_id <= len(final_data[subject][action][cam_id])
        
        temp = np.concatenate((joint_2d, cam_3d, joints_vis[np.newaxis]), axis = -1)
        final_data[subject][action][cam_id].append(temp)
    print(N)
    return final_data

        
        
        
train_data = handle_dataset(train_dataset, is_train = True)
test_data = handle_dataset(valid_dataset, is_train = False)
print(train_data.keys(), test_data.keys())


train_data = fetch(train_data, cameras = train_cams, actions = ['rom', 'walking', 'acting', 'freestyle'], subjects = ['s1', 's2', 's3'])
print('*******************************')
test_data_1 = fetch(test_data, cameras = train_cams, actions = ['walking', 'acting', 'freestyle'], subjects = ['s1', 's2', 's3', 's4', 's5'])
print('*******************************')
test_data_2 = fetch(test_data,cameras = test_cams, actions = ['walking', 'acting', 'freestyle'], subjects = ['s1', 's2', 's3','s4', 's5'])
print('*******************************')
if EVAL:
    test_sets = {'mean_1': test_data_1, 'mean_2':test_data_2}
    for cam_k, cams in {'tr':train_cams, 'te':test_cams}.items():
        for sub_k, subs in {'tr':['s1', 's2', 's3'], 'te':['s4', 's5']}.items():
            for act_k in ['walking', 'acting', 'freestyle']:
                test_data_tmp = fetch(test_data,  cameras = cams, actions = [act_k], subjects = subs)
                test_sets['cam_{}_sub_{}_act_{}'.format(cam_k, sub_k, act_k)] = test_data_tmp
                print('*******************************')
else:
    test_sets ={'mean_1': test_data_1, 'mean_2':test_data_2}

for i in train_data:
    for j in i:
        print(j.shape[0], end = ' ')
    print(' ')
for i in test_data_1:
    for j in i:
        print(j.shape[0], end = ' ')
    print(' ')
receptive_field = args.t_length
pad = receptive_field // 2
causal_shift = 0
use_2d_gt = True
use_inter_loss = True
kps_left  = [4, 5, 6, 10, 11, 12,]
kps_right = [1, 2, 3, 13, 14, 15]
joints_left  =kps_left
joints_right = kps_right
model = VideoMultiViewModel(args, num_view = len(train_cams) + args.add_view, is_train = True, use_inter_loss = use_inter_loss, num_joints =16)
model_test = VideoMultiViewModel(args, num_view = len(train_cams) + args.add_view, is_train = False, num_joints = 16)
if args.vis_complexity:
    model_test.eval()
    for i in range(1,5):
        input = torch.randn(1, receptive_field,17,3,i)
        macs, params = profile(model_test, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
        print('view: {} T: {} MACs:{} params:{}'.format(i, receptive_field, macs, params))


def load_state(model_train, model_test):
    train_state = model_train.state_dict()
    test_state = model_test.state_dict()
    pretrained_dict = {k:v for k, v in train_state.items() if k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)
   
if EVAL:
    chk_filename = args.checkpoint
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    model_test = torch.nn.DataParallel(model_test).cuda()

if True:
    lr = args.learning_rate
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, amsgrad=True)    
    lr_decay = args.lr_decay
    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    train_generator = ChunkedGenerator(args.batch_size, train_data, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=True,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
 
    print('** Starting.')
    best_result = 100
    best_state_dict = None
    best_result_epoch = 0
    data_aug = DataAug(add_view = args.add_view)

    while epoch < 60:
        start_time = time()
        model.train()
        process = tqdm(total = train_generator.num_batches)
        
        for batch_2d in train_generator.next_epoch():
            if EVAL:
                break
            
            process.update(1)
            inputs = torch.from_numpy(batch_2d.astype('float32'))

            assert inputs.shape[-2] == 6
            inputs_2d_gt = inputs[...,:,:2,:]
       
            cam_3d = inputs[..., 2:5,:]
            vis = inputs[...,-1:, :]
            B, T, V, _, _ = vis.shape
            if args.add_view:
                vis = torch.cat((vis, torch.ones(B, T, V, 1, args.add_view)), dim = -1)
            
            inputs_3d_gt = cam_3d
            inputs_3d_gt = inputs_3d_gt.cuda()
            
            inputs_3d_gt_root = copy.deepcopy(inputs_3d_gt[:,:, :1])
            inputs_3d_gt[:,:,0] = 0
            
            view_list = list(range(len(train_cams) + args.add_view))
            if args.add_view > 0:
                pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
                pos_gt_3d_tmp[:,:,:1] = inputs_3d_gt_root
                pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d = inputs_2d_gt, pos_gt_3d = pos_gt_3d_tmp)
                pos_gt_3d[:,:,:1] = 0
                h36_inp = pos_gt_2d[..., view_list]
                h36_gt = pos_gt_3d[..., view_list]
            else:
                if use_2d_gt:
                    h36_inp = inputs_2d_gt[..., view_list]
                else:
                    h36_inp = inputs_2d_pre[..., view_list]
                h36_gt = inputs_3d_gt[..., view_list]
            
            optimizer.zero_grad()
            
            inp = torch.cat((h36_inp, vis), dim = -2)
            pos_gt = h36_gt
            
            if use_inter_loss:
                out, other_out, tran= model(inp)
            else:
                out = model(inp)

            out = out.permute(0, 1, 4, 2,3).contiguous()
            pos_gt = pos_gt.permute(0, 1, 4,2, 3).contiguous()
            
            
            if use_inter_loss:
                for i in range(len(other_out)):
                    other_out[i] = other_out[i].permute(0, 1, 4, 2,3).contiguous()
            
            loss = mpjpe(out , pos_gt[:,pad:pad+1])
            other_a = [0.1, 0.5]
            other_loss = 0
            if use_inter_loss:
                for i in range(len(other_out)):
                    other_loss = other_loss + other_a[i] * mpjpe(other_out[i] , pos_gt)

            if use_inter_loss:
                loss_total = loss + other_loss 
            else:
                loss_total = loss
            loss_total.backward()

            optimizer.step()

        process.close() 

        with torch.no_grad(): 
            load_state(model, model_test)
            model_test.eval()
            NUM_VIEW = 4
            TEST_VIEW = [4]
            USE_FLIP = args.test_time_augmentation
            eval_t = args.eval_n_frames if isinstance(args.eval_n_frames, list) else [args.eval_n_frames]
            for t_len in eval_t:
                pad_t = t_len // 2
                print('t_len:{}'.format(t_len))
                for key, data in test_sets.items():
                    test_generator = ChunkedGenerator(args.batch_size // 2, data, 1,pad=pad_t, causal_shift=causal_shift, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                    epoch_loss_valid = 0 
                    Num =0
                    for num_view in TEST_VIEW:
                        for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                            view_list = list(view_list)
                            for batch_2d in test_generator.next_epoch():
                                    inputs = torch.from_numpy(batch_2d.astype('float32'))

                                    inputs_2d_gt = inputs[...,:2,:]
                                    cam_3d = inputs[..., 2:5,:]
                                    vis = inputs[..., -1:,:]
                                    inputs_3d_gt = cam_3d[:,pad_t:pad_t+1]
                                    if torch.cuda.is_available():  
                                        inputs_3d_gt = inputs_3d_gt.cuda()

                                    inputs_3d_gt[:,:,0] = 0
                                    inp = inputs_2d_gt
                                    inp = inp[...,view_list]
                                    inp = torch.cat((inp, vis[..., view_list]), dim = -2)
                                    B = inp.shape[0]

                                    inp_flip = copy.deepcopy(inp)
                                    inp_flip[:,:,:,0] *= -1
                                    inp_flip[:,:,joints_left + joints_right] = inp_flip[:,:,joints_right + joints_left]
                                    if USE_FLIP:
                                        out, _ = model_test(torch.cat((inp, inp_flip), dim = 0).contiguous())
                                        out[B:,:,:,0] *= -1
                                        out[B:,:,joints_left + joints_right] = out[B:,:,joints_right + joints_left]
                                        out = (out[:B] + out[B:]) / 2
                                    else:
                                        out, _ = model_test(inp)
                                    out[:,:,0] = 0
                                    if EVAL and 0:
                                        show_3d(out[:,0].detach(), inputs_3d_gt[:,0])
                                    out = out.permute(0, 1, 4, 2, 3)
                                    inputs_3d_gt = inputs_3d_gt.permute(0, 1, 4, 2, 3)
                                    loss = mpjpe(out, inputs_3d_gt)
                                    Num += out.shape[0]
                                    epoch_loss_valid += loss.item() * out.shape[0]

                    epoch_loss_valid = epoch_loss_valid / Num * 1000
                    print(key, epoch_loss_valid)
                
        if EVAL:
            exit()
        epoch += 1
        elapsed = (time() - start_time)/60
        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        print('epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.2f}'.format(epoch, elapsed, lr, best_result_epoch, best_result))

        epoch_step = 1
        if epoch % epoch_step == 0:
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            momentum = initial_momentum * np.exp(-epoch/epoch_step/args.epochs * np.log(initial_momentum/final_momentum))
            model.module.set_bn_momentum(momentum)
                                          
        if epoch % 1 == 0:
            chk_path = os.path.join('./checkpoint', 'epoch_{}_total.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model':model.module.state_dict(),
            }, chk_path)
print('epoch:',best_result_epoch, 'best_result:', best_result)
torch.save({'epoch': best_result_epoch, 'model':best_state_dict}, os.path.join('./checkpoint', 'best_total.bin'))
