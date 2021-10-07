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

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)


link = np.array([[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]])

color_left = 'B'
color_right = 'G'
color_other = 'R'
bone_color = [color_right, color_right, color_right, color_left, color_left, color_left, color_other, color_other, color_other, color_other, color_left, color_left, color_left, color_right, color_right, color_right]
RGB_color= {'B':(0, 0, 255), 'G':(0, 255, 0), 'R':(255, 0, 0)}


keypoints = {}
for sub in [1, 5, 6, 7, 8, 9, 11]:
    keypoint = np.load('data/h36m_sub{}.npz'.format(sub), allow_pickle=True)
    keypoints_metadata = keypoint['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints['S{}'.format(sub)] = keypoint['positions_2d'].item()['S{}'.format(sub)]

kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = [kps_left, kps_right]

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

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

actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases','Sitting','SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
train_actions = actions
test_actions = actions
vis_actions = actions
train_cameras = [int(v) for v in args.train_camera.split(',')]
test_cameras = [int(v) for v in args.test_camera.split(',')]

action_frames = {}
for act in actions:
    action_frames[act] = 0
for k,v in N_frame_action_dict.items():
    action_frames[v] += k

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
use_inter_loss = True
receptive_field = args.t_length
pad = receptive_field // 2
causal_shift = 0
model = VideoMultiViewModel(args, num_view = len(train_cameras) + args.add_view, is_train = True, use_inter_loss = use_inter_loss)
model_test = VideoMultiViewModel(args, num_view = len(train_cameras) + args.add_view,is_train = False)

if args.vis_complexity:
    model_test.eval()
    for i in range(1,5):
        input = torch.randn(1, receptive_field,17,3,i)
        macs, params = profile(model_test, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
        print('view: {} T: {} MACs:{} params:{}'.format(i, receptive_field, macs, params))
else:
    total_params = sum(p.numel() for p in model_test.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

EVAL = args.eval
if EVAL and args.vis_3d:
    plt.ion()
    fig = plt.figure()
    ax_views = []
    for i in range(4):
        t = '24{}'.format(i + 1)
        ax = fig.add_subplot(t, projection='3d')
        ax_views.append(ax)
    for i in range(4):
        t = '24{}'.format(i + 1 + 4)
        ax = fig.add_subplot(t)
        ax_views.append(ax)
        
def load_state(model_train, model_test):
    train_state = model_train.state_dict()
    test_state = model_test.state_dict()
    pretrained_dict = {k:v for k, v in train_state.items() if k in test_state}
    test_state.update(pretrained_dict)
    model_test.load_state_dict(test_state)
    
def show_3d(pose_2d, predicted,gt):
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
                
            for bone_id, l in enumerate(link[1:]):
                x = list(pose_2d[i, l, 0, view_id])
                y = list(-pose_2d[i, l, 1, view_id])
                
                ax_views[view_id + 4].plot(x, y, C = 'G')


        plt.pause(0.1)
    
if EVAL:
    chk_filename = args.checkpoint
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['model'])
    
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    model_test = torch.nn.DataParallel(model_test).cuda()

if True:
    poses_train_2d = fetch(subjects_train, train_actions)

    lr = args.learning_rate
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, amsgrad=True)    
    lr_decay = args.lr_decay
    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    train_generator = ChunkedGenerator(args.batch_size, poses_train_2d, 1,pad=pad, causal_shift=causal_shift, shuffle=True, augment=True,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
 
    print('** Starting.')
    best_result = 100
    best_state_dict = None
    best_result_epoch = 0
    # Pos model only
    data_aug = DataAug(add_view = args.add_view)
    while epoch < args.epochs:
        start_time = time()
        model.train()
        process = tqdm(total = train_generator.num_batches)

        for batch_2d in train_generator.next_epoch():
            if EVAL:
                break
            
            process.update(1)
            inputs = torch.from_numpy(batch_2d.astype('float32'))
            assert inputs.shape[-2] == 8
            inputs_2d_gt = inputs[...,:,:2,:]
            inputs_2d_pre = inputs[...,2:4,:]
            cam_3d = inputs[..., 4:7,:]
            B, T, V, C, N = cam_3d.shape
            if use_2d_gt:
                vis = torch.ones(B, T, V, 1, N)
            else:
                vis = inputs[...,7:8, :]
            
            if args.add_view:
                vis = torch.cat((vis, torch.ones(B, T, V, 1, args.add_view)), dim = -1)
            
            
            inputs_3d_gt = cam_3d
            inputs_3d_gt = inputs_3d_gt.cuda()
            
            inputs_3d_gt_root = copy.deepcopy(inputs_3d_gt[:,:, :1])
            inputs_3d_gt[:,:,0] = 0
            
            view_list = list(range(len(train_cameras) + args.add_view))
            
            if args.add_view > 0:
                pos_gt_3d_tmp = copy.deepcopy(inputs_3d_gt)
                pos_gt_3d_tmp[:,:,:1] = inputs_3d_gt_root

                pos_gt_2d, pos_gt_3d = data_aug(pos_gt_2d = inputs_2d_gt, pos_gt_3d = pos_gt_3d_tmp)
                
                pos_gt_3d[:,:,:1] = 0
                pos_pre_2d = torch.cat((inputs_2d_pre, pos_gt_2d[...,inputs_2d_pre.shape[-1]:]), dim = -1)
                
                if use_2d_gt:
                    h36_inp = pos_gt_2d[..., view_list]
                else:
                    h36_inp = pos_pre_2d[..., view_list]
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
            NUM_VIEW = len(test_cameras)
            USE_FLIP = args.test_time_augmentation
            TEST_VIEW = args.eval_n_views if isinstance(args.eval_n_views, list) else [args.eval_n_views]
            eval_t = args.eval_n_frames if isinstance(args.eval_n_frames, list) else [args.eval_n_frames]
            for t_len in eval_t:
                epoch_loss_valid = 0  
                action_mpjpe = {}
                for act in actions:
                    action_mpjpe[act] = [0] * NUM_VIEW
                    for i in range(NUM_VIEW):
                        action_mpjpe[act][i] = [0] * (NUM_VIEW + 1)
                N = [0] * NUM_VIEW
                for i in range(NUM_VIEW):
                    N[i] = [0] * (NUM_VIEW + 1)
                    
                for num_view in TEST_VIEW:
                    pad_t = t_len // 2
                    for view_list in itertools.combinations(list(range(NUM_VIEW)), num_view):
                        view_list = list(view_list)
                        N[num_view - 1][-1] += 1
                        for i in view_list:
                            N[num_view - 1][i] += 1
                        for act in vis_actions if EVAL else actions:
                            poses_valid_2d = fetch(subjects_test, [act], is_test =True)

                            test_generator = ChunkedGenerator(args.batch_size//2, poses_valid_2d, 1,pad=pad_t, causal_shift=causal_shift, shuffle=False, augment=False,kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                            for batch_2d in test_generator.next_epoch():
                                inputs = torch.from_numpy(batch_2d.astype('float32'))
                                inputs_2d_gt = inputs[...,:2,:]
                                inputs_2d_pre = inputs[...,2:4,:]

                                cam_3d = inputs[..., 4:7,:]
                                vis = inputs[...,7:8,:]
                                inputs_3d_gt = cam_3d[:,pad_t:pad_t+1]

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

                                if USE_FLIP:
                                    out, ag, _, _, _ = model_test(torch.cat((inp, inp_flip), dim = 0))

                                    out[B:,:,:,0] *= -1
                                    out[B:,:,joints_left + joints_right] = out[B:,:,joints_right + joints_left]

                                    out = (out[:B] + out[B:]) / 2
                                    out[:,:,0] = 0
                                else:
                                    out, ag = model_test(inp)
                                    out[:,:,0] = 0

                                if EVAL and args.vis_3d:
                                    show_3d(inputs_2d_pre[:,pad], out[:,0].detach(), inputs_3d_gt[:,0])

                                loss = 0

                                for idx, view_idx in enumerate(view_list):
                                    loss_view_tmp = mpjpe(out[..., idx], inputs_3d_gt[..., view_idx])
                                    loss += loss_view_tmp.item()
                                    action_mpjpe[act][num_view - 1][view_idx] += loss_view_tmp.item() * inputs_3d_gt.shape[0]

                                action_mpjpe[act][num_view - 1][-1] += loss * inputs_3d_gt.shape[0]

                print('num_actions :{}'.format(len(action_frames)))
                for num_view in TEST_VIEW:
                    tmp = 0
                    print('num_view:{}'.format(num_view))
                    for act in action_mpjpe:
                        for i in range(NUM_VIEW):
                            action_mpjpe[act][num_view - 1][i] /= (action_frames[act] * N[num_view - 1][i])
                        action_mpjpe[act][num_view - 1][-1] /= (action_frames[act] * N[num_view - 1][-1] * num_view)
                        print('mpjpe of {:18}'.format(act), end = '     ')
                        for i in range(NUM_VIEW):
                            print('view_{}: {:.2f}'.format(i, action_mpjpe[act][num_view - 1][i] * 1000), end = '      ')
                        print('avg: {:.2f}'.format(action_mpjpe[act][num_view - 1][-1] * 1000))
                        tmp += action_mpjpe[act][num_view - 1][-1] * 1000
                    print(tmp / len(action_frames))
                    epoch_loss_valid += tmp / len(action_frames)
                epoch_loss_valid /= len(TEST_VIEW)
                print('t_len:{} avg:{:.2f}'.format(t_len, epoch_loss_valid))
            if EVAL:
                exit()
            
        epoch += 1
        
        if epoch_loss_valid < best_result:
            best_result = epoch_loss_valid
            best_state_dict = copy.deepcopy(model.module.state_dict())
            best_result_epoch = epoch
        elapsed = (time() - start_time)/60
        print('epoch:{:3} time:{:.2f} lr:{:.9f} best_result_epoch:{:3} best_result:{:.2f}'.format(epoch, elapsed, lr, best_result_epoch, best_result))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        model.module.set_bn_momentum(momentum)
                                          
        chk_path = os.path.join('./checkpoint', 'epoch_{}_h36m_2.bin'.format(epoch))
        print('Saving checkpoint to', chk_path)
            
        torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model':model.module.state_dict(),
            }, chk_path)
          
print('epoch:',best_result_epoch, 'best_result:', best_result)
torch.save({'epoch': best_result_epoch, 'model':best_state_dict}, os.path.join('./checkpoint', 'best_h36m2.bin'))
