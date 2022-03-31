# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from itertools import zip_longest
import numpy as np
import torch
from torch.utils.data import Dataset
import sys,os
import copy
import random
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)
N_DCT = 49
K_DCT = 7
class VideoAug:
    def __init__(self, N = 51, K = 7):
        super().__init__()
        self.N = N
        self.K = K
    def __call__(self, x):
        x = torch.from_numpy(x)
        NUM = 0
        N = self.N
        K = self.K
        B, V, C = x.shape
        assert N % 2 == 1
        out = torch.zeros(B, V, C *K)
        pre_tmp = torch.zeros(B + N -1, V, C)
        pad = (N - 1) // 2
        pre_tmp[:pad] = x[:1]
        pre_tmp[pad:-pad] = x
        pre_tmp[-pad:] = x[-1:]

        fixed_bases = [np.ones([N]) * np.sqrt(0.5)]
        x = np.arange(N)
        for i in range(1, K):
            fixed_bases.append(np.cos(i * np.pi * ((x + 0.5) / N)))
        fixed_bases = np.array(fixed_bases)
        bases_tmp = torch.from_numpy(fixed_bases).float()
        bases_tmp = bases_tmp.permute(1,0)#(N,K)
        for i in range(B):
            pos = pre_tmp[i:i+N]#(N, V, C)
            pos = pos.view(N, -1).permute(1, 0)#(V*C, N)      
            tmp = torch.matmul(pos, bases_tmp) / N * 2#(V * C, K), 
            tmp = tmp.view(1, V, C*K)
            out[i] = tmp
        return out.numpy()
    
class ChunkedGenerator(Dataset):
    def __init__(self, batch_size, cameras,  poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, use_dct = False, step = 1):
        pad = 0
        tmp = []
        for i in range(len(poses_2d[0])):
            tmp.append(np.concatenate((poses_2d[0][i][...,np.newaxis], poses_2d[1][i][...,np.newaxis],poses_2d[2][i][...,np.newaxis],poses_2d[3][i][...,np.newaxis]), axis = -1))
        self.db = tmp
        
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for view_idx in range(len(poses_2d)):
            for i in range(len(poses_2d[view_idx])):
                n_chunks = (poses_2d[view_idx][i].shape[0] + chunk_length - 1) // chunk_length

                offset = (n_chunks * chunk_length - poses_2d[view_idx][i].shape[0]) // 2
                bounds = np.arange(n_chunks+1)*chunk_length - offset
                augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                
                pairs += zip(np.repeat(view_idx, len(bounds - 1)),np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
                if augment:
                    pairs += zip(np.repeat(view_idx, len(bounds - 1)),np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)
        
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.videoaug = VideoAug(N = N_DCT, K = K_DCT)
        self.cameras = cameras


        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.step = step
        if use_dct:
            self.hand_db()

        self.batch_2d = np.empty((batch_size, chunk_length + 2*(pad // step), poses_2d[0][0].shape[-2], poses_2d[0][0].shape[-1], 4))

        
    def hand_db(self):
        if self.poses_2d[0].shape[-1] == 18 or self.poses_2d[0].shape[-1] == 19:
            for i in range(len(self.poses_2d)):
                video = self.videoaug(self.poses_2d[i][:,:,15:18])
                #video = self.videoaug(self.poses_2d[i][:,:,2:4])
                self.poses_2d[i] = np.concatenate((self.poses_2d[i], video), axis = -1)
                
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (view_idx, seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    # 2D poses
                    seq_2d = self.db[seq_i]
                
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        tmp = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                        self.batch_2d[i] = tmp[::self.step]
                    else:
                        tmp  =seq_2d[low_2d:high_2d]
                        self.batch_2d[i] = tmp[::self.step]
                    
                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                        if self.batch_2d.shape[-2] == 15:
                            self.batch_2d[i, :, :, 2] *= -1
                            self.batch_2d[i, :, :, 4] *= -1
                        else:
                            print(self.batch_2d.shape[-2])
                            sys.exit()

                if self.endless:
                    self.state = (b_i + 1, pairs)

                yield self.batch_2d[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            
class UnchunkedGenerator:
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None, use_dct = False):
        poses_2d = copy.deepcopy(poses_2d)

        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        tmp = []
        for i in range(len(poses_2d[0])):
            tmp.append(np.concatenate((poses_2d[0][i][...,np.newaxis], poses_2d[1][i][...,np.newaxis],poses_2d[2][i][...,np.newaxis],poses_2d[3][i][...,np.newaxis]), axis = -1))
        self.poses_2d = tmp
        self.num_epoches = len(poses_2d[0])
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
            if p.shape[0] == 2167:
                print('fffffffffff')
                sys.exit()
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_id in range(len(self.poses_2d)):
            seq_2d = self.poses_2d[seq_id]
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0), (0, 0)),
                            'edge'), axis=0)
            
            
            if self.augment:
                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
              
                if batch_2d.shape[-2] == 15:
                    batch_2d[1, :, :, 2] *= -1
                    batch_2d[1, :, :, 4] *= -1    
                else:
                    sys.exit()
            yield batch_2d
            
            
            
            
            
            
            
            
            
            
