import os
import os.path as osp
from tabnanny import check
import yaml
import numpy as np
import argparse
from easydict import EasyDict as edict
from pathlib import Path
import time

# basic hyper-parameters
config = edict()
config.GPU = ['0', '1']
config.LOG_DIR = ''

# Debug
config.DEBUG = False
# Train
config.TRAIN = edict()
config.TRAIN.NUM_AUGMENT_VIEWS = 1
config.TRAIN.NUM_EPOCHES = 60
config.TRAIN.BATCH_SIZE = 720
config.TRAIN.LEARNING_RATE = 0.001
config.TRAIN.LR_DECAY = 0.95
config.TRAIN.CHECKPOINT = ""
config.TRAIN.USE_INTER_LOSS = True
config.TRAIN.INTER_LOSS_NAME = ['before_fuse', 'after_fuse']
config.TRAIN.INTER_LOSS_WEIGHT = [0.1, 0.5]

config.TRAIN.USE_MV_LOSS = False
config.TRAIN.MV_LOSS_WEIGHT = 0.5

config.TRAIN.USE_ROT_LOSS = True
config.TRAIN.ROT_LOSS_WEIGHT = 0.5

config.TRAIN.RESUME = False
config.TRAIN.RESUME_CHECKPOINT = ""


#Network
config.NETWORK = edict()
config.NETWORK.NUM_CHANNELS = 600
config.NETWORK.INPUT_DIM = 2 #不考虑confidence的维度
config.NETWORK.USE_GT_TRANSFORM = False
config.NETWORK.TRANSFORM_DIM = 4
config.NETWORK.TEMPORAL_LENGTH = 7
config.NETWORK.TEMPORAL_MASK = []
config.NETWORK.MASK_RATE = 0.4
config.NETWORK.DROPOUT = 0.1
config.NETWORK.CONFIDENCE_METHOD = 'modulate' #[no, concat, modulate]
config.NETWORK.USE_FEATURE_TRAN = True
config.NETWORK.USE_MFT = True

#### multi-view transformer
config.NETWORK.M_FORMER = edict()
config.NETWORK.M_FORMER.MODE = 'mtf' #('mtf', 'origin', 'point')
config.NETWORK.M_FORMER.NUM_RELATION_LAYERS = 1
config.NETWORK.M_FORMER.GT_TRANSFORM_RES = False #use_gt_transform = True: pred_tran + gt_tran
config.NETWORK.M_FORMER.USE_MEAN_TRANSFORM = False
config.NETWORK.M_FORMER.MASK_SELF = False
config.NETWORK.M_FORMER.GT_TRANSFORM_MODE = 'r' #('r', 'rt')
config.NETWORK.M_FORMER.USE_POSE2D = True
config.NETWORK.M_FORMER.DROPOUT = 0

#### rotation model(use gt transform)
config.NETWORK.ROT_MODEL = edict()
config.NETWORK.ROT_MODEL.NUM_LAYERS = 0
config.NETWORK.ROT_MODEL.NUM_CHANNELS = 300

#### temporal transformer
config.NETWORK.T_FORMER = edict()
config.NETWORK.T_FORMER.NUM_LAYERS = 2
config.NETWORK.T_FORMER.NUM_HEADS = 8
config.NETWORK.T_FORMER.NUM_CHANNELS = 512
# DATA
config.DATA = edict()
config.DATA.USE_GT_2D = False
config.DATA.DATASET_NAME = "h36m"

### H36M Data
config.H36M_DATA = edict()
config.H36M_DATA.ROOT_DIR = '/home/wulele/DataSet/Human36M/images'
config.H36M_DATA.P2D_DETECTOR = 'cpn' #('cpn', 'ada_fuse', 'gt')
config.H36M_DATA.SUBJECTS_TRAIN = ['S1','S5','S6','S7','S8']
config.H36M_DATA.SUBJECTS_TEST = ['S9','S11']
config.H36M_DATA.TRAIN_CAMERAS = [0, 1, 2, 3]
config.H36M_DATA.TEST_CAMERAS = [0, 1, 2, 3]
config.H36M_DATA.BONES = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12, 13], [8, 14],[14, 15], [15, 16]]
config.H36M_DATA.BONES_FLAG = ['m', 'r', 'r', 'm', 'l', 'l', 'm', 'm', 'm', 'm', 'm', 'l', 'l', 'm', 'r', 'r']
config.H36M_DATA.NUM_JOINTS = 17
config.H36M_DATA.JOINT_SYMMETRY = [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]


### Total capture
config.TOTALCAP_DATA = edict()
config.TOTALCAP_DATA.ROOT_DIR = '/home/wulele/DataSet/TotalCapture/TotalCapture-Toolbox-master/data/'
config.TOTALCAP_DATA.POSE2D_DIR = '/home/wulele/code/HRNet-Human-Pose-Estimation-master/result/hrnet_w32_384x288_0_9864626107450614'
config.TOTALCAP_DATA.BONES = [[0, 0],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[8,10],[10,11],[11, 12], [8, 13],[13, 14], [14, 15]]
config.TOTALCAP_DATA.NUM_JOINTS = 16


### KTH
config.KTH_DATA = edict()
config.KTH_DATA.ROOT_DIR = '/home/wulele/DataSet/KTH_MV_Football_2'
#Test
config.TEST = edict()
config.TEST.TRIANGULATE = False
config.TEST.TEST_FLIP = True
config.TEST.TEST_ROTATION = False
config.TEST.CHECKPOINT = "./checkpoint/best_t7_h36m_2.bin"
config.TEST.NUM_VIEWS = [4]
config.TEST.NUM_FRAMES = [1]
config.TEST.BATCH_SIZE = 360
config.TEST.METRIC = "mpjpe" #['mpjpe', 'p_mpjpe', 'n_mpjpe']
config.TEST.METRIC_ALIGN_R = True
config.TEST.METRIC_ALIGN_T = True
config.TEST.METRIC_ALIGN_S = True
config.TEST.ALIGN_TRJ = True
config.TEST.TRJ_ALIGN_R = True
config.TEST.TRJ_ALIGN_T = False
config.TEST.TRJ_ALIGN_S = True

#VIS
config.VIS = edict()
config.VIS.DATASET = 'kth' #('kth', 'h36m')
config.VIS.DEBUG = False
config.VIS.VIS_3D = False
config.VIS.VIS_GRAD = False
config.VIS.VIS_COMPLEXITY = False
config.VIS.BONE_COLOR = {'m':[56, 83, 163, 255], 'r':[238, 31, 35, 255], 'l':[105, 189, 69, 255]}


def _update_dict(k, v, sub_config):
    for vk, vv in v.items():
        if vk in sub_config:
            if isinstance(vv, dict):
                _update_dict(vk, vv, sub_config[vk])
            else:
                sub_config[vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def reset_config(cfg, args):
    if args.h36m_detector != '':
        cfg.H36M_DATA.P2D_DETECTOR = args.h36m_detector
        cfg.DATA.USE_GT_2D = args.h36m_detector == 'gt'
    if args.gpu is not None:
        cfg.GPU = [args.gpu] if isinstance(args.gpu, str) else args.gpu
    if args.eval_n_views is not None:
        cfg.TEST.NUM_VIEWS = [args.eval_n_views] if isinstance(args.eval_n_views, int) else args.eval_n_views
    if args.eval_n_frames is not None:
        cfg.TEST.NUM_FRAMES = [args.eval_n_frames] if isinstance(args.eval_n_frames, int) else args.eval_n_frames
    if args.eval_view_list is not None:
        cfg.H36M_DATA.TEST_CAMERAS = [args.eval_view_list] if isinstance(args.eval_view_list, int) else args.eval_view_list
    if args.eval:
        cfg.TEST.TEST_ROTATION = args.test_rot
        cfg.TEST.TEST_FLIP = args.test_flip
        cfg.VIS.VIS_3D = args.vis_3d
        cfg.VIS.VIS_GRAD = args.vis_grad
        cfg.TEST.TRIANGULATE = args.triangulate
        if not cfg.TEST.TRIANGULATE:
            cfg.NETWORK.TEMPORAL_LENGTH = args.n_frames
            assert args.n_frames is not None, 'args.n_frames 需要在eval时指定，否则会得到错误结果（原因是：时序网络的位置编码）'
    if args.eval_batch_size is not None:
        cfg.TEST.BATCH_SIZE = args.eval_batch_size
    
    cfg.TEST.METRIC = args.metric
    cfg.TEST.METRIC_ALIGN_R = args.align_r
    cfg.TEST.METRIC_ALIGN_T = args.align_t
    cfg.TEST.METRIC_ALIGN_S = args.align_s
    cfg.TEST.ALIGN_TRJ = args.align_trj
    cfg.TEST.TRJ_ALIGN_R = args.trj_align_r
    cfg.TEST.TRJ_ALIGN_T = args.trj_align_t
    cfg.TEST.TRJ_ALIGN_S = args.trj_align_s
    
    ###vis
    cfg.VIS.VIS_COMPLEXITY = args.vis_complexity
    cfg.VIS.DEBUG = args.vis_debug
    cfg.VIS.DATASET = args.vis_dataset
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    cfg_name = osp.basename(args.cfg).split('.')[0]
    cfg_dir = osp.dirname(args.cfg).split('/')[-1]
    
    config.DEBUG = args.debug
    if not args.debug and (not args.eval or args.log):
        tensorboard_log_dir = Path('./log') / (cfg_dir) / (cfg_name + "_" + time_str)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR = tensorboard_log_dir
        
    if not args.debug and not args.eval:
        checkpoint_dir = Path('./checkpoint') / (cfg_dir) / (cfg_name + "_" + time_str)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.TRAIN.CHECKPOINT = checkpoint_dir / ('model.bin')
        
    config.TRAIN.RESUME_CHECKPOINT = args.resume_checkpoint
    if config.TRAIN.RESUME_CHECKPOINT != "":
        config.TRAIN.RESUME = True
    config.TEST.CHECKPOINT = args.checkpoint 
    
def update_config(config_file):
    if config_file == '':
        return
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    for k, v in exp_config.items():
        if k in config:
            if isinstance(v, dict):
                _update_dict(k, v, config[k])
            else:
                config[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))
