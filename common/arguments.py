import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-gpu', '--gpu_ids', default='0', type=str, nargs='+',  help='gpu ids') 
    parser.add_argument('-trc', '--train_camera', default='0,1,2,3', type=str,  help='train cameras') 
    parser.add_argument('-tec', '--test_camera', default='0,1,2,3', type=str,  help='test cameras') 
    parser.add_argument('-av', '--add_view', default=1, type=int, metavar='N', help='number of created new cameras')
    parser.add_argument('-ch', '--channels', default=600, type=int, metavar='N', help='channels')
    parser.add_argument('-d', '--dim', default=3, type=int, metavar='N', help='dim of transform matrix')
    parser.add_argument('-t', '--t_length', default=7, type=int, metavar='N', help='length of sequence')
    parser.add_argument('-mr', '--mask_rate', default=0.4, type=float, metavar='N', help='length of sequence')
    
    parser.add_argument('-drop', '--dropout', default=0.1, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=720, type=int, metavar='N', help='batch size in terms of predicted frames')
    
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')

    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')

    # Experimental
    parser.add_argument('--eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--checkpoint', default='./checkpoint/multi_view_4_mpjpe_att_tran_conf_T1.bin', type=str,  help='path to checkpoint')
    parser.add_argument('--eval_n_views', default='4', type=int, nargs='+',  help='number of views when eval') 
    parser.add_argument('--eval_n_frames', default='1', type=int, nargs='+',  help='number of frames when eval') 
    
    # ablation study
    parser.add_argument('--conf', default='modulate', type=str,  help='how to use confidence [no, concat, modulate]')
    parser.add_argument('-no_fa','--no-feature-alignment',  dest='feature_alignment', action='store_false',  help='disable features align')
    parser.add_argument('-no_mf','--no-multiview-fuse', dest='multiview_fuse', action='store_false',  help='disable multi-view fusion')
    parser.add_argument('-no_att','--no-attention', dest='attention', action='store_false',  help='disable attention')

    # Visualization
    parser.add_argument('--vis_3d', action='store_true', help='if vis 3d pose')
    parser.add_argument('--vis_complexity', action='store_true', help='if vis complexity')
    
    
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)
    parser.set_defaults(feature_alignment=True)
    parser.set_defaults(multiview_fuse=True)
    parser.set_defaults(attention=True)
    
    args = parser.parse_args()
    print(args)


    return args