import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('--cfg', help="Specify the path of the path of the config(*.yaml)", default='')
    parser.add_argument('--h36m_detector', help="p2d detector for human36m", default='')
    parser.add_argument('--resume_checkpoint', help="Specify the path of the path of the resume checkpoint *.bin", default='')
    parser.add_argument('--gpu', type=str, nargs='+',  help='gpu ids') 
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--triangulate', action='store_true', help='')
    parser.add_argument('--no_test_flip', dest = 'test_flip', action='store_false', help='flip augmentation when test')
    parser.add_argument('--no_test_rot', dest = 'test_rot', action='store_false', help='multi-view rot augmentation when test')
    parser.add_argument('--log', action='store_true', help='tensorboard log')
    parser.add_argument('--checkpoint', help="model *.bin", default='')
    parser.add_argument('--debug', action='store_true', help='no tensorboard log')
    parser.add_argument('--eval_n_views', type=int, nargs='+',  help='number of views when eval')
    parser.add_argument('--eval_view_list', type=int, nargs='+',  help='cameras when eval')
    parser.add_argument('--eval_n_frames', type=int, nargs='+',  help='number of frames when eval')
    parser.add_argument('--n_frames', type=int,  help='number of frames when training')
    parser.add_argument('--eval_batch_size', type=int, help='batch size when eval')
    parser.add_argument('--metric', help="eval metric", default='mpjpe') #['mpjpe', 'p_mpjpe', 'n_mpjpe']
    parser.add_argument('--no_align_r', dest = 'align_r', action='store_false', help='align rotation(metric)')
    parser.add_argument('--no_align_t', dest = 'align_t',action='store_false', help='align translatio(metric)n')
    parser.add_argument('--no_align_s', dest = 'align_s', action='store_false', help='align scale(metric)')
    parser.add_argument('--no_align_trj', dest = 'align_trj', action='store_false', help='align triangulation')
    parser.add_argument('--no_trj_align_r', dest = 'trj_align_r', action='store_false', help='align rotation(triangulation)')
    parser.add_argument('--trj_align_t', action='store_true', help='align translation(triangulation)')
    parser.add_argument('--no_trj_align_s', dest = 'trj_align_s', action='store_false', help='align scale(triangulation)')
    # Visualization
    parser.add_argument('--vis_3d', action='store_true', help='if vis 3d pose')
    parser.add_argument('--vis_complexity', action='store_true', help='if vis complexity')
    parser.add_argument('--vis_debug', action='store_true', help='save vis fig')
    parser.add_argument('--vis_grad', action='store_true', help='')
    parser.add_argument('--vis_dataset', help="Specify the name of the vis datast", default='h36m')
    
    
    parser.set_defaults(align_r=True)
    parser.set_defaults(align_t=True)
    parser.set_defaults(align_s=True)
    parser.set_defaults(align_trj=True)
    parser.set_defaults(trj_align_r=True)
    parser.set_defaults(trj_align_s=True)
    parser.set_defaults(test_flip=True)
    parser.set_defaults(test_rot=True)
    
    args = parser.parse_args()


    return args
