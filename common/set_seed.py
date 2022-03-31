import torch
import random
import numpy as np
import sys, os
import torch.backends.cudnn as cudnn
def set_seed():
    seed = 1
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic  = False
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = False
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

