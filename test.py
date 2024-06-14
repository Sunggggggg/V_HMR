import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from lib.dataset._loaders import get_data_loaders
from lib.utils.utils import create_logger

def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed(cfg.SEED_VALUE)
        torch.cuda.manual_seed_all(cfg.SEED_VALUE)
        
    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ========= Dataloaders ========= #
    data_loaders = get_data_loaders(cfg)

    
