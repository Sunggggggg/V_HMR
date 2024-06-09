# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
GLoT_DB_DIR = '/home/dev4/data/SKY/V_HMR/data/preprocessed_data/FullFrame_vitpose_r5064'
AMASS_DIR = '/home/dev4/data/SKY/data/amass'
INSTA_DIR = '/home/dev4/data/SKY/data/V_HMR/insta_variety'
MPII3D_DIR = '/home/dev4/data/SKY/data/V_HMR/mpi_inf_3dhp'
THREEDPW_DIR = '/home/dev4/data/SKY/V_HMR/data/3dpw'
H36M_DIR = '/home/dev4/data/SKY/data/h36m'
PENNACTION_DIR = '/home/dev4/data/SKY/data/penn_action'
POSETRACK_DIR = '/home/dev4/data/SKY/data/posetrack'
BASE_DATA_DIR = '/home/dev4/data/SKY/V_HMR/data/base_data'


# Configuration variables
cfg = CN()
# cfg.clip_norm_num = 10
cfg.TITLE = 'default'
cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.DEBUG = True
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 4
cfg.DEBUG_FREQ = 1000
cfg.SEED_VALUE = -1
cfg.render = False

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_2D = ['Insta']
cfg.TRAIN.DATASETS_3D = ['MPII3D']
cfg.TRAIN.DATASET_EVAL = 'ThreeDPW'
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.OVERLAP = 0.25
cfg.TRAIN.DATA_2D_RATIO = 0.5
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 5
cfg.TRAIN.PRETRAINED_REGRESSOR = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.RESUME = ''
cfg.TRAIN.NUM_ITERS_PER_EPOCH = 1000
cfg.TRAIN.LR_PATIENCE = 5
cfg.TRAIN.val_epoch=5
# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 1e-4
cfg.TRAIN.GEN_MOMENTUM = 0.9

# <====== motion discriminator optimizer
cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 16
cfg.DATASET.OVERLAP = 0.5

            
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/config.yaml', help='cfg file path')
    parser.add_argument('--gpu', type=str, default='1', help='gpu num')
    # evaluation options
    parser.add_argument('--dataset', type=str, default='3dpw', help='pick from 3dpw, mpii3d, h36m')
    parser.add_argument('--seq', type=str, default='', help='render target sequence')
    parser.add_argument('--render', action='store_true', help='render meshes on an rgb video')
    parser.add_argument('--render_plain', action='store_true', help='render meshes on plain background')
    parser.add_argument('--filter', action='store_true', help='apply smoothing filter')
    parser.add_argument('--plot', action='store_true', help='plot acceleration plot graph')
    parser.add_argument('--frame', type=int, default=0, help='render frame start idx')

    args = parser.parse_args()
    print(args, end='\n\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()
    cfg.render = args.render

    return cfg, cfg_file, args
