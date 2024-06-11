import os
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset
from lib.core.config import GLoT_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks, get_single_image_crop

logger = logging.getLogger(__name__)

class Dataset3D(Dataset):
    def __init__(self, load_opt, set, seqlen, overlap=0., folder=None, dataset_name=None, debug=False):
        self.load_opt = load_opt
        self.folder = folder
        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)

        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.debug = debug
        self.db = self.load_db()

        print("is_train: ", (set=='train'))
        self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride, is_train=(set=='train'))

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_db_clip.pt')

        if self.set == 'train':
            if self.load_opt == 'repr_table4_3dpw_model':
                if self.dataset_name == '3dpw':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_occ_db_clip.pt')
                elif self.dataset_name == 'mpii3d':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_occ_db_clip.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_occ_db_clip.pt')

            elif self.load_opt == 'repr_table4_h36m_mpii3d_model':
                if self.dataset_name == '3dpw':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
                elif self.dataset_name == 'mpii3d':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_db.pt')

            elif self.load_opt == 'repr_table6_3dpw_model':
                if self.dataset_name == 'mpii3d':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_occ_db.pt')
                elif self.dataset_name == 'h36m':
                    db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_25fps_occ_db.pt')

        elif self.set == 'val' and self.dataset_name == 'mpii3d':
            db_file = osp.join(GLoT_DB_DIR, f'{self.dataset_name}_{self.set}_scale12_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        img_names = self.get_sequence(start_index, end_index, self.db['img_name'])
        print(img_names)

        return img_names

if __name__ == "__main__" :
    dataset = Dataset3D(load_opt='repr_table4_3dpw_model', set='train', seqlen=16, overlap=0.0, dataset_name='3dpw')
    print(len(dataset))

