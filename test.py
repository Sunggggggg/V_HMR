import os
import os.path as osp
import cv2
import torch
import joblib
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import importlib
from lib.core.Motion.config import BASE_DATA_DIR, parse_args
from lib.data_utils._img_utils import split_into_chunks_test
from lib.data_utils._kp_utils import convert_kps
from lib.models.smpl import SMPL_MODEL_DIR, SMPL, H36M_TO_J14
from lib.utils.demo_utils import convert_crop_cam_to_orig_img, images_to_video
from lib.utils.eval_utils import compute_accel, compute_error_accel, batch_compute_similarity_transform_torch, compute_error_verts, compute_errors, plot_accel
from lib.utils.slerp_filter_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix

def get_sequence(start_index, end_index, seqlen=16):
    if end_index - start_index + 1 == seqlen:
        return [i for i in range(start_index, end_index+1)]
    else:
        seq = []
        if start_index == 0:
            for i in range(seqlen - (end_index - start_index + 1)):
                seq.append(start_index)
            for i in range(start_index, end_index + 1):
                seq.append(i)
        else:
            for i in range(start_index, end_index + 1):
                seq.append(i)
            for i in range(seqlen - (end_index - start_index + 1)):
                seq.append(end_index)
                
        return seq
    

""" Smoothing codes from MEVA (https://github.com/ZhengyiLuo/MEVA) """
def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q-1] - quat[q], axis=0) > np.linalg.norm(quat[q-1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_smooth(quat, ratio = 0.3):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        quat[q] = quaternion_slerp(quat[q-1], quat[q], ratio)
    return quat


def smooth_pose_mat(pose, ratio = 0.3):
    quats_all = []
    for j in range(pose.shape[1]):
        quats = []
        for i in range(pose.shape[0]):
            R = pose[i,j,:,:]
            quats.append(quaternion_from_matrix(R))
        quats = quat_correct(np.array(quats))
        quats = quat_smooth(quats, ratio = ratio)
        quats_all.append(np.array([quaternion_matrix(i)[:3,:3] for i in quats]))

    quats_all = np.stack(quats_all, axis=1)
    return quats_all


if __name__ == "__main__":
    cfg, cfg_file, args = parse_args()
    SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    """ Evaluation Options """
    target_dataset = args.dataset  # 'mpii3d' '3dpw' 'h36m'
    set = 'test'
    target_action = args.seq
    render = args.render or args.render_plain
    render_plain = args.render_plain
    only_img = False
    render_frame_start = args.frame
    plot = args.plot
    avg_filter = args.filter
    gender = 'neutral'

    dtype = torch.float
    J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    """ Data """
    seqlen = 16
    stride = 1  # seqlen
    out_dir = f'./output/{target_dataset}_test_output' # your path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if target_dataset == '3dpw':
        data_path = f'/home/dev4/data/SKY/V_HMR/data/preprocessed_data/FullFrame_vitpose_r5064/3dpw_{set}_db_clip.pt'
        caption_path = f'/mnt/SKY/V_HMR/data/preprocessed_data/Video_caption/3dpw_train_caption.pt'
    elif target_dataset == 'h36m':
        if cfg.TITLE == 'repr_table4_h36m_mpii3d_model':
            data_path = f'/mnt/SKY/preprocessed_data/{target_dataset}_{set}_25fps_db_clip.pt'  # Table 4
        elif cfg.TITLE == 'repr_table6_h36m_model':
            data_path = f'/mnt/SKY/preprocessed_data/{target_dataset}_{set}_front_25fps_tight_db.pt'  # Table 6
    elif target_dataset == 'mpii3d':
        set = 'val'
        data_path = f'/mnt/SKY/preprocessed_data/{target_dataset}_{set}_scale12_db_clip.pt'  #
    else:
        print("Wrong target dataset! Exiting...")
        import sys; sys.exit()

    print(f"Load data from {data_path}")
    dataset_data = joblib.load(data_path)

    print(f"Load caption from {caption_path}")
    caption_data = joblib.load(caption_path)

    full_res = defaultdict(list)

    vid_name_list = dataset_data['vid_name']
    unique_names = np.unique(vid_name_list)
    data_keyed = {}

    # make dictionary with video seqeunce names
    for u_n in unique_names:
        if (target_action != '') and (not target_action in u_n):
            continue
        indexes = vid_name_list == u_n
        if 'valid' in dataset_data:
            valids = dataset_data['valid'][indexes].astype(bool)
        else:
            valids = np.ones(dataset_data['features'][indexes].shape[0]).astype(bool)

        data_keyed[u_n] = {
            'features': dataset_data['features'][indexes][valids],
            'joints3D': dataset_data['joints3D'][indexes][valids],
            'vid_name': dataset_data['vid_name'][indexes][valids],
            'imgname': dataset_data['img_name'][indexes][valids],
            'bbox': dataset_data['bbox'][indexes][valids],
            'vitpose_j2d': dataset_data['vitpose_joint2d'][indexes][valids].astype('float32')
        }

        if 'mpii3d' in data_path:
            data_keyed[u_n]['pose'] = np.zeros((len(valids), 72))
            data_keyed[u_n]['shape'] = np.zeros((len(valids), 10))
            data_keyed[u_n]['valid_i'] = dataset_data['valid_i'][indexes][valids]
            J_regressor = None
        else:
            data_keyed[u_n]['pose'] = dataset_data['pose'][indexes][valids]
            data_keyed[u_n]['shape'] = dataset_data['shape'][indexes][valids]
    dataset_data = data_keyed

    """ Run evaluation """
    with torch.no_grad():
        tot_num_pose = 0
        pbar = tqdm(dataset_data.keys())
        for seq_name in pbar:
            curr_feats = dataset_data[seq_name]['features']         # 
            curr_vitposes = dataset_data[seq_name]['vitpose_j2d']
            curr_text_feats = dataset_data[seq_name]['text_emb']

            res_save = {}
            curr_feat = torch.tensor(curr_feats).to(device)
            curr_vitpose = torch.tensor(curr_vitposes).to(device)

            num_frames = curr_feat.shape[0]
            vid_names = dataset_data[seq_name]['vid_name']

            chunk_idxes = split_into_chunks_test(vid_names, seqlen=seqlen, stride=stride, is_train=False, match_vibe=False)  # match vibe eval number of poses
            if chunk_idxes == []:
                continue

            pred_j3ds, pred_verts, pred_rotmats, pred_thetas, scores = [], [], [], [], []
            for curr_idx in range(0, len(chunk_idxes), 8):
                input_feat = []
                input_vitpose = []
                if (curr_idx + 8) < len(chunk_idxes):
                    for ii in range(8):
                        seq_select = get_sequence(chunk_idxes[curr_idx+ii][0], chunk_idxes[curr_idx+ii][1])
                        input_feat.append(curr_feat[None, seq_select, :])
                        input_vitpose.append(curr_vitpose[None, seq_select, :])
                else:
                    for ii in range(curr_idx, len(chunk_idxes)):
                        seq_select = get_sequence(chunk_idxes[ii][0], chunk_idxes[ii][1])
                        input_feat.append(curr_feat[None, seq_select, :])
                        input_vitpose.append(curr_vitpose[None, seq_select, :])