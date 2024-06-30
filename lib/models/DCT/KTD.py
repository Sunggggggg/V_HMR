import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.models.spin import projection
from lib.utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

ANCESTOR_INDEX = [
    [],
    [0], 
    [0], 
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 1, 4],
    [0, 2, 5],
    [0, 3, 6],
    [0, 1, 4, 7],
    [0, 2, 5, 8],
    [0, 3, 6, 9], 
    [0, 3, 6, 9], 
    [0, 3, 6, 9],
    [0, 3, 6, 9, 12],
    [0, 3, 6, 9, 13],
    [0, 3, 6, 9, 14],
    [0, 3, 6, 9, 13, 16],
    [0, 3, 6, 9, 14, 17],
    [0, 3, 6, 9, 13, 16, 18],
    [0, 3, 6, 9, 14, 17, 19],
    [0, 3, 6, 9, 13, 16, 18, 20],
    [0, 3, 6, 9, 14, 17, 19, 21]
]

Joint3D_INDEX = [
    [17],       # 0 Pelvis_SMP
    [17, 11],   # L Hip
    [17, 12],   # R Hip
    [17],       # 3
    [17, 11, 13],  # 4
    [17, 12, 14],
    [17],  # 6
    [17, 11, 13, 15],
    [17, 12, 14, 16],
    [17, 5, 6, 18],  # 9
    [17, 11, 13, 15],
    [17, 12, 14, 16],
    [17, 18],
    [17, 5],
    [17, 6],
    [18, 0, 1, 2, 3, 4],
    [17, 5],
    [17, 6],
    [17, 5, 7],
    [17, 6, 8],
    [17, 5, 7, 9],
    [17, 6, 8, 10],
    [17, 15, 7, 9],
    [17, 6, 8, 10]
]

class KTD(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=512):
        super(KTD, self).__init__()

        self.feat_dim = feat_dim
        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            create_transl=False,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
        )
        npose_per_joint = 6
        nshape = 10
        ncam = 3

        self.fc1 = nn.Linear(feat_dim + 10 + 3, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        
        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in zip(Joint3D_INDEX, ANCESTOR_INDEX):
            regressor = nn.Linear(hidden_dim + 32 * len(joint_idx) + npose_per_joint * len(ancestor_idx) + 6, npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

        self.decshape = nn.Linear(hidden_dim, nshape)
        self.deccam = nn.Linear(hidden_dim, ncam)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

    def forward(self, f_s, f_p, init_pose, init_shape, init_cam, is_train=False, J_regressor=None):
        """
        f_s : [B, 3, 256]
        f_p : [B, 3, J, d]

        init_pose : [B, T, 144]
        """
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(3):
            x = torch.cat([pred_shape, pred_cam, f_s], dim=-1)
            x = self.fc1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            pred_shape = self.decshape(x) + pred_shape
            pred_cam = self.deccam(x) + pred_cam
        
        pose = []
        cnt = 0
        for ancestor_idx, joint_idx, reg in zip(ANCESTOR_INDEX, Joint3D_INDEX, self.joint_regs):
            ances = torch.cat([x] + [f_p[:, :, i] for i in joint_idx] + [pose[i] for i in ancestor_idx],dim=-1)
            ances = torch.cat((ances, pred_pose[:, :, cnt * 6: cnt * 6 + 6]), dim=-1)

            cnt += 1
            pose.append(reg(ances))

        pred_pose = torch.cat(pose, dim=-1)

        pred_pose = pred_pose.reshape(-1, 144)
        pred_shape = pred_shape.reshape(-1, 10)
        pred_cam = pred_cam.reshape(-1, 3)
        batch_size = pred_pose.shape[0]

        output = self.get_output(pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor)
        return output

    def get_output(self, pred_pose, pred_shape, pred_cam, batch_size, is_train, J_regressor):
        # print("HSCR.py",pred_pose.shape)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            pred_joints = pred_joints[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'rotmat': pred_rotmat
        }]
        return output