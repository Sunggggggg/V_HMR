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
import torch
import torch.nn as nn
from lib.utils.geometry import batch_rodrigues

class Loss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            vel_or_accel_2d_weight=50,
            vel_or_accel_3d_weight=100,
            device='cuda',
    ):
        super(Loss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.vel_or_accel_2d_weight = vel_or_accel_2d_weight
        self.vel_or_accel_3d_weight = vel_or_accel_3d_weight

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_accel = nn.MSELoss('none').to(self.device)

        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d
    ):
        reduce = lambda x: x.contiguous().view((x.shape[0] * x.shape[1],) + x.shape[2:])    # [B, 3, 24, 2] => [B3, 24, 2]
        flatten = lambda x: x.reshape(-1)
        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0) 
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
        
        seq_len = real_2d.shape[1]
        real_3d = data_3d['kp_3d']
        real_3d_theta = data_3d['theta']
        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)

        # 
        real_2d = real_2d[:, seq_len // 2 : seq_len // 2 + 1]
        real_3d = data_3d['kp_3d'][:, seq_len // 2 : seq_len // 2 + 1]
        real_3d_theta = data_3d['theta'][:, seq_len // 2 : seq_len // 2 + 1]
        w_3d = data_3d['w_3d'].type(torch.bool)[:, seq_len // 2 : seq_len // 2 + 1]
        w_smpl = data_3d['w_smpl'].type(torch.bool)[:, seq_len // 2 : seq_len // 2 + 1]

        loss_kp_2d_local, loss_kp_3d_local, loss_pose_global, loss_shape_local = self.cal_loss(sample_2d_count, \
            real_2d, real_3d, real_3d_theta, w_3d, w_smpl, reduce, flatten, generator_outputs)

        # 
        loss_dict = {
            'loss_kp_2d_local': loss_kp_2d_local,
            'loss_kp_3d_local': loss_kp_3d_local,
        }

        if loss_pose_global is not None:
            loss_dict['loss_pose_global'] = loss_pose_global
            loss_dict['loss_shape_local'] = loss_shape_local
                
        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict

    def cal_loss(self, sample_2d_count, real_2d, real_3d, real_3d_theta, w_3d, w_smpl, reduce, flatten, generator_outputs):
        seq_len = real_2d.shape[1]

        real_2d = reduce(real_2d)
        real_3d = reduce(real_3d)
        real_3d_theta = reduce(real_3d_theta)

        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)

        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]
        pred_theta = reduce(pred_theta)
        pred_theta = pred_theta[w_smpl]

        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)
        pred_j3d = pred_j3d[w_3d]

        real_3d_theta = real_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]

        # 3D/2D joint location
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        
        real_shape, pred_shape = real_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = real_3d_theta[:, 3:75], pred_theta[:, 3:75]

        # SMPL parameters
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
        else:
            loss_pose = None
            loss_shape = None

        return loss_kp_2d, loss_kp_3d, loss_pose, loss_shape

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]

        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # gt_keypoints_3d = gt_keypoints_3d
        # conf = conf
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid).mean()
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid).mean()
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def get_vel_input(self, pose_2d, pose_3d, seq_len, reduce, conf_2d_flag=False):
        x0_2d = pose_2d[:, :-1]
        x1_2d = pose_2d[:, 1:]
        vel_2d = x1_2d - x0_2d

        if conf_2d_flag:
            conf_2d = pose_2d[:, :-1, :, -1]
            vel_2d[:, :, :, -1] = conf_2d

        x0_3d = pose_3d[:, :-1]
        x1_3d = pose_3d[:, 1:]
        vel_3d = x1_3d - x0_3d

        pad_2d_accel = torch.zeros(vel_2d.shape[0], 1, vel_2d.shape[2], vel_2d.shape[3]).to(self.device)
        vel_2d = torch.cat((vel_2d, pad_2d_accel), dim=1)
        pad_3d_accel = torch.zeros(vel_3d.shape[0], 1, vel_3d.shape[2], vel_3d.shape[3]).to(self.device)
        vel_3d = torch.cat((vel_3d, pad_3d_accel), dim=1)

        vel_2d = reduce(vel_2d)
        vel_3d = reduce(vel_3d)

        return vel_2d, vel_3d
    
    def accel_3d_loss(self, pred_accel, gt_accel):
        pred_accel = pred_accel[:, 25:39]
        gt_accel = gt_accel[:, 25:39]

        if len(gt_accel) > 0:
            return self.criterion_accel(pred_accel, gt_accel).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
        
def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_wasserstein_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return -1 * disc_value.sum() / k


def batch_adv_disc_wasserstein_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''

    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]

    la = -1 * real_disc_value.sum() / ka
    lb = fake_disc_value.sum() / kb
    return la, lb, la + lb


def batch_smooth_pose_loss(pred_theta):
    pose = pred_theta[:,:,3:75]
    pose_diff = pose[:,1:,:] - pose[:,:-1,:]
    return torch.mean(pose_diff).abs()


def batch_smooth_shape_loss(pred_theta):
    shape = pred_theta[:, :, 75:]
    shape_diff = shape[:, 1:, :] - shape[:, :-1, :]
    return torch.mean(shape_diff).abs()
