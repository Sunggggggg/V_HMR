import os
import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .transformer import FreqTempEncoder
from .FeatureCorrection import ImageFeatureCorrection
from .regressor import HSCR, Regressor

from lib.core.config import BASE_DATA_DIR

class Model(nn.Module):
    def __init__(self, 
                 num_coeff_kept=8,
                 num_frames=16,
                 num_joints=19,
                 embed_dim=512,
                 num_heads=8,
                 depth=4,
                 drop_rate=0.,
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 ):
        super().__init__()
        self.mid_frame = num_frames//2
        self.stride = 4
        # 
        self.joint_tree = JointTree()
        self.lifter = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)

        # Temporal encoder
        self.temp_encoder = ImageFeatureCorrection(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 2048)

        # Regressor
        self.pos_emb = nn.Linear(3, 128)
        self.regressor1 = Regressor()
        self.regressor2 = HSCR(f_dim=embed_dim)
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, f_img, pose2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        
        # 3D Lifting
        pose2d = self.joint_tree.add_joint(pose2d[..., :2])
        pose3d = self.lifter(pose2d, pose2d[:, self.mid_frame-1:self.mid_frame+2])          # [B, 1, 19, 3]
        pose3d = self.pos_emb(pose3d)
        # 
        f_temp = self.temp_encoder(f_img[:, self.mid_frame-self.stride:self.mid_frame+self.stride+1]
                                   , f_img[:, self.mid_frame-1:self.mid_frame+2])           # [B, 1, dim]

        # [B, 1, *]
        f_out = self.output_proj(f_temp)
        _, (pred_pose, pred_shape, pred_cam) = self.regressor1(f_out, is_train=is_train, J_regressor=J_regressor)
        smpl_output = self.regressor2(f_temp, pose3d, pred_pose, pred_shape, pred_cam, is_train=is_train, J_regressor=J_regressor)
    
        scores = None
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)
                s['verts'] = s['verts'].reshape(B, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 1
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)         # [B, 3, 24, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output