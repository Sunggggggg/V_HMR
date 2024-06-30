import os
from lib.core.config import BASE_DATA_DIR
import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .transformer import JointEncoder, FreqTempEncoder, CrossAttention, Transformer
from .regressor import GlobalRegressor, NewLocalRegressor
from .KTD import KTD
"""
PoseformerV2 사용
GMM 사용 X
"""

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
        self.stride = 4
        self.mid_frame = num_frames//2
        self.num_frames = num_frames
        self.jointtree = JointTree()
        # Temp transformer
        self.img_emb = nn.Linear(2048, embed_dim)
        self.temp_encoder = Transformer(depth=3, embed_dim=embed_dim)
        
        # Spatio transformer
        self.joint_encoder = JointEncoder(num_joint=num_joints)

        # Global regre
        self.proj_input = nn.Linear(num_joints*32, embed_dim)
        self.norm_input = nn.LayerNorm(embed_dim)

        self.proj_dec = nn.Linear(embed_dim, embed_dim//2)
        self.global_decoder = Transformer(depth=1, embed_dim=embed_dim//2)
        self.global_regressor = GlobalRegressor(embed_dim//2)

        # Freqtemp transformer
        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)
        self.proj_short_joint = nn.Linear(num_joints*32, embed_dim//2)
        self.proj_short_img = nn.Linear(2048, embed_dim//2)
        self.temp_local_encoder = Transformer(depth=3, embed_dim=embed_dim//2, length=self.stride*2+1)

        #self.local_decoder = CrossAttention(embed_dim//2)
        #self.local_regressor = NewLocalRegressor(embed_dim//2)
        self.local_regressor = KTD()

    def forward(self, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        # Img transformer
        f_temp = self.img_emb(f_img)
        f_temp = self.temp_encoder(f_temp)                          # [B, T, 512]

        # Joint transformer
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])      # [B, T, 19, 2] 
        f_joint = self.joint_encoder(vitpose_2d)                        #                      
        f_joint = self.proj_input(f_joint)
        
        f = self.norm_input(f_joint + f_temp)   # [B, T, 512]
        f = self.proj_dec(f)
        f = self.global_decoder(f)              # [B, T, 256]

        if is_train :
            f_global_output = f
        else :
            f_global_output = f[:, self.mid_frame:self.mid_frame+1]
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        # Joint Local
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint)               # [B, 3, 768]
        short_f_joint = self.proj_short_joint(short_f_joint)
        
        short_f_img = f_img[:, self.mid_frame-self.stride:self.mid_frame+self.stride+1] # [B, 6, 2048]
        short_f_img = self.proj_short_img(short_f_img)
        short_f_img = self.temp_local_encoder(short_f_img)                              # [B, 6, 256]
        short_f_img = short_f_img[:, self.stride-1:self.stride+2]

        #f_st = self.local_decoder(short_f_joint, short_f_img)                           # [B, 3, 256]

        if is_train :
            #f_st = f_st
            short_f_joint = short_f_joint.reshape(B, 3, 19, -1)
            short_f_img = short_f_img

        else :
            #f_st = f_st[:, 1][:, None]
            short_f_joint = short_f_joint[:, 1][:, None].reshape(B, 3, 19, -1)
            short_f_img = short_f_img[:, 1][:, None]

        smpl_output = self.local_regressor(short_f_img, short_f_joint, pred_global[0], pred_global[1], pred_global[2], is_train=is_train, J_regressor=J_regressor)

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
            size = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)
                s['verts'] = s['verts'].reshape(B, size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)         # [B, 3, 24, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, smpl_output_global