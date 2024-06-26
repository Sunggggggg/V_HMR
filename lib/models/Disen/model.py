import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .DTM import DTM
from .transformer import FreqTempEncoder, CrossAttention, Transformer
from .regressor import GlobalRegressor, NewLocalRegressor

class Model(nn.Module):
    def __init__(self, 
                 num_coeff_kept=8,
                 num_frames=16,
                 num_joints=19,
                 embed_dim=512,
                 num_heads=8,
                 depth=4,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,

                 ):
        super().__init__()
        self.stride = 4
        self.mid_frame = num_frames//2
        self.num_frames = num_frames
        
        self.global_network = DTM(depth=3, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4.0,
                                   h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=num_frames)
        
        self.local_network = DTM(depth=2, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2.0,
                                   h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=self.stride*2+1,
                                   decoder=False)
        
        ##########################
        # 2D joint
        ##########################
        self.jointtree = JointTree()
        self.joint_encoder = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)
        self.proj_joint = nn.Linear(num_joints*32, embed_dim)

        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)
        self.proj_short_joint = nn.Linear(num_joints*32, embed_dim//2)
        self.local_decoder = CrossAttention(embed_dim//2)

        ##########################
        # Regressor
        ##########################
        self.global_regressor = GlobalRegressor(embed_dim//2 + embed_dim//4)
        self.local_regressor = NewLocalRegressor(embed_dim//2 + embed_dim//4)

    def forward(self, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        ##########################
        # Global
        ##########################
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])              # [B, T, 19, 2] 
        f_joint = self.joint_encoder(vitpose_2d, vitpose_2d, self.num_frames)   # [B, T, 608]
        f_joint = self.proj_joint(f_joint)

        f_cam, f_pose_shape = self.global_network(f_img, f_joint)
        
        if is_train :
            f_global_output = torch.cat([f_pose_shape, f_cam], dim=-1)         # [B, T, 256+128]
        else :
            f_global_output = torch.cat(f_pose_shape[:, self.mid_frame:self.mid_frame+1], 
                                        f_cam[:, self.mid_frame:self.mid_frame+1], dim=-1) 
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        ##########################
        # Local
        ##########################
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint)               # [B, 3, 19*32]
        short_f_joint = self.proj_short_joint(short_f_joint)                            # [B, 3, 256]

        short_f_img = f_img[:, self.mid_frame-self.stride:self.mid_frame+self.stride+1] # [B, 8, 2048]
        short_f_cam, short_f_pose_shape = self.local_network(short_f_img)               # [B, 8, 128], [B, 8, 256]
        
        short_f_cam = short_f_cam[:, self.stride-1:self.stride+2]
        short_f_pose_shape = short_f_pose_shape[:, self.stride-1:self.stride+2]
        f_st = self.local_decoder(short_f_joint, short_f_pose_shape)

        if is_train :
            f_local_output = torch.cat([short_f_cam, f_st], dim=-1)
        else :
            f_local_output = torch.cat([short_f_cam[:, 1:2], f_st[:, 1:2]], dim=-1)
    
        smpl_output = self.local_regressor(f_local_output, pred_global[0], pred_global[1], pred_global[2], is_train=is_train, J_regressor=J_regressor)

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