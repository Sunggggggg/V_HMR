import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .transformer import TemporalEncoder, JointEncoder, FreqTempEncoder, CrossAttention, Transformer, STEncoder, FreqTempEncoder_img
from .regressor import LocalRegressorThetaBeta, GlobalRegressor, NewLocalRegressor

class Model(nn.Module):
    def __init__(self, 
                 num_coeff_kept=8,
                 num_frames=16,
                 num_joints=19,
                 embed_dim=512,
                 num_heads=8,
                 depth=4,
                 drop_rate=0.,

                 ):
        super().__init__()
        self.mid_frame = num_frames//2
        self.jointtree = JointTree()
        self.stride = 3
        # 
        self.proj_input = nn.Linear(2048, embed_dim)
        self.global_encoder = Transformer(depth=2, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4., h=num_heads, length=num_frames)

        # Camera Network
        self.proj_cam = nn.Linear(embed_dim, embed_dim//4)
        self.decoder_cam = Transformer(depth=1, embed_dim=embed_dim//4, mlp_hidden_dim=embed_dim, h=num_heads, length=num_frames)

        # Pose, shape Network
        self.proj_joint = nn.Linear(num_joints*32, embed_dim)
        self.norm_input = nn.LayerNorm(embed_dim)
        self.proj_dec = nn.Linear(embed_dim, embed_dim//2)

        self.joint_encoder = JointEncoder(num_joint=num_joints)
        self.decoder_pose_shape = Transformer(depth=1, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2., h=num_heads, length=num_frames)

        # Global Regressor
        self.global_regressor = GlobalRegressor(embed_dim//2 + embed_dim//4)

        # Local Network
        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)
        self.proj_short_joint = nn.Linear(num_joints*32, embed_dim//2)
        self.proj_short_img = nn.Linear(2048, embed_dim//2)

        self.temp_local_encoder = Transformer(depth=2, embed_dim=embed_dim//2, length=self.stride*2+1)

        # Local Regressor
        self.local_cam_decoder = nn.Sequential(
            nn.Linear(embed_dim//2, embed_dim//8),
            nn.ReLU(),
            nn.Linear(embed_dim//8, embed_dim//8),
            nn.LayerNorm(embed_dim//8)
        )
        self.local_decoder = CrossAttention(embed_dim//2)
        self.local_regressor = NewLocalRegressor(embed_dim//2+embed_dim//8)


    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        # Global encoder
        f_temp = self.proj_input(f_img)             # [B, T, 512]
        f_temp = self.global_encoder(f_temp)        # [B, T, 512]

        # Camera Network
        f_cam = self.proj_cam(f_temp)
        f_cam = self.decoder_cam(f_cam)         # [B, T, 128]

        # Pose, shape Network
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])  # [B, T, 19, 2] 
        f_joint = self.joint_encoder(vitpose_2d)                    # [B, T, (19*32)]
        f_joint = self.proj_joint(f_joint)                          # [B, T, 512]

        f = self.norm_input(f_joint + f_temp)   # [B, T, 512]
        f = self.proj_dec(f)
        f = self.decoder_pose_shape(f)          # [B, T, 256]

        if is_train :
            f_global_output = torch.cat([f_cam, f], dim=-1)
        else :
            f_global_output = torch.cat([f[:, self.mid_frame:self.mid_frame+1], f_cam[:, self.mid_frame:self.mid_frame+1]], dim=-1)
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        # 
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint).flatten(-2)   # [B, 3, 19*32]
        short_f_joint = self.proj_short_joint(short_f_joint)                            # [B, 3, 256]
        
        short_f_img = f_img[:, self.mid_frame-self.stride:self.mid_frame+self.stride+1] # [B, 6, 2048]
        short_f_img = self.proj_short_img(short_f_img)
        short_f_img = self.temp_local_encoder(short_f_img)                              # [B, 6, 256]
        short_f_img = short_f_img[:, self.mid_frame-1:self.mid_frame+2]

        short_f_cam = self.local_cam_decoder(short_f_img)                               # [B, 3, 64]
        f_st = self.local_decoder(short_f_joint, short_f_img)                           # [B, 3, 256]

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

        return smpl_output, None, smpl_output_global

    
        





        