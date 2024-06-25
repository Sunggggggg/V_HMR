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
                 drop_path_rate=0.2,
                 attn_drop_rate=0.0,

                 ):
        super().__init__()
        self.stride = 4
        self.mid_frame = num_frames//2
        ##########################
        # Camera parameter 
        ##########################
        self.cam_proj = nn.Linear(2048, embed_dim//2)
        self.cam_encoder = Transformer(depth=2, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2.0, h=num_heads, 
                                       drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=num_frames)
        self.cam_dec_proj = nn.Linear(embed_dim//2, embed_dim//4)
        self.cam_decoder = Transformer(depth=1, embed_dim=embed_dim//4, mlp_hidden_dim=embed_dim, h=num_heads, 
                                       drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=num_frames)
        
        ##########################
        # 2D joint
        ##########################
        self.jointtree = JointTree()
        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_coeff_keep=3)
        self.proj_short_joint = nn.Linear(num_joints*32, embed_dim//2)

        ##########################
        # Pose, Shape parameters
        ##########################
        self.pose_shape_proj = nn.Linear(2048, embed_dim)
        self.pose_shape_encoder = Transformer(depth=3, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4.0, h=num_heads, 
                                       drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=num_frames)
        self.input_proj = nn.LayerNorm(embed_dim)
        self.pose_shape_dec_proj = nn.Linear(embed_dim, embed_dim//2)
        self.pose_shape_decoder = CrossAttention(embed_dim//2)

        ##########################
        # Regressor
        ##########################
        self.regressor = GlobalRegressor(embed_dim//2 + embed_dim//4)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cam_encoder.pos_embed, std=.02)
        torch.nn.init.normal_(self.cam_decoder.pos_embed, std=.02)
        torch.nn.init.normal_(self.pose_shape_encoder.pos_embed, std=.02)
        torch.nn.init.normal_(self.joint_refiner.joint_pos_embedding, std=.02)
        torch.nn.init.normal_(self.joint_refiner.freq_pos_embedding, std=.02)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        ##########################
        # Camera parameter 
        ##########################
        f_cam = self.cam_proj(f_img)        # [B, T, 256]
        f_cam = self.cam_encoder(f_cam)     # [B, T, 256]
        f_cam = self.cam_dec_proj(f_cam)    # [B, T, 128]
        f_cam = self.cam_decoder(f_cam)     # [B, T, 128]

        f_cam = f_cam[:, self.mid_frame-1:self.mid_frame+2] # [B, 3, 128]

        ##########################
        # Pose, Shape parameters
        ##########################
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint).flatten(-2)   # [B, 3, 19*32]
        short_f_joint = self.proj_short_joint(short_f_joint)                            # [B, 3, 256]

        f_pose_shape = self.pose_shape_proj(f_img)
        f_pose_shape = self.pose_shape_encoder(f_pose_shape)
        f_pose_shape = self.pose_shape_dec_proj(f_pose_shape)                           # [B, T, 256]
        f_pose_shape = f_pose_shape[:, self.mid_frame-1:self.mid_frame+2]               # [B, 3, 256]
        
        f_st = self.pose_shape_decoder(short_f_joint, f_pose_shape)                     # [B, 3, 256]

        if is_train :
            f_local_output = torch.cat([f_cam, f_st], dim=-1)
        else :
            f_local_output = torch.cat([f_cam[:, 1:2], f_st[:, 1:2]], dim=-1)
    
        smpl_output = self.regressor(f_local_output, is_train=is_train, J_regressor=J_regressor)

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

        return smpl_output