import torch
import torch.nn as nn

from .jointspace import JointTree
from .encoder import TEncoder, STEncoder
from .motion_encoder import MotionEncoder, ContextEncoder
from .regressor import LocalRegressor
from lib.models.GLoT.GMM import GMM
from lib.models.trans_operator import CrossAttention

class Model(nn.Module):
    def __init__(self, 
                 num_frames=16,
                 num_joints=20,
                 embed_dim=512, 
                 depth=3, 
                 num_heads=8, 
                 mlp_ratio=2., 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2,
                 mask_ratio=0.5
                 ) :
        super().__init__()
        self.num_frames = num_frames
        self.mid_frame = num_frames // 2
        self.stride_short = 4
        self.jointtree = JointTree()
        self.img_emb = nn.Linear(2048, embed_dim)
        self.joint_emb = nn.Linear(2*num_joints, embed_dim)

        # 
        self.proj_motion1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.proj_motion2 = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.ReLU()
        )

        self.global_modeling = GMM(num_frames, depth, embed_dim, num_heads, drop_rate, drop_path_rate, attn_drop_rate, mask_ratio)
        self.st_trans = STEncoder(num_frames=3, num_joints=24, depth=depth, embed_dim=embed_dim//2, mlp_ratio=mlp_ratio,
            num_heads=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate)
        
        self.localregressor = LocalRegressor()

        # self.t_trans = TEncoder(embed_dim=embed_dim)
        
        
        # self.motion_enc = MotionEncoder(num_frames, embed_dim)
        # self.context_enc = ContextEncoder(embed_dim//2)

        # self.proj_global = nn.Sequential(
        #     nn.Linear(embed_dim, 2048),
        #     nn.LayerNorm(2048)
        # )
        # self.global_regressor = Regressor(2048)
        
        # self.proj_short = nn.Linear(embed_dim, embed_dim//2)
        # self.short_encoder = CrossAttention(embed_dim//2)
        # self.local_regressor = Regressor(embed_dim//2)
    
    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None):
        B, T = f_img.shape[:2]
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)                    # [B, T, 24, 2]

        # Denoise
        f = self.img_emb(f_img)                # [B, T, D]
        smpl_output_global, mask_ids, mem, pred_global = self.global_modeling(f, is_train=is_train, J_regressor=J_regressor)

        # Joint feat
        f_img_short = f_img[:, self.mid_frame-1 : self.mid_frame+2]             # [B, 3, 2048]
        vitpose_2d_short = vitpose_2d[:, self.mid_frame-1 : self.mid_frame+2]   # [B, 3, 24, 2]
        f_st = self.st_trans(f_img_short, vitpose_2d_short)                     # [B, 3, 24, D]
        
        init_pose, init_shape, init_cam = pred_global[0], pred_global[1], pred_global[2]    # [B, T=3, 144/10/3] if is_train else T=1
        
        if is_train :
            f_st = f_st
        else :
            f_st = f_st[:, 1][:, None]
    
        smpl_output = self.localregressor(f_st, init_pose, init_shape, init_cam, is_train=is_train, J_regressor=J_regressor)

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
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, mask_ids, smpl_output_global

