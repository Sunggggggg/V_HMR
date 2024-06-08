import torch
import torch.nn as nn

from .jointspace import JointTree
from .encoder import TEncoder, STEncoder, CaptionEncoder
from .motion_encoder import MotionEncoder, ContextEncoder
from .regressor import Regressor

class Model(nn.Module):
    def __init__(self, 
                 num_frames=16,
                 num_joints=19,
                 embed_dim=512, 
                 depth=3, 
                 num_heads=8, 
                 mlp_ratio=2., 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2,
                 batch_size=32) :
        super().__init__()
        self.num_frames = num_frames
        self.mid_frame = num_frames // 2
        self.stride_short = 4
        self.joint_space = JointTree()

        self.text_emb = CaptionEncoder(batch=batch_size)
        self.t_trans = TEncoder(embed_dim=embed_dim)
        self.s_trans = STEncoder(depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio,
            num_heads=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate)
        self.motion_enc = MotionEncoder(num_frames, embed_dim)
        self.context_enc = ContextEncoder(embed_dim)

        self.proj_global = nn.Linear(embed_dim, 2048)
        self.global_regressor = Regressor(2048)
        
        self.proj_short = nn.Linear(embed_dim, embed_dim//2)
        self.local_regressor = Regressor(embed_dim//2)
    
    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None):
        B, T = f_img.shape[:2]
        f_joint = self.joint_space(f_joint[..., :2])
        # Global
        f_text = self.text_emb(f_text)
        f_temp = self.t_trans(f_text, f_img)        # [B, T, D]
        f = self.s_trans(f_temp, f_joint)           # [B, T, J, D]
        f_motion = self.motion_enc(f_text, f_joint) # [B, 1, J, 1]
        f_context = self.context_enc(f_text, f_img) # [B, T, 1]
        f = torch.sum(f * f_motion, dim=-2)          # [B, T, D]

        if is_train :
            size = self.num_frames
            f_out = self.proj_global(f) # [B, T, 2048]
        else :
            size = 1
            f_out = self.proj_global(f)[:, self.mid_frame][:, None, :] # [B, 1, 2048]
        
        smpl_output_global, pred_global = self.global_regressor(f_out, is_train=is_train, J_regressor=J_regressor)

        scores = None
        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(B, size, -1)
            s['verts'] = s['verts'].reshape(B, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
            s['scores'] = scores


        f = self.proj_short(f)  # [B, T, d]
        f = f * f_context       # [B, T, d]
        if is_train :
            f_out = f[:, self.mid_frame-1:self.mid_frame+2] # [B, 3, D]
        else :
            f_out = f[:, self.mid_frame][:, None, :]        # [B, 1, D]
        smpl_output, _ = self.local_regressor(f_out, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
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

        return smpl_output, smpl_output_global