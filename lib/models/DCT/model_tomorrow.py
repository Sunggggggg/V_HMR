import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .transformer import TemporalEncoder, JointEncoder, FreqTempEncoder, CrossAttention, Transformer, STEncoder
from .regressor import LocalRegressor, GlobalRegressor

class Model(nn.Module):
    def __init__(self, 
                 num_coeff_kept=8,
                 num_frames=16,
                 num_joints=24,
                 embed_dim=512,
                 num_heads=8,
                 depth=4,
                 drop_rate=0.,

                 ):
        super().__init__()
        self.mid_frame = num_frames//2
        self.jointtree = JointTree()
        # Temp transformer
        self.img_emb = nn.Linear(2048, embed_dim)
        self.temp_encoder = TemporalEncoder(depth=3, embed_dim=embed_dim)
        
        # Spatio transformer
        self.proj_input = nn.Linear(num_joints*32, embed_dim)
        self.joint_encoder = JointEncoder()
        self.norm = nn.LayerNorm(embed_dim)

        # Global regre
        self.global_regressor = GlobalRegressor(embed_dim)

        # Freqtemp transformer
        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        # ST transformer
        self.st_trans = STEncoder(num_frames=3, num_joints=24, depth=depth, embed_dim=embed_dim//2, mlp_ratio=4.,
            num_heads=num_heads, drop_rate=drop_rate)
        self.local_regressor = LocalRegressor(embed_dim//2)
        

    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        # Joint transformer
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)        # [B, T, 24, 2]
        f_joint = self.joint_encoder(vitpose_2d)
        f_joint = self.proj_input(f_joint)
        
        # Temp transformer
        f_temp = self.img_emb(f_img)

        f = self.norm(f_temp + f_joint)
        f, mask_ids, latent = self.temp_encoder(f, is_train=is_train, mask_ratio=0.5)

        if is_train :
            f_global_output = f
        else :
            f_global_output = f[:, self.mid_frame:self.mid_frame+1]
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        # 
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint)      # [B, 3, 24*32]
        f_img_short = f_img[:, self.mid_frame-1:self.mid_frame+2]
        f_st = self.st_trans(f_img_short, short_f_joint)   
    
        if is_train :
            f_st = f_st
        else :
            f_st = f_st[:, 1][:, None]
    
        smpl_output = self.local_regressor(f_st, pred_global[0], pred_global[1], pred_global[2], is_train=is_train, J_regressor=J_regressor)

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

        return smpl_output, mask_ids, smpl_output_global





        