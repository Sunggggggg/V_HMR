import torch
import torch.nn as nn
from functools import partial 
from .jointspace import JointTree
from .transformer import TemporalEncoder, JointEncoder, FreqTempEncoder, CrossAttention, Transformer
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
        self.joint_encoder = JointEncoder()

        # Global regre
        self.proj_input = nn.Linear(embed_dim//2 + num_joints*32, embed_dim)
        self.global_regressor = GlobalRegressor(embed_dim)

        # Freqtemp transformer
        self.joint_refiner = FreqTempEncoder(num_joints, 32, 3, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # Local encoder
        short_d_model = embed_dim//2
        self.proj_short = nn.Linear(2048, short_d_model)
        self.proj_latent = nn.Linear(embed_dim, short_d_model)
        self.local_trans_de = CrossAttention(short_d_model, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.local_trans_en = Transformer(depth=2, embed_dim=short_d_model, mlp_hidden_dim=short_d_model*4, 
                                          h=num_heads, length=3)
        
        self.proj_input2 = nn.Linear(embed_dim//2 + num_joints*32, num_joints*16)
        self.local_regressor = LocalRegressor(16)
        

    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B = f_img.shape[0]
        # Temp transformer
        f_temp = self.img_emb(f_img)
        f_temp, mask_ids, latent = self.temp_encoder(f_temp)

        # Joint
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)        # [B, T, 24, 2]
        f_joint = self.joint_encoder(vitpose_2d)
        f = torch.cat([f_temp, f_joint], dim=-1)
        f = self.proj_input(f)

        if is_train :
            f_global_output = f
        else :
            f_global_output = f[:, self.mid_frame:self.mid_frame+1]
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        # 
        full_2d_joint, short_2d_joint = vitpose_2d, vitpose_2d[:, self.mid_frame-1:self.mid_frame+2]
        short_f_joint = self.joint_refiner(full_2d_joint, short_2d_joint)      # [B, 3, 24, 32]
        
        short_img = f_img[:, self.mid_frame-1:self.mid_frame+2]
        short_img = self.proj_short(short_img)
        short_img = self.local_trans_en(short_img)
        latent = self.proj_latent(latent)
        short_f_temp = self.local_trans_de(short_img, latent)                   # [B, 3, 256]

        f_st = torch.cat([short_f_temp, short_f_joint], dim=-1)                 # [B, 3, 256+24*32]
        f_st = self.proj_input2(f_st)         
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





        