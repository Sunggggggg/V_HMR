import torch
import torch.nn as nn

from .transformer import Transformer
from .regressor import Global_regressor, Local_regressor
from ..trans_operator import CrossAttention

class Model(nn.Module) :
    def __init__(self, 
                 num_frames=16,
                 num_joints=19,
                 embed_dim=512, 
                 depth=3, 
                 num_heads=8, 
                 mlp_ratio=2., 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2
                 ):
        super().__init__()
        self.num_frames = num_frames
        self.mid_frame = num_frames//2
        self.proj_joint = nn.Linear(num_joints*2, embed_dim)
        self.proj_img = nn.Linear(2048, embed_dim)

        self.encoder = Transformer(depth=2, embed_dim=embed_dim, mlp_hidden_dim=mlp_ratio*embed_dim,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=num_frames)
        
        self.proj_decode = nn.Linear(embed_dim, embed_dim//2)
        self.decoder = Transformer(depth=1, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=num_frames)
        
        self.proj_local = nn.Linear(embed_dim, embed_dim//2)
        self.proj_enc_local = nn.Linear(embed_dim, embed_dim//2)
        self.local_encoder = Transformer(depth=3, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=3)
        self.local_decoder = CrossAttention(embed_dim//2)
        
        # Regressor
        self.global_regressor = Global_regressor(embed_dim//2)
        self.local_regressor = Local_regressor(embed_dim//2)
        
    def refine_2djoint(self, vitpose_2d):
        vitpose_j2d_pelvis = vitpose_2d[:,:,[11,12],:2].mean(dim=2, keepdim=True) 
        vitpose_j2d_neck = vitpose_2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)  
        joint_2d_feats = torch.cat([vitpose_2d[... ,:2], vitpose_j2d_pelvis, vitpose_j2d_neck], dim=2)  
        joint_2d_feats = joint_2d_feats.flatten(-2)
        
        return joint_2d_feats

    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None) :
        """
        f_text      : [B, 1, 512]
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B, T = f_img.shape[:2]

        f_joint = self.refine_2djoint(f_joint)  # [B, T, (J+2)*2]
        f_joint = self.proj_joint(f_joint)      # [B, T, D]
        f_img = self.proj_img(f_img)            # [B, T, D]

        f = f_text + f_img + f_joint            # [B, T, D]

        f_enc = self.encoder(f)
        f_dec = self.proj_decode(f_enc)         # [B, T, d]
        f_dec = self.decoder(f_dec)             # [B, T, d]

        if is_train :
            f_global_output = f_dec
        else :
            f_global_output = f_dec[:, self.mid_frame:self.mid_frame+1]

        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)
        
        f_short = f[:, self.mid_frame-1:self.mid_frame+2]
        f_short = self.proj_local(f_short)              # [B, 3, d]
        f_enc_local = self.proj_enc_local(f_enc)
        f_enc_short = self.local_encoder(f_short)       # [B, 3, d]
        f_dec_short = self.local_decoder(f_enc_short, f_enc_local)

        if is_train :
            f_local_output = f_dec_short
        else :
            f_local_output = f_dec_short[:, 1:2]

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
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, smpl_output_global