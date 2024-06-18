import torch
import torch.nn as nn

from .jointspace import JointTree
from .GMM import GMM
from lib.models.trans_operator import Block
from .regressor import GlobalRegressor, LocalRegressor
from .transformer import Transformer
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
        self.mid_frame = num_frames//2
        self.jointtree = JointTree()

        # Spatio transformer
        self.joint_emb = nn.Linear(2, 32)
        self.s_pos_embed = nn.Parameter(torch.zeros(1, num_joints, 32))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spatial_blocks = nn.ModuleList([
            Block(dim=32, num_heads=num_heads, mlp_hidden_dim=32*4.0) for i in range(2)]
        )
        self.s_norm = nn.LayerNorm(32)
        
        # Temporal transformer
        self.proj_img = nn.Linear(2048, embed_dim)
        self.global_modeling = GMM(num_frames, 2, embed_dim, num_heads, drop_rate, drop_path_rate, attn_drop_rate, 0.5)

        # 
        self.proj_input = nn.Linear(embed_dim//2 + num_joints*32, embed_dim//2)
        self.i_norm = nn.LayerNorm(embed_dim)

        self.proj_local = nn.Linear(embed_dim//2, embed_dim//2)
        self.proj_enc_local = nn.Linear(embed_dim//2, embed_dim//2)
        self.local_encoder = Transformer(depth=3, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=3)
        self.local_decoder = CrossAttention(embed_dim//2)

        # Regressor
        self.global_regressor = GlobalRegressor(embed_dim//2)
        self.local_regressor = LocalRegressor(embed_dim//2)
        
    def refine_2djoint(self, vitpose_2d):
        vitpose_j2d_pelvis = vitpose_2d[:,:,[11,12],:2].mean(dim=2, keepdim=True) 
        vitpose_j2d_neck = vitpose_2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)  
        joint_2d_feats = torch.cat([vitpose_2d[... ,:2], vitpose_j2d_pelvis, vitpose_j2d_neck], dim=2)          
        return joint_2d_feats

    def spatio_transformer(self, x):
        B, T, J = x.shape[:-1]

        x = self.joint_emb(x)                   # [B, T, 19, 32]
        x = x.view(B*T, J, -1)                  # [BT, J, 32]
        x = x + self.s_pos_embed                # 
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.s_norm(x)
        x = x.reshape(B, T, -1)                 # [B, T, 19*32]
        return x

    def disentangle_text(self, x):
        """
        x : [B, 1, 512]
        """

    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_text      : [B, 1, 512]
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B, T = f_img.shape[:2]

        # Spatio transformer
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)    # [B, T, 24, 2]
        f_joint = self.spatio_transformer(vitpose_2d)           # [B, T, 608]

        # Temporal transformer
        f_img = self.proj_img(f_img)                            # [B, T, 512]
        f_img = self.global_modeling(f_img, is_train=is_train)  # [B, T, 256]

        f = torch.cat([f_img, f_joint], dim=-1)                 # [B, T, 256+608]
        f = self.proj_input(f)
        
        if is_train :
            f_global_output = f
        else :
            f_global_output = f[:, self.mid_frame:self.mid_frame+1]

        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)
        
        f_short = f[:, self.mid_frame-1:self.mid_frame+2]
        f_short = self.proj_local(f_short)              # [B, 3, d]
        f_enc_local = self.proj_enc_local(f)
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
                s['theta'] = s['theta'].reshape(B, size, -1)            # [B, T, 3+10+24*3]
                s['verts'] = s['verts'].reshape(B, size, -1, 3)         # [B, T, 6890, 3]
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)         # [B, T, 24, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)         # [B, T, 24, 3]
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)    # [B, T, 24, 3, 3]
                s['scores'] = scores

        return smpl_output, smpl_output_global