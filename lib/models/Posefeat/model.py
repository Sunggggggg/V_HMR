import torch
import torch.nn as nn

from lib.models.trans_operator import Block

from .jointspace import JointTree
from .GMM import GMM
from .regressor import LocalRegressor, GlobalRegressor
from .transformer import STEncoder

class Model(nn.Module) :
    def __init__(self, 
                 num_frames=16,
                 num_joints=20,
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
        self.joint_head = nn.Linear(32, 2)
        
        # Temporal transformer
        self.img_emb = nn.Linear(2048, embed_dim)
        self.global_modeling = GMM(num_frames, depth, embed_dim, num_heads, drop_rate, drop_path_rate, attn_drop_rate, 0.5)

        # ST transformer
        self.st_trans = STEncoder(num_frames=3, num_joints=24, depth=depth, embed_dim=embed_dim//2, mlp_ratio=mlp_ratio,
            num_heads=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(embed_dim//2),
            nn.Linear(embed_dim//2, 2)
        )
        self.pose_encoding = nn.Linear(2, embed_dim//2)
        
        self.proj_input = nn.Linear(embed_dim//2 + num_joints*32, embed_dim)
        self.i_norm = nn.LayerNorm(embed_dim)
        self.global_regressor = GlobalRegressor(embed_dim)
        self.localregressor = LocalRegressor()
        
    def spatio_transformer(self, x):
        B, T, J = x.shape[:-1]

        x = self.joint_emb(x)                   # [B, T, J, 32]
        x = x.view(B*T, J, -1)                  # [BT, J, 32]
        x = x + self.s_pos_embed                # 
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.s_norm(x)      # [BT, J, 32]
        x = x.reshape(B, T, -1) # [B, T, 19*32]
        return x

    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_text      : [B, 1, 512]
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B, T = f_img.shape[:2]

        # Spatio transformer
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])  # [B, T, 20, 2]
        f_joint = self.spatio_transformer(vitpose_2d)

        # Temporal transformer
        f_temp = self.img_emb(f_img)
        f_temp, mask_ids = self.global_modeling(f_temp, is_train=is_train)  # [B, T, 256]

        f = torch.cat([f_temp, f_joint], dim=-1)                 # [B, T, 256+608]
        f = self.proj_input(f)
        f = self.i_norm(f) 

        if is_train :
            f_global_output = f
        else :
            f_global_output = f[:, self.mid_frame:self.mid_frame+1]
        smpl_output_global, pred_global = self.global_regressor(f_global_output, n_iter=3, is_train=is_train, J_regressor=J_regressor)

        # Joint feat
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)                    # [B, T, 24, 2]
        f_img_short = f_img[:, self.mid_frame-1 : self.mid_frame+2]             # [B, 3, 2048]
        vitpose_2d_short = vitpose_2d[:, self.mid_frame-1 : self.mid_frame+2]   # [B, 3, 24, 2]
        f_st = self.st_trans(f_img_short, vitpose_2d_short)                     # [B, 3, 24, D]
        vitpose_2d_short = vitpose_2d_short + self.pose_head(f_st)
        f_st = self.pose_encoding(vitpose_2d_short)

        if is_train :
            f_st = f_st
        else :
            f_st = f_st[:, 1][:, None]
    
        smpl_output = self.localregressor(f_st, pred_global[0], pred_global[1], pred_global[2], is_train=is_train, J_regressor=J_regressor)

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

        return smpl_output, mask_ids, smpl_output_global, vitpose_2d_short