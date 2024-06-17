import torch
import torch.nn as nn
import torch_dct as dct
from lib.models.trans_operator import Block
from lib.models.Trans.jointspace import JointTree

class Model(nn.Module):
    def __init__(self, 
                 num_coeff_kept=8,
                 num_joint=20,
                 emb_dim=512,
                 num_heads=8,
                 depth=4,
                 drop_rate=0.,

                 ):
        super().__init__()
        self.jointtree = JointTree()
    
        # Spatio transformer
        self.joint_emb = nn.Linear(2, 32)
        self.s_pos_embed = nn.Parameter(torch.zeros(1, num_joint, 32))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spatial_blocks = nn.ModuleList([
            Block(dim=emb_dim, num_heads=num_heads, mlp_hidden_dim=emb_dim*4.0) for i in range(depth)]
        )
        self.s_norm = nn.LayerNorm()

        # 
        self.freq_embedding = nn.Linear(2048, num_joint*32)



    def spatio_transformer(self, x):
        B, T, J = x.shape[:-1]
        mid_frame = T // 2
        x = x[:, mid_frame-1 : mid_frame+2]     # [B, 3, 19, 32]

        x = self.joint_emb(x)                   # [B, 3, 19, 32]
        x = x.view(B*3, J, -1)                  # [B3, J, 32] 
        x = x + self.s_pos_embed                # 
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.s_norm(x)
        x = x.reshape(B, 3, -1)                 # [B, 3, 19*32]
        return x

    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        B, T = f_img.shape[0]
        f_joint = f_joint[..., :2]
        f_joint = self.jointtree.add_joint(f_joint)

        f_spatio = self.spatio_transformer(f_joint)      # [B, 3, 19*32]

        # 
        f_img_freq = dct.dct(f_img.permute(0, 2, 1))                        
        f_img = f_img_freq.permute(0, 2, 1).contiguous().view(B, T, -1)         # [B, T, 2048]

        # f_joint_freq = dct.dct(f_joint.permute(0, 2, 3, 1))
        # f_joint = f_joint_freq.permute(0, 3, 1, 2).contiguous().view(B, T, -1) # [B, T, J*2]

        f = torch.cat([f_img, f_spatio], dim=-1)

        f_joint = self.joint_emb(f_joint)                                       # [B, T, 19*32]




        