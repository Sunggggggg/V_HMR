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
        
        return x

    def forward(self, f_text, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        f_img       : [B, T, 2048]
        f_joint     : [B, T, J, 2]
        """
        vitpose_2d = self.jointtree.add_joint(vitpose_2d[..., :2])
        vitpose_2d = self.jointtree.map_kp2joint(vitpose_2d)        # [B, T, 24, 2]
        freq_vitpose_2d = dct.dct(vitpose_2d.permute(0, 2, 3, 1))


        
        





        