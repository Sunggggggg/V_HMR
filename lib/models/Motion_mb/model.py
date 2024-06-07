import torch
import torch.nn as nn

from .encoder import TEncoder, STEncoder, CaptionEncoder

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
                 drop_path_rate=0.2) :
        super().__init__()
        self.mid_frame = num_frames // 2

        self.text_emb = CaptionEncoder()
        self.t_trans = TEncoder(embed_dim=embed_dim)
        self.s_trans = STEncoder(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*mlp_ratio,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=num_joints)

    def forward(self, f_text, f_img, f_joint, is_train=False):
        f_text = self.text_emb(f_text)
        f_temp = self.t_trans(f_text, f_img)    # [B, T, D]
        f = self.s_trans(f_temp, f_joint)       # [B, T, J, D]

        return