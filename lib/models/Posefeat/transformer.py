import torch
import torch.nn as nn
from functools import partial
import numpy as np
from lib.models.trans_operator import Block
from einops import rearrange

"""Vanilla transformer"""
class Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, \
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=27,
            ):
        super().__init__()
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) 

    def forward(self, x, pos_embed=None):
        if pos_embed is None :
            x = x + self.pos_embed
        else :
            x = x + pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
class STEncoder(nn.Module):
    def __init__(self, 
                 num_frames=16,
                 num_joints=20,
                 embed_dim=512, 
                 depth=3, 
                 num_heads=8, 
                 mlp_ratio=2, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2
                 ) :
        super().__init__()
        self.depth = depth

        self.joint_embed = nn.Linear(2, embed_dim)
        self.img_embed = nn.Linear(2048, embed_dim)

        self.temp_embed = nn.Linear(embed_dim, embed_dim)
        self.s_norm = nn.LayerNorm(embed_dim)
        self.t_norm = nn.LayerNorm(embed_dim)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.s_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=embed_dim*mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        
        self.t_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=embed_dim*mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])

    def forward(self, f_temp, f_joint):
        """
        f_temp  : [B, T, D]
        f_joint : [B, T, J, 2]
        """
        B, T, J, _ = f_joint.shape
        f_joint = self.joint_embed(f_joint)         # [B, T, J, D]
        f_temp = self.img_embed(f_temp)[:, :, None] # [B, T, 1, D]
        f = f_joint + f_temp                        # [B, T, J, D]

        # Spatial
        f = rearrange(f, 'b t j c  -> (b t) j c')   # [BT, J, D]
        f = f + self.spatial_pos_embed
        f = self.pos_drop(f)
        f = self.s_blocks[0](f)                     # [BT, J, D]
        f = self.s_norm(f)

        # Temporal
        f = rearrange(f, '(b t) j c -> (b j) t c', t=T) #[BJ, T, D]
        f = f + self.temporal_pos_embed             # [BJ, T, D]
        f = self.pos_drop(f)
        f = self.t_blocks[0](f)                     # [BJ, T, D]
        f = self.t_norm(f)

        # Loop
        for i in range(1, self.depth):
            s_block = self.s_blocks[i]
            t_block = self.t_blocks[i]

            f = rearrange(f, '(b j) t c -> (b t) j c', j=J)
            f = s_block(f)
            f = self.s_norm(f)

            f = rearrange(f, '(b t) j c -> (b j) t c', t=T)
            f = t_block(f)
            f = self.t_norm(f)

        f = rearrange(f, '(b j) t c -> b t j c', j=J)
        return f