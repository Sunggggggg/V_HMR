import torch
import torch.nn as nn
from functools import partial
from .transformer import Transformer

"""Disentangle Transformer moudle"""
class DTM(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, 
                 h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16,
                 encoder=True, decoder=True):
        super().__init__()
        self.decoder = decoder
        qkv_bias = True
        qk_scale = None

        if encoder :
            # Camera network
            self.cam_proj = nn.Linear(2048, embed_dim//2)
            self.cam_encoder = Transformer(depth=depth-1, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*2.0, h=h, 
                                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=length)
            # Pose, Shape network
            self.pose_shape_proj = nn.Linear(2048, embed_dim)
            self.pose_shape_encoder = Transformer(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*4.0, h=h, 
                                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=length)
            self.input_norm = nn.LayerNorm(embed_dim)
        
        if decoder:
            # Camera network
            self.cam_dec_proj = nn.Linear(embed_dim//2, embed_dim//4)
            self.cam_decoder = Transformer(depth=1, embed_dim=embed_dim//4, mlp_hidden_dim=embed_dim, h=h, 
                                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=length)
            # Pose, Shape network
            self.pose_shape_dec_proj = nn.Linear(embed_dim, embed_dim//2)
            self.pose_shape_decoder = Transformer(depth=1, embed_dim=embed_dim//2, mlp_hidden_dim=embed_dim*4.0, h=h, 
                                        drop_rate=drop_rate, drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, length=length)

    def forward(self, x, f_joint=None):
        """
        x           : [B, T, 2048]
        f_joint     : [B, T, dim]
        """
        # Encoder
        f_cam = self.cam_proj(x)            # [B, T, 256]
        f_cam = self.cam_encoder(f_cam)     # [B, T, 256]
        
        f_pose_shape = self.pose_shape_proj(x)
        f_pose_shape = self.pose_shape_encoder(f_pose_shape)
        
        # Decoder
        if self.decoder and f_joint is not None :
            f_cam = self.cam_dec_proj(f_cam)    # [B, T, 128]
            f_cam = self.cam_decoder(f_cam)     # [B, T, 128]

            f_pose_shape = self.input_norm(f_joint + f_pose_shape)      # [B, T, 512]
            f_pose_shape = self.pose_shape_dec_proj(f_pose_shape)       # [B, T, 256]
            f_pose_shape = self.pose_shape_decoder(f_pose_shape)        # [B, T, 256]

        return f_cam, f_pose_shape