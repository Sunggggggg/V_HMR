import torch
import torch.nn as nn
from lib.models.trans_operator import CrossAttention

class TextEmbedToken(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.) :
        super().__init__()

        self.motion_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.context_query = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.motion_encoder = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.context_encoder = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        """
        x : [B, 1, 512]
        """
        motion_token = self.motion_encoder(self.motion_query, x)
        context_token = self.context_encoder(self.context_query, x)

        return motion_token, context_token