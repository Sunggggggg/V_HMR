import torch
import torch.nn as nn
from .transformer import Attention, DropPath, Mlp, CrossAttention

class MixedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, num_imgs=3):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :num_imgs] + self.drop_path(self.mlp1(self.norm2(x[:, :num_imgs])))
        x2 = x[:, num_imgs:] + self.drop_path(self.mlp2(self.norm3(x[:, num_imgs:])))
        return torch.cat((x1, x2), dim=1)

class ImageFeatureCorrection(nn.Module):
    def __init__(self, embed_dim, depth=3, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_frames=3, num_frames_keep=9) :
        super().__init__()

        self.local_embedding = nn.Linear(2048, embed_dim)
        self.seq_embedding = nn.Linear(2048, embed_dim)

        self.local_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.seq_pos_embedding = nn.Parameter(torch.zeros(1, num_frames_keep, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
        for i in range(depth)])

        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frames, out_channels=1, kernel_size=1)
        self.weighted_mean_ = torch.nn.Conv1d(in_channels=num_frames_keep, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim),
        )

    def forward(self, full_f_img, short_f_img, num_imgs=3):
        """
        short_f_img : [B, 3, 2048]
        full_f_img  : [B, k, 2048]
        """
        B = short_f_img.shape[0]
        f_img = self.local_embedding(short_f_img) + self.local_pos_embedding
        f_seq = self.seq_embedding(full_f_img) + self.seq_pos_embedding

        f = torch.cat([f_img, f_seq], dim=1)                        # [B, 3+k, dim]

        for blk in self.blocks:
            f = blk(f, num_imgs)                                    # [B, 3+k, dim]
        
        f_img, f_seq = f[:, :num_imgs], f[:, num_imgs:]   # [B, 3, dim], [B, k, dim]
        x = torch.cat((self.weighted_mean(f_img), self.weighted_mean_(f_seq)), dim=-1) # [B, 1, 2dim]

        x = self.head(x).view(B, 1, -1)
        return x