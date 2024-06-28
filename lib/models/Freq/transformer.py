import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BPF_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        x       : [B, N, C] 
        mask    : [N, N]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, h, N, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]                # [B, h, N, dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, h, N, N]
        if mask is not None:
            if attn.dim() == 4:
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn.masked_fill_(mask, -float('inf'))
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, h, N, dim] -> [B, N, h, dim]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

if __name__ == "__main__" :
    x = torch.randn((1, 19, 2))
