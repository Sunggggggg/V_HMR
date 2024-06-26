import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch_dct as dct

from .jointspace import JointTree
from lib.models.smpl import SMPL_MEAN_PARAMS
from lib.models.trans_operator import Block
from timm.models.layers import DropPath
from einops import rearrange

class Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, \
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16,
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

        
    def forward(self, x):
        
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class JointEncoder(nn.Module) :
    def __init__(self, num_joint=24, emb_dim=32, depth=3, num_heads=4, drop_rate=0.) :
        super().__init__()
        self.jointtree = JointTree()

        self.joint_emb = nn.Linear(2, emb_dim)
        self.s_pos_embed = nn.Parameter(torch.zeros(1, num_joint, emb_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spatial_blocks = nn.ModuleList([
            Block(dim=emb_dim, num_heads=num_heads, mlp_hidden_dim=emb_dim*4.0) for i in range(depth)]
        )
        self.s_norm = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        B, T, J = x.shape[:-1]

        x = self.joint_emb(x)                   # [B, 3, 19, 32]
        x = x.view(B*T, J, -1)                  # [BT, J, 32] 
        x = x + self.s_pos_embed                # 
        x = self.pos_drop(x)

        for blk in self.spatial_blocks:
            x = blk(x)

        x = self.s_norm(x)
        x = x.reshape(B, T, -1)                 # [B, 3, 19*32]

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = qk_scale or head_dim ** -0.5

        self.kv_l = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_l = nn.Linear(dim, dim, bias=qkv_bias)
    
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mem, mask=None):
        B, N, C = mem.shape
        kv = self.kv_l(mem).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] 
        q = self.q_l(x).reshape(B, x.shape[1], 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if attn.dim() == 4:
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn.masked_fill_(mask, -float('inf'))
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, x.shape[1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x

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
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, num_imgs=3):
        #B, T = f_temp.shape[:2]

        #x = torch.cat([f_temp, f_freq], dim=1)  # [B, 3+k, dim]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :num_imgs] + self.drop_path(self.mlp1(self.norm2(x[:, :num_imgs])))
        x2 = x[:, num_imgs:] + self.drop_path(self.mlp2(self.norm3(x[:, num_imgs:])))
        return torch.cat((x1, x2), dim=1)

class FreqTempEncoder(nn.Module) :
    def __init__(self, num_joints, embed_dim, depth, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_coeff_keep=3) :
        super().__init__()
        self.num_coeff_keep = num_coeff_keep 

        # spatial patch embedding
        self.joint_embedding = nn.Linear(2, embed_dim)
        self.freq_embedding = nn.Linear(2*num_joints, embed_dim*num_joints)

        self.joint_pos_embedding = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.freq_pos_embedding = nn.Parameter(torch.zeros(1, num_coeff_keep, embed_dim*num_joints))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim*num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        for i in range(depth)])

        self.head = CrossAttention(embed_dim*num_joints)
        
    def LBF(self, x) :
        """
        x : [B, T, J, 2]
        """
        B, T, J = x.shape[:-1]
        x = dct.dct(x.permute(0, 2, 3, 1))[..., :self.num_coeff_keep]
        x = x.permute(0, 3, 1, 2).contiguous().view(B, self.num_coeff_keep, -1) # [B, k, J*2]

        return x

    def forward(self, full_2d_joint, short_2d_joint, num_imgs=3):
        B, T, J = short_2d_joint.shape[:3]

        freq_feat = self.LBF(full_2d_joint)     # [B, t, J*2]
        joint_feat = short_2d_joint             # [B, 3, J, 2]

        freq_feat = self.freq_embedding(freq_feat)
        freq_feat = freq_feat + self.freq_pos_embedding     # [B, k, J*32]

        joint_feat = self.joint_embedding(joint_feat).reshape(B*T, J, -1)   # [B3, J, 32]
        joint_feat = joint_feat + self.joint_pos_embedding                  # [B3, J, 32]
        joint_feat = joint_feat.reshape(B, T, J, -1).view(B, T, -1)         # [B, T, J*32]
        f = torch.cat([joint_feat, freq_feat], dim=1)                       # [B, T+k, J*32]

        for blk in self.blocks:
            f = blk(f, num_imgs)                         # [B, T+k, J*32]
        
        joint_feat, freq_feat = f[:, :num_imgs], f[:, num_imgs:]   # [B, 3, J*32], [B, k, J*32]
        joint_feat = joint_feat + self.head(joint_feat, freq_feat)
        return joint_feat

class FreqTempEncoder_img(nn.Module) :
    def __init__(self, embed_dim, depth, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_coeff_keep=3) :
        super().__init__()
        self.num_coeff_keep = num_coeff_keep 

        # spatial patch embedding
        self.img_embedding = nn.Linear(2048, embed_dim)
        self.freq_embedding = nn.Linear(2048, embed_dim)

        self.img_pos_embedding = nn.Parameter(torch.zeros(1, 3, embed_dim))
        self.freq_pos_embedding = nn.Parameter(torch.zeros(1, num_coeff_keep, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        for i in range(depth)])

        self.head = CrossAttention(embed_dim)
        
    def LBF(self, x) :
        """
        x : [B, T, 2048]
        """
        B = x.shape[0]
        x = dct.dct(x.permute(0, 2, 1))[..., :self.num_coeff_keep]
        x = x.permute(0, 2, 1).contiguous().view(B, self.num_coeff_keep, -1) # [B, k, 2048]

        return x

    def forward(self, full_f_img, short_f_img):
        """
        full_f_img, short_f_img : [B, T, 2048], [B, 3, 2048]
        """

        freq_feat = self.LBF(full_f_img)        # [B, k, 2048]
        img_feat = short_f_img                  # [B, 3, 2048]

        freq_feat = self.freq_embedding(freq_feat)          # [B, k, 256]
        freq_feat = freq_feat + self.freq_pos_embedding     # [B, k, 256]

        img_feat = self.img_embedding(img_feat)             # [B, 3, 256]
        img_feat = img_feat + self.img_pos_embedding        # [B, 3, 256]
        f = torch.cat([img_feat, freq_feat], dim=1)         # [B, 3+k, 256]

        for blk in self.blocks:
            f = blk(f)                   
        
        img_feat, freq_feat = f[:, :3], f[:, 3:]                # [B, 3, 256], [B, k, 256]
        img_feat = img_feat + self.head(img_feat, freq_feat)    # []
        return img_feat