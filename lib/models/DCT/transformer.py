import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch_dct as dct

from .jointspace import JointTree
from lib.models.smpl import SMPL_MEAN_PARAMS
from lib.models.trans_operator import Block
from timm.models.layers import DropPath

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

        
    def forward(self, x):
        
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class TemporalEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024,
            h=8, drop_rate=0.1, drop_path_rate=0.2, attn_drop_rate=0., length=16):
        super().__init__()
        qkv_bias = True
        qk_scale = None
        # load mean smpl pose, shape and cam
        mean_params = np.load(SMPL_MEAN_PARAMS)
        self.init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).to('cuda')
        self.init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0).to('cuda')
        self.init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0).to('cuda')
        
        self.mask_token_mlp = nn.Sequential(
            nn.Linear(24 * 6 + 13, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim // 2)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim // 2, num_heads=h, mlp_hidden_dim=embed_dim * 2, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim // 2))
        self.decoder_norm = norm_layer(embed_dim // 2)

    def forward(self, x, is_train=True, mask_ratio=0.):
        if is_train:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=True, mask_ratio=mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        else:
            latent, mask, ids_restore = self.forward_encoder(x, mask_flag=False,mask_ratio=0.)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask, latent

    def forward_encoder(self, x, mask_flag=False, mask_ratio=0.):
        x = x + self.pos_embed
        if mask_flag:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            # print('mask')
        else:
            mask = None
            ids_restore = None

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        if ids_restore is not None:
            mean_pose = torch.cat((self.init_pose, self.init_shape, self.init_cam), dim=-1)
            # append mask tokens to sequence
            mask_tokens = self.mask_token_mlp(mean_pose)
            mask_tokens = mask_tokens.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        else:
            x_ = x
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1) # assgin value from ids_restore
        
        return x_masked, mask, ids_restore

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

class FreqTempBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # Temp1
        self.norm_t = norm_layer(dim)
        self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Freq1
        self.norm_f = norm_layer(dim)
        self.attn_f = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # TempFreq
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, f_temp, f_freq):
        """
        f_temp : [B, 3, J, dim]
        f_freq : [B, t, J, dim]
        """
        B, t, J = f_freq.shape[:-1]
        f_temp = f_temp + self.drop_path(self.attn_t(self.norm_t(f_temp)))
        f_freq = f_freq + self.drop_path(self.attn_f(self.norm_f(f_freq)))

        f_freq = dct.idct(f_freq.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()  # [B, t, J, dim]
        f = torch.cat([f_temp, f_freq], dim=1)
        f = f + self.drop_path(self.attn(self.norm1(f)))
        f = f + self.drop_path(self.mlp(self.norm2(f)))        

        f_temp, f_freq = f[:, :3], f[:, 3:]
        f_freq = dct.dct(f_freq.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous().view(B, t, J, -1)
        return f_temp, f_freq

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

class FreqTempEncoder(nn.Module) :
    def __init__(self, num_joints, embed_dim, depth, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_coeff_keep=3) :
        super().__init__()
        self.num_coeff_keep = num_coeff_keep 

        # spatial patch embedding
        self.joint_embedding = nn.Linear(2, embed_dim)
        self.freq_embedding = nn.Linear(2, embed_dim)

        self.joint_pos_embedding = nn.Parameter(torch.zeros(1, 3, num_joints, embed_dim))
        self.freq_pos_embedding = nn.Parameter(torch.zeros(1, 3, num_joints, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            FreqTempBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        for i in range(depth)])

        self.joint_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            CrossAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop_rate, drop_rate)
        )
        
    def LBF(self, x) :
        """
        x : [B, T, J, 2]
        """
        B, T, J = x.shape[:-1]
        x = dct.dct(x.permute(0, 2, 3, 1))[..., :self.num_coeff_keep]
        x = x.permute(0, 3, 1, 2).contiguous().view(B, self.num_coeff_keep, J, -1)

        return x

    def forward(self, full_2d_joint, short_2d_joint):
        ######### Input #########
        freq_feat = self.LBF(full_2d_joint)     # [B, t, J, 2]
        joint_feat = short_2d_joint             # [B, 3, J, 2]

        freq_feat = freq_feat + self.freq_pos_embedding
        joint_feat = joint_feat + self.joint_pos_embedding

        for blk in self.blocks:
            joint_feat, freq_feat = blk(joint_feat, freq_feat)
        
        joint_feat = self.joint_head(joint_feat, freq_feat) # [B, 3, 24, 32]
        joint_feat = joint_feat.flatten(-2)
        
        return joint_feat