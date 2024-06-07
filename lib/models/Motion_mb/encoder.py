import torch
import torch.nn as nn

from lib.models.trans_operator import 

"""From PMCE models.PoseEstimation"""
class JointEncoder(nn.Module) :
    def __init__(self, num_frames=16, num_joints=17, embed_dim=256, depth=3, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, pretrained=False):
        super().__init__()
        
        in_dim = 2
        out_dim = 3    
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.joint_embed = nn.Linear(in_dim, embed_dim)
        self.imgfeat_embed = nn.Linear(2048, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=embed_dim*mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=embed_dim*mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.fusion = torch.nn.Conv2d(in_channels=num_frames, out_channels=1, kernel_size=1)
        
        if pretrained:
            self._load_pretrained_model()
        
    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = torch.load(cfg.MODEL.posenet_path, map_location='cuda')
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def SpaTemHead(self, x, img_feat):
        b, t, j, c = x.shape
        x = rearrange(x, 'b t j c  -> (b t) j c')
        x = self.joint_embed(x)
        x = x + rearrange(self.imgfeat_embed(img_feat), 'b t c  -> (b t) 1 c')
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        spablock = self.SpatialBlocks[0]
        x = spablock(x)
        x = self.norm_s(x)
        
        x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        temblock = self.TemporalBlocks[0]
        x = temblock(x)
        x = self.norm_t(x)
        return x

    def forward(self, x, img_feat):
        b, t, j, c = x.shape
        x = self.SpaTemHead(x, img_feat) # bj t c
        
        for i in range(1, self.depth):
            SpaAtten = self.SpatialBlocks[i]
            TemAtten = self.TemporalBlocks[i]
            x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
            x = SpaAtten(x)
            x = self.norm_s(x)
            x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
            x = TemAtten(x)
            x = self.norm_t(x)

        x = rearrange(x, '(b j) t c -> b t j c', j=j)
        x = self.regression(x) # (b t (j * 3))
        x = x.view(b, t, j, -1)
        xout = self.fusion(x)
        xout = xout.squeeze(1)

        return xout

class Encoder(nn.Module) :
    def __init__(self, emb_dim):
        super().__init__()
        self.pos_enc = nn.Linear(2, emb_dim)


    def forward(self, f_img, f_joint):
        """
        f_img   : [B, T, d]
        f_joint : [B, T, J, 2]
        """
        f_joint = self.pos_enc(f_joint)

        f = torch.cat([f_img, f_joint], dim=-1)
        
