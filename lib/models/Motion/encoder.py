import os
import cv2
import torch
import torch.nn as nn
import clip
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from einops import rearrange
from functools import partial
from lib.models.trans_operator import Block
from lib.models.Motion.transformer import Transformer

from lib.core.config import cfg, PMCE_POSE_DIR


# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 30, 
    "num_beams": 8,
}

class TempEncoder(nn.Module):
    def __init__(self,
                 seqlen, 
                 embed_dim=512,
                 mlp_hidden_dim=1024,
                 depth=3
                 ) :
        super().__init__()
        self.input_proj = nn.Linear(2048, embed_dim)
        self.temp_encoder = Transformer(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim, length=seqlen)
        
    def forward(self, f_img) :
        f_img = self.input_proj(f_img)
        f_img = self.temp_encoder(f_img)
        return f_img
    
class CaptionEncoder(nn.Module):
    def __init__(self) :
        super().__init__()
        # Video captioning
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning")

        # Clip
        self.clip = clip.load("ViT-B/32")

    def video_caption(self, seq_path):
        """
        seq_path : list 16
        """
        frames = []
        for path in seq_path :
            path = path.numpy().tobytes().decode('utf-8')
            img = cv2.imread(path)
            H, W = img.shape[:2]
            H, W = H//4, W//4
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            frames.append(img)
        
        pixel_values = self.image_processor(frames, return_tensors="pt").pixel_values # [B, N, 3, H, W]
        tokens = self.model.generate(pixel_values, **gen_kwargs)
        caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

        del frames
        return caption

    def forward(self, seq_path):
        caption = self.video_caption(seq_path) # [B, dim]

        # Text embedding
        clip_token = self.clip.tokenize(caption)
        f_text = self.clip.encode_text(clip_token)
        return f_text
    
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
    def __init__(self, 
                 seqlen,
                 num_joint=19,
                 embed_dim=512,
                 t_encoder_depth=3,
                 j_encoder_depth=3,
                 lifter_pretrained=os.path.join(PMCE_POSE_DIR, 'pose_3dpw.pth.tar')
                 ) :
        super().__init__()
        self.temp_encoder = TempEncoder(seqlen=seqlen, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*2, depth=t_encoder_depth)
        self.text_encoder = CaptionEncoder()
        self.lifter = JointEncoder(num_frames=seqlen, num_joints=num_joint, embed_dim=embed_dim, depth=j_encoder_depth, pretrained=lifter_pretrained)

    def forward(self, img_feat, vitpose_j2d, img_path) :
        """
        img_feat    : [B, T, 2048]
        joint_2d    : [B, T, J, 2]
        text_emb    : [B, 1, 512]

        return
        img_feat    : [B, T, 512]
        vitpose_3d  : [B, J, 3]
        text_emb    : [B, 1, 512]
        """
        f_img = self.temp_encoder(img_feat)
        f_joint = self.lifter(vitpose_j2d, img_feat)    # [B, J, 3]

        if img_path is None :
            f_text = None
        else : 
            f_text = self.text_encoder(img_path)            # [B, 1, 512]
        
        return f_img, f_joint, f_text 