import cv2
import clip
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from .transformer import Transformer
from lib.models.trans_operator import CrossAttention
 
# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 30, 
    "num_beams": 8,
}

class CaptionEncoder(nn.Module):
    def __init__(self, batch, seqlen=16) :
        super().__init__()
        self.batch = batch
        self.seqlen = seqlen
        # Video captioning
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning")

        # Clip
        self.clip_model, preprocess = clip.load("ViT-B/32")

    def video_caption(self, seq_path):
        """
        seq_path : list 16 

        """
        frames = []
        temp = []
        seq_path = seq_path.detach().cpu().numpy().tobytes().decode('utf-8')
        for f in seq_path :
            if f != '@':
                temp.append(f)
            else :
                if len(temp) != 0 :
                    frames.append(''.join(temp))
                temp = []
                continue
        
        text_emb = []
        for b in range(self.batch):
            b_frames = []
            for path in frames[b*self.seqlen : (b+1)*self.seqlen] :
                img = cv2.imread(path)
                H, W = img.shape[:2]
                H, W = H//4, W//4
                img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
                b_frames.append(img)
        
            pixel_values = self.image_processor(b_frames, return_tensors="pt").pixel_values.cuda() # [B, N, 3, H, W]
            tokens = self.model.generate(pixel_values, **gen_kwargs)
            caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

            # Text embedding
            clip_token = clip.tokenize(caption).cuda()
            f_text = self.clip_model.encode_text(clip_token).float()
            text_emb.append(f_text)
        
        text_emb = torch.stack(text_emb, dim=0)
        return text_emb

    def forward(self, seq_path):
        f_text = self.video_caption(seq_path) # [B, 1, dim]
        return f_text

class TEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(2048, embed_dim)
        self.atten = CrossAttention(dim=embed_dim)

    def forward(self, f_text, f_img):
        """
        f_text : [B, 1, 512]
        f_img : [B, T, 512]
        """
        f_img = self.proj(f_img)
        print(f_img.shape, f_text.shape)
        f = self.atten(f_img, f_text)
        return f

class STEncoder(nn.Module):
    def __init__(self, 
                 num_frames=16,
                 num_joints=19,
                 embed_dim=256, 
                 depth=3, 
                 num_heads=8, 
                 mlp_ratio=2, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.2
                 ) :
        super().__init__()
        self.joint_embed = nn.Linear(2, embed_dim)
        self.temp_embed = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.s_trans = Transformer(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*mlp_ratio,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=num_joints)
        
        self.t_trans = Transformer(depth=depth, embed_dim=embed_dim, mlp_hidden_dim=embed_dim*mlp_ratio,
            h=num_heads, drop_rate=drop_rate, drop_path_rate=drop_path_rate, 
            attn_drop_rate=attn_drop_rate, length=num_frames)

    def forward(self, f_temp, f_joint):
        """
        f_temp  : [B, T, D]
        f_joint : [B, T, J, 2]
        """
        B, T, J, _ = f_joint.shape
        # Spatial
        f_joint = self.joint_embed(f_joint) # [B, T, J, D]
        f_joint = rearrange(f_joint, 'b t j c  -> (b t) j c')
        f_joint = self.s_trans(f_joint, self.spatial_pos_embed)
        f_joint = self.pos_drop(f_joint)
        f_joint = rearrange(f_joint, '(b t) j c -> b t j c', t=T)

        # Temporal
        f_temp = self.temp_embed(f_temp)
        f_temp = f_temp[:, :, None]     # [B, T, 1, D]
        f = f_temp + f_joint            # [B, T, J, D]
        f = self.norm(f)
        
        f = rearrange(f, 'b t j c  -> (b j) t c')   # [BJ, T, D]
        f = self.t_trans(f)
        f = self.pos_drop(f)
        f = rearrange(f, '(b j) t c  -> b t j c', j=J)

        return f
        
