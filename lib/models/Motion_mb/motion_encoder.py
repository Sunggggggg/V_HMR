import torch
import torch.nn as nn
from lib.models.trans_operator import CrossAttention

class MotionEncoder(nn.Module):
    def __init__(self, seqlen, emb_dim=256, text_emb_dim=512) :
        super().__init__()
        self.joint_embed = nn.Linear(2*seqlen, emb_dim)
        self.text_embed = nn.Linear(text_emb_dim, emb_dim)
        self.motion_encoder = CrossAttention(dim=emb_dim)
    
    def forward(self, f_text, f_joint) :
        """
        f_joint : [B, T, J, 2]
        f_text  : [B, 1, 512]
        """
        B, T, J, _ = f_joint.shape
        f_joint = f_joint.permute(0, 2, 1, 3).flatten(-2)   # [B, J, 2T]
        f_joint = self.joint_embed(f_joint)
        f_text = self.text_embed(f_text)
        
        f_motion = self.motion_encoder(f_joint, f_text)     # [B, J, d]
        f_motion = f_motion.softmax(dim=1)                  # [B, J, d]
        return f_motion

class ContextEncoder(nn.Module):
    def __init__(self, emb_dim=256, img_emb_dim=2048, text_emb_dim=512) :
        super().__init__()
        self.img_embed = nn.Linear(img_emb_dim, emb_dim)
        self.text_embed = nn.Linear(text_emb_dim, emb_dim)
        self.motion_encoder = CrossAttention(dim=emb_dim)
    
    def forward(self, f_text, f_img) :
        """
        f_img   : [B, T, 2048]
        f_text  : [B, 1, 512]
        """
        f_img = self.img_embed(f_img)       # [B, T, d]
        f_text = self.text_embed(f_text)    # [B, 1, d]
        
        f_context = self.motion_encoder(f_img, f_text)      # [B, T, d]
        return f_context