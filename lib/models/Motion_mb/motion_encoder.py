import torch
import torch.nn as nn
from lib.models.trans_operator import CrossAttention

class MotionEncoder(nn.Module):
    def __init__(self, seqlen, emb_dim) :
        super().__init__()
        self.proj = nn.Linear(2*seqlen, emb_dim)
        self.motion_encoder = CrossAttention(dim=emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(1)
        )
    
    def forward(self, f_text, f_joint) :
        """
        f_joint : [B, T, J, 2]
        f_text  : [B, 1, 512]
        """
        B, T, J, _ = f_joint.shape
        f_joint = f_joint.permute(0, 2, 1, 3).flatten(-2)   # [B, J, 2T]
        f_joint = self.proj(f_joint)
        
        f_motion = self.motion_encoder(f_joint, f_text)     # [B, J, d]
        f_motion = self.fusion(f_motion)            # [B, J, 1]
        f_motion = f_motion.view(B, 1, J, 1)

        return f_motion

class ContextEncoder(nn.Module):
    def __init__(self, emb_dim) :
        super().__init__()
        self.proj = nn.Linear(2048, emb_dim)
        self.motion_encoder = CrossAttention(dim=emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
            nn.Softmax(1)
        )
    
    def forward(self, f_text, f_img) :
        """
        f_img   : [B, T, 2048]
        f_text  : [B, 1, 512]
        """
        f_img = self.proj(f_img)
        
        f_context = self.motion_encoder(f_img, f_text)      # [B, T, d]
        f_context = self.fusion(f_context)                  # [B, T, 1]

        return f_context