import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from lib.models.trans_operator import Block

from lib.models.Motion.joint_lifter import JointLift


"""From PMCE models.PoseEstimation"""
class MotionEnc(nn.Module) :
    def __init__(self, num_frames=16, num_joints=17, embed_dim=256, depth=3, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, pretrained=False):
        
        self.caption_emb = CaptionEmb()
        self.lifter = JointLift()


    def forward(self, img_feat, joint_2d, text_emb) :
        """
        img_feat    : [B, T, 2048]
        joint_2d    : [B, T, J, 2]
        text_emb    : [B, 1, 512]
        """
        joint_3d = self.lifter(joint_2d, img_feat) # [B, J, 3]
        text_emb
