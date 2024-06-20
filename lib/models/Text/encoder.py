import torch
import torch.nn as nn

class MotionToken(nn.Module):
    def __init__(self, embed_dim) :
        super().__init__()

        self.motion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        """
        x : [B, 1, 512]
        """
        