import torch
import torch.nn as nn
import torch_dct as dct

from lib.models.Motion.text import CaptionEmb

class Model(nn.Module) :
    def __init__(self, ) :
        super().__init__()
        self.text_encoder = CaptionEmb()

        self.encoder = 
        self.decoder = 

    def forward(self, x, vitpose_j2d, img_path, is_train=False, J_regressor=None):
        """
        x       : [B, T, 2048]
        vitpose : [B, T, J, 3]
        """
        text_emb = self.text_encoder(img_path)
        

