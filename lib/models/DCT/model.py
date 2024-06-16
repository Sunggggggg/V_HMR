import torch
import torch.nn as nn
import torch_dct as dct

class Model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None) :
        f_img 