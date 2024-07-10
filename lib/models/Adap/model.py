import os
import torch
import torch.nn as nn

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor

class Model(nn.Module):
    def __init__(self, 
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar')):
        super().__init__()

        self.regressor = Regressor()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, f_img, is_train=False, J_regressor=None):
        smpl_output_global, pred_global = self.regressor(f_img, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        
