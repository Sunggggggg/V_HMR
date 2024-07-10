import os
import torch
import torch.nn as nn

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor

class Model(nn.Module):
    def __init__(self, 
                 pretrained=os.path.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
                 ) -> None:
        super().__init__()

        self.regressor = Regressor()
        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')
    
    def ext_param(self, pred, T=16):
        theta = pred[-1]['theta']
        B = theta.shape[0] //T
        theta = theta.reshape(B, T, -1)

        pred_cam, pred_pose, pred_shape = theta[..., 0:3], theta[..., 3:75], theta[..., 75:85]
        return pred_cam, pred_pose, pred_shape
    
    def forward(self, f_img, vitpose_2d, is_train=False, J_regressor=None) :
        """
        """
        self.B, self.T = f_img.shape[:2]
        init_output, _ = self.regressor(f_img, is_train=is_train, J_regressor=J_regressor)
        init_pred_cam, init_pred_pose, init_pred_shape = self.ext_param(init_output)    # [B, T, -]




        