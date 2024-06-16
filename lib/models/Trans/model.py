import torch
import torch.nn as nn
from .GMM import GMM

class Model(nn.Module):
    def __init__(
            self,
            seqlen,
            n_layers=1,
            d_model=2048,
            num_head=8, 
            dropout=0., 
            drop_path_r=0.,
            atten_drop=0.,
            mask_ratio=0.,

    ):
        super(Model, self).__init__()
        self.global_modeling = GMM(seqlen, n_layers, d_model, num_head, dropout, drop_path_r, atten_drop, mask_ratio)

    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None) :
        """
        """
        smpl_output_global, mask_ids, mem, pred_global = self.global_modeling(f_img, is_train=is_train, J_regressor=J_regressor)
        

        return smpl_output, mask_ids, smpl_output_global