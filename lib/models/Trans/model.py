import torch
import torch.nn as nn

from .GMM import GMM
from .jointspace import JointTree

class Model(nn.Module):
    def __init__(
            self,
            seqlen,
            n_layers=1,
            d_model=512,
            num_head=8, 
            dropout=0., 
            drop_path_r=0.,
            atten_drop=0.,
            mask_ratio=0.,

    ):
        super(Model, self).__init__()
        self.proj_motion = nn.Parameter(torch.randn(d_model, d_model//2))
        self.global_modeling = GMM(seqlen, n_layers, d_model, num_head, dropout, drop_path_r, atten_drop, mask_ratio)
        self.jointtree = JointTree()


    def forward(self, f_text, f_img, f_joint, is_train=False, J_regressor=None) :
        """
        """
        f_joint = self.jointtree.add_joint(f_joint[..., :2])    # [B, 16, 20, 2]
        f_joint = self.jointtree.map_kp2joint(f_joint)          # [B, 16, 24, 2]

        pred, mask_ids, mem = self.global_modeling(f_img, is_train=is_train, J_regressor=J_regressor)
        
        