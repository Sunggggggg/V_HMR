import os
import torch
import torch.nn as nn
import torch_dct as dct

from lib.models.Motion.jointspace import JointTree
from lib.models.Motion.encoder import Encoder
from lib.models.Motion.decoder import Decoder

class Model(nn.Module) :
    def __init__(self, 
                 seqlen,
                 num_joint=19,
                 embed_dim=512,
                 j_encoder_depth=3,
                 t_encoder_depth=3,
                 lifter_pretrained=os.path.join()

                 ) :
        super().__init__()
        self.joint_space = JointTree()

        self.encoder = Encoder(seqlen, num_joint=num_joint, embed_dim=embed_dim, t_encoder_depth=t_encoder_depth, j_encoder_depth=j_encoder_depth)
        self.decoder = Decoder(seqlen=16, num_joint=19, embed_dim=64)

    def forward(self, img_feat, vitpose_j2d, img_path):
        """
        x       : [B, T, 2048]
        vitpose : [B, T, J, 3]
        """
        vitpose_j2d = self.joint_space(vitpose_j2d[..., :2])

        # Encoder
        f_img, f_joint, f_text  = self.encoder(img_feat, vitpose_j2d, img_path)
        # Decoder
        joints3d, vertxs_w_res = self.decoder(f_img, f_joint, f_text)

        return joints3d, vertxs_w_res  # B x 6890 x 3
        


        

