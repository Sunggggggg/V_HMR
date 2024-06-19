import torch
import os.path as osp
import torch.nn as nn
from lib.core.config import BASE_DATA_DIR
from lib.models.GLoT.transformer_global import Transformer
import torch.nn.functional as F

class GMM(nn.Module):
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
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(GMM, self).__init__()
            
        self.proj = nn.Linear(512, d_model)
        self.trans = Transformer(depth=n_layers, embed_dim=d_model, \
                mlp_hidden_dim=d_model*4, h=num_head, drop_rate=dropout, \
                drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=seqlen)
        self.mask_ratio = mask_ratio

    def forward(self, input, is_train=False):
        #input = self.proj(input)
        if is_train:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=True, mask_ratio=self.mask_ratio)
        else:
            mem, mask_ids, ids_restore = self.trans.forward_encoder(input, mask_flag=False, mask_ratio=0.)
        pred = self.trans.forward_decoder(mem, ids_restore)  # [N, L, p*p*3]

        return pred, mask_ids

    def initialize_weights(self):
        torch.nn.init.normal_(self.trans.pos_embed, std=.02)
        torch.nn.init.normal_(self.trans.decoder_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
