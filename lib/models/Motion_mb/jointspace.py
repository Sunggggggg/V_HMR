import torch
import torch.nn as nn

class JointTree(nn.Module) :
    def __init__(self, ) :
        super().__init__()
        self.pelvis_index = 17

    def xy2polar(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        rho = torch.sqrt(x**2+y**2)
        phi = torch.arctan2(y, x)
        return torch.stack([rho, phi], dim=-1)
    
    def polar2xy(self, polar):
        rho, phi = polar[..., 0], polar[..., 1]
        x = rho * torch.cos(phi)
        y = rho * torch.sin(phi)
        return torch.stack([x, y], dim=-1)
        
    def pelvis_coordi(self, vitpose_2d):
        return vitpose_2d[:, :, self.pelvis_index:self.pelvis_index+1]
    
    def refine_joint(self, vitpose_2d) :
        vitpose_j2d_pelvis = vitpose_2d[:,:,[11,12],:2].mean(dim=2, keepdim=True) 
        vitpose_j2d_neck = vitpose_2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)  
        joint_2d_feats = torch.cat([vitpose_2d[... ,:2], vitpose_j2d_neck], dim=2)   # [B, T, J, 2]
        joint_2d_feats = joint_2d_feats- vitpose_j2d_pelvis
        return joint_2d_feats

    def forward(self, vitpose_2d) :
        """
        vitpose_2d : [B, T, J, 2]
        """
        body_xy = self.refine_joint(vitpose_2d)
        body_rp = self.xy2polar(body_xy)
        return body_rp
