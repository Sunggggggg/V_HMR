import torch
import torch.nn as nn

joint2kp = [
    15, # 0
    15, # 1
    15, # 2
    15, # 3
    15, # 4
    16, # 5
    17, # 6
    18, # 7
    19, # 8
    22, # 9
    23, # 10
    1,  # 11
    2,  # 12
    4,  # 13
    5,  # 14
    10, # 15
    11, # 16
    0,  # 17
    12  # 18
]

kp2joint = [
    17, # 0(Pelvis)
    11, # 1(Left Hip)
    12, # 2(Right Hip)
    17, ## 3(Spine3)
    13, # 4(Left Knee)
    14, # 5(Right Knee)
    19, ## 6(Spine6)
    15, # 7(Left Ankle)
    16, # 8(Right Ankle)
    19, ## 9(Spine9)
    15, ## 10(Left Toe)
    16, ## 11(Right Toe)
    18, # 12(Neck)
    5,  ## 13
    6,  ## 14
    0,  # 15
    5,  # 16
    6,  # 17
    7,  # 18
    8,  # 19
    9,  # 20
    10, # 21
    9,  ## 22
    10  ## 23
]
Learnable_joint = [3, 6, 9, 10, 11, 13, 14, 22, 23]


class JointTree():
    def __init__(self, num_joint_init=17) :
        self.num_joint_init = num_joint_init
        self.num_joint_out = num_joint_init + 3 # pelvis, neck, spin

        self.jointtoken = nn.ModuleList()
        for i in range(len(Learnable_joint)):
            self.jointtoken.append(nn.Linear(19*2, 2))

    def cal_pelvis(self, vitpose_2d):
        return vitpose_2d[:,:,[11,12],:2].mean(dim=2, keepdim=True)
    
    def cal_neck(self, vitpose_2d) :
        return vitpose_2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)
    
    def cal_spin(self, vitpose_2d):
        pelvis = self.cal_pelvis(vitpose_2d)    # [B, T, 1, 2]
        neck = self.cal_neck(vitpose_2d)        # [B, T, 1, 2]
        pelvis_neck = torch.cat([pelvis, neck], dim=2)
        return pelvis_neck.mean(dim=2, keepdim=True)
    
    def add_joint(self, vitpose_2d) :
        """
        vitpose_2d : [B, T, 17, 2]
        """
        pelvis = self.cal_pelvis(vitpose_2d)    # [B, T, 1, 2]
        neck = self.cal_neck(vitpose_2d)        # [B, T, 1, 2]
        #spin = self.cal_spin(vitpose_2d)        # [B, T, 1, 2]
        #total_joint = torch.cat([vitpose_2d, pelvis, neck, spin], dim=2)
        total_joint = torch.cat([vitpose_2d, pelvis, neck], dim=2)

        return total_joint
    
    def map_joint2kp(self, ):
        
        return

    def map_kp2joint(self, vitpose_2d):
        """
        vitpose_2d : [B, T, 20, 2]
        """
        kp = torch.stack([vitpose_2d[:, :, i] for i in kp2joint], dim=2) # [B, T, 24, 2]
        return kp
    
    def mapping(self, vitpose_2d):
        """
        vitpose_2d : [B, T, 19, 2]
        """
        joint_list = []
        for idx, joint_idx in enumerate(kp2joint) :
            if idx in Learnable_joint :
                joint = vitpose_2d * self.jointtoken[Learnable_joint.index(idx)].unsqueeze(-1)  # [B, T, 2]
                joint = torch.sum(joint, dim=-2)
            else :
                joint = vitpose_2d[:, :, joint_idx] # [B, T, 2]

            print(joint.shape)
            joint_list.append(joint)
        joint_list = torch.stack(joint_list, dim=-2)

        return joint_list