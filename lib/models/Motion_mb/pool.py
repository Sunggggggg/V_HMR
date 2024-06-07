import torch
import torch.nn as nn

s_pooling = {
    "Head" : [0, 1, 2, 3, 4,],
    "To" : [17, 18],
    "Rarm" : [6, 8, 10],
    "Larm" : [5, 7, 9],
    "Rleg" : [12, 14, 16],
    "Lleg" : [11, 13, 15]
}

class Tpooling(nn.Module) :
    def __init__(self, 
                 seqlen=16) :
        super().__init__()
        self.mid_frame = seqlen // 2
        

    def forward(self, x) :
        return