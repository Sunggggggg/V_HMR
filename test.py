import torch
from lib.models.Motion.model import Model

model = Model()
x = torch.rand((1, 16, 2048))
y = torch.rand((1, 16, 17, 2))

model(x, y)
