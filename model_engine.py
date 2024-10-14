import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers=nn.Sequential(
            nn.Linear(in_features=12,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=1),
            nn.ReLU()
        )
    def forward(self,x):
        return self.Layers(x)