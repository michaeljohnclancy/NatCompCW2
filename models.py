
import torch
from torch import nn
import torch.nn.functional as F

class SpiralClassifier(nn.Module):

    def __init__(self):
        super(SpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
