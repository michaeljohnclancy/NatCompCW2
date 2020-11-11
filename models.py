
import torch
from torch import nn
import torch.nn.functional as F

class BaselineSpiralClassifier(nn.Module):

    def __init__(self):
        super(BaselineSpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class LinearInputsSpiralClassifier(nn.Module):

    def __init__(self):
        super(LinearInputsSpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
