
import torch
from torch import nn
import torch.nn.functional as F

class SingleLayerPerceptron(nn.Module):

    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(6, 2)


    def forward(self, x):
        x = self.fc(x)
        return x


class SpiralClassifier(nn.Module):

    def __init__(self):
        super(SpiralClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
