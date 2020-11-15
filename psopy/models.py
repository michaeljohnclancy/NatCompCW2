
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


class GenericSpiralClassifier(nn.Module):

    def __init__(self, network_structure):

        super(GenericSpiralClassifier, self).__init__()


        self.layers = nn.ModuleList()
        for i in range(len(network_structure) - 1):
            self.layers.append(make_layer(network_structure[i], network_structure[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def make_layer(input_shape, output_shape):
    return nn.Linear(input_shape, output_shape)

