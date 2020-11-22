
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
    
    def __init__(self, network_structure, nonlinearity_keys=None):
        super(GenericSpiralClassifier, self).__init__()
        
        nonlinearity_dict = {"A": torch.tanh, "B": torch.relu, "C": torch.sigmoid}
        
        print(len(network_structure))
        print(network_structure)
        print(len(nonlinearity_keys))
        print(nonlinearity_keys)
        assert(len(network_structure)-2 == len(nonlinearity_keys))
        
        if nonlinearity_keys is None:
            nonlinearities = [lambda x: torch.tanh(x) for layer in self.layers[:-1]]
        else:
            nonlinearities = [nonlinearity_dict[nonlinearity_keys[i]] for i in range(len(nonlinearity_keys))]
            
        self.nonlinearities = nonlinearities

        self.layers = nn.ModuleList()
        for i in range(len(network_structure) - 1):
            self.layers.append(make_layer(network_structure[i], network_structure[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.nonlinearities[i](layer(x))
        x = self.layers[-1](x)
        return x

def make_layer(input_shape, output_shape):
    return nn.Linear(input_shape, output_shape)

