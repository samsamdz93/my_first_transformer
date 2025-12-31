import torch
import torch.nn as nn

# Definition of a MLP
class MLP(nn.Module):
    def __init__(self, sequence : list, acti = nn.functional.relu):
        super(MLP, self).__init__()
        self.activation = acti
        self.layers = []
        for i in range(len(sequence)-1):
            self.layers.append(nn.Linear(sequence[i], sequence[i+1]))
        self.layers = nn.ParameterList(self.layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x