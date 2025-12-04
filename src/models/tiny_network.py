import torch
import torch.nn as nn

class TinyNetwork(nn.Module):
    def __init__(self):
        super(TinyNetwork, self).__init__()
        self.relu_layer = nn.Linear(2, 3)
        self.relu_layer.weight.data = torch.tensor([[1.0, -2.0],
                                                    [-1.0, 0.5],
                                                    [1.0, 1.5]])
        self.relu_layer.bias.data = torch.tensor([0.5, 1.0, -0.5])
        self.output_layer = nn.Linear(3, 2)
        self.output_layer.weight.data = torch.tensor([[-1.0, -1.0, 1.0],
                                                     [2.0, 1.0, -0.5]])
        self.output_layer.bias.data = torch.tensor([-0.2, -1.0])
    def forward(self, x):
        x = torch.relu(self.relu_layer(x))
        x = self.output_layer(x)
        return x
