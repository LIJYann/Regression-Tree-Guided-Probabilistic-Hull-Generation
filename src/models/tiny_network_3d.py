import torch
import torch.nn as nn

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
class TinyNetwork3D(nn.Module):
    def __init__(self, hidden_dim=4, num_layers=1, output_dim=2):
        super(TinyNetwork3D, self).__init__()
        layers = []
        input_dim = 3
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        # 初始化权重
        self._initialize_weights()
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x) 