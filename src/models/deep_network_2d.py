import torch
import torch.nn as nn

class DeepNetwork2D(nn.Module):
    def __init__(self, hidden_dim=50, num_layers=6, output_dim=2, seed=0):
        super(DeepNetwork2D, self).__init__()
        
        # 设置随机种子以确保权重初始化一致性
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        layers = []
        input_dim = 2  # 2D输入
        
        # 构建隐藏层
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用确定性初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用固定的初始化方法
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x) 