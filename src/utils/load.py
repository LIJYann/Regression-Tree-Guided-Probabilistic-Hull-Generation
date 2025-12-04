import os
import numpy as np
import torch
import torch.nn as nn
import math
import sys


def readNNet(nnetFile, withNorm=True):
    '''
    Read a .nnet file and return list of weight matrices and bias vectors
    
    Inputs:
        nnetFile: (string) .nnet file to read
        withNorm: (bool) If true, return normalization parameters
        
    Returns: 
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''
    try:
        # Open NNet file
        with open(nnetFile, 'r') as f:
            # Skip header lines
            line = f.readline()
            while line[:2] == "//":
                line = f.readline()

            # Extract information about network architecture
            record = line.split(',')
            numLayers = int(record[0])
            inputSize = int(record[1])

            line = f.readline()
            layerSizes = [int(x) for x in line.strip().split(',') if x]

            # Ensure that the architecture information is correct
            assert len(layerSizes) == numLayers + 1, "Layer sizes don't match number of layers."

            # Skip extra obsolete parameter line
            f.readline()

            # Read the normalization information
            inputMins = [float(x) for x in f.readline().strip().split(",") if x]
            inputMaxes = [float(x) for x in f.readline().strip().split(",") if x]
            means = [float(x) for x in f.readline().strip().split(",") if x]
            ranges = [float(x) for x in f.readline().strip().split(",") if x]

            # Read weights and biases
            weights = []
            biases = []
            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]

                weight_matrix = np.zeros((currentLayerSize, previousLayerSize))
                for i in range(currentLayerSize):
                    line = f.readline()
                    weight_matrix[i] = [float(x) for x in line.strip().split(",")[:-1]]
                weights.append(weight_matrix)

                bias_vector = np.zeros(currentLayerSize)
                for i in range(currentLayerSize):
                    line = f.readline()
                    bias_vector[i] = float(line.strip().split(",")[0])
                biases.append(bias_vector)

            if withNorm:
                return weights, biases, inputMins, inputMaxes, means, ranges
            return weights, biases
    except Exception as e:
        print(f"Error reading NNet file: {e}")
        raise



class ACAS(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        layers = []
        num_layers = len(weights)
        
        for i in range(num_layers):
            # 添加线性层（注意PyTorch的权重需要转置）
            in_features = weights[i].shape[1]
            out_features = weights[i].shape[0]
            linear = nn.Linear(in_features, out_features)
            
            # 设置权重和偏置（注意转置权重矩阵）
            with torch.no_grad():
                linear.weight.data = torch.from_numpy(weights[i].astype(np.float32))
                linear.bias.data = torch.from_numpy(biases[i].astype(np.float32))
            
            layers.append(linear)
            
            # 最后一层不加ReLU
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)  

class ACAS_tanh(nn.Module):
    """ACAS网络使用tanh激活函数的版本"""
    def __init__(self, weights, biases):
        super().__init__()
        layers = []
        num_layers = len(weights)
        
        for i in range(num_layers):
            # 添加线性层（注意PyTorch的权重需要转置）
            in_features = weights[i].shape[1]
            out_features = weights[i].shape[0]
            linear = nn.Linear(in_features, out_features)

            
            layers.append(linear)
            
            # 最后一层不加激活函数，其他层使用tanh
            if i < num_layers - 1:
                layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)



def load_ACASXU(x, y, spec_id):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @C: unsafe matrix, i.e., unsafe region of the outputs
           @goal: unsafe vector: 
           ***unsafe region: (C* y <= goal)
    """
    # 网络路径
    nnet_path = f"./checkpoints/acas/ACASXU_run2a_{x}_{y}_batch_2000.nnet"
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnet_path)
    net = ACAS(weights, biases)
    
    net.cuda()
    net.eval()

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    n_classes = 5
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])
        
        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
           
            C = torch.zeros(size=(1, 1, n_classes), device='cuda')
            groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
            C.scatter_(dim=2, index=groundtruth, value=-1.0)
            goal = torch.tensor([-3.9911]).to('cuda')
            
            #unsafe_mat = np.array([-1, 0, 0, 0, 0])
            #unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            # unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            # unsafe_vec = np.array([0., 0., 0., 0.])
            C = torch.zeros(size=(1, 4, n_classes), device='cuda')
            groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
            
            C[:, :, groundtruth] = -1.0
            for i in range(4): 
                C[:, i, i+1] = 1.0
            goal = torch.tensor([0., 0., 0., 0.]).to('cuda')

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        # unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        # unsafe_vec = np.array([0., 0., 0., 0.])
        C = torch.zeros(size=(1, 4, n_classes), device='cuda')
        groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
        
        C[:, :, groundtruth] = 1.0
        for i in range(4): 
            C[:, i, i+1] = -1.0
        goal = torch.tensor([0., 0., 0., 0.]).to('cuda')
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        # unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        # unsafe_vec = np.array([0., 0., 0., 0.])
        C = torch.zeros(size=(1, 4, n_classes), device='cuda')
        groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
        
        C[:, :, groundtruth] = 1.0
        for i in range(4): 
            C[:, i, i+1] = -1.0
        goal = torch.tensor([0., 0., 0., 0.]).to('cuda')
    
    elif spec_id == 0:
        lb = inputMins
        ub = inputMaxes
        C = None
        goal = None
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means[i])/ranges[i]
        ub[i] = (ub[i] - means[i])/ranges[i]
    return net, np.array(lb), np.array(ub), C, goal 


def load_ACASXU_tanh(x, y, spec_id):
    """Load trained ACASXU tanh networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: trained tanh network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @C: unsafe matrix, i.e., unsafe region of the outputs
           @goal: unsafe vector: 
           ***unsafe region: (C* y <= goal)
    """
    # 加载训练好的tanh网络
    model_path = f"./checkpoints/acas_tanh/acas_tanh_{x}_{y}.pth"
    checkpoint = torch.load(model_path, map_location='cuda')
    nnet_path = f"./checkpoints/acas/ACASXU_run2a_{x}_{y}_batch_2000.nnet"
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnet_path)
    net = checkpoint['model']
    net.cuda()
    net.eval()

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    n_classes = 5
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])
        
        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
           
            C = torch.zeros(size=(1, 1, n_classes), device='cuda')
            groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
            C.scatter_(dim=2, index=groundtruth, value=-1.0)
            goal = torch.tensor([-3.9911]).to('cuda')
            
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            C = torch.zeros(size=(1, 4, n_classes), device='cuda')
            groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
            
            C[:, :, groundtruth] = -1.0
            for i in range(4): 
                C[:, i, i+1] = 1.0
            goal = torch.tensor([0., 0., 0., 0.]).to('cuda')

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        # unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        # unsafe_vec = np.array([0., 0., 0., 0.])
        C = torch.zeros(size=(1, 4, n_classes), device='cuda')
        groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
        
        C[:, :, groundtruth] = 1.0
        for i in range(4): 
            C[:, i, i+1] = -1.0
        goal = torch.tensor([0., 0., 0., 0.]).to('cuda')
       
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        # unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        # unsafe_vec = np.array([0., 0., 0., 0.])
        C = torch.zeros(size=(1, 4, n_classes), device='cuda')
        groundtruth = torch.tensor([0]).to('cuda').unsqueeze(1).unsqueeze(1)
        
        C[:, :, groundtruth] = 1.0
        for i in range(4): 
            C[:, i, i+1] = -1.0
        goal = torch.tensor([0., 0., 0., 0.]).to('cuda')
    
    else:
        raise Exception('Invalid Specification ID')
    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means[i])/ranges[i]
        ub[i] = (ub[i] - means[i])/ranges[i]
    return net, np.array(lb), np.array(ub), C, goal


def load_rocketnet(net_id, spec_id):
    """Load trained rocketnet networks
       Args:
           @ network id (net_id)
           @ specification_id: 1, 2
    """
    model_path = f"./checkpoints/RocketNetReLU/unsafe_agent{net_id}.pt"
    checkpoint = torch.load(model_path, map_location='cuda').float()
    
    lb_p0 = np.array([-0.2, 0.02, -0.5, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180])
    ub_p0 = np.array([0.2, 0.5, 0.5, 1.0, -6 * math.pi / 180, -0.0, 0.0, 0.0, 1.0, 0.0, 0 * math.pi / 180])
    lb_p1 = np.array([-0.2, 0.02, -0.5, -1.0, 6 * math.pi / 180, 0.0, 0.0, 0.0, 0.0, 0.0, 0 * math.pi / 180])
    ub_p1 = np.array([0.2, 0.5, 0.5, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180])

    # A_unsafe0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # d_unsafe0 = np.array([0.0, 0.0])
    # A_unsafe1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # d_unsafe1 = np.array([0.0, 0.0])
    n_classes = 3

    if spec_id == 1:
        lb = lb_p0
        ub = ub_p0
        C = torch.zeros(size=(1, 2, n_classes), device='cuda')
        for i in range(2): 
            C[:, i, i+1] = -1.0
        goal = torch.tensor([0.0, 0.0]).to('cuda')
    elif spec_id == 2:
        lb = lb_p1
        ub = ub_p1
        C = torch.zeros(size=(1, 2, n_classes), device='cuda')
        for i in range(2): 
            C[:, i, i+1] = 1.0
        goal = torch.tensor([0.0, 0.0]).to('cuda')
    return checkpoint, lb, ub, C, goal