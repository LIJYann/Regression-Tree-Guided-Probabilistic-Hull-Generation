import os
import torch
import numpy as np
import time
import pandas as pd
from models.tiny_network import TinyNetwork
from samplers.distribution_boundary_sampler import DistributionBoundarySampler
from samplers.uniform_boundary_sampler import UniformBoundarySampler
from regression_tree.tree_builder import CustomRegionSplitter
from utils.utils import CustomDecisionTreeRegressor
from visualization.vis_utils import visualize_regions, visualize_regions_no_points
from matplotlib import font_manager
import matplotlib.pyplot as plt
np.random.seed(0)

font = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_training_data(model, sample_points, goal=0.0):
    X = sample_points
    with torch.no_grad():
        y = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    y = y[:, 0] - goal
    return X, y

if __name__ == "__main__":
    input_range = [(-2, 2), (-1, 1)]
    mean = [0.0, 0.0]
    scale = [0.8, 0.4]
    net = TinyNetwork().to(device)
    n_classes = 2
    C = torch.zeros(size=(1, 1, n_classes), device=device)
    groundtruth = torch.tensor([0]).to(device).unsqueeze(1).unsqueeze(1)
    C.scatter_(dim=2, index=groundtruth, value=1.0)

    total_samples = 500
    criterion = "mse"
    max_depth = 10
    results = []
    strategies = [
        'mixed'
    ]

    splitters = {}
    for strategy in strategies:
        print(f"\n==== {strategy} sampling ====")
        t0 = time.time()
        if strategy == 'boundary_aware_distribution':
            sampler = DistributionBoundarySampler(
                model=net,
                input_range=input_range,
                C=C,
                mean=mean,
                scale=scale,
                goal=-2.0,
                n_samples=total_samples,
                rejection_threshold=0.1,
                max_rejection_ratio=0.8
            )
            boundary_points = sampler.sample()
        elif strategy == 'boundary_aware_uniform':
            sampler = UniformBoundarySampler(
                model=net,
                input_range=input_range,
                C=C,
                goal=-2.0,
                n_samples=total_samples,
                rejection_threshold=0.1,
                max_rejection_ratio=0.8
            )
            boundary_points = sampler.sample()
        else:
            sampler = DistributionBoundarySampler(
                model=net,
                input_range=input_range,
                C=C,
                mean=mean,
                scale=scale,
                goal=-2.0,
                n_samples=total_samples//2,
                rejection_threshold=0.1,
                max_rejection_ratio=0.8
            )
            boundary_points = sampler.sample()
            sampler = UniformBoundarySampler(
                model=net,
                input_range=input_range,
                C=C,
                goal=-2.0,
                n_samples=total_samples//2,
                rejection_threshold=0.1,
                max_rejection_ratio=0.8
            )
            boundary_points = np.concatenate((boundary_points, sampler.sample()), axis=0)

        t1 = time.time()
        X, y = generate_training_data(net, boundary_points, goal=-2.0)
        t2 = time.time()
        splitter = CustomRegionSplitter(net, -2.0, boundary_points, input_range, C, mean, scale, X, y, criterion, coef=0.3)
        splitter.train_decision_tree(max_depth=max_depth)
        splitters[strategy] = (splitter, y)
        t3 = time.time()

        # Region statistics
        regions = splitter.get_tree_regions()
        safe_prob = 0.0
        unsafe_prob = 0.0
        unknown_prob = 0.0
        safe_count = 0
        unsafe_count = 0
        unknown_count = 0
        for region in regions:
            _, _, depth, safety, prob,_ = region
            if float(safety) == 1:
                safe_prob += float(prob)
                safe_count += 1
            elif float(safety) == 0:
                unsafe_prob += float(prob)
                unsafe_count += 1
            else:
                unknown_prob += float(prob)
                unknown_count += 1
        num_regions = len(regions)

        # Time statistics
        sampling_time = t1 - t0
        data_time = t2 - t1
        train_time = t3 - t2
        total_time = t3 - t0

        results.append({
            'strategy': strategy,
            'safe_prob': safe_prob,
            'unsafe_prob': unsafe_prob,
            'unknown_prob': unknown_prob,
            'safe_count': safe_count,
            'unsafe_count': unsafe_count,
            'unknown_count': unknown_count,
            'num_regions': num_regions,
            'sampling_time': sampling_time,
            'data_time': data_time,
            'train_time': train_time,
            'total_time': total_time
        })
    # Create single plot without points
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Use the mixed strategy splitter for visualization
    mixed_splitter, mixed_y = splitters['mixed']
    visualize_regions_no_points(mixed_splitter, ax, font, show_boundary_line=False)
    
    plt.tight_layout()
    folder_name = os.path.join("./results", "comparison")
    os.makedirs(folder_name, exist_ok=True)
    save_path = os.path.join(folder_name, f'illustration.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    