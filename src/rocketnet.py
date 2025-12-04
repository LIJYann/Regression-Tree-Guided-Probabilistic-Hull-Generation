import os
import torch
import numpy as np
import time
import pandas as pd
from utils.load import load_rocketnet
from samplers.uniform_boundary_sampler import UniformBoundarySampler

from samplers.distribution_boundary_sampler import DistributionBoundarySampler

from regression_tree.tree_builder import CustomRegionSplitter
from utils.utils import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def generate_training_data(model, sample_points, C, goal):
    X = sample_points
    with torch.no_grad():
        #print(model(torch.tensor(X, dtype=torch.float32).cuda()).shape)
        #print(C)
        y = (C @ model(torch.tensor(X, dtype=torch.float32).cuda()).unsqueeze(-1)).squeeze(-1).cpu().numpy()
        
    y = y - goal.cpu().numpy()
    
    return X, y

def run_rocketnet_experiment(
    net_id: int,
    spec_id: int,
    coef:float=0.3,
    uniform_ratio: float = 0.5,
    result_dir: str = "./results/rocketnet_experiments",
    result_prefix: str = "rocketnet_parallel_results",
    parallel: bool = True
) -> pd.DataFrame:
    torch.manual_seed(1024)
    np.random.seed(1024)
    import random
    random.seed(1024)
    print(f"准备进行RocketNet实验{net_id}-spec{spec_id}")
    os.makedirs(result_dir, exist_ok=True)
    model, lb_norm, ub_norm, C, goal = load_rocketnet(net_id, spec_id)
    input_range = [(float(lb_norm[i]), float(ub_norm[i])) for i in range(11)]
    mean = (lb_norm + ub_norm) / 2
    scale = (ub_norm - mean) / 3.0
    scale = [s if s!=0.0 else 1e-5 for s in scale]
    total_samples = 9000
    n_to_sample = 900
    max_depth = 5
    unknown_prob_threshold = 1e-5
    impurity_threshold = 0.0
    criterion = "mse"  # 可选"mse"或"prob_mse"
    
    # 根据比例计算初始采样数量
    distribution_ratio = 1.0 - uniform_ratio
    n_uniform_init = int(total_samples * uniform_ratio)
    n_distribution_init = int(total_samples * distribution_ratio)
    # 确保总数正确（处理舍入误差）
    if n_uniform_init + n_distribution_init < total_samples:
        n_uniform_init += (total_samples - n_uniform_init - n_distribution_init)
    
    t0 = time.time()
    points_list = []
    
    if n_uniform_init > 0:
        sampler_uniform = UniformBoundarySampler(
            model=model,
            input_range=input_range,
            C=C,
            goal=goal,
            n_samples=n_uniform_init,
        )
        points_list.append(sampler_uniform.sample())
    
    if n_distribution_init > 0:
        sampler_distribution = DistributionBoundarySampler(
            model=model,
            input_range=input_range,
            mean=mean,
            scale=scale,
            C=C,
            goal=goal,
            n_samples=n_distribution_init,
        )
        points_list.append(sampler_distribution.sample())
    
    if len(points_list) > 0:
        points = np.concatenate(points_list, axis=0)
    else:
        raise ValueError("采样点数不能为0")
    X, y_val = generate_training_data(model, points, C=C, goal=goal)
    all_X = X.copy()
    all_y = y_val.copy()
    sample_unknown_regions = []
    determined_regions = []
    splitter = CustomRegionSplitter(model, goal, points, input_range, C, mean, scale, all_X, all_y, criterion, coef=None)
    if parallel:
        splitter.train_decision_tree_parallel(max_depth=max_depth, input_region=input_range)
    else:
        splitter.train_decision_tree(max_depth=max_depth, input_region=input_range)
    iter_idx = -1
    # 根据比例计算迭代采样数量
    n_uniform_iter = int(n_to_sample * uniform_ratio)
    n_distribution_iter = int(n_to_sample * distribution_ratio)
    # 确保总数正确（处理舍入误差）
    if n_uniform_iter + n_distribution_iter < n_to_sample:
        n_uniform_iter += (n_to_sample - n_uniform_iter - n_distribution_iter)
    
    sampler_u = None
    sampler_d = None
    
    if n_uniform_iter > 0:
        sampler_u = UniformBoundarySampler(
            model=model,
            input_range=input_range,
            C=C,
            goal=goal,
            n_samples=n_uniform_iter
        )
    
    if n_distribution_iter > 0:
        sampler_d = DistributionBoundarySampler(
            model=model,
            input_range=input_range,
            mean=mean,
            scale=scale,
            C=C,
            goal=goal,
            n_samples=n_distribution_iter,
        )
    target_region = None
    coefficient = None
    method = 'crown'
    
    while True:
        iter_idx += 1
        if parallel:
            regions = splitter.get_tree_regions_parallel()
        else:
            regions = splitter.get_tree_regions()
        new_sample_unknown = [region for region in regions if region[3] == -1]
        new_determined_region = [region for region in regions if region[3] !=-1]
        
        
        sample_unknown_regions.extend(new_sample_unknown)
        determined_regions.extend(new_determined_region)
        total_unknown_prob = sum([r[4] for r in sample_unknown_regions])
        if iter_idx % 100 == 0:
            print(f"RocketNet实验{net_id}-spec{spec_id}迭代{iter_idx}，未知区域总概率: {total_unknown_prob:.4f}，花费时间：{time.time()-t0}")
        
        all_regions = sample_unknown_regions + determined_regions
        
        safe_prob = sum(float(r[4]) for r in all_regions if r[3] == 1)
        unsafe_prob = sum(float(r[4]) for r in all_regions if r[3] == 0)
        unknown_prob = sum(float(r[4]) for r in all_regions if r[3] not in [0, 1])
        if iter_idx % 100 == 0:
            print(f"Ls: {safe_prob:.8f} Us: {1.0-unsafe_prob:.8f}，Us-Ls: {1.0-safe_prob-unsafe_prob:.8f}")
        if sample_unknown_regions:
            target_region = max(sample_unknown_regions, key=lambda r: r[4])
            region_box = target_region[0]
            if target_region[4] <= unknown_prob_threshold:
                print("未知区域概率已达阈值，终止递进采样。")
                break
            #print(f"选定局部采样分割的区域概率为{target_region[4]}，体积为{compute_box_volume(target_region[0])/compute_box_volume(input_range)}")
            if n_to_sample > 0:
                new_points_list = []
                if sampler_u is not None:
                    new_points_list.append(sampler_u.sample(region_box=region_box))
                if sampler_d is not None:
                    new_points_list.append(sampler_d.sample(region_box=region_box))
                
                if len(new_points_list) > 0:
                    new_points = np.concatenate(new_points_list, axis=0)
                else:
                    raise ValueError("迭代采样点数不能为0")
                
                new_X, new_y = generate_training_data(model, new_points, C, goal)
                if safe_prob+unsafe_prob>=impurity_threshold:
                    coefficient=coef
                splitter = CustomRegionSplitter(model, goal, new_points, region_box, C, mean, scale, new_X, new_y, criterion, coef=coefficient, method=method)
                if parallel:
                    splitter.train_decision_tree_parallel(max_depth=max_depth, input_region=region_box)
                else:
                    splitter.train_decision_tree(max_depth=max_depth, input_region=region_box)
            sample_unknown_regions.remove(target_region)
        else:
            print("无样本数不足的未知区域，终止递进采样。")
            break
    t3 = time.time()
    #regions = splitter.get_tree_regions()
    safe_prob = 0.0
    unsafe_prob = 0.0
    unknown_prob = 0.0
    safe_count = 0
    unsafe_count = 0
    unknown_count = 0
    reason_counter = {}
    all_regions = sample_unknown_regions + determined_regions
    for region in all_regions:
        _, _, depth, safety, prob, reason = region[:6]
        if reason not in reason_counter:
            reason_counter[reason] = 0
        reason_counter[reason] += 1
        if safety == 1:
            safe_prob += float(prob)
            safe_count += 1
        elif safety == 0:
            unsafe_prob += float(prob)
            unsafe_count += 1
        else:
            unknown_prob += float(prob)
            unknown_count += 1
    num_regions = len(all_regions)
    total_time = t3 - t0

    results = [{
        'Ls': safe_prob,
        'Us': 1.0-unsafe_prob,
        'Us-Ls': 1.0-safe_prob-unsafe_prob,
        'time': total_time,
    }]
    df = pd.DataFrame(results)
    # 自动生成不覆盖的结果文件名
    
    result_file = os.path.join(
        result_dir,
        f"{result_prefix}_{net_id}_spec{spec_id}.csv"
    )
    df.to_csv(result_file, index=False)
    print(f"结果已保存到: {result_file}")
    print(df)
    return df

def run_rocketnet_experiment_parallel(net_id, spec_id, coef=0.3, uniform_ratio=0.5, result_dir="./results/rocketnet_experiments", result_prefix="rocketnet_parallel_results"):
    return run_rocketnet_experiment(net_id, spec_id, coef, uniform_ratio, result_dir, result_prefix, parallel=True)

def run_rocketnet_experiment_serial(net_id, spec_id, coef=0.3, uniform_ratio=0.5, result_dir="./results/rocketnet_experiments", result_prefix="rocketnet_serial_results"):
    return run_rocketnet_experiment(net_id, spec_id, coef, uniform_ratio, result_dir, result_prefix, parallel=False)

if __name__ == "__main__":
    all_results = []    
    
    for net_id in [0,1]:
        for spec_id in [1,2]:
            df = run_rocketnet_experiment_parallel(net_id, spec_id, coef=0.05, uniform_ratio=0.25)
            all_results.append(df)
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df.to_csv("./results/rocketnet_experiments/rocketnet_parallel_all_summary.csv", index=False)
    print("所有实验已完成，汇总结果已保存。") 
    
    all_results = []    
    for net_id in [0,1]:
        for spec_id in [1,2]:
            df = run_rocketnet_experiment_serial(net_id, spec_id, coef=0.05, uniform_ratio=0.25)
            all_results.append(df)
        
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df.to_csv("./results/rocketnet_experiments/rocketnet_serial_all_summary.csv", index=False)
    print("所有实验已完成，汇总结果已保存。") 