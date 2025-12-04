import torch
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime
from typing import List, Tuple, Optional
from scipy import stats
from utils.utils import compute_crown_bound, compute_box_prob
from utils.load import *

class NaiveBaB:
    """
    Naive Branch and Bound implementation for probabilistic verification
    
    This serves as a baseline comparison method that uses uniform binary subdivision 
    instead of intelligent region partitioning.
    """
    
    def __init__(self, model, goal, input_range, C, mean, scale, method='crown'):
        """
        Initialize Naive BaB verifier
        
        Args:
            model: Neural network model
            goal: Safety threshold 
            input_range: List of (min, max) bounds for each input dimension
            C: Specification matrix
            mean: Mean of input distribution for probability computation
            scale: Scale of input distribution for probability computation  
            method: Verification method ('crown' or 'alpha-crown')
        """
        self.model = model
        self.goal = goal
        self.input_range = input_range
        self.C = C
        self.mean = mean
        self.scale = scale
        self.method = method
        self.regions_safe = []
        self.regions_unsafe = []
        self.regions_unknown = []
        
    def verify(self, epsilon=1e-5, selection_strategy='largest'):
        """
        Run naive BaB verification
        
        Args:
            epsilon: Termination threshold for unknown probability mass
            max_iterations: Maximum number of iterations
            selection_strategy: Strategy for selecting regions to split 
                              ('largest', 'highest_prob', 'random')
        
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Initialize with entire input space
        initial_region = [(r[0], r[1]) for r in self.input_range]
        self.regions_unknown = [initial_region]
        
        iteration = 0
        
        print(f"Starting Naive BaB verification (strategy: {selection_strategy})")
        
        while time.time()-start_time < 7200:
            # Check termination condition
            total_unknown_prob = sum(self._compute_region_prob(r) for r in self.regions_unknown)
            
            
                
            if not self.regions_unknown:
                print("No unknown regions remaining")
                break
                
            print(f"Iteration {iteration} Time {time.time()-start_time:.2f}s: {len(self.regions_unknown)} unknown regions, "
                  f"unknown prob: {total_unknown_prob:.6f}")
            
            # Select region to split
            region_to_split = self._select_region_to_split(selection_strategy)
            if region_to_split is None:
                print("No suitable region found for splitting")
                break
                
            if self._compute_region_prob(region_to_split) <= epsilon:
                print(f"All region smaller than epsilon{epsilon}")
                break
            # Remove selected region from unknown list
            self.regions_unknown.remove(region_to_split)
            
            # Split the region uniformly along longest dimension
            child_regions = self._split_region_uniform(region_to_split)
            
            safety_status = self._verify_region(child_regions)
            
            # Verify each child region
            for child_region, safety_status in zip(child_regions, safety_status):
                
                
                if safety_status == 1:  # Safe
                    self.regions_safe.append(child_region)
                elif safety_status == 0:  # Unsafe
                    self.regions_unsafe.append(child_region)
                else:  # Unknown
                    self.regions_unknown.append(child_region)
            
            iteration += 1
        
        total_time = time.time() - start_time
        
        # Compute final statistics
        safe_prob = sum(self._compute_region_prob(r) for r in self.regions_safe)
        unsafe_prob = sum(self._compute_region_prob(r) for r in self.regions_unsafe) 
        unknown_prob = sum(self._compute_region_prob(r) for r in self.regions_unknown)
        if not total_time>=7200.0:
            results = {
                'Ls': safe_prob,
                'Us': 1.0-unsafe_prob,
                'Us-Ls': 1.0-safe_prob-unsafe_prob,
                'time': total_time,
            }
        
            print(f"Final results: Ls={safe_prob:.4f}, Us={1.0-unsafe_prob:.4f}, "
                f"Us-Ls={1.0-safe_prob-unsafe_prob:.4f}, Time={total_time:.2f}s")
        else:
            results = {
                'Ls': -1,
                'Us': -1,
                'Us-Ls': -1,
                'time': 'T.O.',
            }
            print("Time out!")
        return results
    
    def _select_region_to_split(self, strategy: str):
        """
        Select region to split based on specified strategy
        """
        if not self.regions_unknown:
            return None
            
        if strategy == 'largest':
            # Select region with largest volume
            return max(self.regions_unknown, key=self._compute_region_volume)
        elif strategy == 'highest_prob':
            # Select region with highest probability mass
            return max(self.regions_unknown, key=self._compute_region_prob)
        elif strategy == 'random':
            # Random selection
            return np.random.choice(self.regions_unknown)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _split_region_uniform(self, region: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Split region uniformly along the longest dimension
        """
        # Find dimension with largest range
        dim_lengths = [(ub - lb) for lb, ub in region]
        split_dim = np.argmax(dim_lengths)
        
        # Split at midpoint
        lb, ub = region[split_dim]
        midpoint = (lb + ub) / 2
        
        # Create two child regions
        left_region = [list(bound) for bound in region]
        right_region = [list(bound) for bound in region]
        
        left_region[split_dim] = (lb, midpoint)
        right_region[split_dim] = (midpoint, ub)
        
        return [left_region, right_region]
    
    def _verify_region(self, regions) -> List[int]:
        """
        Verify multiple regions using CROWN bounds
        
        Args:
            regions: List of regions to verify
            
        Returns:
            List of safety status for each region:
            1: Safe
            0: Unsafe  
            -1: Unknown
        """
        # Convert regions to tensors
        lower = torch.tensor([bound[0] for bound in regions[0]], device='cuda', dtype=torch.float32).unsqueeze(0)
        left_upper = torch.tensor([bound[1] for bound in regions[0]], device='cuda', dtype=torch.float32).unsqueeze(0)
        right_lower = torch.tensor([bound[0] for bound in regions[1]], device='cuda', dtype=torch.float32).unsqueeze(0)
        upper = torch.tensor([bound[1] for bound in regions[1]], device='cuda', dtype=torch.float32).unsqueeze(0)
        
        # Compute CROWN bounds
        try:
            l = torch.vstack((lower.clone(), right_lower.clone()))
            u = torch.vstack((left_upper.clone(), upper.clone()))
            current_C = torch.vstack((self.C.clone(), self.C.clone()))
            lb, ub = compute_crown_bound(self.model, l, u, C=current_C, method='crown')
            
            # Check safety conditions for each region
            safety_status = []
            for i in range(len(regions)):
                if (lb[i] >= self.goal).any():
                    safety_status.append(1)  # Safe
                elif (ub[i] <= self.goal).all():
                    safety_status.append(0)  # Unsafe
                else:
                    safety_status.append(-1)  # Unknown
                    
            return safety_status
                
        except Exception as e:
            print(f"CROWN verification failed: {e}")
            return [-1, -1]  # Treat both as unknown if verification fails
    
    def _compute_region_volume(self, region: List[Tuple[float, float]]) -> float:
        """Compute volume of a region"""
        volume = 1.0
        for lb, ub in region:
            volume *= (ub - lb) if ub > lb else 1e-10
        return volume
    
    def _compute_region_prob(self, region: List[Tuple[float, float]]) -> float:
        """Compute probability mass of a region under input distribution"""
        return compute_box_prob(region, self.mean, self.scale)
    
    def get_all_regions(self):
        """
        Get all regions with their classifications
        
        Returns:
            List of tuples (region, safety_status, probability)
        """
        all_regions = []
        
        for region in self.regions_safe:
            prob = self._compute_region_prob(region)
            all_regions.append((region, 1, prob))
            
        for region in self.regions_unsafe:
            prob = self._compute_region_prob(region)
            all_regions.append((region, 0, prob))
            
        for region in self.regions_unknown:
            prob = self._compute_region_prob(region)
            all_regions.append((region, -1, prob))
            
        return all_regions


def run_naive_bab_experiment(model, goal, input_range, C, mean, scale, 
                           epsilon=1e-5, 
                           selection_strategy='largest', method='crown'):
    """
    Convenience function to run naive BaB experiment
    
    Args:
        model: Neural network model
        goal: Safety threshold
        input_range: Input space bounds
        C: Specification matrix
        mean: Distribution mean
        scale: Distribution scale
        epsilon: Termination threshold
        selection_strategy: Region selection strategy
        method: Verification method
    
    Returns:
        Dictionary with results
    """
    verifier = NaiveBaB(model, goal, input_range, C, mean, scale, method)
    results = verifier.verify(epsilon, selection_strategy)
    return results, verifier


if __name__ == "__main__":
    # Example usage with dummy 2D network
    print("Testing Naive BaB implementation...")
    
    # 创建结果保存目录
    results_dir = "./results/naive_bab"
    os.makedirs(results_dir, exist_ok=True)
    
    # 收集所有实验结果用于汇总
    all_results = []
    
    # This is just a test - replace with actual model/parameters
    import torch.nn as nn
    spec_id = 2
    test_combinations = [(1,6),(2,2),(2,9),(3,1),(3,6),(3,7),(4,1),(4,7),(5,3)]
    
    print(f"开始运行 {len(test_combinations)} 个Naive BaB实验...")
    
    for i, (x, y) in enumerate(test_combinations):
        print(f"\n=== 实验 {i+1}/{len(test_combinations)}: ACAS网络 {x}_{y}, spec_id={spec_id} ===")
        
        try:
            model, lb_norm, ub_norm, C, goal = load_ACASXU(x, y, spec_id)
            input_range = [(float(lb_norm[i]), float(ub_norm[i])) for i in range(5)]
            mean = (lb_norm + ub_norm) / 2
            scale = (ub_norm - mean) / 3.0
            
            results, verifier = run_naive_bab_experiment(
                model, goal, input_range, C, mean, scale,
                epsilon=1e-5, selection_strategy='highest_prob'
            )
            
            # 添加实验标识信息
            results['network_x'] = x
            results['network_y'] = y  
            results['spec_id'] = spec_id
            results['experiment_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存单个实验结果
            individual_result_file = os.path.join(
                results_dir, 
                f"naive_bab_{x}_{y}_spec{spec_id}.csv"
            )
            individual_df = pd.DataFrame([results])
            individual_df.to_csv(individual_result_file, index=False)
            print(f"单个实验结果已保存到: {individual_result_file}")
            
            # 添加到汇总结果中
            all_results.append(results)
            
            print("Naive BaB实验完成!")
            
        except Exception as e:
            print(f"实验 {x}_{y} 运行失败: {e}")
            # 记录失败的实验
            failed_result = {
                'network_x': x,
                'network_y': y,
                'spec_id': spec_id,
                'Ls': -1,
                'Us': -1, 
                'Us-Ls': -1,
                'time': -1
            }
            all_results.append(failed_result)
    
    # 保存汇总结果
    if all_results:
        summary_file = os.path.join(results_dir, "naive_bab_all_summary.csv")
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存到: {summary_file}")
        

    
    print(f"\nNaive BaB批量实验完成! 结果保存在: {results_dir}") 