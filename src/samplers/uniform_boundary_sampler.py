import numpy as np
import torch
from typing import List, Tuple

class UniformBoundarySampler:
    """
    基于均匀分布的边界感知采样器
    
    使用初步采样估计模型预测值的实际上下界来调整拒绝概率的阈值。
    基于均匀分布进行基础采样，然后通过指数衰减的拒绝机制偏向边界区域。
    
    通过初步采样获得更紧的边界估计，避免CROWN过近似导致的边界过于宽松。
    """
    
    def __init__(self, 
                 model, 
                 input_range: List[Tuple[float, float]], 
                 C,
                 goal,
                 n_samples: int = 1000,
                 rejection_threshold: float = 0.1,
                 max_rejection_ratio: float = 0.8,
                 enable_rejection_sampling: bool = True):
        """
        初始化基于均匀分布的边界感知采样器
        
        Args:
            model: 神经网络模型
            input_range: 每个维度的取值范围 [(min1, max1), (min2, max2), ...]
            C: 线性变换矩阵
            goal: 目标边界值，默认为0.0
            n_samples: 目标采样点数
            rejection_threshold: 拒绝阈值，控制拒绝概率的敏感度
            max_rejection_ratio: 最大拒绝比例，防止无限循环
            enable_rejection_sampling: 是否启用拒绝采样（边界细分机制）
        """
        self.model = model
        self.input_range = input_range
        self.goal = goal
        self.n_samples = n_samples
        self.rejection_threshold = rejection_threshold
        self.max_rejection_ratio = max_rejection_ratio
        self.enable_rejection_sampling = enable_rejection_sampling
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dim = len(input_range)
        self.C = C
        
        # 初始化边界信息
        self._initialize_bounds()
    
    def _initialize_bounds(self):
        """
        初始化边界信息，通过初步采样估计模型预测值的实际上下界
        """
        print("初始化边界信息...")
        
        # 通过初步采样估计模型预测值的实际上下界
        n_preliminary = self.n_samples*10  # 初步采样点数
        print(f"进行初步采样 ({n_preliminary} 个点) 来估计模型预测边界...")
        
        # 从输入范围均匀采样
        preliminary_samples = []
        for i, (low, high) in enumerate(self.input_range):
            sample = np.random.uniform(low, high, n_preliminary)
            preliminary_samples.append(sample)
        preliminary_samples = np.stack(preliminary_samples, axis=1)
        
        # 计算模型预测值
        with torch.no_grad():
            sample_tensor = torch.tensor(preliminary_samples, dtype=torch.float32, device=self.device)
            outputs = (self.C @ self.model(sample_tensor).unsqueeze(-1)).squeeze(-1)
            # 检查每个样本是否满足条件
            sample_satisfies = (outputs>=self.goal).any(dim=1)
            
            # 判断是否全为True或全为False
            all_true = sample_satisfies.all()
            all_false = (~sample_satisfies).all()
            
            if all_true or all_false:
                print("不需要进行拒绝采样")
                self.reject=False
            else:
                print("需要进行拒绝采样")
                self.reject=True
            
            # 如果禁用了拒绝采样，强制设置为False
            if not self.enable_rejection_sampling:
                self.reject = False
                print("已禁用拒绝采样（边界细分机制）")
            predictions = (outputs - self.goal).cpu().numpy()
        
        # 兼容一维输出，强制二维化
        if predictions.ndim == 1:
            predictions = predictions[:, None]  # shape: [n_samples, 1]
        
        # 计算预测值的统计信息
        pred_min = np.min(predictions, axis=0)
        pred_max = np.max(predictions, axis=0)
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        #print(f"预测值统计:")
        #print(f"  最小值: {np.array2string(pred_min, precision=4)}")
        #print(f"  最大值: {np.array2string(pred_max, precision=4)}")
        #print(f"  均值: {np.array2string(pred_mean, precision=4)}")
        #print(f"  标准差: {np.array2string(pred_std, precision=4)}")
        
        # 使用预测值的实际上下界作为归一化因子
        # 相对于目标值的偏移
        self.global_lb = pred_min  # shape: [n_outputs]
        self.global_ub = pred_max
        self.normalization_factor = pred_std  # shape: [n_outputs]
        # print(f"相对目标值的边界: [{np.array2string(self.global_lb, precision=4)}, {np.array2string(self.global_ub, precision=4)}]")
        # print(f"归一化因子: {np.array2string(self.normalization_factor, precision=4)}")
        
        # print("边界初始化完成！")
    
    def compute_boundary_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算点到决策边界的距离
        
        Args:
            x: 输入点，形状为 (batch_size, dim)
            
        Returns:
            distance: 到边界的距离，形状为 (batch_size,)
        """
        x.requires_grad_(True)
        output = self.C @ self.model(x).unsqueeze(-1)
        output = output.squeeze(-1)
        f1 = output - self.goal  # 到边界的函数值
        return f1.detach()
    
    def compute_rejection_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于初步采样估计的边界信息计算拒绝概率
        
        Args:
            x: 输入点，形状为 (batch_size, dim)
            
        Returns:
            rejection_prob: 拒绝概率，形状为 (batch_size,)
        """
        distances = self.compute_boundary_distance(x)
        # 兼容一维输出，强制二维化
        if distances.ndim == 1:
            distances = distances.unsqueeze(1)  # shape: [batch_size, 1]
        norm = torch.tensor(self.normalization_factor, device=x.device)
        if norm.ndim == 0:
            norm = norm.unsqueeze(0)  # shape: [1]
        distance_norms = torch.abs(torch.max(distances/norm, dim=-1).values)
        # rejection_prob = 1.0 - torch.exp(-dist)
        # rejection_prob = torch.clamp(rejection_prob, 0.0, 0.95)
        
        # # 基于距离排序进行拒绝采样
        # # 计算距离无穷范数（L∞范数）
        # distance_norms = torch.norm(torch.abs(distances), p=float('inf'), dim=1)
        
        # 基于距离排序计算拒绝概率
        # 距离越小，排名越靠前，拒绝概率越低
        sorted_indices = torch.argsort(distance_norms)
        ranks = torch.zeros_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(len(sorted_indices), device=x.device)
        
        # 将排名归一化到[0,1]
        normalized_ranks = ranks.float() / (len(ranks) - 1 + 1e-8)
        
        # 基于排名计算拒绝概率
        # 排名越靠前（距离越小），拒绝概率越低
        rejection_prob = 1.0 - torch.exp(-5.0*normalized_ranks)
        rejection_prob = torch.clamp(rejection_prob, 0.0, 0.95)
        
        return rejection_prob  # shape: [batch_size]
    
    def sample_base_distribution(self, n_samples: int, region_box=None) -> np.ndarray:
        """
        从均匀分布采样
        
        Args:
            n_samples: 采样点数
            region_box: 可选的区域限制，如果提供则在该区域内采样
            
        Returns:
            samples: 采样点，形状为 (n_samples, dim)
        """
        arr = np.zeros((n_samples, self.dim))
        
        if region_box is not None:
            # 在指定区域内均匀采样
            for i, (low, high) in enumerate(region_box):
                arr[:, i] = np.random.uniform(low, high, n_samples)
        else:
            # 在原始输入范围内均匀采样
            for i, (low, high) in enumerate(self.input_range):
                arr[:, i] = np.random.uniform(low, high, n_samples)
        
        return arr
    
    def sample(self, region_box=None) -> np.ndarray:
        """
        执行边界感知采样
        
        Args:
            region_box: 可选的区域限制，如果提供则在该区域内采样
            
        Returns:
            accepted_samples: 被接受的采样点，形状为 (n_accepted, dim)
        """
        accepted_samples = []
        total_generated = 0
        max_total = int(self.n_samples / (1 - self.max_rejection_ratio))  # 防止无限循环
        
        #print(f"开始均匀分布边界感知采样，目标点数: {self.n_samples}")
        
        with torch.no_grad():
            batch_samples = self.sample_base_distribution(self.n_samples, region_box=region_box)
            sample_tensor = torch.tensor(batch_samples, dtype=torch.float32, device=self.device)
            outputs = (self.C @ self.model(sample_tensor).unsqueeze(-1)).squeeze(-1)
            # 检查每个样本是否满足条件
            sample_satisfies = (outputs>=self.goal).any(dim=1)
            
            # 判断是否全为True或全为False
            all_true = sample_satisfies.all()
            all_false = (~sample_satisfies).all()
            
            if all_true or all_false:
                #print("不需要进行拒绝采样")
                self.reject=False
                total_generated += self.n_samples
                accepted_samples.extend(batch_samples)
            else:
                #print("需要进行拒绝采样")
                total_generated += self.n_samples//2
                accepted_samples.extend(self.sample_base_distribution(self.n_samples//2, region_box=region_box))
                self.reject=True
                self.n_samples = self.n_samples*2
        
        while len(accepted_samples) < self.n_samples:
            # 计算需要生成的样本数
            remaining = self.n_samples - len(accepted_samples)
            batch_size = max(remaining * 3, 100)  # 增加批量大小，提高效率
            
            # 从均匀分布采样
            batch_samples = self.sample_base_distribution(batch_size, region_box=region_box)
            total_generated += batch_size
            
            # 转换为torch张量
            batch_tensor = torch.tensor(batch_samples, dtype=torch.float32, device=self.device)
            
            # 计算拒绝概率（使用初步采样估计的边界信息）
            # 计算拒绝概率（使用初步采样估计的边界信息）
            if self.reject:
                rejection_probs = self.compute_rejection_probability(batch_tensor)
            else:
                rejection_probs = torch.zeros_like(batch_tensor[:, 0])
            
            # 执行拒绝采样
            torch.manual_seed(0)
            random_values = torch.rand_like(rejection_probs)
            accepted_mask = random_values > rejection_probs
            
            # 收集被接受的样本
            accepted_batch = batch_samples[accepted_mask.cpu().numpy()]
            accepted_samples.extend(accepted_batch)
            
            # 打印进度
            if total_generated % 2000 == 0:
                acceptance_rate = len(accepted_samples) / total_generated
                #print(f"已生成 {total_generated} 个候选点，接受 {len(accepted_samples)} 个点，接受率: {acceptance_rate:.3f}")
        
        # 如果采样点不足，通过均匀分布采样补充
        if len(accepted_samples) < self.n_samples:
            print(f"警告：只采样到 {len(accepted_samples)} 个点，少于目标 {self.n_samples} 个")
            # 通过均匀分布采样补充
            remaining_count = self.n_samples - len(accepted_samples)
            additional_samples = self.sample_base_distribution(remaining_count, region_box=region_box)
            accepted_samples.extend(additional_samples)
        
        # 确保返回正确数量的样本
        accepted_samples = np.array(accepted_samples[:self.n_samples])
        
        final_acceptance_rate = len(accepted_samples) / total_generated
        #print(f"均匀分布边界感知采样完成：生成 {total_generated} 个候选点，接受 {len(accepted_samples)} 个点，最终接受率: {final_acceptance_rate:.3f}")
        if self.reject:
            self.n_samples = self.n_samples//2
        return accepted_samples 