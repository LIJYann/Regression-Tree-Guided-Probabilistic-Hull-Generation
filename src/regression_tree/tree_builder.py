from scipy import stats
import torch
import numpy as np
from utils.utils import compute_crown_bound
class CustomRegionSplitter:
    def __init__(self, model, goal, boundary_points, input_range, C, mean, scale, X, y, criterion,coef=1.0, method='crown'):
        self.model = model
        self.goal = goal
        self.boundary_points = boundary_points
        self.input_range = input_range
        self.tree = None
        self.C = C
        self.mean = mean
        self.scale = scale
        self.X = X
        self.y = y
        self.criterion = criterion
        self.coef = coef
        self.method = method
    def train_decision_tree(self, max_depth=10, input_region=None):
        from utils.utils import CustomDecisionTreeRegressor
        #print("生成训练数据...")
        X = self.X
        y = self.y
        #print("训练自定义回归树...")
        if input_region is None:
            input_region = self.input_range
        self.tree = CustomDecisionTreeRegressor(self.model, self.goal, max_depth=max_depth, min_samples_split=10, min_samples_leaf=5, criterion=self.criterion, coef=self.coef, method=self.method)
        self.tree.fit(X, y, input_region, self.C, self.mean, self.scale)
        #print(f"自定义回归树训练完成。")

    def train_decision_tree_parallel(self, max_depth=10, input_region=None):
        from utils.utils import CustomDecisionTreeRegressor
        #print("生成训练数据...")
        X = self.X
        y = self.y
        #print("训练自定义回归树（纯回归版本）...")
        if input_region is None:
            input_region = self.input_range
        self.tree = CustomDecisionTreeRegressor(self.model, self.goal, max_depth=max_depth, min_samples_split=10, min_samples_leaf=5, criterion=self.criterion, coef=self.coef, method=self.method)
        self.tree.fit_raw(X, y, input_region, self.mean, self.scale)
        #print(f"自定义回归树训练完成（纯回归版本）。")
    def get_tree_regions_parallel(self):
        def get_node_regions(node, bounds, depth):
            regions = []
            if node.is_leaf:
                value = node.value
                safety = node.safety
                stop_reason = node.stop_reason
                regions.append((bounds, value, depth, safety, self._compute_box_prob(bounds), stop_reason))
            else:
                feature = node.feature
                threshold = node.threshold
                left_bounds = [list(b) for b in bounds]
                left_bounds[feature][1] = threshold
                regions.extend(get_node_regions(node.left, left_bounds, depth+1))
                right_bounds = [list(b) for b in bounds]
                right_bounds[feature][0] = threshold
                regions.extend(get_node_regions(node.right, right_bounds, depth+1))
            return regions
        initial_bounds = [[r[0], r[1]] for r in self.input_range]
        regions = get_node_regions(self.tree.root, initial_bounds, 0)
        
        # 优化：预分配tensor，避免反复vstack
        n_regions = len(regions)
        n_features = len(self.input_range)
        
        # 批量创建所有bounds tensor
        lower_bounds = torch.zeros((n_regions, n_features), device='cuda', dtype=torch.float32)
        upper_bounds = torch.zeros((n_regions, n_features), device='cuda', dtype=torch.float32)
        
        for i, region in enumerate(regions):
            bounds = region[0]
            for j, bound in enumerate(bounds):
                lower_bounds[i, j] = bound[0]
                upper_bounds[i, j] = bound[1]
        
        # 批量复制C矩阵 - 处理4维C矩阵 [1, 1, 4, 5]
        # self.C shape: [1, 1, 4, 5] -> 需要扩展为 [n_regions, 1, 4, 5]
       
        C_batch = self.C.expand(n_regions, -1, -1).contiguous().view(n_regions, self.C.shape[-2], self.C.shape[-1])
        
        
        
        lb, ub = compute_crown_bound(self.model, lower_bounds, upper_bounds, C=C_batch, method=self.method)
        
        # 确保goal在正确的设备上（处理float和tensor两种情况）
        if isinstance(self.goal, torch.Tensor):
            goal = self.goal.to(lb.device)
        else:
            goal = torch.tensor(self.goal, device=lb.device, dtype=lb.dtype)
        
        # 批量更新safety - 使用向量化操作
        lb_safe = (lb >= goal).any(dim=1)
        ub_safe = (ub <= goal).all(dim=1)
        
        # 转移到CPU进行向量化条件赋值，避免GPU-CPU传输开销
        lb_safe_cpu = lb_safe.cpu()
        ub_safe_cpu = ub_safe.cpu()
        
        # 使用numpy进行更快的向量化操作
        safety_values = np.where(lb_safe_cpu, 1, np.where(ub_safe_cpu, 0, -1))
        
        # 批量重构regions，直接使用numpy数组索引
        regions = [(region[0], region[1], region[2], int(safety_values[i]), region[4], region[5]) 
                  for i, region in enumerate(regions)]
        
        return regions
    
    def get_tree_regions(self):
        def get_node_regions(node, bounds, depth):
            regions = []
            if node.is_leaf:
                value = node.value
                safety = node.safety
                stop_reason = node.stop_reason
                regions.append((bounds, value, depth, safety, self._compute_box_prob(bounds), stop_reason))
            else:
                feature = node.feature
                threshold = node.threshold
                left_bounds = [list(b) for b in bounds]
                left_bounds[feature][1] = threshold
                regions.extend(get_node_regions(node.left, left_bounds, depth+1))
                right_bounds = [list(b) for b in bounds]
                right_bounds[feature][0] = threshold
                regions.extend(get_node_regions(node.right, right_bounds, depth+1))
            return regions
        initial_bounds = [[r[0], r[1]] for r in self.input_range]
        return get_node_regions(self.tree.root, initial_bounds, 0)
    def _compute_box_prob(self, bounds) -> float:
        lower = [b[0] for b in bounds]
        upper = [b[1] for b in bounds]
        probability =  [(stats.norm.cdf(u,mu,s)-stats.norm.cdf(l,mu,s)) if u!=l else 1.0 for l, u, mu, s in zip(lower, upper, self.mean, self.scale)]
        prob = 1.0
        for p in probability:
            prob *= p
        return prob
