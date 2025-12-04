import torch
import numpy as np
import random
from collections import defaultdict
import sys
from scipy import stats
import time
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

def set_seed(seed):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_crown_bound(model, lower, upper, C=None, return_A=False, method='crown'):
    device = 'cuda'
    lower = lower.to(device)
    upper = upper.to(device)
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper)
    x = BoundedTensor(lower/2+upper/2, ptb)
    lirpa_model = BoundedModule(model, torch.empty_like(x), device=device)
    #print(C @ lirpa_model(lower).unsqueeze(-1))
    if method=='alpha-crown':
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 10, 'lr_alpha': 0.1}})
    if return_A:
        required_A = defaultdict(set)
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        lb, ub, A = lirpa_model.compute_bounds(x=(x,), method=method, return_A=True, needed_A_dict=required_A, C=C)
        lA = A[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']
        uA = A[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['uA']
        lbias = A[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
        ubias = A[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['ubias']
        return lb, ub, lA, uA, lbias, ubias
    else:
        lb, ub = lirpa_model.compute_bounds(x=(x,), method=method, C=C)
        return lb, ub

def compute_linear_bound(A, bias, lower, upper):
    device = 'cuda'
    A = A.to(device)
    bias = bias.to(device)
    lower = lower.to(device)
    upper = upper.to(device)
    lower_bound = torch.maximum(A, torch.zeros_like(A)) @ lower.unsqueeze(-1) + torch.minimum(A, torch.zeros_like(A)) @ upper.unsqueeze(-1) + bias
    upper_bound = torch.minimum(A, torch.zeros_like(A)) @ lower.unsqueeze(-1) + torch.maximum(A, torch.zeros_like(A)) @ upper.unsqueeze(-1) + bias
    return lower_bound.squeeze(0), upper_bound.squeeze(0)

def compute_box_volume(bounds):
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    volume = 1.0
    for (low, high) in zip(lower, upper):
        if high==low:
            pass
        else:
            volume *= (high - low)
    return volume

def compute_box_prob(bounds, mean, scale):
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    probability = [ (stats.norm.cdf(u, mu, s) - stats.norm.cdf(l, mu, s)) if u!=l else 1.0 for l, u, mu, s in zip(lower, upper, mean, scale)]
    prob = 1.0
    for p in probability:
        prob *= p
    return prob

class CustomTreeNode:
    def __init__(self, is_leaf, value=None, feature=None, threshold=None, left=None, right=None, bounds=None, safety=None, stop_reason=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.bounds = bounds
        self.safety = safety
        self.stop_reason = stop_reason

class CustomDecisionTreeRegressor:
    def __init__(self, model, goal=0.0, max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion='squared_error', coef=1.0, method='crown'):
        self.model = model
        self.goal = goal
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.tree = None
        self.criterion = criterion
        self.n_outputs = 1  # 输出维度，fit时自动判断
        self.coef = coef
        self.method = method

    def fit(self, X: np.ndarray, y: np.ndarray, input_range, C=None, mean=None, scale=None):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,) 或 (n_samples, n_outputs)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.n_outputs = 1
        else:
            self.n_outputs = y.shape[1]
        lower = torch.tensor([r[0] for r in input_range], device='cuda', dtype=torch.float32).unsqueeze(0)
        upper = torch.tensor([r[1] for r in input_range], device='cuda', dtype=torch.float32).unsqueeze(0)
        self.input_range = input_range
        self.mean = mean
        self.scale = scale
        lb = ub =None
        #prob = compute_box_prob(bounds=(lower.cpu().numpy()[0], upper.cpu().numpy()[0]), mean=self.mean, scale=self.scale)
        
        #lb, ub = compute_crown_bound(self.model, lower, upper, C=C, method=self.method)
        self.root = self._build_tree(lb, ub, X, y, lower, upper, depth=0, C=C)

    def fit_raw(self, X: np.ndarray, y: np.ndarray, input_range, mean=None, scale=None):
        """
        纯回归树拟合，不涉及CROWN bounds计算
        X: shape (n_samples, n_features)
        y: shape (n_samples,) 或 (n_samples, n_outputs)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.n_outputs = 1
        else:
            self.n_outputs = y.shape[1]
        lower = torch.tensor([r[0] for r in input_range], device='cuda', dtype=torch.float32).unsqueeze(0)
        upper = torch.tensor([r[1] for r in input_range], device='cuda', dtype=torch.float32).unsqueeze(0)
        self.input_range = input_range
        self.mean = mean
        self.scale = scale
        
        self.root = self._build_tree_raw(X, y, lower, upper, depth=0)

    def _build_tree(self, lb, ub, X: np.ndarray, y: np.ndarray, lower, upper, depth: int, C=None):
                
        stop_reason = None
        if lb is not None and ub is not None:
            if (ub<=self.goal).all() or (lb>=self.goal).any():
                stop_reason = 'crown_bound'
            elif depth >= self.max_depth:
                stop_reason = 'depth'
            elif len(X) < self.min_samples_split:
                stop_reason = 'min_samples_split'
            elif len(X) < 2 * self.min_samples_leaf:
                stop_reason = 'min_samples_leaf'
        if stop_reason is not None:
            # if compute_box_prob((lower.cpu().numpy()[0], upper.cpu().numpy()[0]), self.mean, self.scale)<=1e-5:
            #     lb, ub = compute_crown_bound(self.model, lower, upper, C=C, method='alpha-crown')    
            if (lb>=self.goal).any():
                safety = 1
                stop_reason = 'crown_bound'
            elif (ub<=self.goal).all():
                safety = 0
                stop_reason = 'crown_bound'
            else:
                safety = -1
            # 叶子节点value始终为向量
            value = np.mean(y, axis=0)
            return CustomTreeNode(is_leaf=True, value=value, bounds=(lower.cpu().numpy(), upper.cpu().numpy()), safety=safety, stop_reason=stop_reason)
        dim_lengths = [upper[0, i].item() - lower[0, i].item() for i in range(X.shape[1])]
        best_feature, best_threshold, best_score = None, None, float('inf')
        if self.coef is None:
            feature_candidates = [np.argmax(dim_lengths)]
            coef = 1.0
        else:
            feature_candidates = range(X.shape[1])
            coef = self.coef
        
        for feature in feature_candidates:#range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            dim_length = dim_lengths[feature]
            # 向量化mse/var分裂标准
            if self.criterion not in ['prob_mse', 'mixed', 'absolute_error']:
                sorted_idx = np.argsort(X[:, feature])
                X_sorted = X[sorted_idx, feature]
                y_sorted = y[sorted_idx]
                n = len(y_sorted)
                if n <= 1:
                    continue
                diff = X_sorted[1:] != X_sorted[:-1]
                if not np.any(diff):
                    continue
                thresholds_vec = (X_sorted[1:] + X_sorted[:-1]) / 2
                valid_idx = np.where(diff)[0]
                thresholds_vec = thresholds_vec[valid_idx]
                # 多目标累积和
                cumsum_y = np.cumsum(y_sorted, axis=0)
                cumsum_y2 = np.cumsum(y_sorted ** 2, axis=0)
                left_count = np.arange(1, n)[valid_idx]
                right_count = n - left_count
                left_sum = cumsum_y[:-1][valid_idx]
                right_sum = cumsum_y[-1] - cumsum_y[:-1][valid_idx]
                left_sum2 = cumsum_y2[:-1][valid_idx]
                right_sum2 = cumsum_y2[-1] - cumsum_y2[:-1][valid_idx]
                # 多目标方差
                left_var = (left_sum2 - left_sum ** 2 / left_count[:, None]) / left_count[:, None]
                right_var = (right_sum2 - right_sum ** 2 / right_count[:, None]) / right_count[:, None]
                # 聚合所有目标的方差
                left_var_sum = np.sum(left_var, axis=1)
                right_var_sum = np.sum(right_var, axis=1)
                # 计算每个分割点对应的左右长度
                left_length = thresholds_vec - lower[0, feature].item()
                right_length = upper[0, feature].item() - thresholds_vec
                # 避免除零，添加小的epsilon
                epsilon = 1e-8
                left_length = np.maximum(left_length, epsilon)
                right_length = np.maximum(right_length, epsilon)
                # 分别除以对应长度
                score_vec = (left_var_sum * left_count)  + (right_var_sum * right_count) 
                score_vec /= dim_length**coef
                mask = (left_count >= self.min_samples_leaf) & (right_count >= self.min_samples_leaf)
                if not np.any(mask):
                    continue
                score_vec = score_vec[mask]
                thresholds_vec = thresholds_vec[mask]
                left_count = left_count[mask]
                right_count = right_count[mask]
                min_idx = np.argmin(score_vec)
                score = score_vec[min_idx]
                threshold = thresholds_vec[min_idx]
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                continue
            
            # prob_mse分裂标准批量优化
            if self.criterion == 'prob_mse':
                sorted_idx = np.argsort(X[:, feature])
                X_sorted = X[sorted_idx, feature]
                y_sorted = y[sorted_idx]
                n = len(y_sorted)
                if n <= 1:
                    continue
                diff = X_sorted[1:] != X_sorted[:-1]
                if not np.any(diff):
                    continue
                thresholds_vec = (X_sorted[1:] + X_sorted[:-1]) / 2
                valid_idx = np.where(diff)[0]
                thresholds_vec = thresholds_vec[valid_idx]
                cumsum_y = np.cumsum(y_sorted, axis=0)
                cumsum_y2 = np.cumsum(y_sorted ** 2, axis=0)
                left_count = np.arange(1, n)[valid_idx]
                right_count = n - left_count
                left_sum = cumsum_y[:-1][valid_idx]
                right_sum = cumsum_y[-1] - cumsum_y[:-1][valid_idx]
                left_sum2 = cumsum_y2[:-1][valid_idx]
                right_sum2 = cumsum_y2[-1] - cumsum_y2[:-1][valid_idx]
                left_var = (left_sum2 - left_sum ** 2 / left_count[:, None]) / left_count[:, None]
                right_var = (right_sum2 - right_sum ** 2 / right_count[:, None]) / right_count[:, None]
                left_var_sum = np.sum(left_var, axis=1)
                right_var_sum = np.sum(right_var, axis=1)
                # 批量生成box
                left_bounds_list = []
                right_bounds_list = []
                for t in thresholds_vec:
                    left_upper = upper.clone()
                    left_upper[0, feature] = float(t)
                    right_lower = lower.clone()
                    right_lower[0, feature] = float(t)
                    left_bounds = [[float(l), float(u)] for l, u in zip(lower.cpu().numpy()[0], left_upper.cpu().numpy()[0])]
                    right_bounds = [[float(l), float(u)] for l, u in zip(right_lower.cpu().numpy()[0], upper.cpu().numpy()[0])]
                    left_bounds_list.append(left_bounds)
                    right_bounds_list.append(right_bounds)
                prob_left_arr = np.array([compute_box_prob(b, self.mean, self.scale) for b in left_bounds_list])
                prob_right_arr = np.array([compute_box_prob(b, self.mean, self.scale) for b in right_bounds_list])
                # 计算每个分割点对应的左右长度
                left_length = thresholds_vec - lower[0, feature].item()
                right_length = upper[0, feature].item() - thresholds_vec
                # 避免除零，添加小的epsilon
                epsilon = 1e-8
                left_length = np.maximum(left_length, epsilon)
                right_length = np.maximum(right_length, epsilon)
                # 分别除以对应长度
                score_vec = (left_var_sum / prob_left_arr * left_count) / left_length + (right_var_sum / prob_right_arr * right_count) / right_length
                score_vec /= dim_length**coef
                mask = (left_count >= self.min_samples_leaf) & (right_count >= self.min_samples_leaf)
                if not np.any(mask):
                    continue
                score_vec = score_vec[mask]
                thresholds_vec = thresholds_vec[mask]
                left_count = left_count[mask]
                right_count = right_count[mask]
                min_idx = np.argmin(score_vec)
                score = score_vec[min_idx]
                threshold = thresholds_vec[min_idx]
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                continue
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_upper = upper.clone()
        left_upper[0, best_feature] = float(best_threshold)
        right_lower = lower.clone()
        right_lower[0, best_feature] = float(best_threshold)
        
        l = torch.vstack((lower.clone(), right_lower.clone()))
        u = torch.vstack((left_upper.clone(), upper.clone()))
        current_C = torch.vstack((C.clone(), C.clone()))
        lb, ub = compute_crown_bound(self.model, l, u, C=current_C, method=self.method)
        #lb, ub = compute_crown_bound(self.model, lower, left_upper, C=C, method=self.method)
        left_child = self._build_tree(lb[0], ub[0], X[left_mask], y[left_mask], lower, left_upper, depth+1, C=C)
        
        #lb, ub = compute_crown_bound(self.model, right_lower, upper, C=C, method=self.method)
        right_child = self._build_tree(lb[1], ub[1], X[right_mask], y[right_mask], right_lower, upper, depth+1, C=C)
        
        return CustomTreeNode(is_leaf=False, feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, bounds=(lower.cpu().numpy(), upper.cpu().numpy()))

    def _build_tree_raw(self, X: np.ndarray, y: np.ndarray, lower, upper, depth: int):
        """
        纯回归树构建，不涉及CROWN bounds验证
        """
        stop_reason = None
        
        # 仅基于树参数的停止条件
        if depth >= self.max_depth:
            stop_reason = 'depth'
        elif len(X) < self.min_samples_split:
            stop_reason = 'min_samples_split'
        elif len(X) < 2 * self.min_samples_leaf:
            stop_reason = 'min_samples_leaf'
            
        if stop_reason is not None:
            # 叶子节点，不设置safety属性
            value = np.mean(y, axis=0)
            return CustomTreeNode(is_leaf=True, value=value, bounds=(lower.cpu().numpy(), upper.cpu().numpy()), safety=None, stop_reason=stop_reason)
        
        dim_lengths = [upper[0, i].item() - lower[0, i].item() for i in range(X.shape[1])]
        best_feature, best_threshold, best_score = None, None, float('inf')
        if self.coef is None:
            feature_candidates = [np.argmax(dim_lengths)]
            coef = 1.0
        else:
            feature_candidates = [i for i in range(X.shape[1]) if dim_lengths[i] != 0.0]
            coef = self.coef
        
        for feature in feature_candidates:
            thresholds = np.unique(X[:, feature])
            dim_length = dim_lengths[feature]
            # 向量化mse/var分裂标准
            if self.criterion not in ['prob_mse', 'mixed', 'absolute_error']:
                sorted_idx = np.argsort(X[:, feature])
                X_sorted = X[sorted_idx, feature]
                y_sorted = y[sorted_idx]
                n = len(y_sorted)
                if n <= 1:
                    continue
                diff = X_sorted[1:] != X_sorted[:-1]
                if not np.any(diff):
                    continue
                thresholds_vec = (X_sorted[1:] + X_sorted[:-1]) / 2
                valid_idx = np.where(diff)[0]
                thresholds_vec = thresholds_vec[valid_idx]
                # 多目标累积和
                cumsum_y = np.cumsum(y_sorted, axis=0)
                cumsum_y2 = np.cumsum(y_sorted ** 2, axis=0)
                left_count = np.arange(1, n)[valid_idx]
                right_count = n - left_count
                left_sum = cumsum_y[:-1][valid_idx]
                right_sum = cumsum_y[-1] - cumsum_y[:-1][valid_idx]
                left_sum2 = cumsum_y2[:-1][valid_idx]
                right_sum2 = cumsum_y2[-1] - cumsum_y2[:-1][valid_idx]
                # 多目标方差
                left_var = (left_sum2 - left_sum ** 2 / left_count[:, None]) / left_count[:, None]
                right_var = (right_sum2 - right_sum ** 2 / right_count[:, None]) / right_count[:, None]
                # 聚合所有目标的方差
                left_var_sum = np.sum(left_var, axis=1)
                right_var_sum = np.sum(right_var, axis=1)
                # 计算每个分割点对应的左右长度
                left_length = thresholds_vec - lower[0, feature].item()
                right_length = upper[0, feature].item() - thresholds_vec
                # 避免除零，添加小的epsilon
                epsilon = 1e-8
                left_length = np.maximum(left_length, epsilon)
                right_length = np.maximum(right_length, epsilon)
                # 分别除以对应长度
                score_vec = (left_var_sum * left_count)  + (right_var_sum * right_count) 
                score_vec /= dim_length**coef
                mask = (left_count >= self.min_samples_leaf) & (right_count >= self.min_samples_leaf)
                if not np.any(mask):
                    continue
                score_vec = score_vec[mask]
                thresholds_vec = thresholds_vec[mask]
                left_count = left_count[mask]
                right_count = right_count[mask]
                min_idx = np.argmin(score_vec)
                score = score_vec[min_idx]
                threshold = thresholds_vec[min_idx]
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                continue
            
            # prob_mse分裂标准批量优化
            if self.criterion == 'prob_mse':
                sorted_idx = np.argsort(X[:, feature])
                X_sorted = X[sorted_idx, feature]
                y_sorted = y[sorted_idx]
                n = len(y_sorted)
                if n <= 1:
                    continue
                diff = X_sorted[1:] != X_sorted[:-1]
                if not np.any(diff):
                    continue
                thresholds_vec = (X_sorted[1:] + X_sorted[:-1]) / 2
                valid_idx = np.where(diff)[0]
                thresholds_vec = thresholds_vec[valid_idx]
                cumsum_y = np.cumsum(y_sorted, axis=0)
                cumsum_y2 = np.cumsum(y_sorted ** 2, axis=0)
                left_count = np.arange(1, n)[valid_idx]
                right_count = n - left_count
                left_sum = cumsum_y[:-1][valid_idx]
                right_sum = cumsum_y[-1] - cumsum_y[:-1][valid_idx]
                left_sum2 = cumsum_y2[:-1][valid_idx]
                right_sum2 = cumsum_y2[-1] - cumsum_y2[:-1][valid_idx]
                left_var = (left_sum2 - left_sum ** 2 / left_count[:, None]) / left_count[:, None]
                right_var = (right_sum2 - right_sum ** 2 / right_count[:, None]) / right_count[:, None]
                left_var_sum = np.sum(left_var, axis=1)
                right_var_sum = np.sum(right_var, axis=1)
                # 批量生成box
                left_bounds_list = []
                right_bounds_list = []
                for t in thresholds_vec:
                    left_upper = upper.clone()
                    left_upper[0, feature] = float(t)
                    right_lower = lower.clone()
                    right_lower[0, feature] = float(t)
                    left_bounds = [[float(l), float(u)] for l, u in zip(lower.cpu().numpy()[0], left_upper.cpu().numpy()[0])]
                    right_bounds = [[float(l), float(u)] for l, u in zip(right_lower.cpu().numpy()[0], upper.cpu().numpy()[0])]
                    left_bounds_list.append(left_bounds)
                    right_bounds_list.append(right_bounds)
                prob_left_arr = np.array([compute_box_prob(b, self.mean, self.scale) for b in left_bounds_list])
                prob_right_arr = np.array([compute_box_prob(b, self.mean, self.scale) for b in right_bounds_list])
                # 计算每个分割点对应的左右长度
                left_length = thresholds_vec - lower[0, feature].item()
                right_length = upper[0, feature].item() - thresholds_vec
                # 避免除零，添加小的epsilon
                epsilon = 1e-8
                left_length = np.maximum(left_length, epsilon)
                right_length = np.maximum(right_length, epsilon)
                # 分别除以对应长度
                score_vec = (left_var_sum * prob_left_arr * left_count) / left_length + (right_var_sum * prob_right_arr * right_count) / right_length
                score_vec /= dim_length**coef
                mask = (left_count >= self.min_samples_leaf) & (right_count >= self.min_samples_leaf)
                if not np.any(mask):
                    continue
                score_vec = score_vec[mask]
                thresholds_vec = thresholds_vec[mask]
                left_count = left_count[mask]
                right_count = right_count[mask]
                min_idx = np.argmin(score_vec)
                score = score_vec[min_idx]
                threshold = thresholds_vec[min_idx]
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                continue
                
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_upper = upper.clone()
        left_upper[0, best_feature] = float(best_threshold)
        right_lower = lower.clone()
        right_lower[0, best_feature] = float(best_threshold)
        
        # 递归构建子树，不计算CROWN bounds
        left_child = self._build_tree_raw(X[left_mask], y[left_mask], lower, left_upper, depth+1)
        right_child = self._build_tree_raw(X[right_mask], y[right_mask], right_lower, upper, depth+1)
        
        return CustomTreeNode(is_leaf=False, feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, bounds=(lower.cpu().numpy(), upper.cpu().numpy()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (n_samples, n_features)
        返回: 单目标 shape=(n_samples,); 多目标 shape=(n_samples, n_outputs)
        """
        preds = np.zeros((X.shape[0], self.n_outputs))
        for i in range(X.shape[0]):
            preds[i, :] = self._predict_one(self.root, X[i])
        if self.n_outputs == 1:
            return preds.ravel()
        return preds

    def _predict_one(self, node: CustomTreeNode, x: np.ndarray) -> np.ndarray:
        if node.is_leaf:
            return node.value  # shape=(n_outputs,)
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x) 
        
    