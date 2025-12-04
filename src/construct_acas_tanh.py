from samplers.uniform_sampler import UniformSampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from utils.load import readNNet, ACAS_tanh, load_ACASXU
import sys

def construct_acas_tanh(x, y):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           ***unsafe region: (C* y <= goal)
    """
    # 网络路径
    nnet_path = f"./checkpoints/acas/ACASXU_run2a_{x}_{y}_batch_2000.nnet"
    weights, biases, inputMins, inputMaxes, means, ranges = readNNet(nnet_path)
    net = ACAS_tanh(weights, biases)
    
    net.cuda()
    net.eval()
    lb = [0] * 5
    ub = [0] * 5
    for i in range(0, 5):
        lb[i] = (inputMins[i] - means[i])/ranges[i]
        ub[i] = (inputMaxes[i] - means[i])/ranges[i]
    return net, lb, ub

def generate_training_data(teacher_net, student_net, lb, ub, num_samples=10000):
    """生成训练数据：使用teacher_net的输出作为student_net的训练目标"""
    print(f"生成 {num_samples} 个训练样本...")
    
    # 创建输入范围
    input_range = [[lb[i], ub[i]] for i in range(len(lb))]
    
    # 使用均匀采样器生成输入数据
    sampler = UniformSampler(input_range, n_samples=num_samples)
    X = sampler.sample()
    
    # 使用teacher_net生成目标输出
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_target = teacher_net(X_tensor).cpu().numpy()
    
    print(f"训练数据生成完成，输入形状: {X.shape}, 输出形状: {y_target.shape}")
    return X, y_target

def knowledge_distillation_loss(student_outputs, teacher_outputs, temperature=4.0, alpha=0.7):
    """
    知识蒸馏损失函数
    Args:
        student_outputs: 学生网络输出
        teacher_outputs: 教师网络输出
        temperature: 温度参数，用于软化概率分布
        alpha: 平衡参数，控制硬标签和软标签的权重
    """
    # 软标签损失 (KL散度)
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 硬标签损失 (MSE)
    hard_loss = F.mse_loss(student_outputs, teacher_outputs)
    
    # 组合损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss, soft_loss, hard_loss

def evaluate_similarity(student_net, teacher_net, lb, ub, num_samples=1000, verbose=True):
    """评估学生网络与教师网络的相似度"""
    if verbose:
        print(f"评估网络相似度 (使用 {num_samples} 个测试样本)...")
    
    # 生成测试数据
    input_range = [[lb[i], ub[i]] for i in range(len(lb))]
    sampler = UniformSampler(input_range, n_samples=num_samples)
    X_test = sampler.sample()
    
    # 转换为tensor
    X_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
    
    # 获取两个网络的输出
    with torch.no_grad():
        student_net.eval()
        teacher_net.eval()
        
        y_student = student_net(X_tensor).cpu().numpy()
        y_teacher = teacher_net(X_tensor).cpu().numpy()
    
    # 计算各种相似度指标
    mse = np.mean((y_student - y_teacher) ** 2)
    rmse = np.sqrt(mse)
    
    # 相对误差
    relative_error = np.mean(np.abs(y_student - y_teacher) / (np.abs(y_teacher) + 1e-8))
    
    # 最大绝对误差
    max_abs_error = np.max(np.abs(y_student - y_teacher))
    
    # 平均绝对误差
    mae = np.mean(np.abs(y_student - y_teacher))
    
    # 相关系数
    correlation = np.corrcoef(y_student.flatten(), y_teacher.flatten())[0, 1]
    
    # 输出相似度
    if verbose:
        print(f"相似度评估结果:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  相对误差: {relative_error:.6f}")
        print(f"  最大绝对误差: {max_abs_error:.6f}")
        print(f"  相关系数: {correlation:.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'relative_error': relative_error,
        'max_abs_error': max_abs_error,
        'correlation': correlation
    }

def train_network(student_net, teacher_net, lb, ub, 
                 num_epochs=200, batch_size=128, learning_rate=0.001,
                 eval_interval=10, temperature=4.0, alpha=1.0):
    """训练student_net模仿teacher_net，使用知识蒸馏损失函数"""
    print(f"开始训练网络...")
    print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"知识蒸馏参数: temperature={temperature}, alpha={alpha}")
    print(f"相似度评估间隔: 每 {eval_interval} 个epoch")
    
    # 生成训练数据
    X_train, y_train = generate_training_data(teacher_net, student_net, lb, ub, num_samples=20000)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    # 设置优化器
    optimizer = optim.Adam(student_net.parameters(), lr=learning_rate)
    
    # 训练历史
    train_losses = []
    soft_losses = []
    hard_losses = []
    similarity_history = []
    
    # 训练循环
    student_net.train()
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_soft_loss = 0.0
        epoch_hard_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            
            # 获取教师网络的输出
            with torch.no_grad():
                teacher_net.eval()
                teacher_outputs = teacher_net(batch_X)
            
            # 学生网络前向传播
            optimizer.zero_grad()
            student_outputs = student_net(batch_X)
            
            # 计算知识蒸馏损失
            total_loss, soft_loss, hard_loss = knowledge_distillation_loss(
                student_outputs, teacher_outputs, temperature, alpha
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_soft_loss += soft_loss.item()
            epoch_hard_loss += hard_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_total_loss = epoch_total_loss / num_batches
        avg_soft_loss = epoch_soft_loss / num_batches
        avg_hard_loss = epoch_hard_loss / num_batches
        
        train_losses.append(avg_total_loss)
        soft_losses.append(avg_soft_loss)
        hard_losses.append(avg_hard_loss)
        
        # 定期评估相似度
        if (epoch + 1) % eval_interval == 0:
            student_net.eval()
            similarity = evaluate_similarity(student_net, teacher_net, lb, ub, 
                                          num_samples=1000, verbose=False)
            similarity_history.append(similarity)
            student_net.train()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Total Loss: {avg_total_loss:.6f}, "
                  f"Soft Loss: {avg_soft_loss:.6f}, "
                  f"Hard Loss: {avg_hard_loss:.6f}, "
                  f"RMSE: {similarity['rmse']:.6f}, "
                  f"相对误差: {similarity['relative_error']:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Total Loss: {avg_total_loss:.6f}, "
                  f"Soft Loss: {avg_soft_loss:.6f}, "
                  f"Hard Loss: {avg_hard_loss:.6f}")
    
    # 最终相似度评估
    print(f"\n训练完成！最终损失: {avg_total_loss:.6f}")
    print("最终相似度评估:")
    final_similarity = evaluate_similarity(student_net, teacher_net, lb, ub, 
                                         num_samples=2000, verbose=True)
    
    return avg_total_loss, final_similarity, similarity_history, {
        'total_losses': train_losses,
        'soft_losses': soft_losses,
        'hard_losses': hard_losses
    }

def save_trained_network(net, final_loss, network_id, lb, ub, similarity_metrics, loss_history, save_path):
    """保存训练好的网络，支持AutoLiRPA验证"""
    # 确保网络处于eval模式
    net.eval()
    
    # 保存完整网络对象和相关信息
    torch.save({
        'model': net,  # 保存完整网络对象，可直接用于AutoLiRPA
        'model_state_dict': net.state_dict(),  # 也保存权重，以备需要
        'final_loss': final_loss,
        'network_id': network_id,
        'input_bounds': (lb, ub),
        'network_type': 'ACAS_tanh',
        'activation': 'tanh',
        'similarity_metrics': similarity_metrics,  # 保存相似度指标
        'loss_history': loss_history  # 保存损失历史
    }, save_path)
    
    print(f"网络已保存到: {save_path}")
    print(f"网络类型: ACAS_tanh with tanh activation")
    print(f"输入维度: {len(lb)}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"最终相似度 - RMSE: {similarity_metrics['rmse']:.6f}, "
          f"相对误差: {similarity_metrics['relative_error']:.6f}")

def load_trained_network_for_verification(model_path, device='cuda'):
    """加载训练好的网络用于AutoLiRPA验证"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载训练好的网络用于验证: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 直接加载完整网络对象
    net = checkpoint['model']
    net = net.to(device)
    net.eval()
    
    # 获取其他信息
    final_loss = checkpoint['final_loss']
    network_id = checkpoint['network_id']
    lb, ub = checkpoint['input_bounds']
    similarity_metrics = checkpoint.get('similarity_metrics', {})
    loss_history = checkpoint.get('loss_history', {})
    
    print(f"网络加载成功！")
    print(f"网络ID: {network_id}")
    print(f"最终损失: {final_loss:.6f}")
    print(f"输入边界: {lb} ~ {ub}")
    print(f"网络类型: {checkpoint.get('network_type', 'Unknown')}")
    print(f"激活函数: {checkpoint.get('activation', 'Unknown')}")
    
    if similarity_metrics:
        print(f"相似度指标:")
        print(f"  RMSE: {similarity_metrics.get('rmse', 'N/A'):.6f}")
        print(f"  相对误差: {similarity_metrics.get('relative_error', 'N/A'):.6f}")
        print(f"  相关系数: {similarity_metrics.get('correlation', 'N/A'):.6f}")
    
    return net, lb, ub, final_loss, network_id, similarity_metrics, loss_history

def verify_with_autolirpa(model_path, spec_id=1):
    """使用AutoLiRPA验证保存的网络"""
    try:
        # 导入AutoLiRPA相关模块
        sys.path.append('/home/lizong/alpha-beta-CROWN/complete_verifier/')
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        
        # 加载网络
        net, lb, ub, final_loss, network_id, similarity_metrics, loss_history = load_trained_network_for_verification(model_path)
        
        # 获取验证规格
        x, y = network_id
        acas_net, lb_acas, ub_acas, C, goal = load_ACASXU(x, y, spec_id)
        
        print(f"\n=== 使用AutoLiRPA验证网络 ACASXU_{x}_{y} ===")
        print(f"规格ID: {spec_id}")
        
        # 创建输入边界
        lower = torch.tensor(lb, dtype=torch.float32).unsqueeze(0).cuda()
        upper = torch.tensor(ub, dtype=torch.float32).unsqueeze(0).cuda()
        
        # 创建BoundedModule
        x_center = (lower + upper) / 2
        lirpa_model = BoundedModule(net, torch.empty_like(x_center), device='cuda')
        
        # 设置扰动
        norm = float("inf")
        ptb = PerturbationLpNorm(norm=norm, x_L=lower, x_U=upper)
        x = BoundedTensor(x_center, ptb)
        
        # 计算边界
        print("计算网络边界...")
        lb_output, ub_output = lirpa_model.compute_bounds(x=(x,), method='crown')
        
        print(f"输出边界:")
        print(f"下界: {lb_output}")
        print(f"上界: {ub_output}")
        
        # 如果有规格矩阵，计算规格验证
        if C is not None and goal is not None:
            print(f"\n规格验证:")
            print(f"规格矩阵C: {C.shape}")
            print(f"目标值goal: {goal}")
            
            lb_spec, ub_spec = lirpa_model.compute_bounds(x=(x,), method='crown', C=C)
            print(f"规格边界:")
            print(f"下界: {lb_spec}")
            print(f"上界: {ub_spec}")
            
            # 检查是否满足规格
            if torch.all(lb_spec > goal):
                print("✅ 规格验证通过：网络满足安全要求")
            else:
                print("❌ 规格验证失败：网络可能不满足安全要求")
        
        return True
        
    except ImportError as e:
        print(f"❌ AutoLiRPA导入失败: {e}")
        print("请确保AutoLiRPA已正确安装")
        return False
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    # 确保保存目录存在
    save_dir = "/home/lizong/ProbabilisticVerification/AAAI/checkpoints/acas_tanh"
    os.makedirs(save_dir, exist_ok=True)
    
    for x in range(1, 6):
        for y in range(1,10):
            print(f"\n=== 处理网络 ACASXU_{x}_{y} ===")
            
            tanh_model_path = f"{save_dir}/acas_tanh_{x}_{y}.pth"
            
            # 如果模型已存在，跳过
            if os.path.exists(tanh_model_path):
                print(f"模型 {tanh_model_path} 已存在，跳过...")
                continue
            
            # 构建tanh网络
            net, lb, ub = construct_acas_tanh(x, y)
            
            # 加载原始ACAS网络作为教师网络
            acas_net, lb_acas, ub_acas, C, goal = load_ACASXU(x, y, 0)
            
            # 训练前评估初始相似度
            print("训练前相似度评估:")
            initial_similarity = evaluate_similarity(net, acas_net, lb, ub, num_samples=1000)
            
            # 训练tanh网络模仿原始网络（使用知识蒸馏损失）
            final_loss, final_similarity, similarity_history, loss_history = train_network(
                net, acas_net, lb, ub, num_epochs=1000, temperature=30.0, alpha=0.05
            )
            
            # 保存训练好的网络（支持AutoLiRPA验证）
            save_trained_network(net, final_loss, (x, y), lb, ub, final_similarity, loss_history, tanh_model_path)
            
            print(f"=== 网络 ACASXU_{x}_{y} 处理完成 ===\n")