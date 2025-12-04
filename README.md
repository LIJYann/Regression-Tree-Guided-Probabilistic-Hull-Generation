# Regression-Tree-Guided Probabilistic Hull Generation

本项目实现了基于回归树引导的概率外壳生成方法，用于神经网络概率验证。

This project implements a regression-tree-guided probabilistic hull generation method for neural network probabilistic verification.

## 📋 目录 / Table of Contents

- [项目简介 / Project Introduction](#项目简介--project-introduction)
- [环境配置 / Environment Setup](#环境配置--environment-setup)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [核心算法 / Core Algorithms](#核心算法--core-algorithms)
- [实验结果 / Experimental Results](#实验结果--experimental-results)

## 🎯 项目简介 / Project Introduction

本项目提出了一种回归树引导的概率外壳生成方法，用于神经网络的概率验证。主要特点包括：

This project proposes a regression-tree-guided probabilistic hull generation method for neural network probabilistic verification. Key features include:

- **智能采样策略 / Intelligent Sampling Strategies**：边界感知采样和分布引导采样
  - Boundary-aware sampling and distribution-guided sampling
- **回归树引导分区 / Regression Tree-Guided Partitioning**：自适应区域划分，基于概率质量引导的分区策略
  - Adaptive region partitioning with probability mass-guided partitioning strategies
- **概率验证 / Probabilistic Verification**：基于 CROWN 边界计算的安全概率估计
  - Safety probability estimation based on CROWN bound computation

## 🔧 环境配置 / Environment Setup

### 系统要求 / System Requirements

- Python 3.8+ (auto_LiRPA 要求 Python 3.7+ / auto_LiRPA requires Python 3.7+)
- PyTorch >=1.11.0, <2.3.0 (auto_LiRPA 的严格版本要求 / Strict version requirement from auto_LiRPA)
- CUDA 11.1+ (可选，用于 GPU 加速 / Optional, for GPU acceleration)

### 安装方法 / Installation Methods

#### 方法 1: 使用 Conda (推荐) / Method 1: Using Conda (Recommended)

```bash
# 创建 conda 环境 / Create conda environment
conda env create -f environment.yml

# 激活环境 / Activate environment
conda activate prob-verification

# 安装 auto_LiRPA (作为子模块) / Install auto_LiRPA (as submodule)
cd auto_LiRPA
pip install -e .

# 可选：构建 CUDA 模块以加速计算 / Optional: Build CUDA modules for faster computation
python auto_LiRPA/cuda_utils.py install
```

#### 方法 2: 使用 pip / Method 2: Using pip

```bash
# 创建虚拟环境 / Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖 / Install dependencies
pip install -r requirements.txt

# 安装 auto_LiRPA (作为子模块) / Install auto_LiRPA (as submodule)
cd auto_LiRPA
pip install -e .

# 可选：构建 CUDA 模块 / Optional: Build CUDA modules
python auto_LiRPA/cuda_utils.py install
```

**重要提示 / Important Note**：auto_LiRPA 需要作为子模块包含在项目中。如果使用 Git 克隆，请运行：

auto_LiRPA needs to be included as a submodule in the project. If cloning via Git, run:

```bash
git submodule update --init --recursive
```

## 🚀 快速开始 / Quick Start

### 运行实验 / Running Experiments

1. **ACAS Xu 实验 (ReLU 网络) / ACAS Xu Experiments (ReLU Networks)**:
   ```bash
   cd src
   python acas.py
   ```

2. **ACAS Xu 实验 (Tanh 网络) / ACAS Xu Experiments (Tanh Networks)**:
   ```bash
   cd src
   python acas_tanh.py
   ```

3. **RocketNet 实验 / RocketNet Experiments**:
   ```bash
   cd src
   python rocketnet.py
   ```

### 实验结果 / Experimental Results

所有实验结果将保存在 `results/` 目录中：

All experimental results will be saved in the `results/` directory:

- `results/acas_experiments/` - ACAS 实验结果 / ACAS experiment results
- `results/tanh_experiments/` - Tanh 网络实验结果 / Tanh network experiment results
- `results/rocketnet_experiments/` - RocketNet 实验结果 / RocketNet experiment results

## 📁 项目结构 / Project Structure

```
├── src/                          # 源代码目录 / Source code directory
│   ├── acas.py                  # ACAS Xu ReLU 网络实验主程序 / ACAS Xu ReLU network experiment main program
│   ├── acas_tanh.py             # ACAS Xu Tanh 网络实验主程序 / ACAS Xu Tanh network experiment main program
│   ├── rocketnet.py             # RocketNet 实验主程序 / RocketNet experiment main program
│   ├── construct_acas_tanh.py   # ACAS Tanh 网络构建工具 / ACAS Tanh network construction tool
│   │
│   ├── utils/                   # 工具函数 / Utility functions
│   │   ├── load.py              # 模型加载函数 / Model loading functions
│   │   ├── utils.py             # 核心计算工具 / Core computational tools
│   │   └── __init__.py
│   │
│   ├── samplers/                # 采样器模块 / Sampler modules
│   │   ├── uniform_boundary_sampler.py      # 均匀边界采样 / Uniform boundary sampling
│   │   ├── distribution_boundary_sampler.py # 分布边界采样 / Distribution boundary sampling
│   │   └── __init__.py
│   │
│   ├── regression_tree/         # 回归树模块 / Regression tree modules
│   │   ├── tree_builder.py      # 决策树构建器 / Decision tree builder
│   │   └── __init__.py
│   │
│   └── models/                  # 网络模型 / Network models
│       ├── tiny_network.py     # 2D 示例网络 / 2D example network
│       ├── deep_network_2d.py  # 深度 2D 网络 / Deep 2D network
│       └── __init__.py
│
├── checkpoints/                 # 预训练模型 / Pre-trained models (see checkpoints/README.md)
│   ├── acas/                   # ACAS ReLU 模型 / ACAS ReLU models
│   ├── acas_tanh/              # ACAS Tanh 模型 / ACAS Tanh models
│   └── RocketNetReLU/          # RocketNet 模型 / RocketNet models
│
├── docs/                        # 文档 / Documentation
│   └── artifacts.md            # 资源获取说明 / Resource acquisition guide
│
├── requirements.txt             # Python 依赖 / Python dependencies
├── environment.yml              # Conda 环境配置 / Conda environment configuration
└── README.md                    # 本文件 / This file
```

## 🔬 核心算法 / Core Algorithms

本项目实现了以下关键算法：

This project implements the following key algorithms:

1. **智能采样策略 / Intelligent Sampling Strategies**：
   - 边界感知采样 / Boundary-aware sampling
   - 分布引导采样 / Distribution-guided sampling
   - 混合采样策略 / Mixed sampling strategies

2. **回归树引导分区 / Regression Tree-Guided Partitioning**：
   - 自适应区域划分 / Adaptive region partitioning
   - 概率质量引导的分区策略 / Probability mass-guided partitioning strategies
   - 并行化树构建 / Parallelized tree construction

3. **概率验证 / Probabilistic Verification**：
   - CROWN 边界计算 / CROWN bound computation
   - 安全概率估计 / Safety probability estimation
   - 收敛检查 / Convergence checking

## 📊 实验结果 / Experimental Results

每个实验输出以下关键指标：

Each experiment outputs the following key metrics:

- **Ls**: 下界安全概率 / Lower safe probability
- **Us**: 上界安全概率 / Upper safe probability
- **Us-Ls**: 概率区间宽度 / Probability interval width
- **time**: 运行时间 (秒) / Runtime (seconds)

较小的 `Us-Ls` 值表示更精确的概率估计。

Smaller `Us-Ls` values indicate more precise probability estimates.

### 实验参数 / Experimental Parameters

关键实验参数（在脚本中固定）：

Key experimental parameters (fixed in scripts):

- **总采样数 / Total samples**: 1000 (ACAS), 9000 (RocketNet)
- **增量采样数 / Incremental samples**: 100 (ACAS), 900 (RocketNet)
- **最大树深度 / Maximum tree depth**: 5
- **未知概率阈值 / Unknown probability threshold**: 1e-5 (ACAS), 1e-3 (RocketNet)
- **系数参数 / Coefficient parameter**: 0.3

## 📄 许可证 / License

本项目采用 [MIT License](LICENSE) 许可证。详情请参阅 LICENSE 文件。

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

## 🔗 相关资源 / Related Resources

- **模型权重 / Model Weights**: 请参阅 [checkpoints/README.md](checkpoints/README.md) 了解如何获取预训练模型 / See [checkpoints/README.md](checkpoints/README.md) for how to obtain pre-trained models
- **实验资源 / Experimental Resources**: 请参阅 [docs/artifacts.md](docs/artifacts.md) 了解额外的实验数据和资源 / See [docs/artifacts.md](docs/artifacts.md) for additional experimental data and resources
