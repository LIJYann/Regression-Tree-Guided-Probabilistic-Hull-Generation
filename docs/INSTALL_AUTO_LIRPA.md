# auto_LiRPA 安装说明 / auto_LiRPA Installation Guide

本项目依赖于 [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) 库进行线性松弛传播和边界计算。

This project depends on the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library for linear relaxation propagation and bound computation.

## 📦 安装方法 / Installation Methods

### 方法 1: 作为 Git 子模块 (推荐) / Method 1: As Git Submodule (Recommended)

如果通过 Git 克隆项目：

If cloning the project via Git:

```bash
# 初始化并更新子模块 / Initialize and update submodules
git submodule update --init --recursive

# 进入 auto_LiRPA 目录 / Enter auto_LiRPA directory
cd auto_LiRPA

# 安装 auto_LiRPA / Install auto_LiRPA
pip install -e .

# 可选：构建 CUDA 模块以加速计算 / Optional: Build CUDA modules for faster computation
python auto_LiRPA/cuda_utils.py install
```

### 方法 2: 手动克隆 / Method 2: Manual Clone

如果项目不包含子模块，可以手动克隆：

If the project does not include submodules, you can manually clone:

```bash
# 克隆 auto_LiRPA 到项目根目录 / Clone auto_LiRPA to project root directory
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git

# 进入 auto_LiRPA 目录 / Enter auto_LiRPA directory
cd auto_LiRPA

# 安装 / Install
pip install -e .

# 可选：构建 CUDA 模块 / Optional: Build CUDA modules
python auto_LiRPA/cuda_utils.py install
```

### 方法 3: 使用 pip (不推荐，可能版本不匹配) / Method 3: Using pip (Not Recommended, Version May Not Match)

```bash
pip install auto-LiRPA
```

**注意 / Note**: 使用 pip 安装可能无法保证版本兼容性，建议使用方法 1 或 2。

Using pip installation may not guarantee version compatibility. It is recommended to use Method 1 or 2.

## ⚠️ 版本要求 / Version Requirements

- **Python**: >= 3.7 (推荐 3.8+ / Recommended 3.8+)
- **PyTorch**: >= 1.11.0, < 2.3.0 (严格版本要求 / Strict version requirement)
- **torchvision**: >= 0.12.0, < 0.18.0

## 🔍 验证安装 / Verify Installation

安装完成后，可以通过以下方式验证：

After installation, you can verify it by:

```python
import auto_LiRPA
print(auto_LiRPA.__version__)
```

如果导入成功，说明安装正确。

If the import succeeds, the installation is correct.

## 🐛 常见问题 / Common Issues

1. **导入错误 / Import Error**: 确保在 auto_LiRPA 目录中运行了 `pip install -e .` / Ensure you ran `pip install -e .` in the auto_LiRPA directory
2. **PyTorch 版本 / PyTorch Version**: 确保 PyTorch 版本在 1.11.0 到 2.3.0 之间 / Ensure PyTorch version is between 1.11.0 and 2.3.0
3. **CUDA 错误 / CUDA Error**: 如果使用 GPU，确保 CUDA 版本与 PyTorch 兼容 / If using GPU, ensure CUDA version is compatible with PyTorch

## 📚 更多信息 / More Information

- auto_LiRPA 官方仓库 / auto_LiRPA Official Repository: https://github.com/Verified-Intelligence/auto_LiRPA
- 文档 / Documentation: https://github.com/Verified-Intelligence/auto_LiRPA/wiki
