# auto_LiRPA 安装说明

本项目依赖于 [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) 库进行线性松弛传播和边界计算。

## 📦 安装方法

### 方法 1: 作为 Git 子模块 (推荐)

如果通过 Git 克隆项目：

```bash
# 初始化并更新子模块
git submodule update --init --recursive

# 进入 auto_LiRPA 目录
cd auto_LiRPA

# 安装 auto_LiRPA
pip install -e .

# 可选：构建 CUDA 模块以加速计算
python auto_LiRPA/cuda_utils.py install
```

### 方法 2: 手动克隆

如果项目不包含子模块，可以手动克隆：

```bash
# 克隆 auto_LiRPA 到项目根目录
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git

# 进入 auto_LiRPA 目录
cd auto_LiRPA

# 安装
pip install -e .

# 可选：构建 CUDA 模块
python auto_LiRPA/cuda_utils.py install
```

### 方法 3: 使用 pip (不推荐，可能版本不匹配)

```bash
pip install auto-LiRPA
```

**注意**: 使用 pip 安装可能无法保证版本兼容性，建议使用方法 1 或 2。

## ⚠️ 版本要求

- **Python**: >= 3.7 (推荐 3.8+)
- **PyTorch**: >= 1.11.0, < 2.3.0 (严格版本要求)
- **torchvision**: >= 0.12.0, < 0.18.0

## 🔍 验证安装

安装完成后，可以通过以下方式验证：

```python
import auto_LiRPA
print(auto_LiRPA.__version__)
```

如果导入成功，说明安装正确。

## 🐛 常见问题

1. **导入错误**: 确保在 auto_LiRPA 目录中运行了 `pip install -e .`
2. **PyTorch 版本**: 确保 PyTorch 版本在 1.11.0 到 2.3.0 之间
3. **CUDA 错误**: 如果使用 GPU，确保 CUDA 版本与 PyTorch 兼容

## 📚 更多信息

- auto_LiRPA 官方仓库: https://github.com/Verified-Intelligence/auto_LiRPA
- 文档: https://github.com/Verified-Intelligence/auto_LiRPA/wiki

