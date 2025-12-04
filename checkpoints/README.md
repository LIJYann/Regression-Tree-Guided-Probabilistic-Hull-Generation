# Checkpoints 目录说明 / Checkpoints Directory Guide

本目录包含项目使用的所有预训练模型文件。

This directory contains all pre-trained model files used in this project.

## 📁 目录结构 / Directory Structure

```
checkpoints/
├── acas/              # ACAS Xu ReLU 网络模型 / ACAS Xu ReLU network models
│   └── 45 个 .nnet 文件 / 45 .nnet files (~7.3MB)
├── acas_tanh/         # ACAS Xu Tanh 网络模型 / ACAS Xu Tanh network models
│   └── 45 个 .pth 文件 / 45 .pth files (~4.1MB)
└── RocketNetReLU/     # RocketNet ReLU 模型 / RocketNet ReLU models
    └── 3 个 .pt 文件 / 3 .pt files (~76KB)
```

**总计 / Total**: 93 个模型文件，约 12MB / 93 model files, approximately 12MB

## 📥 获取模型文件 / Obtaining Model Files

### ✅ 模型文件已包含在仓库中 / Model Files Included in Repository

所有必需的模型文件已经包含在此 GitHub 仓库中。如果您通过 Git 克隆了项目，模型文件会自动下载。

All required model files are already included in this GitHub repository. If you clone the project via Git, model files will be automatically downloaded.

### 如果模型文件缺失 / If Model Files Are Missing

如果由于某些原因模型文件未正确下载，您可以通过以下方式获取：

If model files are not downloaded correctly for some reason, you can obtain them through the following methods:

#### 方法 1: 重新克隆仓库 / Method 1: Re-clone Repository

```bash
git clone --recursive <repository-url>
```

#### 方法 2: 从原始来源获取 / Method 2: Obtain from Original Sources

- **ACAS Xu 模型 / ACAS Xu Models**: 可以从 [ACAS Xu 官方资源](https://github.com/verivital/nnv) 获取 `.nnet` 文件 / Can be obtained from [ACAS Xu official resources](https://github.com/verivital/nnv) as `.nnet` files
- **RocketNet 模型 / RocketNet Models**: 请参考论文或联系作者获取 / Please refer to the paper or contact the authors

#### 方法 3: 自行构建 / Method 3: Build Yourself

对于 ACAS Tanh 模型，可以使用 `src/construct_acas_tanh.py` 脚本自行构建。

For ACAS Tanh models, you can build them yourself using the `src/construct_acas_tanh.py` script.

## 🔍 文件格式 / File Formats

- **`.nnet` 文件 / `.nnet` files**: ACAS Xu 标准格式的网络文件，用于 ACAS ReLU 网络实验 / ACAS Xu standard format network files for ACAS ReLU network experiments
- **`.pth` 文件 / `.pth` files**: PyTorch 模型文件，用于 ACAS Tanh 网络实验 / PyTorch model files for ACAS Tanh network experiments
- **`.pt` 文件 / `.pt` files**: PyTorch 模型文件，用于 RocketNet 实验 / PyTorch model files for RocketNet experiments

## 📊 模型文件列表 / Model File List

### ACAS 模型 (45 个文件) / ACAS Models (45 files)

所有 ACAS 模型文件命名格式为：`ACASXU_run2a_X_Y_batch_2000.nnet`，其中 X 和 Y 表示网络编号（1-5）。

All ACAS model files follow the naming format: `ACASXU_run2a_X_Y_batch_2000.nnet`, where X and Y represent network numbers (1-5).

### ACAS Tanh 模型 (45 个文件) / ACAS Tanh Models (45 files)

所有 ACAS Tanh 模型文件命名格式为：`acas_tanh_X_Y.pth`，其中 X 和 Y 表示网络编号（1-5）。

All ACAS Tanh model files follow the naming format: `acas_tanh_X_Y.pth`, where X and Y represent network numbers (1-5).

### RocketNet 模型 (3 个文件) / RocketNet Models (3 files)

- `unsafe_agent0.pt`
- `unsafe_agent1.pt`
- `unsafe_agent2.pt`

## ⚠️ 注意事项 / Important Notes

1. **模型文件位置 / Model File Location**: 确保模型文件放在正确的子目录中 / Ensure model files are placed in the correct subdirectories
2. **文件名匹配 / File Name Matching**: 模型文件名应与代码中的加载逻辑匹配 / Model file names should match the loading logic in the code
3. **存储空间 / Storage Space**: 所有模型文件总大小约 12MB，请确保有足够的存储空间 / Total size of all model files is approximately 12MB, ensure sufficient storage space
4. **Git LFS / Git LFS**: 如果使用 Git LFS，确保已正确安装和配置 / If using Git LFS, ensure it is properly installed and configured

## 📚 相关文档 / Related Documentation

- **主 README / Main README**: 了解如何使用这些模型运行实验 / See how to use these models to run experiments
- **[docs/artifacts.md](../docs/artifacts.md)**: 了解更详细的资源获取说明 / See more detailed resource acquisition guide

## 🚀 快速开始 / Quick Start

模型文件已包含在仓库中，您可以直接运行实验：

Model files are included in the repository, you can directly run experiments:

```bash
cd src
python acas.py          # 使用 ACAS ReLU 模型 / Uses ACAS ReLU models
python acas_tanh.py     # 使用 ACAS Tanh 模型 / Uses ACAS Tanh models
python rocketnet.py     # 使用 RocketNet 模型 / Uses RocketNet models
```
