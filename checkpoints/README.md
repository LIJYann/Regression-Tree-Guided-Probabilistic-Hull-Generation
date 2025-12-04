# Checkpoints 目录说明 / Checkpoints Directory Guide

本目录用于存储预训练模型文件。

This directory is used to store pre-trained model files.

## 📁 目录结构 / Directory Structure

```
checkpoints/
├── acas/              # ACAS Xu ReLU 网络模型 / ACAS Xu ReLU network models
├── acas_tanh/         # ACAS Xu Tanh 网络模型 / ACAS Xu Tanh network models
└── RocketNetReLU/     # RocketNet ReLU 模型 / RocketNet ReLU models
```

## 📥 获取模型文件 / Obtaining Model Files

由于模型文件较大，GitHub 仓库可能不包含所有模型文件。请通过以下方式获取：

Due to the large size of model files, the GitHub repository may not include all model files. Please obtain them through the following methods:

### 方法 1: 从发布版本下载 / Method 1: Download from Releases

如果作者提供了模型文件的下载链接，请：

If the authors provide download links for model files:

1. 访问发布页面或文档中提供的链接 / Visit the release page or links provided in the documentation
2. 下载对应的模型文件 / Download the corresponding model files
3. 将文件解压到相应的子目录中 / Extract files to the corresponding subdirectories

### 方法 2: 从原始来源获取 / Method 2: Obtain from Original Sources

- **ACAS Xu 模型 / ACAS Xu Models**: 可以从 [ACAS Xu 官方资源](https://github.com/verivital/nnv) 获取 `.nnet` 文件 / Can be obtained from [ACAS Xu official resources](https://github.com/verivital/nnv) as `.nnet` files
- **RocketNet 模型 / RocketNet Models**: 请参考论文或联系作者获取 / Please refer to the paper or contact the authors

### 方法 3: 自行构建 / Method 3: Build Yourself

对于 ACAS Tanh 模型，可以使用 `src/construct_acas_tanh.py` 脚本自行构建。

For ACAS Tanh models, you can build them yourself using the `src/construct_acas_tanh.py` script.

## 🔍 文件格式 / File Formats

- **`.nnet` 文件 / `.nnet` files**: ACAS Xu 标准格式的网络文件 / ACAS Xu standard format network files
- **`.pt` 文件 / `.pt` files**: PyTorch 模型文件 / PyTorch model files

## ⚠️ 注意事项 / Important Notes

1. 确保模型文件放在正确的子目录中 / Ensure model files are placed in the correct subdirectories
2. 模型文件名应与代码中的加载逻辑匹配 / Model file names should match the loading logic in the code
3. 某些模型文件可能很大，请确保有足够的存储空间 / Some model files may be very large, ensure you have sufficient storage space

## 📚 相关文档 / Related Documentation

- 主 README / Main README: 了解如何使用这些模型运行实验 / See how to use these models to run experiments
- [docs/artifacts.md](../docs/artifacts.md): 了解更详细的资源获取说明 / See more detailed resource acquisition guide
