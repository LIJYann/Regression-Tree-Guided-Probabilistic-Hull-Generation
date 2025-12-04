# 实验资源说明 / Experimental Resources Guide

本文档说明如何获取和使用本项目的实验资源，包括预训练模型、实验结果等。

This document explains how to obtain and use experimental resources for this project, including pre-trained models, experimental results, etc.

## 📦 预训练模型 / Pre-trained Models

本项目使用的预训练模型存储在 `checkpoints/` 目录中。由于模型文件较大，GitHub 仓库可能不包含所有模型文件。

Pre-trained models used in this project are stored in the `checkpoints/` directory. Due to the large size of model files, the GitHub repository may not include all model files.

### ACAS Xu 模型 / ACAS Xu Models

- **位置 / Location**: `checkpoints/acas/`
- **格式 / Format**: `.nnet` 文件 / `.nnet` files
- **说明 / Description**: ACAS Xu ReLU 网络模型，用于 ACAS 实验 / ACAS Xu ReLU network models for ACAS experiments

### ACAS Tanh 模型 / ACAS Tanh Models

- **位置 / Location**: `checkpoints/acas_tanh/`
- **格式 / Format**: `.pt` 文件 (PyTorch 模型) / `.pt` files (PyTorch models)
- **说明 / Description**: ACAS Xu Tanh 网络模型，用于 Tanh 网络实验 / ACAS Xu Tanh network models for Tanh network experiments

### RocketNet 模型 / RocketNet Models

- **位置 / Location**: `checkpoints/RocketNetReLU/`
- **格式 / Format**: `.pt` 文件 (PyTorch 模型) / `.pt` files (PyTorch models)
- **说明 / Description**: RocketNet ReLU 模型，用于 RocketNet 实验 / RocketNet ReLU models for RocketNet experiments

### 获取模型 / Obtaining Models

如果模型文件未包含在仓库中，您可以通过以下方式获取：

If model files are not included in the repository, you can obtain them through the following methods:

1. **从论文作者处获取 / From Paper Authors**: 请联系论文作者获取模型文件下载链接 / Please contact the paper authors for model file download links
2. **自行训练 / Self-training**: 参考 `src/construct_acas_tanh.py` 了解如何构建 ACAS Tanh 模型 / Refer to `src/construct_acas_tanh.py` to learn how to build ACAS Tanh models
3. **使用公开数据集 / Using Public Datasets**: ACAS Xu 模型可以从 [ACAS Xu 官方资源](https://github.com/verivital/nnv) 获取 / ACAS Xu models can be obtained from [ACAS Xu official resources](https://github.com/verivital/nnv)

## 📊 实验结果 / Experimental Results

论文中的实验结果可以通过运行实验脚本复现。所有结果将保存在 `results/` 目录中。

Experimental results from the paper can be reproduced by running the experiment scripts. All results will be saved in the `results/` directory.

### 结果文件格式 / Result File Formats

- **CSV 文件 / CSV files**: 包含实验指标（Ls, Us, Us-Ls, time 等） / Contains experimental metrics (Ls, Us, Us-Ls, time, etc.)
- **JSON 文件 / JSON files**: 包含详细的实验配置和结果 / Contains detailed experimental configurations and results
- **可视化图像 / Visualization images**: PNG 格式的图表和可视化结果 / Charts and visualization results in PNG format

### 复现实验 / Reproducing Experiments

要复现论文中的实验结果，请：

To reproduce the experimental results from the paper:

1. 确保已安装所有依赖（见主 README） / Ensure all dependencies are installed (see main README)
2. 确保模型文件在 `checkpoints/` 目录中 / Ensure model files are in the `checkpoints/` directory
3. 运行相应的实验脚本（见主 README 的"快速开始"部分） / Run the corresponding experiment scripts (see "Quick Start" section in main README)
4. 结果将自动保存到 `results/` 目录 / Results will be automatically saved to the `results/` directory

## 🔧 实验配置 / Experimental Configuration

实验的关键参数在脚本中已固定，以确保可重现性：

Key experimental parameters are fixed in scripts to ensure reproducibility:

- **随机种子 / Random seed**: 1024
- **采样参数 / Sampling parameters**: 见各脚本中的配置 / See configuration in each script
- **树深度 / Tree depth**: 见各脚本中的配置 / See configuration in each script
- **概率阈值 / Probability threshold**: 见各脚本中的配置 / See configuration in each script

## 📝 注意事项 / Important Notes

1. **模型文件大小 / Model File Size**: 某些模型文件可能很大（数百 MB），请确保有足够的存储空间 / Some model files may be very large (hundreds of MB), ensure sufficient storage space
2. **GPU 内存 / GPU Memory**: 某些实验可能需要较大的 GPU 内存（16GB+） / Some experiments may require large GPU memory (16GB+)
3. **运行时间 / Runtime**: 完整实验可能需要数小时，请耐心等待 / Complete experiments may take several hours, please be patient

## 🐛 问题反馈 / Issue Reporting

如果您在获取或使用实验资源时遇到问题，请：

If you encounter issues when obtaining or using experimental resources:

1. 检查主 README 中的"常见问题"部分 / Check the "Common Issues" section in the main README
2. 联系论文作者 / Contact the paper authors
3. 在 GitHub Issues 中报告问题 / Report issues in GitHub Issues
