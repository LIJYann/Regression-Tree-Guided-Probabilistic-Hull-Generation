# Checkpoints 目录说明

本目录用于存储预训练模型文件。

## 📁 目录结构

```
checkpoints/
├── acas/              # ACAS Xu ReLU 网络模型
├── acas_tanh/         # ACAS Xu Tanh 网络模型
└── RocketNetReLU/     # RocketNet ReLU 模型
```

## 📥 获取模型文件

由于模型文件较大，GitHub 仓库可能不包含所有模型文件。请通过以下方式获取：

### 方法 1: 从发布版本下载

如果作者提供了模型文件的下载链接，请：
1. 访问发布页面或文档中提供的链接
2. 下载对应的模型文件
3. 将文件解压到相应的子目录中

### 方法 2: 从原始来源获取

- **ACAS Xu 模型**: 可以从 [ACAS Xu 官方资源](https://github.com/verivital/nnv) 获取 `.nnet` 文件
- **RocketNet 模型**: 请参考论文或联系作者获取

### 方法 3: 自行构建

对于 ACAS Tanh 模型，可以使用 `src/construct_acas_tanh.py` 脚本自行构建。

## 🔍 文件格式

- **`.nnet` 文件**: ACAS Xu 标准格式的网络文件
- **`.pt` 文件**: PyTorch 模型文件

## ⚠️ 注意事项

1. 确保模型文件放在正确的子目录中
2. 模型文件名应与代码中的加载逻辑匹配
3. 某些模型文件可能很大，请确保有足够的存储空间

## 📚 相关文档

- 主 README: 了解如何使用这些模型运行实验
- [docs/artifacts.md](../docs/artifacts.md): 了解更详细的资源获取说明

