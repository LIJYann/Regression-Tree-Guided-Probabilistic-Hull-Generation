# 发布说明

## 📦 发布包内容

本发布包包含以下核心内容：

### ✅ 包含的内容

1. **核心源代码** (`src/`)
   - 所有实验脚本（ACAS, RocketNet, Baseline 等）
   - 工具函数模块（utils, samplers, regression_tree, models, visualization）
   - 完整的 Python 源代码

2. **配置文件**
   - `requirements.txt` - Python 依赖列表
   - `environment.yml` - Conda 环境配置
   - `.gitignore` - Git 忽略规则

3. **文档**
   - `README.md` - 主文档（项目介绍、安装、使用）
   - `docs/artifacts.md` - 资源获取说明
   - `docs/INSTALL_AUTO_LIRPA.md` - auto_LiRPA 安装指南
   - `checkpoints/README.md` - 模型文件说明

4. **许可证**
   - `LICENSE` - MIT 许可证

### ❌ 不包含的内容（已排除）

以下内容已从发布包中排除，以减少体积并保持简洁：

1. **实验结果和日志**
   - `results/` 目录
   - 所有 `*.log` 文件
   - `*.out` 文件

2. **实验脚本和临时文件**
   - `grid_search/` 目录
   - `ablation_experiments/` 目录
   - 临时脚本文件

3. **大型依赖**
   - `auto_LiRPA/` 目录（需作为子模块单独获取）
   - `checkpoints/` 中的模型文件（需单独下载）

4. **构建产物**
   - `__pycache__/` 目录
   - `*.egg-info/` 目录
   - 编译产物

## 🚀 快速开始

1. **解压发布包**
   ```bash
   tar -xzf regression-tree-artifact.tar.gz
   cd release_package
   ```

2. **安装依赖**
   ```bash
   conda env create -f environment.yml
   conda activate prob-verification
   ```

3. **安装 auto_LiRPA**
   ```bash
   # 方法 1: 作为子模块
   git submodule update --init --recursive
   cd auto_LiRPA && pip install -e . && cd ..
   
   # 方法 2: 手动克隆
   git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
   cd auto_LiRPA && pip install -e . && cd ..
   ```

4. **下载模型文件**
   - 将模型文件放入 `checkpoints/` 相应子目录
   - 详见 `checkpoints/README.md`

5. **运行实验**
   ```bash
   cd src
   python acas.py
   ```

## 📊 发布包统计

- **总文件数**: ~64 个文件
- **压缩包大小**: ~84 KB
- **解压后大小**: ~544 KB

## 🔗 相关链接

- 主 README: [README.md](README.md)
- 资源说明: [docs/artifacts.md](docs/artifacts.md)
- auto_LiRPA 安装: [docs/INSTALL_AUTO_LIRPA.md](docs/INSTALL_AUTO_LIRPA.md)
- 模型说明: [checkpoints/README.md](checkpoints/README.md)

## 📝 注意事项

1. **auto_LiRPA 依赖**: 必须单独安装 auto_LiRPA，详见安装文档
2. **模型文件**: 模型文件需要单独下载，不包含在发布包中
3. **Python 版本**: 需要 Python 3.8+ 和 PyTorch 1.11.0-2.3.0
4. **GPU 支持**: 可选，但建议使用 GPU 以加速计算

## 🐛 问题反馈

如遇到问题，请：
1. 查阅主 README 的"常见问题"部分
2. 检查所有依赖是否正确安装
3. 联系论文作者或提交 GitHub Issue

