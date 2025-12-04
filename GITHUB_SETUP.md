# GitHub 上传指南

本指南将帮助您将项目上传到 GitHub。

## 📋 前置准备

1. **GitHub 账号**: 确保您已注册 GitHub 账号
2. **Git 已安装**: 确保系统已安装 Git
3. **SSH 密钥或 Personal Access Token**: 用于身份验证

## 🚀 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `Regression-Tree-Guided-Probabilistic-Hull-Generation` (或您喜欢的名称)
   - **Description**: `Regression-Tree-Guided Probabilistic Hull Generation for Neural Network Verification`
   - **Visibility**: 选择 Public (公开) 或 Private (私有)
   - **不要**勾选 "Initialize this repository with a README" (我们已经有了 README.md)
4. 点击 "Create repository"

## 🔧 步骤 2: 初始化本地 Git 仓库

在 `release_package` 目录中执行以下命令：

```bash
cd release_package

# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial release: Regression-Tree-Guided Probabilistic Hull Generation"
```

## 🔗 步骤 3: 连接到 GitHub 仓库

将本地仓库连接到 GitHub（替换 `YOUR_USERNAME` 和 `YOUR_REPO_NAME`）：

```bash
# 添加远程仓库地址
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 或者使用 SSH（如果您配置了 SSH 密钥）
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# 查看远程仓库配置
git remote -v
```

## 📤 步骤 4: 推送到 GitHub

```bash
# 重命名主分支为 main（如果 GitHub 使用 main 作为默认分支）
git branch -M main

# 推送到 GitHub
git push -u origin main
```

如果遇到认证问题，您可能需要：
- 使用 Personal Access Token 代替密码
- 或配置 SSH 密钥

## ✅ 步骤 5: 验证上传

1. 访问您的 GitHub 仓库页面
2. 确认所有文件都已上传
3. 检查 README.md 是否正确显示

## 🔄 后续更新

如果需要更新代码：

```bash
cd release_package

# 查看更改
git status

# 添加更改的文件
git add .

# 提交更改
git commit -m "Update: 描述您的更改"

# 推送到 GitHub
git push
```

## 📝 添加 auto_LiRPA 子模块（可选）

如果您想将 auto_LiRPA 作为子模块包含：

```bash
cd release_package

# 添加 auto_LiRPA 作为子模块
git submodule add https://github.com/Verified-Intelligence/auto_LiRPA.git auto_LiRPA

# 提交子模块
git add .gitmodules auto_LiRPA
git commit -m "Add auto_LiRPA as submodule"
git push
```

## 🐛 常见问题

### 1. 认证失败

**问题**: `remote: Support for password authentication was removed`

**解决**: 使用 Personal Access Token
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. 选择权限：至少勾选 `repo`
4. 生成后复制 token
5. 推送时使用 token 作为密码

### 2. 分支名称问题

**问题**: `error: src refspec main does not match any`

**解决**: 
```bash
# 检查当前分支
git branch

# 如果分支是 master，重命名为 main
git branch -M main

# 或直接推送到 master
git push -u origin master
```

### 3. 大文件问题

**问题**: 如果模型文件太大，GitHub 可能拒绝

**解决**: 
- 使用 Git LFS (Large File Storage)
- 或将大文件放在外部存储，在文档中提供下载链接

## 📚 有用的 Git 命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新更改
git pull

# 查看分支
git branch -a
```

## 🎯 完成后的检查清单

- [ ] GitHub 仓库已创建
- [ ] 本地 Git 仓库已初始化
- [ ] 所有文件已提交
- [ ] 已连接到 GitHub 远程仓库
- [ ] 代码已成功推送
- [ ] README.md 在 GitHub 上正确显示
- [ ] 仓库描述和标签已设置（可选）

