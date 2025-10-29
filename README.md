# 更易用的 AI Toolkit（中文 README）

本项目是 AI Toolkit 的中文与易用性优化版本（ai-toolkit-easy2use）。在保留原功能的基础上，聚焦「更易安装、更易上手、更易维护」。本 README 全文为中文，并对安装、运行与 UI 使用进行汉化说明。

> 原项目作者：Ostris；本仓库维护：DocWorkBox。

---

## 项目简介

- 面向扩散模型（Diffusion Models）的训练与推理一体化工具。
- 支持常见的图像与视频模型，适配消费级硬件。
- 提供命令行（CLI）与 Web 用户界面（UI），上手门槛低同时功能完备。

## 环境要求

- Python ≥ 3.10
- Git（用于拉取仓库）
- NVIDIA GPU（显存需满足你的训练或推理需求）
- Python 虚拟环境（建议）
- Node.js ≥ 18（用于运行 Web UI）

## 安装（Linux / Windows）

### 1）克隆仓库

```bash
git clone https://github.com/DocWorkBox/ai-toolkit-easy2use.git
cd ai-toolkit-easy2use
```

### 2）创建并激活虚拟环境

Linux / macOS：

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows（PowerShell）：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3）安装 PyTorch（示例，CUDA 12.6 对应版本）

根据你的 CUDA / 显卡环境调整版本。以下为参考示例：

```bash
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

### 4）安装项目依赖

```bash
pip install -r requirements.txt
```

## 运行 UI（中文界面）

### 环境要求

- Node.js ≥ 18

UI 为基于 Next.js 的 Web 应用。UI 无需持续运行即可执行训练任务，仅在启动/停止/监控任务时需要使用。

### 开发模式

开发模式运行在 `http://localhost:3000`：

```bash
cd ui
npm install
npm run dev
```

打开浏览器访问：

- `http://localhost:3000/`（首页）
- `http://localhost:3000/dashboard`（仪表盘）
- `http://localhost:3000/jobs/new`（新建任务）

### 生产环境

生产环境运行在端口 `8675`。以下命令将安装/更新 UI 及其依赖并启动 UI：

```bash
cd ui
npm run build_and_start
```

启动后可通过以下地址访问：

- `http://localhost:8675`（本地访问）
- `http://<your-ip>:8675`（服务器部署时的远程访问）

> **注意**：UI 无需持续运行即可执行训练任务。UI 仅用于启动、停止和监控任务。

## 中文版 UI 截图

![仪表盘（中文）](ui/public/screenshots/dashboard_zh.png)
![新建任务（中文）](ui/public/screenshots/jobs_new_zh.png)

## 常见问题（FAQ）

- 显存不足如何处理？
  - 训练大型模型时，如遇显存限制，可在配置中开启低显存选项（例如 `low_vram: true`），或在 CPU 上量化部分模块以降低显存占用。
- Windows 环境安装遇到困难？
  - 建议优先确认 Python、CUDA 与驱动版本匹配；也可以考虑使用 WSL（Windows Subsystem for Linux）以获得更稳定的依赖环境。
- UI 无法访问或接口报错？
  - 请检查 Node.js 版本（≥18）、依赖是否安装完成（`npm install`）、以及开发服务器是否正常运行（`npm run dev`）。

## 致谢与说明

- 本仓库以更易用为目标进行中文化与体验优化，基于原 AI Toolkit 项目实现。
- 如需反馈问题或提交改进，欢迎在本仓库的 Issue 中留言。

---

## 目录指南（简要）

- `config/`：训练或推理相关配置示例与模板。
- `ui/`：Next.js 中文 UI 源码。
- `requirements.txt`：Python 依赖列表。
- 其他训练脚本、适配器与模型相关工具按模块分类存放。

---

## 许可证

本仓库遵循原项目的许可证政策（如有变更将于此处更新）。请在商用或分发前确认模型及数据集的独立许可证要求。