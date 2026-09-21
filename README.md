# 更易用的 AI Toolkit（中文 README）

本项目是 AI Toolkit 的中文与易用性优化版本（`ai-toolkit-easy2use`）。在保留原功能的基础上，聚焦「更易安装、更易上手、更易维护」。本 README 全文为中文，并对安装、运行与 UI 使用进行汉化说明。

> 原项目作者：Ostris；本仓库维护：DocWorkBox。

---

## 项目简介

- 面向扩散模型（Diffusion Models）的训练与推理一体化工具。
- 支持图像、视频、音频模型，以及编辑 / instruction 类模型。
- 提供命令行（CLI）与 Web 用户界面（UI），上手门槛低同时功能完备。
- 本仓库持续跟进上游更新，同时保留中文文档、中文 UI 和更贴近中文用户的使用体验。

## 当前支持概览

- 图像：FLUX.1 / FLUX.2 / FLUX.2 Klein / Qwen-Image / Qwen-Image-2512 / Qwen-Image-2.1 / Z-Image / SDXL / SD1.5 / ERNIE-Image / Nucleus-Image / Krea 2 / Mage-Flow 等
- 编辑：Qwen-Image-Edit / Qwen-Image-Edit-2509 / Qwen-Image-Edit-2511 / HiDream E1 / FireRed-Image-Edit-1.1 / Mage-Flow Edit 预设
- 视频：Wan 2.x / LTX-2 / LTX-2.3 / MiniMax-H3 等
- 音频：ACE-Step 1.5 / ACE-Step 1.5 XL / YuE2
- 多模态文本：Qwen2.5-Omni 训练，以及 Qwen3-Omni / Qwen2.5-Omni / MOSS 音视频打标
- 实验性：Zeta-Chroma 等

## 环境要求

- Python >= 3.10（推荐 3.12）
- Git
- Python 虚拟环境
- Node.js >= 20（运行 Web UI）
- NVIDIA GPU（按训练任务准备足够显存）
## 上游支持模型明细

### 图像
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (FLUX.1)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (FLUX.2)
- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (FLUX.2-klein-base-4B)
- [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (FLUX.2-klein-base-9B)
- [ostris/Flex.1-alpha](https://huggingface.co/ostris/Flex.1-alpha) (Flex.1)
- [ostris/Flex.2-preview](https://huggingface.co/ostris/Flex.2-preview) (Flex.2)
- [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) (Chroma)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) (Lumina2)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Qwen-Image)
- [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Qwen-Image-2512)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1)
- [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) (HiDream I1)
- [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (OmniGen2)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Z-Image Turbo)
- [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Z-Image)
- [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) (Z-Image De-Turbo)
- [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P) (Z-Image L2P)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (SD 1.5)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (ERNIE-Image)
- [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (Nucleus-Image)
- [Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) (Boogu Image 0.1)
- [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) (HiDream O1)
- [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (Ideogram 4 FP8)
- [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) (PRXPixel)
- [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers) (Anima)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo)
- [microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base) (Mage-Flow)

### 指令 / 编辑
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (FLUX.1-Kontext-dev)
- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Qwen-Image-Edit)
- [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (Qwen-Image-Edit-2509)
- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Qwen-Image-Edit-2511)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen-Image-2.1) - 同一模型同时支持文生图与编辑；数据集包含控制图时执行编辑训练
- [HiDream-ai/HiDream-E1-1](https://huggingface.co/HiDream-ai/HiDream-E1-1) (HiDream E1)
- [Boogu/Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) (Boogu Image Edit)
- [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea 2 Edit Training)
- [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) (Krea 2 Turbo Edit Training)
- [microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base) (Mage-Flow Edit)

### 视频
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (Wan 2.1 1.3B)
- [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) (Wan 2.1 I2V 14B-480P)
- [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) (Wan 2.1 I2V 14B-720P)
- [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (Wan 2.1 14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (Wan 2.2 14B)
- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Wan 2.2 I2V 14B)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Wan 2.2 TI2V 5B)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (LTX-2)
- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (LTX-2.3)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (LTX-2.5)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMaxAI/MiniMax-H3)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax-H3 Ref2V) - 参考图生视频

### 音频
- [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (Ace Step 1.5)
- [ACE-Step/acestep-v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base) (Ace Step 1.5 XL)
- [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (YuE2) - 官方音频 token 编码器尚未发布；训练使用 Kytra（[@sin_ceriously](https://x.com/sin_ceriously)）提供的社区 tokenizer：[Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)。

### LLM
- [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (Qwen2.5-Omni)

### 实验性
- [lodestones/Zeta-Chroma](https://huggingface.co/lodestones/Zeta-Chroma) (Zeta Chroma)

### Mac Apple Silicon

上游已加入 Apple Silicon 的实验性支持。若你在 macOS 上使用，可直接尝试：

```bash
chmod +x run_mac.zsh
./run_mac.zsh
```

如果你在 Mac 上遇到兼容性问题，欢迎优先参考上游最新 issue 或在本仓库反馈。

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

### 3）安装 PyTorch（示例：CUDA 12.8）

请根据你的 CUDA / 驱动环境调整版本。以下为当前仓库推荐示例：

```bash
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### 4）安装项目依赖

```bash
pip install -r requirements.txt
```

### 5）DGX OS

DGX OS 设备请参考仓库内的 DGX 说明，并使用：

```bash
pip install -r dgx_requirements.txt
```

## 运行 UI（中文界面）

UI 为基于 Next.js 的 Web 应用。训练任务本身不依赖 UI 持续前台运行，UI 主要用于创建、启动、停止和监控任务。

### 旧版本升级到新版（重要）

如果你是从旧版本直接 `git pull` 到新版本，而不是全新安装，请务必在启动 UI 前同步 Prisma Client 和本地数据库结构。

这是因为上游更新会不定期修改 [`ui/prisma/schema.prisma`](ui/prisma/schema.prisma)。如果代码已经更新，但还没有重新生成 Prisma Client / 执行 `db push`，就可能出现这类问题：

- 页面加载时报 `Application error` 或 client-side exception
- 创建 / 保存任务失败
- 接口报 Prisma 字段不存在，例如 `Unknown argument job_ref`

推荐升级步骤：

```bash
cd ui
npm install
npm run update_db
npm run build
```

其中：

- `npm run update_db` 会执行 `npx prisma generate && npx prisma db push`
- `prisma generate` 用于重建 Prisma Client
- `prisma db push` 用于把本地 SQLite 数据库结构同步到最新 schema

如果你当前已经在运行 UI，请先停止正在运行的 `node` / `next` 进程，再执行上面的命令；否则在 Windows 下可能会遇到 Prisma DLL 被占用、无法更新的问题。

### 开发模式

```bash
cd ui
npm install
npm run dev
```

访问：

- `http://localhost:3000/`
- `http://localhost:3000/dashboard`
- `http://localhost:3000/jobs/new`

### 生产模式

```bash
cd ui
npm run build_and_start
```

默认端口：

- `http://localhost:8675`
- `http://<your-ip>:8675`

### 安全建议

如果你把 UI 暴露在公网，建议设置认证令牌：

```bash
# Linux
AI_TOOLKIT_AUTH=your_token npm run build_and_start

# Windows PowerShell
$env:AI_TOOLKIT_AUTH="your_token"; npm run build_and_start
```

## 近期已并入的重要上游能力

- 新增插件化模型 UI 注册架构，内置模型可通过各扩展目录的 `ui.tsx` 注册训练和生成选项
- 新增独立生成页面、推理引擎、LoRA 浏览 / 上传 / 动态挂载能力
- 新增 YuE2 音乐训练、Qwen2.5-Omni 多模态文本训练和 MOSS 音乐打标支持
- MiniMax-H3 新增 Ref2V、VSA 和参考图呈现方式等训练能力
- 新增 AdamConvRot 优化器，同时保留 Automagic v3、Automagic 实验组和 Singularity
- MiniMax-H3 T2V / I2V 训练支持
- 新增跨平台环境 manager、Windows 无窗口启停和 NVIDIA Spark 构建支持
- 视频 latent 多线程预处理、AV1/PyAV 解码回退和无音轨视频兼容
- 新增 ConvRot/NVFP4/uintx 量化链路，并改进 LoRA 合并往返精度
- Mage-Flow / Mage-Flow Edit 训练支持
- 视频帧顺序解码、视频像素空间 `uint8` 缓存与样本缩略图优化
- Qwen Image 原生增加 `1328` 分辨率支持
- 新增 `ERNIE-Image` 与 `Nucleus-Image` 模型支持
- Flux2 / Flux2 Klein 低显存加载、量化顺序和控制图编码修复
- `compile: true` 真正生效，并修复 compile 时序
- 数据集页支持 duplicate dataset、改进拖拽 / 上传模型、隐藏文件过滤
- 数据集自动 caption、音频数据集 captioning、音频样本下载
- ACE-Step 1.5 / XL 音频模型支持
- `flac` / `ogg` 支持
- Light Mode 支持
- `AdvancedPromptEmbeds` 引入，增强 prompt/embed 兼容能力

## 中文版 UI 截图

![仪表盘（中文）](ui/public/screenshots/dashboard_zh.png)
![新建任务（中文）](ui/public/screenshots/jobs_new_zh.png)

### 手机端 UI 截图

![手机端：适配器页面（1）](ui/public/screenshots/adapter-UI0.png)
![手机端：适配器页面（2）](ui/public/screenshots/adapter-UI1.png)

## Windows 图形启动器

Windows 便携包可使用 `AI Toolkit Launcher.exe` 统一完成环境检查、安装、
依赖修复、程序更新、诊断以及 UI 的启动和停止。启动器只调用仓库内的
`python -m manager`，环境规则仍由当前版本的 manager 维护。

开发者可在仓库根目录执行：

```powershell
.\scripts\build_windows_launcher.ps1
```

默认产物位于 `dist/windows-launcher/<版本>/win-x64/`，为 Windows x64
自包含单文件，不要求用户预装 .NET。向便携包复制产物前，应先同步本次
版本的 `manager/` 代码，避免启动器界面与环境管理协议版本不一致。

部署到便携包时传入其根目录，脚本会同时复制匹配版本的 `manager/` 和
`AI Toolkit Launcher.exe`：

```powershell
.\scripts\build_windows_launcher.ps1 -PortableRoot C:\path\to\ai-toolkit-portable
```

启动器窗口底部会显示当前实际管理的根目录。环境状态来自完整诊断，
不是启动器自身所在机器能否找到任意 Python。

### 便携版更新分支

Windows 便携版代码由 `protable` 分支发布。便携包中的“检查更新”和
“执行更新”会查询并下载该分支的固定提交归档；更新时保留本地
`runtime/`、`models/`、`datasets/`、`output/`、数据库以及 UI 依赖和
构建缓存。启动器 EXE 会先暂存到 `.cache/portable-update/`，待下次启动时
完成自替换，避免覆盖正在运行的程序。

从第一次分支更新开始，更新状态会记录由该分支管理的代码文件。后续版本
删除代码时只清理这些已记录文件，不删除用户自行添加的文件。

便携版的“执行更新”只更新 `protable` 代码，不自动升级 Python、PyTorch
或其他环境依赖。环境差异由“环境诊断”显示，并通过“修复环境”按失败项
处理，避免正常可用的便携环境仅因建议版本变化而下载整套 PyTorch。

## FireRed-Image-Edit-1.1 预设说明

- 训练 UI 已新增 `FireRed-Image-Edit-1.1` 预设，适用于基于该模型的 LoRA 训练起步配置。
- 该预设复用当前仓库中的 `qwen_image_edit_plus` 兼容链路，底层按 `QwenImageEditPlusPipeline` 方式加载。
- 这是一份 ai-toolkit 风格预设，不是 FireRed 原仓 `train_lora.sh` 的逐项复刻；未直接映射的 FireRed 专属训练参数继续沿用 ai-toolkit 现有机制。
- 推荐优先从示例配置 `config/examples/train_lora_firered_image_edit_1_1_32gb.yaml` 启动，再按你的显存和数据集情况微调。

## 常见问题（FAQ）

- 显存不足如何处理？
  - 训练大型模型时，如遇显存限制，可在配置中开启低显存选项（如 `low_vram: true`），或对部分模块量化 / CPU 卸载。
- Windows 安装遇到困难？
  - 优先确认 Python、CUDA、驱动版本匹配；必要时建议使用 WSL 获得更稳定的依赖环境。
- UI 无法访问或接口报错？
  - 请确认 Node.js 版本（>=20）、依赖已安装完成，并检查 `npm run dev` / `npm run build_and_start` 是否正常启动。
- 旧版本升级后页面报错、数据集页打不开、保存任务失败？
  - 很多时候是 Prisma Client 或数据库结构没有同步。请进入 `ui/` 目录后执行：`npm install && npm run update_db && npm run build`。
- 想启用 Hugging Face 高速下载？
  - 可在启动前设置 `HF_HUB_ENABLE_HF_TRANSFER=1`。
- 音频 / 视频数据集读取失败？
  - 请检查 `ffmpeg`、`torchaudio` 与容器格式支持，必要时重新安装依赖并确认系统里可用的解码后端。

## 目录指南（简要）

- `config/`：训练或推理配置示例
- `ui/`：Next.js 中文 UI 源码
- `docs/docker/`：本仓库额外整理的 Docker 说明
- `requirements.txt` / `requirements_base.txt`：Python 依赖
- `dgx_requirements.txt`：DGX OS 专用依赖

## 致谢与说明

- 本仓库以更易用为目标进行中文化与体验优化，基于原 AI Toolkit 项目实现。
- 本仓库欢迎 issue、bug report 和改进建议。
- 自动化生成的 PR 请至少补充清晰的人类说明与验证结果，否则不建议直接合并。

## 许可证

本仓库遵循原项目的许可证政策。请在商用或分发前，额外确认模型、数据集和第三方依赖各自的许可证要求。
