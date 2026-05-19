# 2026-05-20 开发交接记录：上游同步、自定义 API、Anima 训练支持

这份文档用于换电脑后继续开发。它压缩记录本轮聊天中的开发决策、错误现象、已推送提交、验证命令和后续排查方向。

## 仓库与分支

- 本地仓库：`/mnt/c/ai-toolkit-easy2use`
- 推送目标：`origin HEAD:Dev`
- 上游仓库：`git@github.com:ostris/ai-toolkit.git`
- 当前主要开发分支：`main`
- 远端工作分支：`Dev`
- Windows 环境推送命令：

```bat
cd /d C:\ai-toolkit-easy2use
git -c http.https://github.com.proxy= push origin HEAD:Dev
```

## 已完成的上游同步

本轮按仓库既有“双 merge commit”策略做过多次上游同步：

- 使用 `sync/upstream-*` 分支先合入 `upstream/main`
- 再回到 `main` 用 `--no-ff` 合并同步分支
- README 冲突原则：保留中文 README 为主，补入上游新增模型/能力
- Captioner 冲突原则：保留本地 Remote API 打标工作流、中文状态、成功/失败计数、`newline="\n"` 行为

重要要求：

- 不回退本地 caption API/UI 能力
- 不把中文 README 整体替换成英文上游版本
- 新的上游模型、训练器、采样器能力优先接收

## 自定义 API 打标问题

现象：

```text
Backend should be defined in the BACKENDS_MAPPING. Offending backend: tensorflow_text
Backend should be defined in the BACKENDS_MAPPING. Offending backend: tf
```

结论：

- 这不是自定义 API 打标代码本身的问题。
- 云端镜像里的 `transformers==5.5.3` 曾出现导入级异常。
- 重装/刷新依赖后 `transformers` 可导入，但又暴露 `numpy==2.4.4` 与 `scipy==1.12.0`、`dctorch==0.1.2` 的 ABI/版本冲突。

排查命令：

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
which pip
pip -V
python -m pip show transformers tensorflow-text tensorflow_text diffusers huggingface_hub peft sentencepiece
python -m pip freeze | grep -iE 'tensorflow|tf-keras|keras|tensorflow-text|transformers|diffusers|huggingface-hub|peft|numpy|scipy'
python - <<'PY'
import transformers
print("transformers =", transformers.__version__)
from transformers import T5Tokenizer, T5EncoderModel, UMT5EncoderModel
print("import ok")
PY
```

建议：

- 如果云端镜像用于正式训练，固定依赖版本，不要让 `pip install -U transformers` 顺手升级 `numpy` 到 2.x。
- 对当前项目更稳的是保持 `numpy<2`，除非同时升级所有依赖到支持 NumPy 2 的版本。

## 优化器下拉重复项

曾出现 3 个 Prodigy：

- `Prodigy`
- `Prodigy8Bit`
- `Prodigy 8bit`

处理：

- 删除了本地额外添加的重复项 `Prodigy 8bit`
- 保留上游/现有两个选项

区别：

- `Prodigy`：普通 Prodigy 优化器，不依赖 8bit 优化状态。
- `Prodigy8Bit`：8bit 版本，降低优化器状态显存/内存占用，但数值行为与依赖路径可能不同。

## Anima 模型支持概览

目标模型：

- Hugging Face：`circlestone-labs/Anima`
- UI 模型名：`Anima`
- 架构名：`anima`

核心特点：

- Transformer 基于 Cosmos transformer 结构，但权重键名、hidden size、采样方式与通用 Cosmos2 pipeline 不完全一致。
- 文本侧使用 Qwen3 0.6B base 作为 LLM/text encoder，并额外使用 T5 tokenizer 参与 prompt 编码格式。
- 模型还包含 LLM adapter，用来把 Qwen hidden states 映射到 diffusion transformer 期望的 conditioning 空间。
- VAE 使用 Qwen Image VAE。
- 当前主要目标是 LoRA 训练 transformer，不训练 text encoder。

推荐默认模型路径：

```yaml
model:
  arch: anima
  name_or_path: /datasets/studio/huggingface/models/Anima
  model_paths:
    transformer: /datasets/studio/huggingface/models/Anima/split_files/diffusion_models/anima-base-v1.0.safetensors
    vae: /datasets/studio/huggingface/models/Anima/split_files/vae/qwen_image_vae.safetensors
    llm: /datasets/studio/huggingface/models/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors
    tokenizer: Qwen/Qwen3-0.6B-Base
    t5_tokenizer: google-t5/t5-11b
  model_kwargs:
    llm_adapter_lr: 0
```

离线环境注意：

- `tokenizer` 和 `t5_tokenizer` 如果写 Hugging Face repo id，`transformers` 可能仍会尝试联网查 metadata。
- 断网正式训练建议把它们改成本地 snapshot 目录。
- Qwen tokenizer 常见必要文件：`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`，以及 repo 里 tokenizer 依赖的 config 文件。
- T5 tokenizer 常见必要文件：`spiece.model`、`tokenizer_config.json`、`special_tokens_map.json`。

## Anima 已推送提交

本轮 Anima 相关提交已经推送到 `Dev`：

```text
a6eac0f Add-Anima-training-support
4fbb400 Fix-Anima-pipeline-type-annotation
eb06f16 Use-local-Anima-diffusers-configs
c74039d Match-Anima-transformer-hidden-size
1b63394 Disable-meta-loading-for-Anima-single-file
e389346 Avoid-silent-Anima-tokenizer-download-hangs
0fb0a73 Load-Anima-transformer-directly-from-safetensors
5e55da5 Add-Anima-transformer-loading-progress
ac2214d Resolve-Anima-tokenizers-to-local-snapshots
820d2c9 Convert-Anima-transformer-checkpoint-keys
90e78f6 Disable-Anima-learnable-pos-embed
977f194 Fix-Anima-Qwen3-meta-loading
4a6ba1b Remove-Anima-no-init-weights-dependency
1e6a5cd Load-Anima-LLM-adapter-with-net-prefix
16fee3c Guard-Anima-empty-prompt-encoding
ee601e0 Batch-Anima-advanced-prompt-embeds
9244e8f Skip-Anima-quantization-extras
f7b45ec Tune-Anima-sampling-defaults
d48ada7 Add-Anima-tagged-sample-prompts
9397344 Align-Anima-sampling-with-Cosmos2
3a9e97e Fix-Anima-sampling-sigma-schedule
113ae93 Use-Anima-Z-Image-flowmatch-sampling
```

注意：

- `9397344` 和 `3a9e97e` 是排查中的中间方向，后续 `113ae93` 已把采样方向改回 Anima 参考实现的 Z-Image FlowMatch。
- 如果远端测试仍出黑图/花图，优先基于 `113ae93` 继续排查，不要回退到 Cosmos2 EDM preconditioning。

## Anima 调试过程中的关键错误与结论

### 1. 类型注解错误

错误：

```text
NameError: name 'Cosmos2TextToImagePipeline' is not defined
```

处理：

- 改为 Anima pipeline 类型。

### 2. 单文件加载配置联网/客户端关闭

错误：

```text
Cannot send a request, as the client has been closed.
```

原因：

- `from_single_file` 仍尝试从 Hub 拉 config metadata。

处理：

- 增加本地 diffusers config。
- 后续改为直接从 safetensors 读取权重并映射键名。

### 3. transformer hidden size 不匹配

错误：

```text
expected shape torch.Size([256, 4096]), but got torch.Size([256, 2048])
```

处理：

- 本地 Anima transformer config 调整为匹配 Preview 3 权重的 2048 hidden size。

### 4. meta tensor 加载错误

错误：

```text
Cannot copy out of meta tensor; no data!
```

处理：

- transformer 和 Qwen3 text encoder 避免错误 meta loading 路径。
- 去掉对 `transformers.modeling_utils.no_init_weights` 的依赖，因为当前 `transformers==5.5.3` 不提供该导入路径。

### 5. LLM adapter 权重查找失败

错误：

```text
Anima transformer weights did not contain llm_adapter weights
```

处理：

- 支持带 `net.` 前缀的 adapter 键名。

### 6. 空 prompt 导致 Qwen reshape 错误

错误：

```text
cannot reshape tensor of 0 elements into shape [1, 0, -1, 128]
```

处理：

- 对空 unconditional prompt 做保护，避免送入长度为 0 的 token 序列。

### 7. prompt embeds 传入 list

错误：

```text
'list' object has no attribute 'shape'
```

处理：

- 对 `AdvancedPromptEmbeds.text_embeds` 做 batch stack。

### 8. quanto extras 量化导致 5D latent 断言

错误：

```text
AssertionError: assert activations.ndim in (2, 3)
```

处理：

- Anima transformer 只量化 transformer blocks。
- 跳过会直接接收 5D latent 的 extras，例如 patch embedding/out projection 相关模块。

### 9. 黑图/花图采样

现象：

- 先出现 prompt 明显不匹配。
- 然后改为 Cosmos2 EDM preconditioning 后出现黑图。
- 再调整 sigma schedule 后出现彩色噪声/花图。

结论：

- Anima 不能直接套 Cosmos2 EDM preconditioning。
- 参考 DiffSynth-Studio Anima pipeline，应使用 Z-Image FlowMatch 风格：
  - 初始 latent 为标准高斯噪声，不乘 `sigma_max=80`
  - sigma 从 `1 -> 0`
  - Z-Image shift：`sigmas = 3 * sigmas / (1 + 2 * sigmas)`
  - transformer 输出作为 velocity
  - scheduler step 使用 flow velocity
  - VAE decode 使用 Qwen Image latent mean/std 还原，不除以 `sigma_data`

最新修复提交：

```text
113ae93 Use-Anima-Z-Image-flowmatch-sampling
```

## Anima 采样默认值

当前 UI/defaults 已改为更适合 Anima 的动漫 tag 风格样例：

- `sample_steps`: 35
- `guidance_scale`: 4.5
- 负面提示词：`worst quality, low quality, score_1, score_2, score_3, artist name`
- 默认提示词倾向 tag 化动漫内容，例如 `1girl`、`anime screencap`、`best quality`、`score_9`、`highres` 等。

用户测试过的提示词：

```text
@yaegashi nan, best quality, score_9, highres, year 2025, safe, Teresa of the Faint Smile, long wavy blonde hair, calm silver eyes, a slight mysterious smile on her lips, wearing pristine silver warrior armor. She is walking through a field of wildflowers under a bright sun, her heavy sword resting casually on her shoulder. Soft glowing light, vibrant green and floral colors, graceful and powerful mood.
```

当图像完全不跟 prompt 走时，优先排查采样路径和 prompt embedding，而不是 UI prompt 是否传入。UI 截图底部已经显示 prompt，说明前端展示层拿到了 prompt。

## 当前重点文件

```text
extensions_built_in/diffusion_models/anima/anima_model.py
extensions_built_in/diffusion_models/anima/llm_adapter.py
extensions_built_in/diffusion_models/anima/configs/cosmos_transformer/config.json
extensions_built_in/diffusion_models/anima/configs/wan_vae/vae/config.json
config/examples/train_lora_anima_24gb.yaml
ui/src/app/jobs/new/options.ts
ui/src/types.ts
testing/test_anima_model_support.py
```

## 本地验证命令

Windows 本地：

```powershell
cd C:\ai-toolkit-easy2use
.\.venv\Scripts\python.exe testing\test_anima_model_support.py
.\.venv\Scripts\python.exe -m py_compile extensions_built_in\diffusion_models\anima\anima_model.py
git diff --check
```

Linux 云端快速验证：

```bash
python testing/test_anima_model_support.py
python -m py_compile extensions_built_in/diffusion_models/anima/anima_model.py
```

Linux 云端训练启动前环境检查：

```bash
python - <<'PY'
import torch, transformers, diffusers, huggingface_hub
print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("diffusers =", diffusers.__version__)
print("huggingface_hub =", huggingface_hub.__version__)
PY
```

## 下一步建议

1. 在服务器拉取 `Dev` 最新提交后，用同一份 Anima 配置先设置 `skip_first_sample: false` 跑 baseline sample。
2. 如果仍出花图，优先检查 `anima_model.py` 中传给 transformer 的 `timestep` 是否为 `sigma` 本身，而不是 scheduler 内部 0..1000 timestep。
3. 如果 prompt 仍完全不跟随，打印一次 `prompt_embeds` shape、mean/std、非零比例，确认 Qwen3 + LLM adapter 输出不是空或 NaN。
4. 如果 latent 在 step 后爆炸，打印每一步 `latents.mean/std/min/max` 和 `velocity.mean/std/min/max`。
5. 不建议再回到 Cosmos2 EDM `sigma_max=80` 路线，前面的黑图和花图都说明该路线不匹配 Anima 当前权重。

