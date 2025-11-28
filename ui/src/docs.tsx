import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: '训练名称',
    description: (
      <>
        训练任务的名称。用于标识任务并作为最终模型文件名的一部分。必须唯一，仅允许字母、数字、下划线和短横线，不允许空格或特殊字符。
      </>
    ),
  },
  gpuids: {
    title: 'GPU 编号',
    description: (
      <>
        训练所使用的 GPU 编号。目前 UI 每个任务仅支持选择一个 GPU，但可以并行启动多个任务分别使用不同的 GPU。
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: '触发词',
    description: (
      <>
        可选：用于触发你训练概念或角色的词或标记。
        <br />
        <br />
        当启用触发词时，如果数据集字幕未包含触发词，系统会自动在字幕开头加入触发词；若无字幕，将仅使用触发词作为字幕。若希望在不同位置使用触发词，可在字幕中使用占位符{' '}
        <code>{'[trigger]'}</code>，系统会自动替换为你的触发词。
        <br />
        <br />
        触发词不会自动加到测试提示词中，请手动添加或同样使用{' '}
        <code>{'[trigger]'}</code> 作为占位符。
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: '名称或路径',
    description: (
      <>
        HuggingFace 上 diffusers 仓库名，或本地基础模型的文件夹路径。多数模型需要使用 diffusers 格式的文件夹；部分模型（如 SDXL、SD1）可填入整合的 safetensors 检查点路径。
      </>
    ),
  },
  'config.process[0].model.arch': {
    title: '模型架构',
    description: (
      <>
        选择训练所用的基础模型架构，会影响可配置项与训练流程。例如图像、视频、编辑模型的架构不同。
      </>
    ),
  },
  'model.low_vram': {
    title: '低显存',
    description: (
      <>
        启用后以更保守的显存占用进行训练，适合显存较小的显卡。可能降低训练速度但提高稳定性。
      </>
    ),
  },
  'model.layer_offloading_transformer_percent': {
    title: 'Transformer 卸载百分比',
    description: (
      <>
        设置将 Transformer 层权重卸载到 CPU 内存的比例（0–100%）。卸载越多显存占用越少，但训练速度可能下降。
      </>
    ),
  },
  'model.layer_offloading_text_encoder_percent': {
    title: '文本编码器卸载百分比',
    description: (
      <>
        设置将文本编码器权重卸载到 CPU 内存的比例（0–100%）。用于进一步降低显存占用。
      </>
    ),
  },
  'config.process[0].network.type': {
    title: '目标类型',
    description: (
      <>
        训练目标网络类型，常见为 LoRA 或 LoKr。不同类型会影响参数含义与导出格式。
      </>
    ),
  },
  'config.process[0].network.lokr_factor': {
    title: 'LoKr 因子',
    description: (
      <>
        LoKr 的因子设置，影响参数分解与容量。-1 表示自动选择。
      </>
    ),
  },
  'config.process[0].network.linear': {
    title: '线性秩',
    description: (
      <>
        LoRA 的线性层秩（rank），数值越大容量越高但显存与训练难度也增加。
      </>
    ),
  },
  'config.process[0].network.conv': {
    title: '卷积秩',
    description: (
      <>
        卷积分支的秩，控制卷积通道的低秩近似容量。可选项，按需调整。
      </>
    ),
  },
  'train.batch_size': {
    title: '批大小',
    description: (
      <>
        每步训练的样本数量。受显存影响，过大可能 OOM，过小训练不稳定。
      </>
    ),
  },
  'train.gradient_accumulation': {
    title: '梯度累计',
    description: (
      <>
        累计多次小批次的梯度再进行一次优化，相当于放大有效批大小以节省显存。
      </>
    ),
  },
  'train.steps': {
    title: '总步数',
    description: (
      <>
        总训练迭代步数。步数越多通常效果更好，但训练时间更长。
      </>
    ),
  },
  'train.optimizer': {
    title: '优化器',
    description: (
      <>
        选择参数更新算法，如 AdamW8Bit 或 Adafactor。不同优化器对显存与稳定性影响不同。
      </>
    ),
  },
  'train.lr': {
    title: '学习率',
    description: (
      <>
        控制参数更新的步幅。过大易发散，过小收敛慢。建议从 1e-4 等常用值起调。
      </>
    ),
  },
  'train.optimizer_params.weight_decay': {
    title: '权重衰减',
    description: (
      <>
        L2 正则项系数，抑制过拟合并提升泛化能力。通常与优化器搭配调整。
      </>
    ),
  },
  'train.timestep_type': {
    title: '时间步类型',
    description: (
      <>
        噪声时间步分布类型（Sigmoid/Linear/Shift/Weighted），影响训练采样策略与学习重点。
      </>
    ),
  },
  'train.content_or_style': {
    title: '时间步偏向',
    description: (
      <>
        训练偏向形体、结构（高噪声）或细节、纹理（低噪声），Balanced 表示均衡。
      </>
    ),
  },
  'train.loss_type': {
    title: '损失类型',
    description: (
      <>
        选择损失函数（MSE/MAE/Wavelet/Stepped Recovery），影响训练优化目标与收敛特性。
      </>
    ),
  },
  'datasets.control_path': {
    title: '控制数据集',
    description: (
      <>
        控制数据集的文件名需要与训练数据集一一对应，形成成对文件。训练时这些图像作为控制/输入图像使用，
        控制图会自动缩放以匹配目标训练图像的尺寸。
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: '多控制数据集',
    description: (
      <>
        控制数据集的文件名需与训练数据集对应为成对文件，训练时作为控制/输入图像。
        <br />
        <br />
        多控制数据集会按列出的顺序依次应用所有控制图。如果模型不要求与目标图像保持相同长宽比（例如 Qwen/QIE-2509），
        则控制图不必与目标图像尺寸或比例一致，系统会自动缩放到更适合该模型/目标的分辨率。
      </>
    ),
  },
  'datasets.num_frames': {
    title: '帧数',
    description: (
      <>
        用于视频数据集：将每个视频压缩/抽取为固定帧数。如果是图像数据集请设为 1。纯视频数据集会按时间均匀抽帧。
        <br />
        <br />
        建议在训练前将视频剪裁到合适长度。以 Wan 为例，默认 16fps，81 帧约等于 5 秒视频，
        因此将视频统一到约 5 秒更利于训练稳定。
        <br />
        <br />
        示例：若设为 81，且数据集中两个视频分别为 2 秒与 90 秒，都会被均匀抽取为 81 帧，
        因此 2 秒视频看起来更慢，90 秒视频看起来更快。
      </>
    ),
  },
  'datasets.do_i2v': {
    title: '启用 I2V',
    description: (
      <>
        对同时支持 I2V（图到视频）与 T2V（文到视频）的模型，此选项将该数据集按 I2V 方式训练：
        会从视频中提取第一帧作为起始图像。未启用时，默认按 T2V 方式处理。
      </>
    ),
  },
  'datasets.flip': {
    title: '水平/垂直翻转',
    description: (
      <>
        可在训练时动态进行数据增强：按 x（水平）/y（垂直）方向翻转。翻转单一轴会有效扩大数据量（原图+翻转图）。
        需谨慎使用：例如人物上下颠倒或人脸左右互换可能破坏效果；文本翻转通常不可取。
        <br />
        <br />
        控制图也会按相同方式翻转以与训练图像逐像素对应。
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: '卸载文本编码器',
    description: (
      <>
        启用后会缓存触发词与示例提示词，并将文本编码器从 GPU 卸载以节省显存。数据集中提供的字幕在训练时会被忽略。
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: '缓存文本嵌入',
    description: (
      <>
        <small>（实验性）</small>
        <br />
        该选项会预处理并将文本编码器生成的所有文本嵌入缓存到磁盘，同时把文本编码器从 GPU 卸载以降低显存占用。
        对会动态改变提示词的功能（例如触发词、字幕丢弃等）不适用。
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        层级卸载{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> 实验性 )
        </span>
      </>
    ),
    description: (
      <>
        该功能基于{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        ，仍处于早期阶段，后续会频繁更新与调整，因此在不同版本间可能表现不一致，并且只适用于部分模型。
        <br />
        <br />
        层级卸载会使用 CPU 内存来承载模型的大部分权重，而不是使用 GPU 显存。这使得在较小显存的显卡上也能训练更大的模型（前提是拥有足够的 CPU 内存）。
        相比纯 GPU 显存训练，这种方式速度更慢，但 CPU 内存更便宜且可升级。仍然需要一定的 GPU 显存来保存优化器状态与 LoRA 权重，通常仍建议使用较大显存的显卡。
        <br />
        <br />
        你可以选择需要卸载的层所占的百分比。一般来说，为了性能更好，建议尽量少卸载（接近 0%）；如果内存不足，可以适当提高卸载比例。
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: '差异化输出保持',
    description: (
      <>
        差异化输出保持（DOP）用于在训练概念的同时保留其所属类别的知识。需要设置触发词以便将概念与类别区分开来。
        训练过程中会进行一次“先验预测”：禁用 LoRA，将提示词中的触发词替换为类别词（例如“photo of Alice”→“photo of woman”）。
        每一步除了正常训练外，还会基于该先验预测与类别提示执行一次额外训练，以帮助 LoRA 保持对类别的认知。
        这能提升概念的表现，并避免模型把同类对象都生成为同一个概念。
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: '空提示词保持',
    description: (
      <>
        空提示词保持（BPP）用于在无提示词时保留模型的已有知识，提高灵活性与推理质量（尤其在使用 CFG 时）。
        训练每一步都会在禁用 LoRA、使用空提示词的情况下进行一次先验预测，并在额外的空提示训练步中将其作为目标，
        以保持模型在无提示时的泛化能力，避免过度拟合提示词。
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: '差分引导',
    description: (
      <>
        差分引导会在训练期间放大模型预测与目标之间的差异，从而创建一个新的目标。差分引导强度将作为差异的乘数。这仍然是实验性功能，
        但在我的测试中，它能让模型训练得更快，并且在所有我尝试过的场景中学习细节的效果更好。
        <br />
        <br />
        核心思想是：普通训练会逐渐接近目标但实际上永远无法完全达到，因为它受到学习率的限制。使用差分引导，我们会将差异放大到超越实际目标的新目标，
        这样可以让模型学会达到或超越目标，而不是达不到目标。
        <br />
        <br />
        <img src="/imgs/diff_guidance_cn_clean.svg" alt="差分引导原理图" className="max-w-full mx-auto rounded-lg shadow-lg" />
      </>
    ),
  },
  'train.differential_guidance_scale': {
    title: '差分引导强度',
    description: (
      <>
        控制差分引导效果的强度系数。数值越大，模型预测与目标的差异放大效果越明显，训练时模型会更积极地学习达到或超越目标。
        <br />
        <br />
        推荐值：3.0（默认）
        <br />
        调整范围：0.5 - 10.0
        <br />
        <br />
        数值过高可能导致训练不稳定，建议从默认值开始，根据训练效果适当调整。
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
