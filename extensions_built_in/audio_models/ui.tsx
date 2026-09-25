// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import type { ModelArch } from "@/app/jobs/new/options";
import {
  defaultSampleConfig,
  defaultAudioSampleConfig,
  defaultYue2SampleConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "ace_step_15_xl",
    label: "ACE-Step 1.5 XL",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample": [
        defaultAudioSampleConfig,
        defaultSampleConfig,
      ],
    },
    sampleTags: {
      CAPTION: {
        title: "音频提示词",
        type: "text",
        full: true,
      },
      LYRICS: {
        title: "歌词",
        type: "multiline",
        full: true,
      },
      BPM: {
        title: "BPM",
        type: "number",
      },
      KEYSCALE: {
        title: "调性",
        type: "text",
      },
      TIMESIGNATURE: {
        title: "拍号",
        type: "text",
      },
      DURATION: {
        title: "时长（秒）",
        type: "number",
      },
      LANGUAGE: {
        title: "语言",
        type: "text",
      },
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
  {
    name: "yue2",
    label: "YuE2",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Comfy-Org/Yue2/checkpoints/yue2_3b_int8_convrot.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.low_vram": [false, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["sigmoid", "sigmoid"],
      // the int8 repack ships convrot8 layers; requesting convrot8 keeps them as-is (no requantization)
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].sample": [
        defaultYue2SampleConfig,
        defaultSampleConfig,
      ],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, true],
      // audio has no resolution; every bucket would duplicate the whole dataset
      "config.process[0].datasets[x].resolution": [[512], [512, 768, 1024]],
      // blank captions break lyric following; the AR must always see the prefix
      "config.process[0].datasets[x].caption_dropout_rate": [0, 0.05],
      "config.process[0].model.model_kwargs": [
        {
          cot: "full",
          abc_dropout: 0.5,
          sample_ar_repetition_penalty: 1.2,
          ar_kl_weight: 0.2,
        },
        {},
      ],
    },
    // native YuE2 prompt: style text, a [Lyrics] line, the lyrics
    hasMultiLinePrompts: true,
    modelNotes: (
      <div className="space-y-2">
        <p className="font-semibold text-amber-400">
          实验性功能。AR（作曲）模型很容易记住训练数据，不适合小数据集。
          少量歌曲就可能让它背下每首歌的完整 Token 序列，之后会失去泛化能力，
          自由生成结果也会偏离训练素材。要让 AR 学到风格而不是歌曲本身，
          通常需要规模更大、内容更多样的数据集。
        </p>
        <p>
          YuE2 在同一骨干上包含两个专家。<b>AR 专家</b>读取风格和歌词，
          将歌曲写成语义编解码 Token 序列（每秒 25 个）；<b>NAR 专家</b>再通过
          流匹配把这些 Token 渲染为音频潜变量。这里训练 LoRA 会同时训练两者：
          AR 对整首歌计算下一 Token 损失，NAR 对随机窗口计算流损失。
        </p>
        <p>
          重点观察 AR 损失 <code>loss/ar_ce</code>。它通常从约 5 开始；若小数据集在
          数百步内降到接近 0，说明模型正在死记。<code>ar_kl_weight</code> 会把 AR
          约束在基础模型附近，避免只记住训练歌曲；降低 <code>ar_lr_multiplier</code>
          或 AR Rank 也能减慢记忆。风格主要来自 NAR，歌词跟随主要来自 AR。
          不要使用打标丢弃，空提示词会破坏歌词跟随。
        </p>
        <p>
          官方音频转 Token 编码器尚未发布。训练使用 Kytra 提供的社区 Tokenizer（
          <a
            href="https://x.com/sin_ceriously"
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 hover:underline"
          >
            @sin_ceriously
          </a>
          ), a MERT-v2-FullSong head that maps real audio to YuE2 codec tokens:{" "}
          <a
            href="https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 hover:underline"
          >
            Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
          </a>
          ）。首次使用时会自动下载。它生成的是模型原生 Token 的近似值，因此即使
          AR 完整复现了一首歌曲，渲染结果也不会与训练音频逐位一致，但通常非常接近。
        </p>
        <p>
          <b>ABC 生成。</b>生成分为两步：AR 先用 ABC 记谱写出整首歌的主谱
          （段落、和弦、人声和器乐旋律），再根据主谱生成编解码 Token。
          也可以用不生成主谱的 off 模式，此时会使用不同指令。
          <code>cot: full</code> 生成和弦与旋律，<code>cot: melody</code> 只生成旋律，
          <code>cot: off</code> 使用单阶段生成。
        </p>
        <p>
          <b>同时训练两种模式。</b>SheetSage2 会把每首歌转录为 ABC 主谱，AR 同时学习
          歌词到主谱、主谱到 Token。<code>abc_dropout</code>（默认 0.5）表示以 off 模式
          跳过主谱的训练样本比例，使同一个 LoRA 兼容两种模式。设为 0 只训练主谱路径，
          设为 1 只训练 off 模式。
        </p>
        <p>
          <b>必须启用缓存。</b>主谱在潜变量缓存阶段生成，并与潜变量、编解码 Token
          一起保存，所以必须开启磁盘潜变量缓存。否则每个训练步都会重新运行 SheetSage2
          （每首约 12 秒）、MERT Tokenizer 和 VAE。缓存会记录生成模式；修改
          <code>cot</code> 后需删除数据集的 <code>_latent_cache</code> 目录重新构建。
          音频没有分辨率概念，每个数据集只保留一个分辨率桶，否则歌曲会被重复加入。
        </p>
        <p>
          提示词格式：先写一行风格，再写 <code>[Lyrics]</code>，然后填写带段落标题的歌词，
          例如 <code>[Verse 1]</code> 和 <code>[Chorus]</code>。Qwen3-Omni 打标器提供
          YuE2 预设，可直接生成这种格式。采样时长由采样区域的“时长”字段控制。
        </p>
      </div>
    ),
    // no separate text encoder: the prompt side is the AR expert, covered by the transformer quantization
    disableSections: ["network.conv", "model.quantize_te"],
    additionalSections: ["model.low_vram", "sample.duration"],
  },
  {
    name: "ace_step_15",
    label: "ACE-Step 1.5",
    group: "audio",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample": [
        defaultAudioSampleConfig,
        defaultSampleConfig,
      ],
    },
    sampleTags: {
      CAPTION: {
        title: "音频提示词",
        type: "text",
        full: true,
      },
      LYRICS: {
        title: "歌词",
        type: "multiline",
        full: true,
      },
      BPM: {
        title: "BPM",
        type: "number",
      },
      KEYSCALE: {
        title: "调性",
        type: "text",
      },
      TIMESIGNATURE: {
        title: "拍号",
        type: "text",
      },
      DURATION: {
        title: "时长（秒）",
        type: "number",
      },
      LANGUAGE: {
        title: "语言",
        type: "text",
      },
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
];
