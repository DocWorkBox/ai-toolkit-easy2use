// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import type { ModelArch } from "@/app/jobs/new/options";
import {
  defaultSampleConfig,
  defaultQwen25OmniSampleConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "qwen25_omni",
    label: "Qwen2.5-Omni",
    group: "llm",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.low_vram": [false, false],
      // the single-file thinker ships convrot8 layers; requesting convrot8 keeps them as-is
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.batch_size": [1, 1],
      "config.process[0].sample": [
        defaultQwen25OmniSampleConfig,
        defaultSampleConfig,
      ],
      // media is encoded on the GPU per step; the cache is optional and large (~54 MB per 300 s of audio)
      "config.process[0].datasets[x].cache_latents_to_disk": [false, true],
      "config.process[0].datasets[x].resolution": [[512], [512, 768, 1024]],
      // the caption is the training target; a blank one trains nothing
      "config.process[0].datasets[x].caption_dropout_rate": [0, 0.05],
      "config.process[0].model.model_kwargs": [
        { instruction: "Describe this in detail." },
        {},
      ],
    },
    disableSections: [
      "network.conv",
      "trigger_word",
      "train.diff_output_preservation",
      "train.blank_prompt_preservation",
      "train.unload_text_encoder",
      "slider",
    ],
    additionalSections: [
      "model.model_kwargs.instruction",
      "sample.ctrl_img",
      "datasets.num_frames",
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Qwen2.5-Omni 7B Thinker 作为文本生成模型使用：输入音频、图片或视频，
          输出文本。每条数据由媒体文件（mp3、wav、flac、ogg、jpg、png、webp、
          mp4 等）及其同名打标文件组成，打标文本就是模型要学习生成的目标。
          同一个数据集可以混合三类媒体。帧数大于 1 时视频按画面帧读取，
          不读取视频音轨。
        </p>
        <p>
          <code>model_kwargs.instruction</code> 是每条训练数据使用的用户指令；
          用训练后的 LoRA 打标时应使用相同措辞。采样时每条提示词对应一个媒体文件，
          生成文本保存为 .txt 文件。
        </p>
        <p>
          每一步都会由冻结的音频和视觉编码器在 GPU 上编码媒体，因此可以关闭磁盘潜变量缓存
          （300 秒音频的缓存约为 54 MB）。批大小固定为 1，LoRA 只训练文本堆栈。
          重点观察 <code>loss/ce</code>，它是打标文本的下一 Token 交叉熵损失。
        </p>
      </div>
    ),
  },
  {
    name: "qwen38",
    label: "Qwen3.8 27B",
    group: "llm",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/poddata/ai-toolkit-models/text_encoders/qwen3.8_27b_w4a8.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.low_vram": [false, false],
      // the ComfyUI file ships W4A8 (int4 codebook weights, int8 activations); requesting it keeps the layers as-is
      "config.process[0].model.qtype": ["convrotw4a8", "qfloat8"],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.batch_size": [1, 1],
      "config.process[0].sample": [
        defaultQwen25OmniSampleConfig,
        defaultSampleConfig,
      ],
      // media is encoded on the GPU per step; the cache is optional
      "config.process[0].datasets[x].cache_latents_to_disk": [false, true],
      "config.process[0].datasets[x].resolution": [[1024], [512, 768, 1024]],
      // the caption is the training target; a blank one trains nothing
      "config.process[0].datasets[x].caption_dropout_rate": [0, 0.05],
      "config.process[0].model.model_kwargs": [
        { instruction: "Describe this in detail." },
        {},
      ],
    },
    disableSections: [
      "network.conv",
      "trigger_word",
      "train.diff_output_preservation",
      "train.blank_prompt_preservation",
      "train.unload_text_encoder",
      "slider",
    ],
    additionalSections: [
      "model.model_kwargs.instruction",
      "sample.ctrl_img",
      "datasets.num_frames",
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Qwen3.8 27B 是输入图片或视频、输出文本的视觉语言模型。每条数据包含一个媒体文件
          （jpg、png、webp、mp4 等）及其同名打标文件，打标内容就是模型要学习生成的目标文本。
          帧数大于 1 时使用视频输入，只读取视频帧，不包含音频。
        </p>
        <p>
          默认使用 ComfyUI W4A8 权重（4 位码本权重、8 位激活，约 17 GB），并保留该量化格式运行。
          生成时使用融合 int4 解码内核，训练时使用 int8 Tensor Core 矩阵乘法。
          生成采用模型的 MTP 草稿头进行推测解码（<code>model_kwargs.mtp_depth</code>，默认 3），
          并使用 CUDA 图加速解码步骤。
        </p>
        <p>
          <code>model_kwargs.instruction</code> 是每条训练数据的用户指令，使用训练后的 LoRA 打标时
          应保持相同措辞。思考模式默认关闭（<code>model_kwargs.enable_thinking</code>）。
          批大小为 1，LoRA 只训练文本部分；<code>loss/ce</code> 表示打标文本的下一个 token 交叉熵损失。
        </p>
      </div>
    ),
  },
];
