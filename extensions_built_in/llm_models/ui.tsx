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
        "/model/HuggingFace/ai-toolkit/Qwen2.5-Omni-7B/qwen2_5_omni_7b_convrot8.safetensors",
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
];
