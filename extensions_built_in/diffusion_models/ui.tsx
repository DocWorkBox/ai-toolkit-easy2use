// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import Link from "next/link";
import type { ModelArch } from "@/app/jobs/new/options";
import type { JobConfig } from "@/types";
import {
  defaultSampleConfig,
  defaultIdeogramSamplesConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";
const defaultLinearRank = 32;

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: "anima",
    label: "Anima",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/circlestone-labs/Anima-Base-v1.0-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.qtype": ["", "qfloat8"],
      "config.process[0].model.qtype_te": ["", "qfloat8"],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].sample.neg": [
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, signature, artist name",
        "",
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "flux",
    label: "FLUX.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/black-forest-labs/FLUX.1-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
  },
  {
    name: "flux_kontext",
    label: "FLUX.1-Kontext-dev",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/black-forest-labs/FLUX.1-Kontext-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["datasets.control_path", "sample.ctrl_img"],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
  },
  {
    name: "flex1",
    label: "Flex.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/Flex.1-alpha",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.bypass_guidance_embedding": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "chroma",
    label: "Chroma",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "lodestones/Chroma1-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "zeta_chroma",
    label: "Zeta Chroma",
    group: "experimental",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "wan21:1b",
    label: "Wan 2.1 (1.3B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21_i2v:14b480p",
    label: "Wan 2.1 I2V (14B-480P)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21_i2v:14b",
    label: "Wan 2.1 I2V (14B-720P)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21:14b",
    label: "Wan 2.1 (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Wan-AI/Wan2.1-T2V-14B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan22_14b:t2v",
    label: "Wan 2.2 (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
      "config.process[0].model.model_kwargs": [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "model.multistage",
      "model.layer_offloading",
      "datasets.auto_frame_count",
    ],
    accuracyRecoveryAdapters: {
      // '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint3.safetensors',
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
    },
  },
  {
    name: "wan22_14b_i2v",
    label: "Wan 2.2 I2V (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
      "config.process[0].model.model_kwargs": [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "model.multistage",
      "model.layer_offloading",
      "datasets.auto_frame_count",
    ],
    accuracyRecoveryAdapters: {
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors",
    },
  },
  {
    name: "wan22_5b",
    label: "Wan 2.2 TI2V (5B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.do_i2v",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "lumina2",
    label: "Lumina2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Alpha-VLLM/Lumina-Image-2.0",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "qwen_image",
    label: "Qwen-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Qwen/Qwen-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image:2512",
    label: "Qwen-Image-2512",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Qwen/Qwen-Image-2512",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
    // Training an ARA now, the other one will not work
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors",
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors",
    },
  },
  {
    name: "qwen_image_edit",
    label: "Qwen-Image-Edit",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Qwen/Qwen-Image-Edit",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.control_path",
      "sample.ctrl_img",
      "model.low_vram",
      "model.layer_offloading",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_edit_plus",
    label: "Qwen-Image-Edit-2509",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/HuggingFace/Qwen/Qwen-Image-Edit-2509",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_edit_plus:2511",
    label: "Qwen-Image-Edit-2511",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Qwen/Qwen-Image-Edit-2511",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_2",
    label: "Qwen-Image-2.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Comfy-Org/Qwen-Image-2.1",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "/model/ModelScope/Qwen/Qwen-Image-2.1",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // the Comfy-Org weights are pre-quantized int8 convrot; these qtypes
      // match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].sample.guidance_scale": [3.0, 4.0],
      // the VAE is RGBA: images load, encode and decode with their alpha
      "config.process[0].model.model_kwargs": [
        {
          rgba: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    // one model: it edits when the dataset has control paths, and is plain
    // text to image when it does not
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
    customModelSelectOptions: [
      {
        type: "checkbox",
        label: "透明通道（RGBA）",
        getValue: (config: JobConfig) =>
          config?.config?.process?.[0]?.model?.model_kwargs?.rgba ?? false,
        onChange: (
          value: boolean,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value) {
            kwargs.rgba = true;
          } else {
            delete kwargs.rgba;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
        },
        doc: {
          title: "透明通道（RGBA）",
          description: (
            <div className="space-y-2">
              <p>
                此模型的 VAE 原生支持 RGBA。启用后，数据集图片和参考图会读取透明通道，
                VAE 将编码全部四个通道，采样结果也会保存为保留透明度的 PNG。
              </p>
              <p>
                没有透明通道的图片会按完全不透明处理，因此可以混合使用。关闭后将按普通
                RGB 训练和采样：输入时丢弃透明通道，输出时补为完全不透明。
              </p>
              <p>
                切换此选项会重新缓存潜变量，并且不能与使用 <code>alpha_mask</code>
                的数据集同时启用，因为后者会将透明通道用作损失遮罩。
              </p>
            </div>
          ),
        },
      },
    ],
  },
  {
    name: "qwen_image_edit_plus:firered",
    label: "FireRed-Image-Edit-1.1",
    group: "instruction",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/FireRedTeam/FireRed-Image-Edit-1.1",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].train.lr": [0.00002, 0.0001],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].network.linear": [128, 32],
      "config.process[0].network.linear_alpha": [128, 32],
      "config.process[0].sample.width": [512, 1024],
      "config.process[0].sample.height": [512, 1024],
      "config.process[0].model.model_kwargs": [
        { match_target_res: false },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
  },
  {
    name: "hidream",
    label: "HiDream",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-I1-Full",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.lr": [0.0002, 0.0001],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["ff_i.experts", "ff_i.gate"],
        [],
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram"],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/hidream_i1_full_torchao_uint3.safetensors",
    },
  },
  {
    name: "hidream_e1",
    label: "HiDream E1",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-E1-1",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.lr": [0.0001, 0.0001],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["ff_i.experts", "ff_i.gate"],
        [],
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.control_path",
      "sample.ctrl_img",
      "model.low_vram",
    ],
  },
  {
    name: "sdxl",
    label: "SDXL",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "stabilityai/stable-diffusion-xl-base-1.0",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].sample.sampler": ["ddpm", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["ddpm", "flowmatch"],
      "config.process[0].sample.guidance_scale": [6, 4],
    },
    disableSections: ["model.quantize", "train.timestep_type"],
  },
  {
    name: "sd15",
    label: "SD 1.5",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        defaultNameOrPath,
      ],
      "config.process[0].sample.sampler": ["ddpm", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["ddpm", "flowmatch"],
      "config.process[0].sample.width": [512, 1024],
      "config.process[0].sample.height": [512, 1024],
      "config.process[0].sample.guidance_scale": [6, 4],
    },
    disableSections: ["model.quantize", "train.timestep_type"],
  },
  {
    name: "omnigen2",
    label: "OmniGen2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "OmniGen2/OmniGen2",
        defaultNameOrPath,
      ],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["datasets.control_path", "sample.ctrl_img"],
  },
  {
    name: "flux2",
    label: "FLUX.2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/black-forest-labs/FLUX.2-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.2-dev",
  },
  {
    name: "zimage:turbo",
    label: "Z-Image Turbo (w/ Training Adapter)",
    generateNameOverride: "Z-Image Turbo",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Tongyi-MAI/Z-Image-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.assistant_lora_path": [
        "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [9, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
  },
  {
    name: "zimage",
    label: "Z-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Tongyi-MAI/Z-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample.sample_steps": [30, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "zimage:deturbo",
    label: "Z-Image De-Turbo (De-Distilled)",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/zhaotutu12/Z-Image-De-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [25, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "minimax_h3",
    label: "MiniMax-H3",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Comfy-Org/MiniMax-H3",
        defaultNameOrPath,
      ],
      // the Comfy-Org weights are pre-quantized (int8 convrot DiT, nvfp4 TE); these
      // qtypes match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].train.do_guidance_loss": [true, undefined],
      "config.process[0].train.guidance_loss_target": [3.5, undefined],
      "config.process[0].model.assistant_lora_path": [
        "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].network.linear": [16, defaultLinearRank],
      "config.process[0].network.linear_alpha": [16, defaultLinearRank],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["adaln_proj"],
        [],
      ],
      "config.process[0].sample.num_frames": [107, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [28, 25],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].num_frames": [39, undefined],
      "config.process[0].datasets[x].auto_frame_count": [true, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        label: "蒸馏保持方式",
        options: [
          { value: "cg", label: "对比引导" },
          { value: "ta", label: "训练适配器" },
          {
            value: "both",
            label: "对比引导 + 训练适配器（默认）",
          },
          { value: "none", label: "不启用" },
        ],
        getValue: (config: JobConfig) => {
          const assistantLoraPath =
            config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath =
            assistantLoraPath && assistantLoraPath.trim() !== "";
          const hasContrastiveGuidance =
            config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return "both";
          }
          if (hasAssistantLoraPath) {
            return "ta";
          }
          if (hasContrastiveGuidance) {
            return "cg";
          }
          return "none";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          if (value === "cg") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "ta") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
          } else if (value === "both") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "none") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
          }
        },
        doc: {
          title: "MiniMax-H3 蒸馏保持方式",
          description: (
            <div>
              MiniMax H3 是引导蒸馏模型，直接训练会逐渐破坏原有蒸馏能力。
              对比引导和训练适配器都能延缓这种退化：训练适配器速度更快，
              但长时间训练后仍可能失效；对比引导速度较慢，但更不容易失效。
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          权重从设置页中的{" "}
          <Link href="/settings" className="text-blue-400 hover:underline">
            模型目录路径
          </Link>{" "}
          加载。缺失文件会在首次加载时从 <code>Comfy-Org/MiniMax-H3</code>
          下载到该目录（总计约 43GB）。使用以下文件：
        </p>
        <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs overflow-x-auto">
          <code>{`<MODELS_PATH>/
├── diffusion_models/
│   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors`}</code>
        </pre>
        <p>
          这些检查点已经预量化，可直接加载：int8 ConvRot DiT（约 21GB）和 nvfp4
          Qwen3-VL 文本编码器（约 16GB）。默认 qtype（<code>convrot8</code> /{" "}
          <code>nvfp4</code>）与文件完全匹配，因此加载时不会重复量化。
          选择其他量化格式会逐层重新量化已有权重。
        </p>
        <p>
          支持带联合音频的文生视频和首帧图生视频（控制图 / I2V 数据集）。
          模型经过引导蒸馏，引导系数应保持为 1。视频固定为 24 fps，帧数会向下对齐到
          17n+5 网格（5、22、39、56……107、124，约 5 秒）。图片数据集
          （帧数为 1）按单帧训练，采样帧数为 1 时输出单张图片。
        </p>
      </div>
    ),
  },
  {
    name: "minimax_h3_ref2va",
    label: "MiniMax-H3 Ref2V",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Comfy-Org/MiniMax-H3",
        defaultNameOrPath,
      ],
      // pre-quantized weights: matching qtypes keep the load unchanged
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].train.do_guidance_loss": [true, undefined],
      "config.process[0].train.guidance_loss_target": [3.5, undefined],
      "config.process[0].model.assistant_lora_path": [
        "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].network.linear": [16, defaultLinearRank],
      "config.process[0].network.linear_alpha": [16, defaultLinearRank],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["adaln_proj"],
        [],
      ],
      "config.process[0].sample.num_frames": [107, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [28, 25],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].num_frames": [39, undefined],
      "config.process[0].datasets[x].auto_frame_count": [true, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "datasets.multi_control_paths",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        label: "蒸馏保持方式",
        options: [
          { value: "cg", label: "对比引导" },
          { value: "ta", label: "训练适配器" },
          {
            value: "both",
            label: "对比引导 + 训练适配器（默认）",
          },
          { value: "dopsd", label: "D-OPSD" },
          { value: "none", label: "不启用" },
        ],
        getValue: (config: JobConfig) => {
          if (config?.config?.process?.[0]?.model?.model_kwargs?.dopsd) {
            return "dopsd";
          }
          const assistantLoraPath =
            config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath =
            assistantLoraPath && assistantLoraPath.trim() !== "";
          const hasContrastiveGuidance =
            config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return "both";
          }
          if (hasAssistantLoraPath) {
            return "ta";
          }
          if (hasContrastiveGuidance) {
            return "cg";
          }
          return "none";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value === "dopsd") {
            kwargs.dopsd = true;
            kwargs.dopsd_bleed_strength = 1.0;
          } else {
            delete kwargs.dopsd;
            delete kwargs.dopsd_bleed_strength;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
          if (value === "cg") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "ta") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
          } else if (value === "both") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "dopsd" || value === "none") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
          }
        },
        doc: {
          title: "MiniMax-H3 蒸馏保持方式",
          description: (
            <div>
              MiniMax H3 是引导蒸馏模型，直接训练会逐渐破坏原有蒸馏能力。
              训练适配器速度更快，但长时间训练后仍可能退化；对比引导速度较慢，
              但更不容易失效。D-OPSD 使用自蒸馏：无梯度教师前向把训练目标作为自身参考，
              再用教师预测监督无参考前向，将参考信息写入触发词；未设置触发词时则写入打标文本。
              启用后会重新缓存包含像素张量的潜变量。
            </div>
          ),
        },
      },
      {
        label: "参考图呈现方式",
        options: [
          { value: "picture", label: "图片（默认）" },
          { value: "video", label: "静态视频片段" },
        ],
        getValue: (config: JobConfig) => {
          return config?.config?.process?.[0]?.model?.model_kwargs
            ?.image_refs_as_video
            ? "video"
            : "picture";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value === "video") {
            kwargs.image_refs_as_video = true;
          } else {
            delete kwargs.image_refs_as_video;
            delete kwargs.image_ref_video_frames;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
        },
        doc: {
          title: "MiniMax-H3 参考图呈现方式",
          description: (
            <div className="space-y-2">
              <p>
                控制图和采样参考图等静态图片以何种方式呈现给模型。
                视频参考始终使用视频路径。
              </p>
              <p>
                <strong>图片</strong>：使用原生 Ref2VA 流程，以单帧参考块呈现，
                Qwen3-VL 会看到 <code>&lt;Picture i&gt;</code> 块；只允许缩小，不会放大。
              </p>
              <p>
                <strong>静态视频片段</strong>：将图片保持 5 帧（2 个潜变量帧），
                并走完整的视频参考路径，包括按目标像素面积缩放、多帧参考块，以及带时间戳的
                <code>&lt;Video k&gt;</code> 呈现。适合用图片参考训练、但采样时使用视频参考的情况，
                让 LoRA 学到实际使用的路径。每个参考会增加少量序列行。
                可通过 <code>model_kwargs.image_ref_video_frames</code> 调整帧数（17n+5）；
                修改后会重新缓存文本嵌入。
              </p>
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          参考图生视频：控制图片和视频作为主体或风格参考，而不是首帧。
          参考素材保持自身宽高比，并按目标像素面积匹配；图片只缩小不放大，
          同宽高比的视频参考会与目标尺寸完全一致。每个参考都作为独立参考块加入打包序列，
          同时以 <code>&lt;Picture i&gt;</code>（图片）或带时间戳的
          <code>&lt;Video k&gt;</code>（视频）视觉块提供给 Qwen3-VL。
          训练参考来自数据集控制路径，采样参考来自采样控制图。
          “参考图呈现方式”可让静态图片改走短静态视频路径。
        </p>
        <p>
          权重加载方式与 MiniMax-H3 相同，详见该架构说明。文件从{" "}
          <Link href="/settings" className="text-blue-400 hover:underline">
            模型目录路径
          </Link>
          , using{" "}
          <code>
            diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
          </code>{" "}
          ，它是同一版本的 Ref2VA 分区；文本编码器和 VAE 与 FL2VA 架构共用。
          其余行为（预量化加载、24 fps、17n+5 帧网格、引导系数 1、单图模式）
          均与 MiniMax-H3 一致。
        </p>
      </div>
    ),
  },
  {
    name: "ltx2",
    label: "LTX-2",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Lightricks/LTX-2",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "ltx2.3",
    label: "LTX-2.3",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "ltx2.5",
    label: "LTX-2.5",
    gateUrl: "https://huggingface.co/Lightricks/LTX-2.5",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      // comfy-style split files resolve from/download to the models folder;
      // the int8 ConvRot dev transformer is the default
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/Lightricks/LTX-2.5",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "flux2_klein_4b",
    label: "FLUX.2-klein-base-4B",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/black-forest-labs/FLUX.2-klein-base-4B",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
  },
  {
    name: "ernie_image",
    label: "ERNIE-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/PaddlePaddle/ERNIE-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "flux2_klein_9b",
    label: "FLUX.2-klein-base-9B",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/black-forest-labs/FLUX.2-klein-base-9B",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B",
  },
  {
    name: "nucleus_image",
    label: "Nucleus-Image",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "NucleusAI/Nucleus-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["img_mlp.experts", "img_mlp.gate"],
        [],
      ],
      "config.process[0].network.linear": [128, defaultLinearRank],
      "config.process[0].network.linear_alpha": [128, defaultLinearRank],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram"],
  },
  {
    name: "hidream_o1",
    label: "HiDream-O1",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-O1-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].train.max_loss": [1.0, undefined],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["lm_head", "patch_embed", "visual"],
        [],
      ],
      "config.process[0].network.transformer_only": [false, undefined],
      "config.process[0].sample.width": [2048, 1024],
      "config.process[0].sample.height": [2048, 1024],
      "config.process[0].model.model_kwargs": [
        {
          noise_scale_inference: 8.0,
          noise_scale: 8.0,
        },
        {},
      ],
    },
    disableSections: [
      "network.conv",
      "model.quantize_te",
      "train.unload_text_encoder",
    ],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "zimage_l2p",
    label: "Z-Image L2P (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "zhen-nan/L2P/model-1k-merge.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "ideogram4",
    label: "Ideogram4",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/ideogram-ai/ideogram-4-fp8",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample": [
        defaultIdeogramSamplesConfig,
        defaultSampleConfig,
      ],
      "config.process[0].model.unconditional_lora_path": [
        "ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors",
        undefined,
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "ideogram_4_prompt",
      "model.unconditional_lora_path",
    ],
    hasMultiLinePrompts: true,
    gateUrl: "https://huggingface.co/ideogram-ai/ideogram-4-fp8",
  },
  {
    name: "prx_pixel",
    label: "PRXPixel (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Photoroom/prxpixel-t2i",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "krea2",
    label: "Krea 2 Raw",
    group: "image",
    gateUrl: "https://huggingface.co/krea/Krea-2-Raw",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/krea/Krea-2-Raw",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "krea2:turbo",
    label: "Krea 2 Turbo（训练适配器）",
    generateNameOverride: "Krea 2 Turbo",
    group: "image",
    gateUrl: "https://huggingface.co/krea/Krea-2-Turbo",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/krea/Krea-2-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "/model/ModelScope/ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [9, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
  },
  {
    name: "krea2:o_edit",
    label: "Krea 2 Raw（编辑训练）",
    gateUrl: "https://huggingface.co/krea/Krea-2-Raw",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/krea/Krea-2-Raw",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "krea2:o_edit_turbo",
    label: "Krea 2 Turbo（训练适配器，编辑训练）",
    generateNameOverride: "Krea 2 Turbo (Edit)",
    gateUrl: "https://huggingface.co/krea/Krea-2-Turbo",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "/model/ModelScope/krea/Krea-2-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "/model/ModelScope/ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [8, 25],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "mageflow",
    label: "Mage-Flow",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "microsoft/Mage-Flow-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.guidance_scale": [4, 4],
      "config.process[0].sample.sample_steps": [25, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "mageflow_edit",
    label: "Mage-Flow Edit",
    group: "instruction",
    defaults: {
      "config.process[0].model.name_or_path": [
        "microsoft/Mage-Flow-Edit-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.guidance_scale": [4, 4],
      "config.process[0].sample.sample_steps": [25, 25],
      "config.process[0].train.unload_text_encoder": [false, false],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
  {
    name: "boogu_image",
    label: "Boogu Image",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Boogu/Boogu-Image-0.1-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "boogu_image_edit",
    label: "Boogu Image Edit",
    group: "instruction",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Boogu/Boogu-Image-0.1-Edit",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
  },
];
