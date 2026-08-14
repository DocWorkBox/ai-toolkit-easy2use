import React from 'react';
import Link from 'next/link';
import { GroupedSelectOption, SelectOption, JobConfig, ConfigDoc } from '@/types';
import { defaultSliderConfig } from './jobConfig';
import { defaultAudioSampleConfig, defaultSampleConfig, defaultIdeogramSamplesConfig } from '@/helpers/defaultSamples';

type Control = 'depth' | 'line' | 'pose' | 'inpaint';

type DisableableSections =
  | 'model.quantize'
  | 'model.quantize_te'
  | 'train.timestep_type'
  | 'network.conv'
  | 'trigger_word'
  | 'train.diff_output_preservation'
  | 'train.blank_prompt_preservation'
  | 'train.unload_text_encoder'
  | 'slider';

type AdditionalSections =
  | 'datasets.control_path'
  | 'datasets.multi_control_paths'
  | 'datasets.do_i2v'
  | 'datasets.do_audio'
  | 'datasets.audio_normalize'
  | 'datasets.audio_preserve_pitch'
  | 'datasets.auto_frame_count'
  | 'sample.ctrl_img'
  | 'sample.multi_ctrl_imgs'
  | 'train.audio_loss_multiplier'
  | 'datasets.num_frames'
  | 'model.multistage'
  | 'model.layer_offloading'
  | 'model.low_vram'
  | 'model.qie.match_target_res'
  | 'model.assistant_lora_path'
  | 'model.unconditional_lora_path'
  | 'model.model_kwargs.kv_cache'
  | 'ideogram_4_prompt';

type ModelGroup = 'image' | 'instruction' | 'video' | 'experimental' | 'audio';

export interface CustomModelSelectOption {
  label: string;
  options: SelectOption[];
  getValue: (config: JobConfig) => string | undefined;
  onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => void;
  doc?: ConfigDoc;
}

export type SampleTag = {
  title: string;
  type: 'text' | 'multiline' | 'number';
  full?: boolean;
};

export interface SampleTags {
  [key: string]: SampleTag;
}

export interface ModelArch {
  name: string;
  label: string;
  group: ModelGroup;
  controls?: Control[];
  isVideoModel?: boolean;
  hasMultiLinePrompts?: boolean;
  defaults?: { [key: string]: any };
  disableSections?: DisableableSections[];
  additionalSections?: AdditionalSections[];
  accuracyRecoveryAdapters?: { [key: string]: string };
  sampleTags?: SampleTags;
  gateUrl?: string;
  modelNotes?: React.ReactNode;
  customModelSelectOptions?: CustomModelSelectOption[];
}

const defaultNameOrPath = '';
const defaultLinearRank = 32;

// used by the MiniMax-H3 fl2va arch (ref2va is contrastive-guidance only)
const minimaxH3DistillationHandling = {
  label: '蒸馏保持方式',
  options: [
    { value: 'cg', label: '对比引导（默认）' },
    { value: 'ta', label: '训练适配器' },
    { value: 'both', label: '对比引导 + 训练适配器' },
  ],
  getValue: (config: JobConfig) => {
    const assistantLoraPath = config?.config?.process?.[0]?.model?.assistant_lora_path;
    const hasAssistantLoraPath = assistantLoraPath && assistantLoraPath.trim() !== '';
    const hasContrastiveGuidance = config?.config?.process?.[0]?.train?.do_guidance_loss;
    if (hasAssistantLoraPath && hasContrastiveGuidance) {
      return 'both';
    }
    if (hasAssistantLoraPath) {
      return 'ta';
    }
    return 'cg';
  },
  onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => {
    if (value === 'cg') {
      setJobConfig(true, 'config.process[0].train.do_guidance_loss');
      setJobConfig(undefined, 'config.process[0].model.assistant_lora_path');
      if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
        setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
      }
    } else if (value === 'ta') {
      setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
      setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
      setJobConfig(
        './models/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
        'config.process[0].model.assistant_lora_path',
      );
    } else if (value === 'both') {
      setJobConfig(true, 'config.process[0].train.do_guidance_loss');
      setJobConfig(
        './models/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors',
        'config.process[0].model.assistant_lora_path',
      );
      if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
        setJobConfig(3.5, 'config.process[0].train.guidance_loss_target');
      }
    }
  },
  doc: {
    title: 'MiniMax-H3 蒸馏保持方式',
    description: (
      <div>
        MiniMax H3 是经过引导蒸馏的模型，直接训练可能破坏蒸馏效果。训练适配器速度更快，但长时间训练仍可能退化；
        对比引导速度较慢，但稳定性更好。也可以同时启用两者。
      </div>
    ),
  },
};

export const modelArchs: ModelArch[] = [
  {
    name: 'flux',
    label: 'FLUX.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/FLUX.1-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.1-dev',
  },
  {
    name: 'flux_kontext',
    label: 'FLUX.1-Kontext-dev',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/FLUX.1-Kontext-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev',
  },
  {
    name: 'flex1',
    label: 'Flex.1',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Flex.1-alpha', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'flex2',
    label: 'Flex.2',
    group: 'image',
    controls: ['depth', 'line', 'pose', 'inpaint'],
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Flex.2-preview', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.model_kwargs': [
        {
          invert_inpaint_mask_chance: 0.2,
          inpaint_dropout: 0.5,
          control_dropout: 0.5,
          inpaint_random_chance: 0.2,
          do_random_inpainting: true,
          random_blur_mask: true,
          random_dialate_mask: true,
        },
        {},
      ],
      'config.process[0].train.bypass_guidance_embedding': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'chroma',
    label: 'Chroma',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Chroma1-Base/Chroma1-Base.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'zeta_chroma',
    label: 'Zeta Chroma',
    group: 'experimental',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['./models/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'wan21:1b',
    label: 'Wan 2.1 (1.3B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.1-T2V-1.3B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21_i2v:14b480p',
    label: 'Wan 2.1 I2V (14B-480P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.1-I2V-14B-480P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21_i2v:14b',
    label: 'Wan 2.1 I2V (14B-720P)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.1-I2V-14B-720P-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.ctrl_img', 'datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan21:14b',
    label: 'Wan 2.1 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.1-T2V-14B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].datasets[x].fps': [16, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.num_frames', 'model.low_vram', 'datasets.auto_frame_count'],
  },
  {
    name: 'wan22_14b:t2v',
    label: 'Wan 2.2 (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.2-T2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.num_frames',
      'model.low_vram',
      'model.multistage',
      'model.layer_offloading',
      'datasets.auto_frame_count',
    ],
    accuracyRecoveryAdapters: {
      // '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|./models/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_14b_i2v',
    label: 'Wan 2.2 I2V (14B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.2-I2V-A14B-Diffusers-bf16', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [41, 1],
      'config.process[0].sample.fps': [16, 1],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].datasets[x].fps': [16, undefined],
      'config.process[0].model.model_kwargs': [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.low_vram',
      'model.multistage',
      'model.layer_offloading',
      'datasets.auto_frame_count',
    ],
    accuracyRecoveryAdapters: {
      '4 bit with ARA': 'uint4|./models/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors',
    },
  },
  {
    name: 'wan22_5b',
    label: 'Wan 2.2 TI2V (5B)',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Wan2.2-TI2V-5B-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.low_vram',
      'datasets.do_i2v',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'lumina2',
    label: 'Lumina2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Lumina-Image-2.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
    },
    disableSections: ['network.conv'],
  },
  {
    name: 'qwen_image',
    label: 'Qwen-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Qwen-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors',
    },
  },
  {
    name: 'anima',
    label: 'Anima',
    group: 'image',
    defaults: {
      // Use the official model repo while following the upstream Anima preset.
      'config.process[0].model.name_or_path': ['./models/Anima-Base-v1.0-Diffusers', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].model.qtype': ['', 'qfloat8'],
      'config.process[0].model.qtype_te': ['', 'qfloat8'],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].sample.neg': [
        'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, signature, artist name',
        '',
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'qwen_image:2512',
    label: 'Qwen-Image-2512',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Qwen-Image-2512', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
    // Training an ARA now, the other one will not work
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors',
      '4 bit with ARA': 'uint4|./models/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors',
    },
  },
  {
    name: 'qwen_image_edit',
    label: 'Qwen-Image-Edit',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Qwen-Image-Edit', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram', 'model.layer_offloading'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus',
    label: 'Qwen-Image-Edit-2509',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Qwen-Image-Edit-2509', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus:2511',
    label: 'Qwen-Image-Edit-2511',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Qwen-Image-Edit-2511', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors',
    },
  },
  {
    name: 'qwen_image_edit_plus:firered',
    label: 'FireRed-Image-Edit-1.1',
    group: 'instruction',
    defaults: {
      // FireRed-Image-Edit-1.1 uses the Qwen Image Edit Plus compatible pipeline.
      'config.process[0].model.name_or_path': ['./models/FireRed-Image-Edit-1.1', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].train.lr': [0.00002, 0.0001],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].network.linear': [128, 32],
      'config.process[0].network.linear_alpha': [128, 32],
      'config.process[0].sample.width': [512, 1024],
      'config.process[0].sample.height': [512, 1024],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'hidream',
    label: 'HiDream',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/HiDream-I1-Full', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0002, 0.0001],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram'],
    accuracyRecoveryAdapters: {
      '3 bit with ARA': 'uint3|./models/accuracy_recovery_adapters/hidream_i1_full_torchao_uint3.safetensors',
    },
  },
  {
    name: 'hidream_e1',
    label: 'HiDream E1',
    group: 'instruction',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/HiDream-E1-1', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.lr': [0.0001, 0.0001],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.experts', 'ff_i.gate'], []],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img', 'model.low_vram'],
  },
  {
    name: 'sdxl',
    label: 'SDXL',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/stable-diffusion-xl-base-1.0', defaultNameOrPath],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'sd15',
    label: 'SD 1.5',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/stable-diffusion-v1-5', defaultNameOrPath],
      'config.process[0].sample.sampler': ['ddpm', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['ddpm', 'flowmatch'],
      'config.process[0].sample.width': [512, 1024],
      'config.process[0].sample.height': [512, 1024],
      'config.process[0].sample.guidance_scale': [6, 4],
    },
    disableSections: ['model.quantize', 'train.timestep_type'],
  },
  {
    name: 'omnigen2',
    label: 'OmniGen2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/OmniGen2', defaultNameOrPath],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].model.quantize': [false, false],
      'config.process[0].model.quantize_te': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['datasets.control_path', 'sample.ctrl_img'],
  },
  {
    name: 'flux2',
    label: 'FLUX.2',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/FLUX.2-dev', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.2-dev',
  },
  {
    name: 'zimage:turbo',
    label: 'Z-Image Turbo（训练适配器）',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Z-Image-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.assistant_lora_path': [
        './models/zimage_turbo_training_adapter_v2/zimage_turbo_training_adapter_v2.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [9, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'zimage',
    label: 'Z-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Z-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.sample_steps': [30, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'zimage:deturbo',
    label: 'Z-Image De-Turbo (De-Distilled)',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/Z-Image-De-Turbo', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['./models/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample.guidance_scale': [3, 4],
      'config.process[0].sample.sample_steps': [25, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'minimax_h3',
    label: 'MiniMax-H3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/MiniMax-H3', defaultNameOrPath],
      // the Comfy-Org weights are pre-quantized (int8 convrot DiT, nvfp4 TE); these
      // qtypes match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['nvfp4', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].train.do_guidance_loss': [true, undefined],
      'config.process[0].train.guidance_loss_target': [3.5, undefined],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['adaln_proj'], []],
      'config.process[0].sample.num_frames': [107, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [28, 25],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].num_frames': [39, undefined],
      'config.process[0].datasets[x].auto_frame_count': [true, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
      'model.assistant_lora_path',
    ],
    customModelSelectOptions: [minimaxH3DistillationHandling],
    modelNotes: (
      <div className="space-y-2">
        <p>
          权重从设置中的{' '}
          <Link href="/settings" className="text-blue-400 hover:underline">
            模型目录路径
          </Link>{' '}
          加载。首次加载时，缺少的文件会从 <code>Comfy-Org/MiniMax-H3</code> 下载到该目录，总计约 43GB。使用的文件如下：
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
          检查点已经预量化，可直接加载：int8 ConvRot DiT（约 21GB）和 nvfp4 Qwen3-VL 文本编码器（约 16GB）。
          默认量化类型 <code>convrot8</code> / <code>nvfp4</code> 与文件完全匹配，因此加载时不会重新量化。
          选择其他量化类型时，会逐层把预量化权重转换为目标格式。
        </p>
        <p>
          支持文生视频和首帧图生视频（控制图 / I2V 数据集），并可联合训练音频。该模型已做 guidance 蒸馏，
          guidance scale 应保持为 1。视频固定为 24 fps，帧数会向下对齐到 17n+5（5、22、39、56、...、107，
          124 帧约为 5 秒）。图像数据集（num_frames 为 1）按单帧训练，采样时 num_frames 为 1 会输出单张图像。
        </p>
      </div>
    ),
  },
  {
    name: 'minimax_h3_ref2va',
    label: 'MiniMax-H3 Ref2V',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['Comfy-Org/MiniMax-H3', defaultNameOrPath],
      // pre-quantized weights: matching qtypes keep the load unchanged
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['nvfp4', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].train.do_guidance_loss': [true, undefined],
      'config.process[0].train.guidance_loss_target': [3.5, undefined],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['adaln_proj'], []],
      'config.process[0].sample.num_frames': [107, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [28, 25],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['shift', 'sigmoid'],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].num_frames': [39, undefined],
      'config.process[0].datasets[x].auto_frame_count': [true, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.multi_ctrl_imgs',
      'datasets.multi_control_paths',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          参考图生视频：控制图会作为主体或风格参考，而不是首帧。每张参考图保持自身宽高比，并按目标像素面积缩放，
          作为独立参考块加入序列，同时以 <code>&lt;Picture i&gt;</code> 视觉块送入 Qwen3-VL 条件编码器。
          训练时从数据集控制图路径读取，采样时使用采样控制图。目前仅支持图片参考，不支持参考视频或音频。
        </p>
        <p>
          权重与 MiniMax-H3 相同，从设置中的{' '}
          <Link href="/settings" className="text-blue-400 hover:underline">
            模型目录路径
          </Link>
          加载，其中使用 <code>diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors</code> 作为 Ref2V
          分区；文本编码器和 VAE 与 fl2va 架构共用。预量化加载、24 fps、17n+5 帧数规则、guidance scale 1
          和单图模式均与 MiniMax-H3 一致。
        </p>
      </div>
    ),
  },
  {
    name: 'ltx2',
    label: 'LTX-2',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/LTX-2', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'ltx2.3',
    label: 'LTX-2.3',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/LTX-2.3/ltx-2.3-22b-dev.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'ltx2.5',
    label: 'LTX-2.5',
    gateUrl: 'https://huggingface.co/Lightricks/LTX-2.5',
    group: 'video',
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      // comfy-style split files resolve from/download to the models folder;
      // the int8 ConvRot dev transformer is the default
      'config.process[0].model.name_or_path': [
        './models/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors',
        defaultNameOrPath,
      ],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.qtype': ['convrot8', 'qfloat8'],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.qtype_te': ['convrot8', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.num_frames': [121, 1],
      'config.process[0].sample.fps': [24, 1],
      'config.process[0].sample.width': [768, 1024],
      'config.process[0].sample.height': [768, 1024],
      'config.process[0].train.audio_loss_multiplier': [1.0, undefined],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].datasets[x].cache_latents_to_disk': [true, false],
      'config.process[0].datasets[x].do_i2v': [false, undefined],
      'config.process[0].datasets[x].do_audio': [true, undefined],
      'config.process[0].datasets[x].fps': [24, undefined],
      'config.process[0].datasets[x].auto_frame_count': [false, undefined],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'sample.ctrl_img',
      'datasets.num_frames',
      'model.layer_offloading',
      'model.low_vram',
      'datasets.do_audio',
      'datasets.audio_normalize',
      'datasets.audio_preserve_pitch',
      'datasets.do_i2v',
      'train.audio_loss_multiplier',
      'datasets.auto_frame_count',
    ],
  },
  {
    name: 'flux2_klein_4b',
    label: 'FLUX.2-klein-base-4B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/FLUX.2-klein-base-4B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
  {
    name: 'ernie_image',
    label: 'ERNIE-Image',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/ERNIE-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'flux2_klein_9b',
    label: 'FLUX.2-klein-base-9B',
    group: 'image',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/FLUX.2-klein-base-9B', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
    gateUrl: 'https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B',
  },
  {
    name: 'ace_step_15_xl',
    label: 'ACE-Step 1.5 XL',
    group: 'audio',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample': [defaultAudioSampleConfig, defaultSampleConfig],
    },
    sampleTags: {
      CAPTION: {
        title: 'Audio Prompt',
        type: 'text',
        full: true,
      },
      LYRICS: {
        title: 'Lyrics',
        type: 'multiline',
        full: true,
      },
      BPM: {
        title: 'BPM',
        type: 'number',
      },
      KEYSCALE: {
        title: 'Key Scale',
        type: 'text',
      },
      TIMESIGNATURE: {
        title: 'Time Signature',
        type: 'text',
      },
      DURATION: {
        title: 'Duration (sec)',
        type: 'number',
      },
      LANGUAGE: {
        title: 'Language',
        type: 'text',
      },
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.multi_ctrl_imgs', 'model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'ace_step_15',
    label: 'ACE-Step 1.5',
    group: 'audio',
    defaults: {
      // default updates when [selected, unselected] in the UI
      'config.process[0].model.name_or_path': ['./models/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].model.qtype': ['qfloat8', 'qfloat8'],
      'config.process[0].sample': [defaultAudioSampleConfig, defaultSampleConfig],
    },
    sampleTags: {
      CAPTION: {
        title: 'Audio Prompt',
        type: 'text',
        full: true,
      },
      LYRICS: {
        title: 'Lyrics',
        type: 'multiline',
        full: true,
      },
      BPM: {
        title: 'BPM',
        type: 'number',
      },
      KEYSCALE: {
        title: 'Key Scale',
        type: 'text',
      },
      TIMESIGNATURE: {
        title: 'Time Signature',
        type: 'text',
      },
      DURATION: {
        title: 'Duration (sec)',
        type: 'number',
      },
      LANGUAGE: {
        title: 'Language',
        type: 'text',
      },
    },
    disableSections: ['network.conv'],
    additionalSections: ['sample.multi_ctrl_imgs', 'model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'nucleus_image',
    label: 'Nucleus-Image',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Nucleus-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['img_mlp.experts', 'img_mlp.gate'], []],
      'config.process[0].network.linear': [128, defaultLinearRank],
      'config.process[0].network.linear_alpha': [128, defaultLinearRank],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram'],
  },
  {
    name: 'hidream_o1',
    label: 'HiDream-O1',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/HiDream-O1-Image', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].train.max_loss': [1.0, undefined],
      'config.process[0].network.network_kwargs.ignore_if_contains': [['lm_head', 'patch_embed', 'visual'], []],
      'config.process[0].network.transformer_only': [false, undefined],
      'config.process[0].sample.width': [2048, 1024],
      'config.process[0].sample.height': [2048, 1024],
      'config.process[0].model.model_kwargs': [
        {
          noise_scale_inference: 8.0,
          noise_scale: 8.0,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'model.quantize_te', 'train.unload_text_encoder'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'zimage_l2p',
    label: 'Z-Image L2P (pixel space)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/L2P/model-1k-merge.safetensors', defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': ['./models/Z-Image-Turbo', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'ideogram4',
    label: 'Ideogram4',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/ideogram-4-fp8', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample': [defaultIdeogramSamplesConfig, defaultSampleConfig],
      'config.process[0].model.unconditional_lora_path': [
        './models/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors',
        undefined,
      ],
    },
    disableSections: ['network.conv'],
    additionalSections: [
      'model.low_vram',
      'model.layer_offloading',
      'ideogram_4_prompt',
      'model.unconditional_lora_path',
    ],
    hasMultiLinePrompts: true,
    gateUrl: 'https://huggingface.co/ideogram-ai/ideogram-4-fp8',
  },
  {
    name: 'prx_pixel',
    label: 'PRXPixel (pixel space)',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/prxpixel-t2i', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'krea2',
    label: 'Krea 2 Raw',
    group: 'image',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Raw',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'krea2:turbo',
    label: 'Krea 2 Turbo（训练适配器）',
    group: 'image',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Turbo',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        './models/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [9, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading', 'model.assistant_lora_path'],
  },
  {
    name: 'krea2:o_edit',
    label: 'Krea 2 (raw) [Edit Training]',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Raw',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Krea-2-Raw', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
      'model.model_kwargs.kv_cache',
    ],
  },
  {
    name: 'krea2:o_edit_turbo',
    label: 'Krea 2 Turbo (w/ Training Adapter) [Edit Training]',
    gateUrl: 'https://huggingface.co/krea/Krea-2-Turbo',
    group: 'experimental',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Krea-2-Turbo', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.assistant_lora_path': [
        './models/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
        undefined,
      ],
      'config.process[0].sample.guidance_scale': [1, 4],
      'config.process[0].sample.sample_steps': [8, 25],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.assistant_lora_path',
      'model.qie.match_target_res',
      'model.model_kwargs.kv_cache',
    ],
  },
  {
    name: 'mageflow',
    label: 'Mage-Flow',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Mage-Flow-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.guidance_scale': [4, 4],
      'config.process[0].sample.sample_steps': [25, 25],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'mageflow_edit',
    label: 'Mage-Flow Edit',
    group: 'instruction',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Mage-Flow-Edit-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].sample.guidance_scale': [4, 4],
      'config.process[0].sample.sample_steps': [25, 25],
      'config.process[0].train.unload_text_encoder': [false, false],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
    ],
  },
  {
    name: 'boogu_image',
    label: 'Boogu Image',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Boogu-Image-0.1-Base', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
    },
    disableSections: ['network.conv'],
    additionalSections: ['model.low_vram', 'model.layer_offloading'],
  },
  {
    name: 'boogu_image_edit',
    label: 'Boogu Image Edit',
    group: 'instruction',
    defaults: {
      'config.process[0].model.name_or_path': ['./models/Boogu-Image-0.1-Edit', defaultNameOrPath],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [true, false],
      'config.process[0].train.timestep_type': ['linear', 'sigmoid'],
      'config.process[0].network.conv': [undefined, 16],
      'config.process[0].network.conv_alpha': [undefined, 16],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].train.unload_text_encoder': [false, false],
      'config.process[0].model.model_kwargs': [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ['network.conv', 'train.unload_text_encoder'],
    additionalSections: [
      'datasets.multi_control_paths',
      'sample.multi_ctrl_imgs',
      'model.low_vram',
      'model.layer_offloading',
      'model.qie.match_target_res',
    ],
  },
].sort((a, b) => {
  // Sort by label, case-insensitive
  return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedModelOptions: GroupedSelectOption[] = modelArchs.reduce((acc, arch) => {
  const group = acc.find(g => g.label === arch.group);
  if (group) {
    group.options.push({ value: arch.name, label: arch.label });
  } else {
    acc.push({
      label: arch.group,
      options: [{ value: arch.name, label: arch.label }],
    });
  }
  return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
  { value: '', label: '- NONE -' },
  { value: 'qfloat8', label: 'qfloat8 (default)' },
  { value: 'float8', label: 'float8' },
  { value: 'convrot8', label: '8bit convrot' },
  { value: 'convrot4', label: '4bit convrot (nvfp4)' },
  { value: 'nvfp4', label: 'nvfp4 (4bit weight only)' },
  { value: 'convrotint7', label: '7bit convrot' },
  { value: 'convrotint6', label: '6bit convrot' },
  { value: 'convrotint5', label: '5bit convrot' },
  { value: 'convrotint4', label: '4bit convrot' },
  { value: 'convrotint3', label: '3bit convrot' },
  { value: 'convrotint2', label: '2bit convrot' },
  { value: 'convrotbitnet', label: '1.58bit convrot (bitnet)' },
  { value: 'uint7', label: '7 bit' },
  { value: 'uint6', label: '6 bit' },
  { value: 'uint5', label: '5 bit' },
  { value: 'uint4', label: '4 bit' },
  { value: 'uint3', label: '3 bit' },
  { value: 'uint2', label: '2 bit' },
];

export const defaultQtype = 'qfloat8';

interface JobTypeOption extends SelectOption {
  disableSections?: DisableableSections[];
  processSections?: string[];
  onActivate?: (config: JobConfig) => JobConfig;
  onDeactivate?: (config: JobConfig) => JobConfig;
}

export const jobTypeOptions: JobTypeOption[] = [
  {
    value: 'diffusion_trainer',
    label: 'LoRA Trainer',
    disableSections: ['slider'],
  },
  {
    value: 'concept_slider',
    label: 'Concept Slider',
    disableSections: ['trigger_word', 'train.diff_output_preservation'],
    onActivate: (config: JobConfig) => {
      // add default slider config
      config.config.process[0].slider = { ...defaultSliderConfig };
      return config;
    },
    onDeactivate: (config: JobConfig) => {
      // remove slider config
      delete config.config.process[0].slider;
      return config;
    },
  },
];
