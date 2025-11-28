'use client';
import { useMemo } from 'react';
import {
  modelArchs,
  ModelArch,
  groupedModelOptions,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
} from './options';
import { defaultDatasetConfig } from './jobConfig';
import { GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { objectCopy } from '@/utils/basic';
import { TextInput, SelectInput, Checkbox, FormGroup, NumberInput, SliderInput } from '@/components/formInputs';
import Card from '@/components/Card';
import { X } from 'lucide-react';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { IoFlaskSharp } from 'react-icons/io5';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
};

const isDev = process.env.NODE_ENV === 'development';

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
}: Props) {
  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [jobConfig.config.process[0].model.arch]);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');

  const numTopCards = useMemo(() => {
    let count = 4; // job settings, model config, target config, save config
    if (modelArch?.additionalSections?.includes('model.multistage')) {
      count += 1; // add multistage card
    }
    if (!disableSections.includes('model.quantize')) {
      count += 1; // add quantization card
    }
    if (!disableSections.includes('slider')) {
      count += 1; // add slider card
    }
    return count;
  }, [modelArch, disableSections]);

  let topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-6';

  if (numTopCards == 5) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6';
  }
  if (numTopCards == 6) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-6';
  }

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    return count;
  }, [disableSections]);

  let trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  return (
    <>
      <form onSubmit={handleSubmit} className="space-y-8">
        <div className={topBarClass}>
          <Card title="任务">
            <TextInput
              label="训练名称"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="请输入训练名称"
              disabled={runId !== null}
              required
            />
            <SelectInput
              label="GPU 编号"
              value={`${gpuIDs}`}
              docKey="gpuids"
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
            {disableSections.includes('trigger_word') ? null : (
              <TextInput
                label="触发词"
                value={jobConfig.config.process[0].trigger_word || ''}
                docKey="config.process[0].trigger_word"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].trigger_word');
                }}
                placeholder=""
                required
              />
            )}
          </Card>

          {/* Model Configuration Section */}
          <Card title="模型">
            <SelectInput
              label="模型架构"
              value={jobConfig.config.process[0].model.arch}
              docKey="config.process[0].model.arch"
              onChange={value => {
                handleModelArchChange(jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
              }}
              options={groupedModelOptions}
            />
            <TextInput
              label="名称或路径"
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].model.name_or_path');
              }}
              placeholder=""
              required
            />
            {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
              <TextInput
                label="Training Adapter Path"
                value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                docKey="config.process[0].model.assistant_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.additionalSections?.includes('model.low_vram') && (
              <FormGroup label="选项">
                <Checkbox
                  label="低显存"
                  checked={jobConfig.config.process[0].model.low_vram}
                  onChange={value => setJobConfig(value, 'config.process[0].model.low_vram')}
                  docKey="model.low_vram"
                />
              </FormGroup>
            )}
            {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
              <Checkbox
                label="匹配目标分辨率"
                docKey="model.qie.match_target_res"
                checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.layer_offloading') && (
              <>
                <Checkbox
                  label={
                    <>
                      层级卸载 <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                    </>
                  }
                  checked={jobConfig.config.process[0].model.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading')}
                  docKey="model.layer_offloading"
                />
                {jobConfig.config.process[0].model.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Transformer 卸载百分比"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                      docKey="model.layer_offloading_transformer_percent"
                    />
                    <SliderInput
                      label="文本编码器卸载百分比"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                      docKey="model.layer_offloading_text_encoder_percent"
                    />
                  </div>
                )}
              </>
            )}
          </Card>
          {disableSections.includes('model.quantize') ? null : (
            <Card title="量化">
              <SelectInput
                label="Transformer"
                value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype');
                }}
                options={transformerQuantizationOptions}
              />
              <SelectInput
                label="文本编码器"
                value={jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize_te');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize_te');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype_te');
                }}
                options={quantizationOptions}
              />
            </Card>
          )}
          {modelArch?.additionalSections?.includes('model.multistage') && (
            <Card title="多阶段">
              <FormGroup label="训练阶段" docKey={'model.multistage'}>
                <Checkbox
                  label="高噪声"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                />
                <Checkbox
                  label="低噪声"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                />
              </FormGroup>
              <NumberInput
                label="切换间隔(步数)"
                value={jobConfig.config.process[0].train.switch_boundary_every}
                onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                placeholder="eg. 1"
                docKey={'train.switch_boundary_every'}
                min={1}
                required
              />
            </Card>
          )}
          <Card title="目标">
            <SelectInput
              label="目标类型"
              value={jobConfig.config.process[0].network?.type ?? 'lora'}
              docKey="config.process[0].network.type"
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={[
                { value: 'lora', label: 'LoRA' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {jobConfig.config.process[0].network?.type == 'lokr' && (
              <SelectInput
                label="LoKr 因子"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                docKey="config.process[0].network.lokr_factor"
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {jobConfig.config.process[0].network?.type == 'lora' && (
              <>
                <NumberInput
                  label="线性秩"
                  value={jobConfig.config.process[0].network.linear}
                  onChange={value => {
                    console.log('onChange', value);
                    setJobConfig(value, 'config.process[0].network.linear');
                    setJobConfig(value, 'config.process[0].network.linear_alpha');
                  }}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                  docKey="config.process[0].network.linear"
                />
                {disableSections.includes('network.conv') ? null : (
                  <NumberInput
                    label="卷积秩"
                    value={jobConfig.config.process[0].network.conv}
                    onChange={value => {
                      console.log('onChange', value);
                      setJobConfig(value, 'config.process[0].network.conv');
                      setJobConfig(value, 'config.process[0].network.conv_alpha');
                    }}
                    placeholder="eg. 16"
                    min={0}
                    max={1024}
                    docKey="config.process[0].network.conv"
                  />
                )}
              </>
            )}
          </Card>
          {!disableSections.includes('slider') && (
            <Card title="滑条">
              <TextInput
                label="目标类别"
                className=""
                value={jobConfig.config.process[0].slider?.target_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                placeholder="eg. person"
              />
              <TextInput
                label="正向提示词"
                className=""
                value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                placeholder="eg. person who is happy"
              />
              <TextInput
                label="负向提示词"
                className=""
                value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                placeholder="eg. person who is sad"
              />
              <TextInput
                label="锚点类别"
                className=""
                value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                placeholder=""
              />
            </Card>
          )}
          <Card title="保存">
            <SelectInput
              label="数据类型"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="保存间隔(步)"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="保留的最大保存次数"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
          </Card>
        </div>
        <div>
          <Card title="训练">
            <div className={trainingBarClass}>
              <div>
                <NumberInput
                  label="批大小"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                  docKey="train.batch_size"
                />
                <NumberInput
                  label="梯度累计"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                  docKey="train.gradient_accumulation"
                />
                <NumberInput
                  label="总步数"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                  docKey="train.steps"
                />
              </div>
              <div>
                <SelectInput
                  label="优化器"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'adafactor', label: 'Adafactor' },
                  ]}
                  docKey="train.optimizer"
                />
                <NumberInput
                  label="学习率"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                  docKey="train.lr"
                />
                <NumberInput
                  label="权重衰减"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                  docKey="train.optimizer_params.weight_decay"
                />
              </div>
              <div>
                {disableSections.includes('train.timestep_type') ? null : (
                  <SelectInput
                    label="时间步类型"
                    value={jobConfig.config.process[0].train.timestep_type}
                    disabled={disableSections.includes('train.timestep_type') || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      {
                        value: 'linear',
                        label:
                          'linear ｜ 线性 ｜ 均匀采样噪声强度（0→1） ｜ 稳定、最通用 ｜ 人像、风格微调',
                      },
                      {
                        value: 'weighted',
                        label:
                          'weighted ｜ 加权 ｜ 高噪声区域采样权重更高 ｜ 结构保持强 ｜ 背景、动作较复杂的图像',
                      },
                      {
                        value: 'shift',
                        label:
                          'shift ｜ 偏移 ｜ 对采样分布做偏移，偏向低噪或高噪（根据偏移方向） ｜ 可强化细节或轮廓 ｜ 实验调优用',
                      },
                      {
                        value: 'sigmoid',
                        label:
                          'sigmoid ｜ S 型 ｜ 使用 S 型曲线偏置中段时间步 ｜ 细节与稳定性兼顾 ｜ 高质量模型推荐',
                      },
                    ]}
                    docKey="train.timestep_type"
                  />
                )}
                <SelectInput
                  label="时间步偏向"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    {
                      value: 'balanced',
                      label: 'balanced ｜ 均衡 ｜ 兼顾形体结构与细节纹理（通用推荐）',
                    },
                    {
                      value: 'content',
                      label: 'high noise ｜ 高噪声 ｜ 形体/结构偏向 ｜ 强化形体、轮廓与结构稳定性',
                    },
                    {
                      value: 'style',
                      label: 'low noise ｜ 低噪声 ｜ 细节/纹理偏向 ｜ 提升细节锐度与纹理表现',
                    },
                  ]}
                  docKey="train.content_or_style"
                />
                <SelectInput
                  label="损失类型"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    {
                      value: 'mse',
                      label:
                        'mean squared error (MSE) ｜ 均方误差 ｜ 标准 L2 损失，最常见 ｜ 稳定但偏向模糊 ｜ 默认、安全',
                    },
                    {
                      value: 'mae',
                      label:
                        'mean absolute error (MAE) ｜ 平均绝对误差 ｜ L1 损失，更保守 ｜ 较保细节但慢 ｜ 小数据集或高保真 LoRA',
                    },
                    {
                      value: 'wavelet',
                      label:
                        'wavelet ｜ 小波 ｜ 在小波域计算误差，强调纹理 ｜ 能强化细节和锐度 ｜ 人物皮肤、衣料质感 LoRA',
                    },
                    {
                      value: 'stepped',
                      label:
                        'stepped recovery ｜ 阶梯恢复 ｜ 分阶段权重损失，对高噪更容忍 ｜ 学习快但风险高 ｜ 快速训练',
                    },
                  ]}
                  docKey="train.loss_type"
                />
              </div>
              <div>
                <FormGroup label="EMA（指数移动平均）">
                  <Checkbox
                    label="启用 EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA 衰减率"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config?.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="文本编码器优化" className="pt-2">
                  {!disableSections.includes('train.unload_text_encoder') && (
                    <Checkbox
                      label="卸载文本编码器"
                      checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                      docKey={'train.unload_text_encoder'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                        }
                      }}
                    />
                  )}
                  <Checkbox
                    label="缓存文本嵌入"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey={'train.cache_text_embeddings'}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                </FormGroup>
              </div>
              <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="正则化">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="差异化输出保持"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP 损失倍率"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP 保持类别"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="空提示词保持"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP 损失倍率"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
<Card title="高级设置" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="差分引导"
                  docKey={'train.do_differential_guidance'}
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      // set default differential guidance scale to 3.0
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="差分引导强度"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="数据集">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                  <button
                    type="button"
                    onClick={() =>
                      setJobConfig(
                        jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                        'config.process[0].datasets',
                      )
                    }
                    className="absolute top-2 right-2 bg-red-800 hover:bg-red-700 rounded-full p-1 text-sm transition-colors"
                  >
                    <X />
                  </button>
                  <h2 className="text-lg font-bold mb-4">数据集 {i + 1}</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div>
                      <SelectInput
                        label="目标数据集"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={datasetOptions}
                      />
                      {modelArch?.additionalSections?.includes('datasets.control_path') && (
                        <SelectInput
                          label="控制数据集"
                          docKey="datasets.control_path"
                          value={dataset.control_path ?? ''}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                          }
                          options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                        />
                      )}
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <>
                          <SelectInput
                            label="控制数据集 1"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_1 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_1`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="控制数据集 2"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_2 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_2`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="控制数据集 3"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_3 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_3`,
                              )
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                        </>
                      )}
                      <NumberInput
                        label="LoRA 权重"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                    </div>
                    <div>
                      <TextInput
                        label="默认触发词"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <NumberInput
                        label="全局打标丢弃率"
                        className="pt-2"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                      {modelArch?.additionalSections?.includes('datasets.num_frames') && (
                        <NumberInput
                          label="帧数"
                          className="pt-2"
                          docKey="datasets.num_frames"
                          value={dataset.num_frames}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                          placeholder="eg. 41"
                          min={1}
                          required
                        />
                      )}
                    </div>
                    <div>
                      <FormGroup label="设置" className="">
                        <Checkbox
                          label="缓存潜变量"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                          }
                        />
                        <Checkbox
                          label="作为正则化"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                        {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                          <Checkbox
                            label="启用 I2V"
                            checked={dataset.do_i2v || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                            docKey="datasets.do_i2v"
                          />
                        )}
                      </FormGroup>
                      <FormGroup label="翻转" docKey={'datasets.flip'} className="mt-2">
                        <Checkbox
                          label={
                            <>
                              水平翻转 <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_x || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                        />
                        <Checkbox
                          label={
                            <>
                              垂直翻转 <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                            </>
                          }
                          checked={dataset.flip_y || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                        />
                      </FormGroup>
                    </div>
                    <div>
                      <FormGroup label="分辨率" className="pt-2">
                        <div className="grid grid-cols-2 gap-2">
                          {[
                            [256, 512, 768],
                            [1024, 1280, 1536],
                          ].map(resGroup => (
                            <div key={resGroup[0]} className="space-y-2">
                              {resGroup.map(res => (
                                <Checkbox
                                  key={res}
                                  label={res.toString()}
                                  checked={dataset.resolution.includes(res)}
                                  onChange={value => {
                                    const resolutions = dataset.resolution.includes(res)
                                      ? dataset.resolution.filter(r => r !== res)
                                      : [...dataset.resolution, res];
                                    setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                  }}
                                />
                              ))}
                            </div>
                          ))}
                        </div>
                      </FormGroup>
                    </div>
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const newDataset = objectCopy(defaultDatasetConfig);
                  // automaticallt add the controls for a new dataset
                  const controls = modelArch?.controls ?? [];
                  newDataset.controls = controls;
                  setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                }}
                className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                添加数据集
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="采样">
            <div
              className={
                isVideoModel
                  ? 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6'
                  : 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6'
              }
            >
              <div>
                <NumberInput
                  label="采样间隔(步)"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <SelectInput
                  label="采样器"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={[
                    { value: 'flowmatch', label: 'FlowMatch' },
                    { value: 'ddpm', label: 'DDPM' },
                  ]}
                />
                <NumberInput
                  label="引导强度"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="采样步数"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>
              <div>
                <NumberInput
                  label="宽度"
                  value={jobConfig.config.process[0].sample.width}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                  placeholder="eg. 1024"
                  min={0}
                  required
                />
                <NumberInput
                  label="高度"
                  value={jobConfig.config.process[0].sample.height}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                  placeholder="eg. 1024"
                  className="pt-2"
                  min={0}
                  required
                />
                {isVideoModel && (
                  <div>
                    <NumberInput
                      label="帧数"
                      value={jobConfig.config.process[0].sample.num_frames}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                      placeholder="eg. 0"
                      className="pt-2"
                      min={0}
                      required
                    />
                    <NumberInput
                      label="帧率"
                      value={jobConfig.config.process[0].sample.fps}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                      placeholder="eg. 0"
                      className="pt-2"
                      min={0}
                      required
                    />
                  </div>
                )}
              </div>

              <div>
                <NumberInput
                  label="随机种子"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="种子递增"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="高级采样" className="pt-2">
                  <div>
                    <Checkbox
                      label="跳过首次采样"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="强制首次采样"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="禁用采样"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            <FormGroup label={`采样提示词 (${jobConfig.config.process[0].sample.samples.length})`} className="pt-2">
              <div></div>
            </FormGroup>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        <TextInput
                          label={`提示词`}
                          value={sample.prompt}
                          onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                          placeholder="输入提示词"
                          required
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
                          <TextInput
                            label={`宽度`}
                            value={sample.width ? `${sample.width}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].width;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                } else {
                                  console.warn('Invalid width value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                          />
                          <TextInput
                            label={`高度`}
                            value={sample.height ? `${sample.height}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].height;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                } else {
                                  console.warn('Invalid height value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                          />
                          <TextInput
                            label={`随机种子`}
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                } else {
                                  console.warn('Invalid seed value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          <TextInput
                            label={`LoRA 强度`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />
                        </div>
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="控制图" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'].map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={`添加控制图 ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              添加提示词
            </button>
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <AddSingleImageModal />
    </>
  );
}
