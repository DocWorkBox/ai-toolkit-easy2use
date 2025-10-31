import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    // English localization: training job name
    title: 'Training Name',
    description: (
      <>
        The name of the training job. Used to identify the job and becomes part of the final model filename. Must be
        unique and may only contain letters, numbers, underscores, and dashes; no spaces or special characters.
      </>
    ),
  },
  gpuids: {
    // English localization: GPU selection
    title: 'GPU Index',
    description: (
      <>
        The GPU to use for this job. The UI supports selecting a single GPU per job, but you can run multiple jobs in
        parallel on different GPUs.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    // English localization: trigger word behavior
    title: 'Trigger Word',
    description: (
      <>
        Optional token used to invoke your trained concept or character.
        <br />
        <br />
        When enabled, if a caption in the dataset does not contain the trigger, it will be inserted at the beginning;
        if no caption exists, the trigger alone will be used. To place the trigger elsewhere in a caption, use the
        placeholder <code>{'[trigger]'}</code> which will be replaced by your trigger.
        <br />
        <br />
        The trigger is not added to sampling prompts automatically. Please include it manually or use the same
        placeholder <code>{'[trigger]'}</code>.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    // English localization: model identifier
    title: 'Name or Path',
    description: (
      <>
        A Hugging Face diffusers repository name or a local folder path to a base model. Most models expect a diffusers
        folder; some (e.g., SDXL/SD1) may use a merged <code>safetensors</code> checkpoint path.
      </>
    ),
  },
  'config.process[0].model.arch': {
    // English localization: base model architecture
    title: 'Model Architecture',
    description: (
      <>
        The base model architecture for training. This determines available options and the training pipeline. Image,
        video, and editing models have different architectures.
      </>
    ),
  },
  'model.low_vram': {
    // English localization: low VRAM mode
    title: 'Low VRAM',
    description: (
      <>
        Trains with more conservative VRAM usage. Useful for GPUs with less memory; may reduce speed but improve
        stability.
      </>
    ),
  },
  'model.layer_offloading_transformer_percent': {
    // English localization: transformer offload percent
    title: 'Transformer Offload Percent',
    description: (
      <>
        Percentage of transformer block weights to offload to CPU RAM (0–100%). More offloading reduces VRAM usage but
        can slow training.
      </>
    ),
  },
  'model.layer_offloading_text_encoder_percent': {
    // English localization: text encoder offload percent
    title: 'Text Encoder Offload Percent',
    description: (
      <>
        Percentage of text encoder weights to offload to CPU RAM (0–100%) to further reduce VRAM usage.
      </>
    ),
  },
  'config.process[0].network.type': {
    // English localization: target adapter type
    title: 'Target Type',
    description: (
      <>
        Network adapter type to train, commonly LoRA or LoKr. Different types affect parameter meaning and export
        formats.
      </>
    ),
  },
  'config.process[0].network.lokr_factor': {
    // English localization: LoKr factor
    title: 'LoKr Factor',
    description: (
      <>
        Factor parameter for LoKr affecting decomposition and capacity. Use -1 to auto-select.
      </>
    ),
  },
  'config.process[0].network.linear': {
    // English localization: LoRA linear rank
    title: 'Linear Rank',
    description: (
      <>
        Rank for linear layers in LoRA. Higher values increase capacity and VRAM usage and may make training harder.
      </>
    ),
  },
  'config.process[0].network.conv': {
    // English localization: convolution rank
    title: 'Convolution Rank',
    description: (
      <>
        Rank for the convolution branch controlling low-rank capacity of conv channels. Optional; adjust as needed.
      </>
    ),
  },
  'train.batch_size': {
    // English localization: batch size
    title: 'Batch Size',
    description: (
      <>
        Number of samples per training step. Constrained by VRAM; too large may OOM, too small can reduce stability.
      </>
    ),
  },
  'train.gradient_accumulation': {
    // English localization: gradient accumulation
    title: 'Gradient Accumulation',
    description: (
      <>
        Accumulate gradients over several micro-batches before an optimizer step. Increases effective batch size to
        save VRAM.
      </>
    ),
  },
  'train.steps': {
    // English localization: total training steps
    title: 'Total Steps',
    description: (
      <>
        Total number of training iterations. More steps usually improve quality but take longer to train.
      </>
    ),
  },
  'train.optimizer': {
    // English localization: optimizer selection
    title: 'Optimizer',
    description: (
      <>
        Choose the parameter update algorithm (e.g., AdamW8Bit or Adafactor). Different optimizers affect VRAM usage
        and training stability.
      </>
    ),
  },
  'train.lr': {
    // English localization: learning rate
    title: 'Learning Rate',
    description: (
      <>
        Step size for parameter updates; too large may diverge, too small may converge slowly. Common starting values
        include <code>1e-4</code>.
      </>
    ),
  },
  'train.optimizer_params.weight_decay': {
    // English localization: weight decay (L2 regularization)
    title: 'Weight Decay',
    description: (
      <>
        L2 regularization coefficient to reduce overfitting and improve generalization. Typically tuned alongside the
        optimizer.
      </>
    ),
  },
  'train.timestep_type': {
    // English localization: training timestep distribution
    title: 'Timestep Type',
    description: (
      <>
        Noise timestep distribution (Sigmoid/Linear/Shift/Weighted) that affects sampling strategy and learning focus
        during training.
      </>
    ),
  },
  'train.content_or_style': {
    // English localization: bias toward structure vs detail
    title: 'Timestep Bias',
    description: (
      <>
        Bias training toward shape/structure (high-noise) or details/texture (low-noise). <code>Balanced</code>
        indicates a neutral setting.
      </>
    ),
  },
  'train.loss_type': {
    // English localization: loss function type
    title: 'Loss Type',
    description: (
      <>
        Select the loss function (MSE/MAE/Wavelet/Stepped Recovery). This impacts optimization targets and convergence
        behavior.
      </>
    ),
  },
  'datasets.control_path': {
    // English localization: single control dataset path
    title: 'Control Dataset',
    description: (
      <>
        Filenames in the control dataset must pair one-to-one with the training dataset, forming matched files. During
        training, these are used as control/input images. Control images are automatically resized to match the target
        image dimensions.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    // English localization: multiple control datasets
    title: 'Multiple Control Datasets',
    description: (
      <>
        Filenames in each control dataset must pair with the training dataset and are used as control/input images.
        <br />
        <br />
        Multiple control datasets are applied in the listed order. If the model does not require the same aspect ratio
        as the target (e.g., Qwen/QIE-2509), control images do not need to match the target size or ratio; they will be
        resized to a resolution suitable for the model/target.
      </>
    ),
  },
  'datasets.num_frames': {
    // English localization: number of frames for video datasets
    title: 'Frames',
    description: (
      <>
        For video datasets: compress/extract each video to a fixed number of frames. Set to 1 for image datasets. Pure
        video datasets sample frames uniformly over time.
        <br />
        <br />
        It is recommended to trim videos to an appropriate length before training. For Wan, 16 fps and 81 frames is
        about 5 seconds, so unifying videos to ~5 seconds helps stability.
        <br />
        <br />
        Example: If set to 81, two videos of 2s and 90s in the dataset are both sampled to 81 frames. The 2-second
        video will appear slower; the 90-second video will appear faster.
      </>
    ),
  },
  'datasets.do_i2v': {
    // English localization: enable image-to-video training
    title: 'Enable I2V',
    description: (
      <>
        For models that support both image-to-video (I2V) and text-to-video (T2V), this treats the dataset as I2V: the
        first frame of each video is used as the starting image. When disabled, the dataset is processed as T2V.
      </>
    ),
  },
  'datasets.flip': {
    // English localization: flip augmentation
    title: 'Flip X/Y',
    description: (
      <>
        Data augmentation: flip along the x (horizontal) and/or y (vertical) axis. Flipping a single axis effectively
        increases data (original + flipped). Use with care: upside-down subjects or mirrored faces can harm quality;
        flipping text is usually undesirable.
        <br />
        <br />
        Control images will be flipped in the same manner to align pixel-wise with training images.
      </>
    ),
  },
  'train.unload_text_encoder': {
    // English localization: Unload text encoder to save VRAM
    title: 'Unload Text Encoder',
    description: (
      <>
        When enabled, the text encoder is unloaded from the GPU to reduce VRAM usage. Prompt tokens used for training
        (trigger words or sample prompts) will be cached so training can proceed without the encoder loaded. Captions
        supplied in the dataset are ignored while this option is active.
      </>
    ),
  },
  'train.cache_text_embeddings': {
    // English localization: Cache all text embeddings and unload the encoder
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(Experimental)</small>
        <br />
        Pre-process and cache all embeddings produced by the text encoder to disk, then unload the encoder from the GPU
        to lower VRAM usage. Not compatible with features that modify prompts dynamically (for example trigger words
        insertion or caption dropout).
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
    // English localization: Layer Offloading (RamTorch-based)
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental )
        </span>
      </>
    ),
    description: (
      <>
        Powered by{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        . This feature is early-stage and will change frequently; behavior may differ across versions and it only works
        with certain models.
        <br />
        <br />
        Layer Offloading stores most model weights in CPU RAM instead of GPU VRAM. This enables training larger models
        on GPUs with limited VRAM, provided you have sufficient system memory. Training speed is slower than pure-GPU,
        but RAM is cheaper and upgradable. You still need some GPU VRAM for optimizer state and LoRA parameters, so a
        higher-VRAM GPU remains recommended.
        <br />
        <br />
        You can choose the percentage of layers to offload. For best performance, offload as little as possible (near
        0%). Increase the percentage only when memory is insufficient.
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
    // English localization: DOP concept/category preservation
    title: 'Differential Output Preservation (DOP)',
    description: (
      <>
        Preserves the model’s knowledge of a broader category while learning your specific concept. Requires a trigger
        word to distinguish the concept from its category.
        <br />
        During training, a "prior prediction" step runs with LoRA disabled and the trigger word replaced by the
        category term (e.g., "photo of Alice" → "photo of woman"). In each iteration, an additional loss against this
        prior is applied to help the LoRA retain awareness of the category.
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    // English localization: BPP prior under blank prompt
    title: 'Blank Prompt Preservation (BPP)',
    description: (
      <>
        Helps retain the model’s general knowledge under a blank prompt, improving flexibility and inference quality
        (especially with CFG).
        <br />
        Each iteration includes a prior prediction step with LoRA disabled and an empty prompt, plus an extra blank
        prompt loss to maintain generalization and prevent overfitting to the prompt.
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
