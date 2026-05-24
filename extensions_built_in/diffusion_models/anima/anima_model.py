import os
from typing import List, Optional

import numpy as np
import torch
import yaml
from scipy import stats
from diffusers.pipelines.cosmos.pipeline_output import CosmosImagePipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from optimum.quanto import freeze
from tqdm import tqdm

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import get_qtype, quantize


ANIMA_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
HF_TOKEN = os.getenv("HF_TOKEN", None)
ANIMA_BETA_ALPHA = 0.6
ANIMA_BETA_BETA = 0.6
ANIMA_SAMPLING_SHIFT = 1.0
ANIMA_COMPONENT_LOAD_ORDER = (
    "tokenizer",
    "t5_tokenizer",
    "text_encoder",
    "text_conditioner",
    "transformer",
    "vae",
)
ANIMA_TORCH_DTYPE_COMPONENTS = {"text_encoder", "text_conditioner", "transformer", "vae"}

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    return t * alpha / (t * (alpha - 1.0) + 1.0)


def _build_anima_beta_sigmas(
    num_inference_steps: int,
    device: torch.device,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    t = torch.arange(1, num_train_timesteps + 1, dtype=torch.float32, device=device) / float(num_train_timesteps)
    base_sigmas = _time_snr_shift(ANIMA_SAMPLING_SHIFT, t)
    total_timesteps = len(base_sigmas) - 1
    ts = 1.0 - np.linspace(0.0, 1.0, num_inference_steps, endpoint=False)
    mapped = stats.beta.ppf(ts, ANIMA_BETA_ALPHA, ANIMA_BETA_BETA) * float(total_timesteps)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=float(total_timesteps), neginf=0.0)
    indices = np.clip(np.rint(mapped).astype(np.int64), 0, total_timesteps)

    sigmas = []
    last_index = None
    for index in indices:
        if last_index is None or index != last_index:
            sigmas.append(float(base_sigmas[int(index)].item()))
        last_index = int(index)
    sigmas.append(0.0)
    return torch.tensor(sigmas, device=device, dtype=torch.float32)


class _NoSafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, _text):
        return True

    def check_video_safety(self, video):
        return video


class AnimaTextToImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->llm_adapter->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(self, text_encoder, tokenizer, t5_tokenizer, llm_adapter, transformer, vae, scheduler):
        super().__init__()
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            llm_adapter=llm_adapter,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _encode_prompt(self, prompt, device, dtype, max_sequence_length=512):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if all(p.strip() == "" for p in prompts):
            return torch.zeros(
                len(prompts), 512, self.llm_adapter.config.target_dim, device=device, dtype=dtype
            )
        qwen_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        ).to(device)
        qwen_outputs = self.text_encoder(**qwen_inputs)
        qwen_hidden_states = qwen_outputs.last_hidden_state.to(dtype=dtype)
        t5_inputs = self.t5_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        ).to(device)
        adapted_embeds = self.llm_adapter(
            source_hidden_states=qwen_hidden_states,
            target_input_ids=t5_inputs.input_ids,
        )
        if adapted_embeds.shape[1] < 512:
            adapted_embeds = torch.nn.functional.pad(adapted_embeds, (0, 0, 0, 512 - adapted_embeds.shape[1]))
        return adapted_embeds[:, :512]

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompts = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompts) if prompts is not None else prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(prompts, device, dtype, max_sequence_length)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompts = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._encode_prompt(negative_prompts, device, dtype, max_sequence_length)
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, num_frames=1, dtype=None, device=None, generator=None, latents=None):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        height=768,
        width=1360,
        num_inference_steps=35,
        guidance_scale=7.0,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        max_sequence_length=512,
        eta=1.0,
        s_noise=1.0,
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if prompt is not None else prompt_embeds.shape[0]
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        sigmas = _build_anima_beta_sigmas(num_inference_steps, device=device)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height,
            width,
            1,
            torch.float32,
            device,
            generator,
            latents,
        )
        latents = latents * sigmas[0].to(latents.dtype)
        transformer_dtype = self.transformer.dtype
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        with self.progress_bar(total=len(sigmas) - 1) as progress_bar:
            for i in range(len(sigmas) - 1):
                sigma = sigmas[i].to(device=latents.device, dtype=latents.dtype)
                sigma_next = sigmas[i + 1].to(device=latents.device, dtype=latents.dtype)
                timestep = sigma.expand(latents.shape[0]).float()
                latent_model_input = latents.to(transformer_dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0].float()
                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                denoised = latents - sigma.float() * noise_pred
                if sigma_next.item() == 0.0:
                    latents = denoised
                    progress_bar.update()
                    continue

                downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
                sigma_down = sigma_next * downstep_ratio
                alpha_next = 1.0 - sigma_next
                alpha_down = 1.0 - sigma_down
                renoise_sq = sigma_next**2 - sigma_down**2 * alpha_next**2 / (alpha_down**2)
                renoise_coeff = renoise_sq.clamp_min(0).sqrt()
                sigma_down_ratio = sigma_down / sigma
                latents = sigma_down_ratio.to(latents.dtype) * latents + (1.0 - sigma_down_ratio).to(
                    latents.dtype
                ) * denoised
                if eta > 0:
                    noise = randn_tensor(
                        latents.shape,
                        generator=generator,
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    latents = (alpha_next / alpha_down).to(latents.dtype) * latents + noise * s_noise * renoise_coeff.to(
                        latents.dtype
                    )
                progress_bar.update()

        if output_type == "latent":
            image = latents[:, :, 0]
        else:
            if not torch.isfinite(latents).all():
                raise RuntimeError("Anima latents contain NaN/Inf before VAE decode")
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            video = self.vae.decode((latents / latents_std + latents_mean).to(self.vae.dtype), return_dict=False)[0]
            if not torch.isfinite(video).all():
                raise RuntimeError("Anima VAE decode output contains NaN/Inf")
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            image = [batch[0] for batch in video]
            if isinstance(video, torch.Tensor):
                image = torch.stack(image)
            elif isinstance(video, np.ndarray):
                image = np.stack(image)
        if not return_dict:
            return (image,)
        return CosmosImagePipelineOutput(images=image)


class AnimaModel(BaseModel):
    arch = "anima"

    def __init__(
        self,
        device=None,
        model_config: ModelConfig = None,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["CosmosTransformer3DModel"]

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def load_diffusers_pipeline(self, model_path: str, dtype: torch.dtype):
        kwargs = {"dtype": dtype}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        try:
            self.print_and_status_update("Loading Anima official Diffusers pipeline")
            return DiffusionPipeline.from_pretrained(model_path, **kwargs)
        except TypeError as e:
            if "dtype" not in str(e):
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.print_and_status_update("Loading Anima official Diffusers pipeline with torch_dtype")
            return DiffusionPipeline.from_pretrained(model_path, **kwargs)
        except OSError as e:
            if "model_index.json" not in str(e):
                raise
            if not os.path.isfile(os.path.join(model_path, "modular_model_index.json")):
                raise

        self.print_and_status_update("Loading Anima modular Diffusers pipeline fallback")
        try:
            from diffusers import ModularPipeline
        except ImportError as e:
            raise RuntimeError(
                "Anima Diffusers requires a diffusers build with ModularPipeline support. "
                "Install the diffusers Anima PR pinned in requirements_base.txt."
            ) from e

        modular_pipe = ModularPipeline.from_pretrained(model_path, token=HF_TOKEN)
        local_files_only = os.path.isdir(model_path)
        for component_name in ANIMA_COMPONENT_LOAD_ORDER:
            self.print_and_status_update(f"Loading Anima component: {component_name}")
            load_kwargs = {
                "token": HF_TOKEN,
                "local_files_only": local_files_only,
                "pretrained_model_name_or_path": model_path,
            }
            if component_name in ANIMA_TORCH_DTYPE_COMPONENTS:
                load_kwargs["torch_dtype"] = dtype
            try:
                modular_pipe.load_components(component_name, **load_kwargs)
            except TypeError as e:
                if "torch_dtype" not in str(e):
                    raise
                load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
                modular_pipe.load_components(component_name, **load_kwargs)
            if getattr(modular_pipe, component_name, None) is None:
                component_spec = modular_pipe.get_component_spec(component_name)
                self.print_and_status_update(f"Retrying Anima component with direct spec load: {component_name}")
                try:
                    loaded_component = component_spec.load(**load_kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load Anima component `{component_name}`. Component spec: {component_spec}"
                    ) from e
                modular_pipe.update_components(**{component_name: loaded_component})
            self.print_and_status_update(f"Loaded Anima component: {component_name}")
        return modular_pipe

    def _quantize_transformer_blocks(self, transformer):
        quantization_type = get_qtype(self.model_config.qtype)
        all_blocks = []
        for name in self.get_transformer_block_names() or []:
            block_list = getattr(transformer, name, None)
            if block_list is not None:
                all_blocks.extend(list(block_list))

        self.print_and_status_update(f" - quantizing {len(all_blocks)} transformer blocks")
        for block in tqdm(all_blocks):
            block.to(self.device_torch, dtype=self.torch_dtype, non_blocking=True)
            quantize(block, weights=quantization_type)
            freeze(block)
            block.to("cpu", non_blocking=True)

        self.print_and_status_update(" - skipping Anima transformer extras quantization")

    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path or ANIMA_REPO
        self.print_and_status_update("Loading Anima model")

        self.print_and_status_update("Loading Anima Diffusers pipeline")
        diffusers_pipe = self.load_diffusers_pipeline(model_path, dtype)
        transformer = diffusers_pipe.transformer
        text_encoder = diffusers_pipe.text_encoder
        tokenizer = diffusers_pipe.tokenizer
        t5_tokenizer = diffusers_pipe.t5_tokenizer
        llm_adapter = diffusers_pipe.text_conditioner
        vae = diffusers_pipe.vae

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            self._quantize_transformer_blocks(transformer)
            flush()

        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        self.print_and_status_update("Preparing Anima text encoder")
        text_encoder.to(self.device_torch, dtype=dtype)
        llm_adapter.to(self.device_torch, dtype=dtype)

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        if self.model_config.layer_offloading and self.model_config.layer_offloading_text_encoder_percent > 0:
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        self.print_and_status_update("Preparing Anima VAE")
        vae.to(self.device_torch, dtype=dtype)

        self.noise_scheduler = AnimaModel.get_train_scheduler()
        pipe = AnimaTextToImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            llm_adapter=llm_adapter,
            vae=vae,
            transformer=transformer,
        )

        self.vae = vae
        self.text_encoder = [pipe.text_encoder]
        self.tokenizer = [pipe.tokenizer]
        self.t5_tokenizer = pipe.t5_tokenizer
        self.llm_adapter = pipe.llm_adapter
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        pipeline = AnimaTextToImagePipeline(
            scheduler=AnimaModel.get_train_scheduler(),
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            t5_tokenizer=self.t5_tokenizer,
            llm_adapter=unwrap_model(self.llm_adapter),
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )
        return pipeline.to(self.device_torch)

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        images = torch.stack([image.to(device, dtype=dtype) for image in image_list]).unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        return latents.squeeze(2).to(device, dtype=dtype)

    def decode_latents(self, latents, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        latents = latents.to(device, dtype=dtype).unsqueeze(2)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0].squeeze(2)

    @staticmethod
    def _batch_text_embeds(text_embeds):
        if isinstance(text_embeds, (list, tuple)):
            if len(text_embeds) == 0:
                raise ValueError("Anima text embeddings cannot be empty")
            if len(text_embeds) == 1 and text_embeds[0].ndim == 3:
                return text_embeds[0]
            return torch.stack(list(text_embeds), dim=0)
        if text_embeds.ndim == 2:
            return text_embeds.unsqueeze(0)
        return text_embeds

    def generate_single_image(
        self,
        pipeline: AnimaTextToImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        image = pipeline(
            prompt_embeds=self._batch_text_embeds(conditional_embeds.text_embeds),
            negative_prompt_embeds=self._batch_text_embeds(unconditional_embeds.text_embeds),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return image

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        latents_5d = latent_model_input.unsqueeze(2).to(self.device_torch, self.torch_dtype)
        batch_size, _, _, height, width = latents_5d.shape
        timestep = (timestep / 1000).to(self.device_torch, self.torch_dtype)
        if timestep.ndim == 1:
            timestep = timestep.view(batch_size, 1, 1, 1, 1)
        padding_mask = latents_5d.new_zeros(1, 1, height, width)

        pred = self.transformer(
            hidden_states=latents_5d,
            timestep=timestep,
            encoder_hidden_states=self._batch_text_embeds(text_embeddings.text_embeds).to(
                self.device_torch, self.torch_dtype
            ),
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        return pred.squeeze(2)

    def get_prompt_embeds(self, prompt: str) -> AdvancedPromptEmbeds:
        if self.pipeline.text_encoder.device == torch.device("cpu"):
            self.pipeline.text_encoder.to(self.device_torch)
        if self.pipeline.llm_adapter.device == torch.device("cpu"):
            self.pipeline.llm_adapter.to(self.device_torch)
        prompts = [prompt] if isinstance(prompt, str) else prompt
        # Qwen3 tokenizers can return a zero-length sequence for an empty string.
        # Training calls encode_prompt("") for unconditional embeds, so keep that path non-empty.
        prompts = [" " if text is None or not str(text).strip() else text for text in prompts]
        inputs = self.pipeline.tokenizer(
            prompts,
            padding=True,
            max_length=1024,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device_torch)
        outputs = self.pipeline.text_encoder(**inputs, return_dict=True)
        qwen_hidden_states = outputs.last_hidden_state.to(self.device_torch, self.torch_dtype)
        t5_inputs = self.pipeline.t5_tokenizer(
            prompts,
            padding=True,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device_torch)
        embeds = self.pipeline.llm_adapter(
            source_hidden_states=qwen_hidden_states,
            target_input_ids=t5_inputs.input_ids,
        )
        if embeds.shape[1] < 512:
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, 512 - embeds.shape[1]))
        embeds = embeds[:, :512].to(self.device_torch, self.torch_dtype)
        pe = AdvancedPromptEmbeds(text_embeds=[embed for embed in embeds])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "anima"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("lora_transformer_"):
                new_key = "lora_unet_" + new_key[len("lora_transformer_"):]
            new_key = new_key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("lora_unet_"):
                new_key = "lora_transformer_" + new_key[len("lora_unet_"):]
            new_key = new_key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
