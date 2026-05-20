import os
from typing import List, Optional

import huggingface_hub
import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel
from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
from diffusers.pipelines.cosmos.pipeline_output import CosmosImagePipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from optimum.quanto import freeze
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, Qwen3Model, T5TokenizerFast
from tqdm import tqdm

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management import MemoryManager
from toolkit.models.base_model import BaseModel
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.util.quantize import get_qtype, quantize
from .llm_adapter import AnimaLLMAdapter


ANIMA_REPO = "circlestone-labs/Anima"
ANIMA_TRANSFORMER_FILENAME = "anima-base-v1.0.safetensors"
ANIMA_VAE_FILENAME = "qwen_image_vae.safetensors"
ANIMA_LLM_FILENAME = "qwen_3_06b_base.safetensors"
ANIMA_QWEN_CONFIG = "Qwen/Qwen3-0.6B-Base"
ANIMA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
ANIMA_TRANSFORMER_CONFIG = os.path.join(ANIMA_CONFIG_DIR, "cosmos_transformer")
ANIMA_VAE_CONFIG = os.path.join(ANIMA_CONFIG_DIR, "wan_vae")
HF_TOKEN = os.getenv("HF_TOKEN", None)

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
        self._comfy_debug = False
        self._comfy_debug_max_steps = 4

    def _encode_prompt(self, prompt, device, dtype, max_sequence_length=512):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompts = ["" if p is None else str(p) for p in prompts]
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
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        self._comfy_debug = bool(kwargs.pop("comfy_debug", False))
        self._comfy_debug_max_steps = int(kwargs.pop("comfy_debug_max_steps", 4))
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
        if self._comfy_debug:
            print(f"[AnimaDebug] cfg={guidance_scale} steps={num_inference_steps} shape(prompt)={tuple(prompt_embeds.shape)} shape(neg)={tuple(negative_prompt_embeds.shape) if negative_prompt_embeds is not None else None}")
            print(f"[AnimaDebug] prompt mean/std={prompt_embeds.float().mean().item():.6f}/{prompt_embeds.float().std().item():.6f} neg mean/std={negative_prompt_embeds.float().mean().item():.6f}/{negative_prompt_embeds.float().std().item():.6f}")

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1]
        sigmas = 3 * sigmas / (1 + 2 * sigmas)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
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
        transformer_dtype = self.transformer.dtype
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                sigma = self.scheduler.sigmas[i].to(device=latents.device, dtype=latents.dtype)
                timestep = sigma.expand(latents.shape[0]).to(transformer_dtype)
                latent_model_input = latents.to(transformer_dtype)
                if self.do_classifier_free_guidance:
                    latent_model_input_cfg = torch.cat([latent_model_input, latent_model_input], dim=0)
                    timestep_cfg = torch.cat([timestep, timestep], dim=0)
                    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    velocity_cfg = self.transformer(
                        hidden_states=latent_model_input_cfg,
                        timestep=timestep_cfg,
                        encoder_hidden_states=prompt_embeds_cfg,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    velocity_uncond, velocity_cond = velocity_cfg.chunk(2, dim=0)
                    velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)
                else:
                    velocity = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0].float()
                    velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)
                    if self._comfy_debug and i < self._comfy_debug_max_steps:
                        delta = (velocity - velocity_uncond).float()
                        print(
                            f"[AnimaDebug][step={i}] sigma={sigma.item():.6f} "
                            f"latent(mean/std)=({latents.float().mean().item():.6f}/{latents.float().std().item():.6f}) "
                            f"cond(mean/std)=({(velocity_uncond + delta).mean().item():.6f}/{(velocity_uncond + delta).std().item():.6f}) "
                            f"uncond(mean/std)=({velocity_uncond.mean().item():.6f}/{velocity_uncond.std().item():.6f}) "
                            f"delta(mean/std)=({delta.mean().item():.6f}/{delta.std().item():.6f})"
                        )
                if (not self.do_classifier_free_guidance) and self._comfy_debug and i < self._comfy_debug_max_steps:
                    print(
                        f"[AnimaDebug][step={i}] sigma={sigma.item():.6f} "
                        f"latent(mean/std)=({latents.float().mean().item():.6f}/{latents.float().std().item():.6f}) "
                        f"vel(mean/std)=({velocity.mean().item():.6f}/{velocity.std().item():.6f})"
                    )
                latents = self.scheduler.step(velocity, timesteps[i], latents, return_dict=False)[0]
                if self._comfy_debug and i < self._comfy_debug_max_steps:
                    print(f"[AnimaDebug][step={i}] post-latent(mean/std)=({latents.float().mean().item():.6f}/{latents.float().std().item():.6f})")
                progress_bar.update()

        if output_type == "latent":
            image = latents[:, :, 0]
        else:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            video = self.vae.decode((latents / latents_std + latents_mean).to(self.vae.dtype), return_dict=False)[0]
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

    def _resolve_component_path(self, key: str, filename: str) -> str:
        configured = self.model_config.model_paths.get(key)
        if configured:
            if os.path.exists(configured):
                return configured
            if "/" in configured and configured.endswith(".safetensors"):
                parts = configured.split("/")
                try:
                    return huggingface_hub.hf_hub_download(
                        repo_id="/".join(parts[:2]),
                        filename="/".join(parts[2:]),
                        token=HF_TOKEN,
                    )
                except Exception:
                    pass

        model_path = self.model_config.name_or_path or ANIMA_REPO
        if os.path.isdir(model_path):
            candidates = [
                os.path.join(model_path, filename),
                os.path.join(model_path, "split_files", filename),
                os.path.join(model_path, "split_files", "diffusion_models", filename),
                os.path.join(model_path, "split_files", "vae", filename),
                os.path.join(model_path, "split_files", "text_encoders", filename),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
        if os.path.isfile(model_path) and key == "transformer":
            return model_path

        repo_id = model_path if "/" in model_path and not model_path.endswith(".safetensors") else ANIMA_REPO
        remote_candidates = [
            filename,
            f"split_files/diffusion_models/{filename}",
            f"split_files/vae/{filename}",
            f"split_files/text_encoders/{filename}",
        ]
        for remote_name in remote_candidates:
            try:
                return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=remote_name, token=HF_TOKEN)
            except Exception:
                continue
        raise FileNotFoundError(f"Unable to resolve Anima {key} file: {filename}")

    def _load_qwen3_llm(self, llm_path: str, dtype: torch.dtype):
        if os.path.isdir(llm_path):
            return Qwen3Model.from_pretrained(llm_path, torch_dtype=dtype)

        config = AutoConfig.from_pretrained(ANIMA_QWEN_CONFIG)
        text_encoder = Qwen3Model(config)
        state_dict = load_file(llm_path, device="cpu")
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
        for key in state_dict:
            state_dict[key] = state_dict[key].to(dtype)
        missing_keys, unexpected_keys = text_encoder.load_state_dict(state_dict, strict=False, assign=True)
        if missing_keys:
            self.print_and_status_update(f"Anima Qwen3 missing {len(missing_keys)} non-critical keys")
        if unexpected_keys:
            self.print_and_status_update(f"Anima Qwen3 ignored {len(unexpected_keys)} unexpected keys")
        return text_encoder

    def _load_llm_adapter(self, transformer_path: str, dtype: torch.dtype):
        adapter = AnimaLLMAdapter()
        prefixes = [
            "llm_adapter.",
            "net.llm_adapter.",
            "model.llm_adapter.",
            "net.model.llm_adapter.",
            "diffusion_model.llm_adapter.",
            "net.diffusion_model.llm_adapter.",
            "model.diffusion_model.llm_adapter.",
            "net.model.diffusion_model.llm_adapter.",
            "transformer.llm_adapter.",
            "net.transformer.llm_adapter.",
        ]
        state_dict = {}
        adapter_like_keys = []
        with safe_open(transformer_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ("adapter" in key or "llm" in key) and len(adapter_like_keys) < 10:
                    adapter_like_keys.append(key)
                for prefix in prefixes:
                    if key.startswith(prefix):
                        state_dict[key[len(prefix):]] = f.get_tensor(key).to(dtype)
                        break
        if not state_dict:
            hint = ", ".join(adapter_like_keys) if adapter_like_keys else "none"
            raise ValueError(
                f"Anima transformer weights did not contain llm_adapter weights. Adapter-like keys: {hint}"
            )
        adapter.load_state_dict(state_dict, strict=False)
        return adapter

    def _load_transformer(self, transformer_path: str, dtype: torch.dtype):
        self.print_and_status_update("Building Anima transformer module")
        config = CosmosTransformer3DModel.load_config(ANIMA_TRANSFORMER_CONFIG)
        with torch.device("meta"):
            transformer = CosmosTransformer3DModel.from_config(config)

        target_keys = set(transformer.state_dict().keys())
        self.print_and_status_update(f"Reading Anima transformer weights: {transformer_path}")
        checkpoint = load_file(transformer_path, device="cpu")
        self.print_and_status_update("Converting Anima transformer weights to diffusers format")
        checkpoint = convert_cosmos_transformer_checkpoint_to_diffusers(checkpoint)
        state_dict = {
            key: value.to(dtype)
            for key, value in checkpoint.items()
            if key in target_keys
        }
        self.print_and_status_update(f"Loaded Anima transformer weights {len(state_dict)}/{len(target_keys)}")

        missing_keys = sorted(target_keys - set(state_dict.keys()))
        if missing_keys:
            preview = ", ".join(missing_keys[:10])
            raise ValueError(
                f"Anima transformer weights are missing {len(missing_keys)} expected keys. First missing keys: {preview}"
            )

        self.print_and_status_update("Assigning Anima transformer weights")
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        self.print_and_status_update("Anima transformer weights loaded")
        return transformer

    def _load_tokenizer(self, tokenizer_cls, source: str, label: str):
        allow_download = self.model_config.model_kwargs.get("allow_tokenizer_download", False)
        local_files_only = os.path.isdir(self.model_config.name_or_path or "") and not allow_download
        try:
            tokenizer_source = source
            if local_files_only and not os.path.isdir(source):
                tokenizer_source = huggingface_hub.snapshot_download(
                    repo_id=source,
                    local_files_only=True,
                    token=HF_TOKEN,
                )
            return tokenizer_cls.from_pretrained(tokenizer_source, local_files_only=local_files_only)
        except Exception as e:
            if local_files_only:
                raise RuntimeError(
                    f"Unable to load Anima {label} tokenizer from local cache/path: {source}. "
                    "Pre-cache it on the server or set model.model_kwargs.allow_tokenizer_download=true "
                    "to permit Hugging Face downloads during model load."
                ) from e
            raise

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

        self.print_and_status_update("Loading Anima transformer")
        transformer_path = self._resolve_component_path("transformer", ANIMA_TRANSFORMER_FILENAME)
        transformer = self._load_transformer(transformer_path, dtype)

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

        self.print_and_status_update("Loading Anima Qwen3 text encoder")
        llm_path = self._resolve_component_path("llm", ANIMA_LLM_FILENAME)
        tokenizer_source = self.model_config.model_paths.get("tokenizer", ANIMA_QWEN_CONFIG)
        t5_tokenizer_source = self.model_config.model_paths.get("t5_tokenizer", "google-t5/t5-11b")
        self.print_and_status_update("Loading Anima Qwen3 tokenizer")
        tokenizer = self._load_tokenizer(AutoTokenizer, tokenizer_source, "Qwen3")
        self.print_and_status_update("Loading Anima T5 tokenizer")
        t5_tokenizer = self._load_tokenizer(T5TokenizerFast, t5_tokenizer_source, "T5")
        self.print_and_status_update("Loading Anima Qwen3 weights")
        text_encoder = self._load_qwen3_llm(llm_path, dtype)
        text_encoder.to(self.device_torch, dtype=dtype)
        self.print_and_status_update("Loading Anima LLM adapter")
        llm_adapter = self._load_llm_adapter(transformer_path, dtype).to(self.device_torch, dtype=dtype)

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

        self.print_and_status_update("Loading Anima VAE")
        vae_path = self.model_config.vae_path or self._resolve_component_path("vae", ANIMA_VAE_FILENAME)
        vae = AutoencoderKLWan.from_single_file(
            vae_path,
            config=ANIMA_VAE_CONFIG,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=False,
        ).to(self.device_torch, dtype=dtype)

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
        embeds = self.pipeline._encode_prompt(
            prompts,
            device=self.device_torch,
            dtype=self.torch_dtype,
            max_sequence_length=512,
        )
        pe = AdvancedPromptEmbeds(text_embeds=[embed for embed in embeds])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: CosmosTransformer3DModel = unwrap_model(self.model)
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
        return {key.replace("transformer.", "diffusion_model."): value for key, value in state_dict.items()}

    def convert_lora_weights_before_load(self, state_dict):
        return {key.replace("diffusion_model.", "transformer."): value for key, value in state_dict.items()}
