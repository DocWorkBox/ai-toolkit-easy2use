import os
import random
from collections import OrderedDict
from typing import Union, Literal, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from PIL import Image
from torchvision.transforms import functional as TF


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
        
        if self.train_config.blank_prompt_preservation:
            if self.network_config is None:
                raise ValueError("blank_prompt_preservation requires a network to be set")
        
        if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
            self.do_prior_prediction = True
        
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")


    def before_model_load(self):
        pass
    
    def cache_sample_prompts(self):
        if self.train_config.disable_sampling:
            return
        if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
            self.sd.sample_prompts_cache = []
            sample_folder = os.path.join(self.save_root, 'samples')
            output_path = os.path.join(sample_folder, 'test.jpg')
            for i in range(len(self.sample_config.prompts)):
                sample_item = self.sample_config.samples[i]
                prompt = self.sample_config.prompts[i]

                gen_img_config = GenerateImageConfig(
                    prompt=prompt,
                    negative_prompt=sample_item.neg,
                    output_path=output_path,
                    ctrl_img=sample_item.ctrl_img,
                    ctrl_img_1=sample_item.ctrl_img_1,
                    ctrl_img_2=sample_item.ctrl_img_2,
                    ctrl_img_3=sample_item.ctrl_img_3,
                )
                
                has_control_images = False
                if gen_img_config.ctrl_img is not None or gen_img_config.ctrl_img_1 is not None or gen_img_config.ctrl_img_2 is not None or gen_img_config.ctrl_img_3 is not None:
                    has_control_images = True
                if self.sd.encode_control_in_text_embeddings and has_control_images:
                    ctrl_img_list = []
                    if gen_img_config.ctrl_img is not None:
                        ctrl_img = Image.open(gen_img_config.ctrl_img).convert("RGB")
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img)
                    if gen_img_config.ctrl_img_1 is not None:
                        ctrl_img_1 = Image.open(gen_img_config.ctrl_img_1).convert("RGB")
                        ctrl_img_1 = (
                            TF.to_tensor(ctrl_img_1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_1)
                    if gen_img_config.ctrl_img_2 is not None:
                        ctrl_img_2 = Image.open(gen_img_config.ctrl_img_2).convert("RGB")
                        ctrl_img_2 = (
                            TF.to_tensor(ctrl_img_2)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_2)
                    if gen_img_config.ctrl_img_3 is not None:
                        ctrl_img_3 = Image.open(gen_img_config.ctrl_img_3).convert("RGB")
                        ctrl_img_3 = (
                            TF.to_tensor(ctrl_img_3)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_3)
                    if self.sd.has_multiple_control_images:
                        ctrl_img = ctrl_img_list
                    else:
                        ctrl_img = ctrl_img_list[0] if len(ctrl_img_list) > 0 else None
                    positive = self.sd.encode_prompt(
                        gen_img_config.prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                    negative = self.sd.encode_prompt(
                        gen_img_config.negative_prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                else:
                    positive = self.sd.encode_prompt(gen_img_config.prompt).to('cpu')
                    negative = self.sd.encode_prompt(gen_img_config.negative_prompt).to('cpu')
                self.sd.sample_prompts_cache.append({
                    'conditional': positive,
                    'unconditional': negative
                })
        

    def before_dataset_load(self):
        self.assistant_adapter = None
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.is_caching_text_embeddings:
            self.sd.unet.to('cpu')
        
        with torch.no_grad():
            kwargs = {}
            if self.sd.encode_control_in_text_embeddings:
                control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                if self.sd.has_multiple_control_images:
                    control_image = [control_image]
                kwargs['control_images'] = control_image
            self.unconditional_embeds = self.sd.encode_prompt(
                [self.train_config.unconditional_prompt],
                long_prompts=self.do_long_prompts,
                **kwargs
            ).to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache
                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")
                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                self.sd.text_encoder_to(self.device_torch)
                encode_kwargs = {}
                if self.sd.encode_control_in_text_embeddings:
                    control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.sd.has_multiple_control_images:
                        control_image = [control_image]
                    encode_kwargs['control_images'] = control_image
                self.cached_blank_embeds = self.sd.encode_prompt("", **encode_kwargs)
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word, **encode_kwargs)
                if self.train_config.diff_output_preservation:
                    self.diff_output_preservation_embeds = self.sd.encode_prompt(self.train_config.diff_output_preservation_class)
                
                self.cache_sample_prompts()
                
                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                if self.is_caching_text_embeddings:
                    print_acc("Embeddings cached to disk. We dont need the text encoder anymore")
                else:
                    print_acc("This will train only with a blank prompt or trigger word, if set")
                    print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")

                if self.is_caching_text_embeddings:
                    unload_text_encoder(self.sd)
                else:
                    self.sd.text_encoder_to("cpu")
                flush()
        
        if self.train_config.blank_prompt_preservation and self.cached_blank_embeds is None:
            self.cached_blank_embeds = self.sd.encode_prompt("").to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.diffusion_feature_extractor_path is not None:
            vae = self.sd.vae
            self.dfe = load_dfe(self.train_config.diffusion_feature_extractor_path, vae=vae)
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            else:
                self.dfe.eval()
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except:
                    pass


    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()
            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)
                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]
            residual_noise = (noise_chunks[i] * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            denoised_latent = denoised_latent - residual_noise
            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            with torch.no_grad():
                show_tensors(output)

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    def calculate_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor, noisy_latents: torch.Tensor, timesteps: torch.Tensor, batch: 'DataLoaderBatchDTO', mask_multiplier: Union[torch.Tensor, float] = 1.0, prior_pred: Union[torch.Tensor, None] = None, **kwargs):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0
        prior_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)
        has_mask = batch.mask_tensor is not None
        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)
        if self.train_config.match_noise_norm:
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)
        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler
        target = None
        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier
        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)
                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std
                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier
                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust
                        eps = 1e-5
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()
            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 
                    prior_mask_multiplier = 1.0 - prior_mask
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps).detach()
        elif self.sd.is_flow_matching:
            target = (noise - batch.latents).detach()
        else:
            target = noise
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1]
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                pred_features = self.dfe(stepped_latents.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    target_features = self.dfe(bl.float())
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()
            elif self.dfe.version in [2, 3, 4, 5]:
                pass
        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                unconditional_embeds = concat_prompt_embeds([self.unconditional_embeds] * noisy_latents.shape[0],)
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                is_video = len(target.shape) == 5
                alpha = 1.0
                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)
                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)
        if target is None:
            target = noise
        pred = noise_pred
        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)
        ignore_snr = False
        if self.train_config.loss_type == "mae":
            loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
        elif self.train_config.loss_type == "wavelet":
            loss = wavelet_loss(pred, batch.latents, noise)
        elif self.train_config.loss_type == "stepped":
            loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
            loss = loss * 10.0
        else:
            loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        try:
            loss = loss * loss_multiplier
        except:
            pass
        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma, fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)
        loss = loss.mean()
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:
            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None
        if self.train_config.target_norm_std:
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss
        return loss + additional_loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(self, noisy_latents: torch.Tensor, conditional_embeds: PromptEmbeds, match_adapter_assist: bool, network_weight_list: list, timesteps: torch.Tensor, pred_kwargs: dict, batch: 'DataLoaderBatchDTO', noise: torch.Tensor, unconditional_embeds: Optional[PromptEmbeds] = None, **kwargs):
        return get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )
    
    def get_prior_prediction(self, noisy_latents: torch.Tensor, conditional_embeds: PromptEmbeds, match_adapter_assist: bool, network_weight_list: list, timesteps: torch.Tensor, pred_kwargs: dict, batch: 'DataLoaderBatchDTO', noise: torch.Tensor, unconditional_embeds: Optional[PromptEmbeds] = None, conditioned_prompts=None, **kwargs):
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or isinstance(self.adapter, ReferenceAdapter) or (isinstance(self.adapter, CustomAdapter))):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False
        if self.train_config.unload_text_encoder and self.adapter is not None and not isinstance(self.adapter, CustomAdapter):
            raise ValueError("Prior predictions currently do not support unloading text encoder with adapter")
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            embeds_to_use = conditional_embeds.clone().detach()
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''
                triggers = ['[trigger]', '[name]']
                remove_tokens = []
                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name
                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name
                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt
                if batch.prompt_embeds is not None:
                    embeds_to_use = batch.prompt_embeds.clone().to(self.device_torch, dtype=dtype)
                else:
                    prompt_kwargs = {}
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    embeds_to_use = self.sd.encode_prompt(prompt_list, long_prompts=self.do_long_prompts).to(self.device_torch, dtype=dtype, **prompt_kwargs).detach()
            self.sd.unet.eval()
            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux and not self.sd.is_lumina2:
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]
            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch
            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **pred_kwargs
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']
            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def predict_noise(self, noisy_latents: torch.Tensor, timesteps: Union[int, torch.Tensor] = 1, conditional_embeds: Union[PromptEmbeds, None] = None, unconditional_embeds: Union[PromptEmbeds, None] = None, batch: Optional['DataLoaderBatchDTO'] = None, is_primary_pred: bool = False, **kwargs,):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )
    

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        with torch.no_grad():
            self.timer.start('preprocess_batch')
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_raw(batch)
            batch = self.preprocess_batch(batch)
            if isinstance(self.adapter, CustomAdapter):
                batch = self.adapter.edit_batch_processed(batch)
            dtype = get_torch_dtype(self.train_config.dtype)
            if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
            if isinstance(self.sd.text_encoder, list):
                for encoder in self.sd.text_encoder:
                    if encoder.dtype != self.sd.te_torch_dtype:
                        encoder.to(self.sd.te_torch_dtype)
            else:
                if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                    self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
            if self.train_config.do_cfg or self.train_config.do_random_cfg:
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            if self.adapter and isinstance(self.adapter, CustomAdapter):
                conditioned_prompts = self.adapter.condition_prompt(conditioned_prompts)

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True
                    has_clip_image = False

            do_reg_prior = False
            if any([batch.file_items[idx].prior_reg for idx in range(len(batch.file_items))]):
                do_reg_prior = True

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
                raise ValueError(
                    "IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

            match_adapter_assist = False
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand((1,), device=self.device_torch, dtype=dtype) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    if len(noisy_latents.shape) == 5:
                        h = noisy_latents.shape[3]
                        w = noisy_latents.shape[4]
                    else:
                        h = noisy_latents.shape[2]
                        w = noisy_latents.shape[3]
                    mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(h, w))
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                return 1.0
            elif match_adapter_assist:
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1
            adapter_conditioning_scale = torch.rand((1,), device=self.device_torch, dtype=dtype)
            adapter_conditioning_scale = value_map(adapter_conditioning_scale, 0.0, 1.0, adapter_strength_min, adapter_strength_max)
            return adapter_conditioning_scale

        with self.timer('grad_setup'):
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True
            if self.embedding is not None:
                grad_on_text_encoder = True
            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True
            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()
            network.multiplier = network_weight_list

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]
        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list, noise_list, timesteps_list, conditioned_prompts_list, imgs_list, adapter_images_list, clip_images_list, mask_multiplier_list, prompt_2_list):

            with (network):
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True
                            )
                        else:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros((noisy_latents.shape[0], 3, 512, 512), device=self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True,
                                drop=True
                            )
                        self.adapter(conditional_clip_embeds)

                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or is_reg):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    self.adapter.trigger_pre_te(
                        tensors_preprocessed=clip_images if not is_reg else None,
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count,
                        batch_tensor=batch.tensor if not is_reg else None,
                        batch_size=noisy_latents.shape[0]
                    )

                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    prompt_kwargs = {}
                    # 支持多控图：当存在 control_tensor_list 时按控图索引聚合为批次
                    if self.sd.encode_control_in_text_embeddings:
                        if getattr(self.sd, 'has_multiple_control_images', False) and getattr(batch, 'control_tensor_list', None) is not None:
                            num_controls = len(batch.control_tensor_list[0])
                            batched_controls: List[torch.Tensor] = []
                            for c_idx in range(num_controls):
                                per_items = []
                                for item_controls in batch.control_tensor_list:
                                    ctrl = item_controls[c_idx]
                                    if ctrl is not None and len(ctrl.shape) == 3:
                                        ctrl = ctrl.unsqueeze(0)
                                    per_items.append(ctrl.to(self.sd.device_torch, dtype=self.sd.torch_dtype))
                                batched = torch.cat(per_items, dim=0)
                                batched_controls.append(batched)
                            prompt_kwargs['control_images'] = batched_controls
                        elif batch.control_tensor is not None:
                            prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)

                    if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
                        with torch.set_grad_enabled(False):
                            if batch.prompt_embeds is not None:
                                conditional_embeds = batch.prompt_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                            else:
                                embeds_to_use = self.cached_blank_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                                if self.cached_trigger_embeds is not None and not is_reg:
                                    embeds_to_use = self.cached_trigger_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                                conditional_embeds = concat_prompt_embeds([embeds_to_use] * noisy_latents.shape[0])
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.cached_blank_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                                unconditional_embeds = concat_prompt_embeds([unconditional_embeds] * noisy_latents.shape[0])
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(self.device_torch, dtype=dtype)

                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(self.device_torch, dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                    else:
                        with torch.set_grad_enabled(False):
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(self.device_torch, dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(self.device_torch, dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(conditional_embeds.text_embeds)
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(unconditional_embeds.text_embeds, is_unconditional=True)

                pred_kwargs = {}
                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    down_block_additional_residuals = [sample.to(dtype=dtype).detach() * adapter_multiplier for sample in down_block_additional_residuals]
                                else:
                                    down_block_additional_residuals = [sample.to(dtype=dtype) * adapter_multiplier for sample in down_block_additional_residuals]
                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals

                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        quad_count = random.randint(1, 4)
                        image_size = self.adapter.input_size
                        if has_clip_image_embeds:
                            if is_reg:
                                embeds = [load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in range(noisy_latents.shape[0])]
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(embeds, quad_count=quad_count)
                                if self.train_config.do_cfg:
                                    embeds = [load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in range(noisy_latents.shape[0])]
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(embeds, quad_count=quad_count)
                            else:
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(batch.clip_image_embeds, quad_count=quad_count)
                                if self.train_config.do_cfg:
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(batch.clip_image_embeds_unconditional, quad_count=quad_count)
                        elif is_reg:
                            clip_images = torch.zeros((noisy_latents.shape[0], 3, image_size, image_size), device=self.device_torch, dtype=dtype).detach()
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(clip_images, drop=True, is_training=True, has_been_preprocessed=False, quad_count=quad_count)
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(torch.zeros((noisy_latents.shape[0], 3, image_size, image_size), device=self.device_torch, dtype=dtype).detach(), is_training=True, drop=True, has_been_preprocessed=False, quad_count=quad_count)
                        elif has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(clip_images.detach().to(self.device_torch, dtype=dtype), is_training=True, has_been_preprocessed=True, quad_count=quad_count)
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(clip_images.detach().to(self.device_torch, dtype=dtype), is_training=True, drop=True, has_been_preprocessed=True, quad_count=quad_count)
                        else:
                            print_acc("No Clip Image")
                            print_acc([file_item.path for file_item in batch.file_items])
                            raise ValueError("Could not find clip image")
                    if not self.adapter_config.train_image_encoder:
                        conditional_clip_embeds = conditional_clip_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_clip_embeds = unconditional_clip_embeds.detach()
                    with self.timer('encode_adapter'):
                        self.adapter.train()
                        conditional_embeds = self.adapter(conditional_embeds.detach(), conditional_clip_embeds, is_unconditional=False)
                        if self.train_config.do_cfg:
                            unconditional_embeds = self.adapter(unconditional_embeds.detach(), unconditional_clip_embeds, is_unconditional=True)
                        else:
                            self.adapter.last_unconditional = None

                if self.adapter and isinstance(self.adapter, ReferenceAdapter):
                    self.adapter.noise_scheduler = self.lr_scheduler
                    if has_clip_image or has_adapter_img:
                        img_to_use = clip_images if has_clip_image else adapter_images
                        reference_images = ((img_to_use - 0.5) * 2).detach().to(self.device_torch, dtype=dtype)
                        self.adapter.set_reference_images(reference_images)
                        self.adapter.noise_scheduler = self.sd.noise_scheduler
                    elif is_reg:
                        self.adapter.set_blank_reference_images(noisy_latents.shape[0])
                    else:
                        self.adapter.set_reference_images(None)

                prior_pred = None
                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True
                do_correct_pred_norm_prior = self.train_config.correct_pred_norm
                do_guidance_prior = False
                if batch.unconditional_latents is not None:
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True
                if ((has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        if self.train_config.diff_output_preservation:
                            prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        if self.train_config.blank_prompt_preservation:
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                            prior_embeds_to_use = concat_prompt_embeds([blank_embeds] * noisy_latents.shape[0])
                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=prior_embeds_to_use,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                            unconditional_embeds=unconditional_embeds,
                            conditioned_prompts=conditioned_prompts
                        )
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or self.adapter_config.type in ['llm_adapter', 'text_encoder']):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    conditional_embeds = self.adapter.condition_encoded_embeds(tensors_0_1=clip_images, prompt_embeds=conditional_embeds, is_training=True, has_been_preprocessed=True, quad_count=quad_count)
                    if self.train_config.do_cfg and unconditional_embeds is not None:
                        unconditional_embeds = self.adapter.condition_encoded_embeds(tensors_0_1=clip_images, prompt_embeds=unconditional_embeds, is_training=True, has_been_preprocessed=True, is_unconditional=True, quad_count=quad_count)

                if self.adapter and isinstance(self.adapter, CustomAdapter) and batch.extra_values is not None:
                    self.adapter.add_extra_values(batch.extra_values.detach())
                    if self.train_config.do_cfg:
                        self.adapter.add_extra_values(torch.zeros_like(batch.extra_values.detach()), is_unconditional=True)

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample
                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    self._guidance_loss_target_batch = [random.uniform(self.train_config.guidance_loss_target[0], self.train_config.guidance_loss_target[1]) for _ in range(batch_size)]

                self.before_unet_predict()
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                with self.timer('condition_noisy_latents'):
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)
                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        noisy_latents = self.adapter.condition_noisy_latents(noisy_latents, batch)
                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():
                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(next_sample_pred, noisy_latents, timesteps, self.sd.noise_scheduler)
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )
                elif self.train_config.loss_type == 'mean_flow':
                    loss = self.get_mean_flow_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        prior_pred=prior_pred,
                    )
                else:
                    with self.timer('predict_unet'):
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            is_primary_pred=True,
                            **pred_kwargs
                        )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        doing_preservation = self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation
                        if doing_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )
                if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
                    self.accelerator.backward(loss)
                    normal_loss = loss.detach()
                    with torch.no_grad():
                        if self.train_config.diff_output_preservation:
                            preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        elif self.train_config.blank_prompt_preservation:
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(self.device_torch, dtype=dtype)
                            preservation_embeds = concat_prompt_embeds([blank_embeds] * noisy_latents.shape[0])
                    preservation_pred = self.predict_noise(
                        noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                        timesteps=timesteps,
                        conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                        unconditional_embeds=unconditional_embeds,
                        batch=batch,
                        **pred_kwargs
                    )
                    multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                    preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier
                    self.accelerator.backward(preservation_loss)
                    loss = normal_loss + preservation_loss
                    loss = loss.clone().detach()
                    loss.requires_grad_(True)

                if torch.isnan(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    loss = loss * loss_multiplier.mean()
                    self.accelerator.backward(loss)
        return loss.detach()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            if self.sd.is_multistage:
                if self.steps_this_boundary >= self.train_config.switch_boundary_every or self.current_boundary_index not in self.sd.trainable_multistage_boundaries:
                    while True:
                        self.steps_this_boundary = 0
                        self.current_boundary_index += 1
                        if self.current_boundary_index >= len(self.sd.multistage_boundaries):
                            self.current_boundary_index = 0
                        if self.current_boundary_index in self.sd.trainable_multistage_boundaries:
                            break
            loss = self.train_single_accumulation(batch)
            self.steps_this_boundary += 1
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()

        if not self.is_grad_accumulation_step:
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            with self.timer('optimizer_step'):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.adapter and isinstance(self.adapter, CustomAdapter):
                    self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
        else:
            pass

        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict({'loss': (total_loss / len(batch_list)).item()})
        self.end_of_training_loop()
        return loss_dict

