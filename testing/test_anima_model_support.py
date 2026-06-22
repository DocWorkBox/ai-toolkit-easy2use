from pathlib import Path
import pytest


def test_anima_model_class_is_registered():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from toolkit.config_modules import ModelConfig
    from toolkit.util.get_model import get_model_class

    config = ModelConfig(name_or_path="circlestone-labs/Anima-Base-v1.0-Diffusers", arch="anima")

    model_class = get_model_class(config)

    assert model_class.__name__ == "AnimaModel"
    assert model_class.arch == "anima"


def test_anima_ui_defaults_match_reference_training_config():
    options = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")

    assert "name: 'anima'" in options
    assert "label: 'Anima'" in options
    assert "/model/ModelScope/circlestone-labs/Anima-Base-v1.0-Diffusers" in options
    assert "circlestone-labs/Anima'" not in options
    assert "anima-base-v1.0.safetensors" not in options
    assert "qwen_image_vae.safetensors" not in options
    assert "qwen_3_06b_base.safetensors" not in options
    assert "'config.process[0].model.model_paths'" not in options
    assert "llm_adapter_lr: 0" in options
    assert "'config.process[0].network.linear': [32, defaultLinearRank]" in options
    assert "'config.process[0].train.lr': [2e-5" in options
    assert "'config.process[0].sample.guidance_scale': [4.5, 4]" in options
    assert "'config.process[0].sample.sample_steps': [35, 25]" in options
    assert "worst quality, low quality, score_1, score_2, score_3, artist name" in options
    assert "'config.process[0].sample.samples'" in options
    assert "defaultSampleConfig.samples" in options
    assert "anime key visual" in options
    assert "anime screencap" in options
    assert "anime fantasy art" in options


def test_anima_uses_official_diffusers_components_directly():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")
    requirements = Path("requirements_base.txt").read_text(encoding="utf-8")

    assert 'ANIMA_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"' in source
    assert "DocWorkBox/diffusers.git@8153646f890ba45513058e21c2847db042dde1f7" in requirements
    assert "diffusers/pull/13732" in requirements
    assert "Qwen Image txt_seq_lens" in requirements
    assert "Ernie Image dtype fixes" in requirements
    assert "DiffusionPipeline.from_pretrained" in source
    assert "ModularPipeline.from_pretrained" in source
    assert "Loading Anima official Diffusers pipeline" in source
    assert "Loading Anima modular Diffusers pipeline fallback" in source
    assert "modular_model_index.json" in source
    assert "ANIMA_COMPONENT_LOAD_ORDER" in source
    assert "Loading Anima component: {component_name}" in source
    assert "Loaded Anima component: {component_name}" in source
    assert "Retrying Anima component with direct spec load: {component_name}" in source
    assert "component_spec.load(**load_kwargs)" in source
    assert "modular_pipe.update_components(**{component_name: loaded_component})" in source
    assert "local_files_only" in source
    assert '"pretrained_model_name_or_path": model_path' in source
    assert "AnimaModularPipeline" not in source
    assert "diffusers_module.AnimaModularPipeline" not in source
    assert "load_components" in source
    assert "text_conditioner" in source
    assert "diffusers_pipe.transformer" in source
    assert "diffusers_pipe.text_encoder" in source
    assert "diffusers_pipe.tokenizer" in source
    assert "diffusers_pipe.t5_tokenizer" in source
    assert "diffusers_pipe.vae" in source
    assert "_resolve_component_path" not in source
    assert "_load_transformer" not in source
    assert "_load_qwen3_llm" not in source
    assert "_load_llm_adapter" not in source
    assert "_load_tokenizer" not in source
    assert "CosmosTransformer3DModel.from_single_file" not in source
    assert "CosmosTransformer3DModel.from_config" not in source
    assert "convert_cosmos_transformer_checkpoint_to_diffusers" not in source
    assert "assign=True" not in source
    assert '"nvidia/Cosmos-Predict2-2B-Text2Image"' not in source
    assert '"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"' not in source


def test_anima_diffusers_pipeline_owns_tokenizer_loading():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "Loading Anima Diffusers pipeline" in source
    assert "Loading Anima official Diffusers pipeline" in source
    assert "Loading Anima modular Diffusers pipeline fallback" in source
    assert "Loading Anima component: {component_name}" in source
    assert "Loading Anima Qwen3 tokenizer" not in source
    assert "Loading Anima T5 tokenizer" not in source
    assert "allow_tokenizer_download" not in source
    assert "snapshot_download" not in source
    assert "not found in local cache/path, downloading" not in source


def test_anima_uses_diffusers_text_conditioner_component():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "llm_adapter = diffusers_pipe.text_conditioner" in source
    assert "llm_adapter=llm_adapter" in source
    assert "Adapter-like keys" not in source


def test_anima_prompt_encoding_guards_empty_unconditional_prompt():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert 'encode_prompt("")' in source
    assert 'not str(text).strip()' in source
    assert '" " if text is None' in source


def test_anima_advanced_prompt_embeds_are_batched_for_training_and_sampling():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "def _batch_text_embeds" in source
    assert "torch.stack(list(text_embeds), dim=0)" in source
    assert "prompt_embeds=self._batch_text_embeds(conditional_embeds.text_embeds)" in source
    assert "negative_prompt_embeds=self._batch_text_embeds(unconditional_embeds.text_embeds)" in source
    assert "encoder_hidden_states=self._batch_text_embeds(text_embeddings.text_embeds)" in source
    assert "AdvancedPromptEmbeds(text_embeds=[embed for embed in embeds])" in source


def test_anima_quantization_skips_extras_that_receive_5d_latents():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "def _quantize_transformer_blocks" in source
    assert "quantize(block, weights=quantization_type)" in source
    assert "skipping Anima transformer extras quantization" in source
    assert "quantize_model(self, transformer)" not in source


def test_anima_sampling_matches_diffusers_anima_default_reference():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "ANIMA_BETA_ALPHA = 0.6" in source
    assert "ANIMA_BETA_BETA = 0.6" in source
    assert "def _build_anima_beta_sigmas" in source
    assert "stats.beta.ppf" in source
    assert "latents = latents * sigmas[0].to(latents.dtype)" in source
    assert "noise_pred = self.transformer(" in source
    assert "noise_pred_uncond = self.transformer(" in source
    assert "noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)" in source
    assert "denoised = latents - sigma.float() * noise_pred" in source
    assert "downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta" in source
    assert "renoise_coeff = renoise_sq.clamp_min(0).sqrt()" in source
    assert "vae.decode((latents / latents_std + latents_mean)" in source
    assert "sigma_max = 80.0" not in source
    assert "c_skip" not in source
    assert "c_out" not in source
    assert "self.scheduler.step(velocity" not in source
    assert "noise_pred = (latents - noise_pred.float()) / current_sigma" not in source


def test_anima_uses_remote_diffusers_transformer_config():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "ANIMA_TRANSFORMER_CONFIG" not in source
    assert "ANIMA_VAE_CONFIG" not in source


def test_anima_lora_export_uses_comfyui_generic_diffusion_model_prefix():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert 'key.startswith("lora_transformer_")' in source
    assert '"lora_unet_" + key[len("lora_transformer_"):]' in source
    assert 'key.startswith("lora_unet_")' in source
    assert '"lora_transformer_" + key[len("lora_unet_"):]' in source
    assert '("transformer_blocks_", "blocks_")' in source
    assert '("_attn1_to_out_0", "_self_attn_output_proj")' in source
    assert '("_attn2_to_out_0", "_cross_attn_output_proj")' in source
    assert '("_ff_net_0_proj", "_mlp_layer1")' in source
    assert '("_norm1_linear_1", "_adaln_modulation_self_attn_1")' in source
    assert '(".attn1.to_out.0.", ".self_attn.output_proj.")' in source
    assert '(".norm3.linear_2.", ".adaln_modulation_mlp.2.")' in source


if __name__ == "__main__":
    test_anima_model_class_is_registered()
    test_anima_ui_defaults_match_reference_training_config()
    test_anima_uses_official_diffusers_components_directly()
    test_anima_diffusers_pipeline_owns_tokenizer_loading()
    test_anima_uses_diffusers_text_conditioner_component()
    test_anima_prompt_encoding_guards_empty_unconditional_prompt()
    test_anima_advanced_prompt_embeds_are_batched_for_training_and_sampling()
    test_anima_quantization_skips_extras_that_receive_5d_latents()
    test_anima_sampling_matches_diffusers_anima_default_reference()
    test_anima_uses_remote_diffusers_transformer_config()
    test_anima_lora_export_uses_comfyui_generic_diffusion_model_prefix()
