from pathlib import Path
import json

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class


def test_anima_model_class_is_registered():
    config = ModelConfig(name_or_path="circlestone-labs/Anima", arch="anima")

    model_class = get_model_class(config)

    assert model_class.__name__ == "AnimaModel"
    assert model_class.arch == "anima"


def test_anima_ui_defaults_match_reference_training_config():
    options = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")

    assert "name: 'anima'" in options
    assert "label: 'Anima'" in options
    assert "circlestone-labs/Anima" in options
    assert "anima-base-v1.0.safetensors" not in options
    assert "qwen_image_vae.safetensors" not in options
    assert "qwen_3_06b_base.safetensors" not in options
    assert "tokenizer: 'Qwen/Qwen3-0.6B-Base'" in options
    assert "t5_tokenizer: 'google-t5/t5-11b'" in options
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


def test_anima_uses_local_diffusers_component_configs():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert 'config=ANIMA_TRANSFORMER_CONFIG' in source
    assert 'config=ANIMA_VAE_CONFIG' in source
    assert 'local_files_only=True' in source
    assert 'low_cpu_mem_usage=False' in source
    assert "CosmosTransformer3DModel.from_single_file" not in source
    assert "CosmosTransformer3DModel.from_config" in source
    assert "convert_cosmos_transformer_checkpoint_to_diffusers" in source
    assert "assign=True" in source
    assert '"nvidia/Cosmos-Predict2-2B-Text2Image"' not in source
    assert '"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"' not in source


def test_anima_tokenizer_loading_does_not_silently_hang_for_local_models():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert "Building Anima transformer module" in source
    assert "Reading Anima transformer weights" in source
    assert "Assigning Anima transformer weights" in source
    assert "Loading Anima Qwen3 tokenizer" in source
    assert "Loading Anima T5 tokenizer" in source
    assert "allow_tokenizer_download" in source
    assert "snapshot_download" in source
    assert "local_files_only=True" in source
    assert "not found in local cache/path, downloading" in source
    assert "local_files_only=False" in source


def test_anima_llm_adapter_loader_accepts_raw_checkpoint_prefixes():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert '"net.llm_adapter."' in source
    assert '"net.diffusion_model.llm_adapter."' in source
    assert "Adapter-like keys" in source


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


def test_anima_transformer_config_matches_preview3_weight_shape():
    config = json.loads(
        Path("extensions_built_in/diffusion_models/anima/configs/cosmos_transformer/config.json").read_text(
            encoding="utf-8"
        )
    )

    # Anima Preview 3 transformer weights use a 2048 hidden size. The public Cosmos 2B config is 4096.
    assert config["num_attention_heads"] * config["attention_head_dim"] == 2048
    # The Anima checkpoint does not ship diffusers' optional learnable positional embedding parameters.
    assert config["extra_pos_embed_type"] is None


if __name__ == "__main__":
    test_anima_model_class_is_registered()
    test_anima_ui_defaults_match_reference_training_config()
    test_anima_uses_local_diffusers_component_configs()
    test_anima_tokenizer_loading_does_not_silently_hang_for_local_models()
    test_anima_llm_adapter_loader_accepts_raw_checkpoint_prefixes()
    test_anima_prompt_encoding_guards_empty_unconditional_prompt()
    test_anima_advanced_prompt_embeds_are_batched_for_training_and_sampling()
    test_anima_quantization_skips_extras_that_receive_5d_latents()
    test_anima_sampling_matches_diffusers_anima_default_reference()
    test_anima_transformer_config_matches_preview3_weight_shape()
