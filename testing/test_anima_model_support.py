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
    assert "anima-base-v1.0.safetensors" in options
    assert "qwen_image_vae.safetensors" in options
    assert "qwen_3_06b_base.safetensors" in options
    assert "llm_adapter_lr: 0" in options
    assert "'config.process[0].network.linear': [32, defaultLinearRank]" in options
    assert "'config.process[0].train.lr': [2e-5" in options


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
    assert "local_files_only=local_files_only" in source


def test_anima_llm_adapter_loader_accepts_raw_checkpoint_prefixes():
    source = Path("extensions_built_in/diffusion_models/anima/anima_model.py").read_text(encoding="utf-8")

    assert '"net.llm_adapter."' in source
    assert '"net.diffusion_model.llm_adapter."' in source
    assert "Adapter-like keys" in source


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
    test_anima_transformer_config_matches_preview3_weight_shape()
