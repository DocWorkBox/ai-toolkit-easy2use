from pathlib import Path

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
    assert '"nvidia/Cosmos-Predict2-2B-Text2Image"' not in source
    assert '"Wan-AI/Wan2.1-T2V-1.3B-Diffusers"' not in source


if __name__ == "__main__":
    test_anima_model_class_is_registered()
    test_anima_ui_defaults_match_reference_training_config()
    test_anima_uses_local_diffusers_component_configs()
