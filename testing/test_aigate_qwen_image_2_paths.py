from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_REPO = "/model/ModelScope/Qwen/Qwen-Image-2.1"
COMFY_MODEL_DIR = "/model/ModelScope/Comfy-Org/Qwen-Image-2.1"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_qwen_image_2_uses_compshare_local_configs_with_comfy_weights():
    model_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py"
    )
    assert f'BASE_REPO = "{CONFIG_REPO}"' in model_source
    assert 'COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"' in model_source

    for path in (
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/vae.py",
    ):
        assert f'aitk_config_repo = "{CONFIG_REPO}"' in _source(path)

    text_encoder_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py"
    )
    assert f'aitk_processor_repo = "{CONFIG_REPO}"' in text_encoder_source

    ui_source = _source("extensions_built_in/diffusion_models/ui.tsx")
    assert f'"{COMFY_MODEL_DIR}"' in ui_source
    assert f'"{CONFIG_REPO}"' in ui_source


def test_qwen_image_2_resolves_split_weights_from_local_comfy_root():
    model_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py"
    )
    assert "def _resolve_local_comfy_component" in model_source
    assert "QwenImage21Transformer2DModel._COMFY_FILES" in model_source
    assert "QwenImage21TextEncoder._COMFY_FILES" in model_source
    assert "AutoencoderKLQwenImage21._COMFY_FILES" in model_source
    assert "config_path=base_model_path" in model_source
