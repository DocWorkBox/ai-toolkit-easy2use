from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_CONFIG_PATH = "./models/Qwen-Image-2.1"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_qwen_image_2_uses_local_configs_with_comfy_weights():
    model_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py"
    )
    assert f'BASE_REPO = "{PORTABLE_CONFIG_PATH}"' in model_source
    assert 'COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"' in model_source
    assert "component_path = COMFY_REPO" in model_source
    assert "base_model_path, dtype=dtype, subfolder=\"text_encoder\"" not in model_source
    assert "config_path=base_model_path" in model_source

    for path in (
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/vae.py",
    ):
        source = _source(path)
        assert f'aitk_config_repo = "{PORTABLE_CONFIG_PATH}"' in source
        assert "/datasets/" not in source

    text_encoder_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py"
    )
    assert f'aitk_processor_repo = "{PORTABLE_CONFIG_PATH}"' in text_encoder_source
