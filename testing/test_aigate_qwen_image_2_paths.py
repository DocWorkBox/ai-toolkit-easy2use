from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG_REPO = "Qwen/Qwen-Image-2.1"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_qwen_image_2_uses_official_configs_with_comfy_weights():
    model_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py"
    )
    assert f'BASE_REPO = "{OFFICIAL_CONFIG_REPO}"' in model_source
    assert 'COMFY_REPO = "Comfy-Org/Qwen-Image-2.1"' in model_source

    for path in (
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/vae.py",
    ):
        source = _source(path)
        assert f'aitk_config_repo = "{OFFICIAL_CONFIG_REPO}"' in source
        assert "/datasets/" not in source

    text_encoder_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/src/text_encoder.py"
    )
    assert f'aitk_processor_repo = "{OFFICIAL_CONFIG_REPO}"' in text_encoder_source


def test_qwen_image_2_resolves_comfy_turbo_weights_and_adapter_before_hub():
    mixin_source = _source("toolkit/models/v2/_mixin.py")
    model_source = _source(
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py"
    )

    assert "from toolkit.models.v2.resolver import resolve_named_file" in mixin_source
    assert "return resolve_named_file(name_or_path)" in mixin_source
    assert "from toolkit.models.v2.resolver import resolve_lora_file" in model_source
    assert "lora_path = resolve_lora_file(" in model_source


def test_comfy_resolvers_find_turbo_weights_and_adapter(tmp_path, monkeypatch):
    from toolkit.models.v2 import resolver

    models_root = tmp_path / "models"
    transformer = (
        models_root
        / "diffusion_models"
        / "qwen_image_2.1_turbo_int8_convrot.safetensors"
    )
    adapter = (
        models_root
        / "loras"
        / "qwen_image_2_turbo_training_adapter_v1.safetensors"
    )
    transformer.parent.mkdir(parents=True)
    adapter.parent.mkdir(parents=True)
    transformer.touch()
    adapter.touch()
    monkeypatch.setattr(resolver, "MODELS_PATH", str(models_root))

    resolved_transformer = resolver.resolve_named_file(
        "Comfy-Org/Qwen-Image-2.1/"
        "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
    )
    resolved_adapter = resolver.resolve_lora_file(
        "ostris/qwen_image_2_turbo_training_adapter/"
        "qwen_image_2_turbo_training_adapter_v1.safetensors"
    )

    assert Path(resolved_transformer).resolve() == transformer.resolve()
    assert Path(resolved_adapter).resolve() == adapter.resolve()


def test_new_model_adapters_share_local_first_resolver():
    for path in (
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py",
        "extensions_built_in/diffusion_models/ming_image/ming_image.py",
        "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py",
    ):
        source = _source(path)
        assert "resolve_lora_file(" in source
        assert 'TOOLKIT_ROOT, "models", "loras", "training_adapters"' in source
