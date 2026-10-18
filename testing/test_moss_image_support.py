from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "extensions_built_in/diffusion_models"


def test_moss_image_registers_custom_components_and_four_model_options():
    registry = (MODEL_ROOT / "__init__.py").read_text(encoding="utf-8")
    model = (MODEL_ROOT / "moss_image/moss_image.py").read_text(encoding="utf-8")
    options = (MODEL_ROOT / "ui.tsx").read_text(encoding="utf-8")
    assert "from .moss_image import MossImageModel" in registry
    assert "from .src.transformer import MossImageTransformer" in model
    assert "from .src.pipeline import (" in model
    for name in ("moss_image", "moss_image:base", "moss_image:edit", "moss_image:base_edit"):
        assert f'name: "{name}"' in options
    assert options.count('"OpenMOSS-Team/MOSS-Image-SFT"') == 2
    assert options.count('"OpenMOSS-Team/MOSS-Image-Base"') == 2


def test_moss_image_edit_instructions_are_localized_without_renaming_models():
    options = (MODEL_ROOT / "ui.tsx").read_text(encoding="utf-8")
    assert "使用这些 LoRA" in options
    assert "需要安装" in options
    assert "Using these LoRAs in ComfyUI requires" not in options
    assert 'label: "MOSS-Image SFT [编辑训练]"' in options
    assert 'label: "MOSS-Image Base [编辑训练]"' in options
    assert 'label: "MOSS-Image SFT"' in options
    assert 'label: "MOSS-Image Base"' in options


def test_existing_kv_cache_control_and_help_are_localized():
    simple_job = (ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")
    docs = (ROOT / "ui/src/docs.tsx").read_text(encoding="utf-8")
    assert 'label="KV 缓存"' in simple_job
    assert "'model.model_kwargs.kv_cache':" in docs
    assert "为支持的模型开启控制图 KV 缓存" in docs


def test_upstream_sync_preserves_aigate_version_number():
    assert (ROOT / "version.py").read_text(encoding="utf-8").strip() == 'VERSION = "1.18.9"'
