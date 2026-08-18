from pathlib import Path


REGISTRY_SOURCE = Path("extensions_built_in/diffusion_models/__init__.py")
MODEL_SOURCE = Path("extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py")
OPTIONS_SOURCE = Path("ui/src/app/jobs/new/options.tsx")
SIMPLE_JOB_SOURCE = Path("ui/src/app/jobs/new/SimpleJob.tsx")
DOCS_SOURCE = Path("ui/src/docs.tsx")
VERSION_SOURCE = Path("version.py")
SETTINGS_SOURCE = Path("ui/src/app/settings/page.tsx")
UI_PATHS_SOURCE = Path("ui/src/paths.ts")
CRON_PATHS_SOURCE = Path("ui/cron/paths.ts")
SETTINGS_ROUTE_SOURCE = Path("ui/src/app/api/settings/route.ts")


def test_minimax_h3_is_registered_with_custom_training_components():
    registry = REGISTRY_SOURCE.read_text(encoding="utf-8")
    model = MODEL_SOURCE.read_text(encoding="utf-8")

    assert registry.count("from .minimax_h3 import MinimaxH3Model") == 1
    assert registry.count("    MinimaxH3Model,") == 1
    assert 'arch = "minimax_h3"' in model
    assert 'COMFY_REPO = "Comfy-Org/MiniMax-H3"' in model
    assert 'ORIGINAL_REPO = "/datasets/studio/huggingface/models/MiniMax-H3"' in model
    assert "from .src.transformer import MiniMaxH3Transformer" in model
    assert "from .src.vae import MiniMaxH3VideoVAE" in model
    assert "def get_frame_count_snapper" in model


def test_minimax_h3_ui_defaults_and_notes_are_localized():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    simple_job = SIMPLE_JOB_SOURCE.read_text(encoding="utf-8")
    settings = SETTINGS_SOURCE.read_text(encoding="utf-8")

    assert "name: 'minimax_h3'" in options
    assert "name: 'minimax_h3_ref2va'" in options
    assert "'Comfy-Org/MiniMax-H3'" in options
    assert "'config.process[0].datasets[x].do_audio': [true, undefined]" in options
    assert "'config.process[0].datasets[x].do_i2v': [false, undefined]" in options
    assert "模型目录路径" in options
    assert "模型说明" in simple_job
    assert "模型目录路径" in settings
    assert "输入模型目录路径" in settings
    assert "Models Folder Path" not in options
    assert "Reference-to-video" not in options
    assert "Model notes" not in simple_job


def test_minimax_h3_training_adapter_uses_aigate_defaults_and_localized_help():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")
    version = VERSION_SOURCE.read_text(encoding="utf-8")

    adapter_path = "/datasets/ComfyUI/models/loras/minimax_h3_training_adapter_v1.safetensors"
    ref2va_adapter_path = "/datasets/ComfyUI/models/loras/minimax_h3_ref2va_training_adapter_v1.safetensors"
    assert adapter_path in options
    assert ref2va_adapter_path in options
    assert "label: '蒸馏保持方式'" in options
    assert "label: '对比引导'" in options
    assert "label: '训练适配器'" in options
    assert "label: '对比引导 + 训练适配器（默认）'" in options
    assert "'config.process[0].train.do_guidance_loss': [true, undefined]" in options
    assert "'config.process[0].train.guidance_loss_target': [4.0, undefined]" not in options
    assert "'config.process[0].model.assistant_lora_path': {" in docs
    assert "训练适配器路径" in docs
    assert 'VERSION = "1.18.3"' in version


def test_minimax_h3_ref2va_supports_video_references():
    model = MODEL_SOURCE.read_text(encoding="utf-8")

    assert "self.supports_video_control_images = True" in model
    assert "load_ref_video_latent" in model


def test_aigate_comfy_models_path_is_the_effective_ui_default():
    ui_paths = UI_PATHS_SOURCE.read_text(encoding="utf-8")
    cron_paths = CRON_PATHS_SOURCE.read_text(encoding="utf-8")
    settings_route = SETTINGS_ROUTE_SOURCE.read_text(encoding="utf-8")

    expected_default = "export const defaultModelsFolder = '/datasets/ComfyUI/models';"
    assert expected_default in ui_paths
    assert expected_default in cron_paths
    assert "let modelsPath = defaultModelsFolder" in cron_paths
    assert "row.value !== legacyDefaultModelsFolder" in cron_paths
    assert "settingsObject.MODELS_PATH === legacyDefaultModelsFolder" in settings_route
