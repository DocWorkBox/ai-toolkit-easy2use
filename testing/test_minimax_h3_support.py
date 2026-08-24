from pathlib import Path


REGISTRY_SOURCE = Path("extensions_built_in/diffusion_models/__init__.py")
MODEL_SOURCE = Path("extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py")
LTX25_MODEL_SOURCE = Path("extensions_built_in/diffusion_models/ltx2/ltx2.py")
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
    assert 'ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"' in model
    assert "from .src.transformer import MiniMaxH3Transformer" in model
    assert "from .src.vae import MiniMaxH3VideoVAE" in model
    assert "def get_frame_count_snapper" in model


def test_minimax_h3_ui_defaults_and_notes_are_localized():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    simple_job = SIMPLE_JOB_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")
    settings = SETTINGS_SOURCE.read_text(encoding="utf-8")

    assert "name: 'minimax_h3'" in options
    assert "name: 'minimax_h3_ref2va'" in options
    assert options.count("'/model/ModelScope/Comfy-Org/MiniMax-H3'") == 2
    assert "'config.process[0].datasets[x].do_audio': [true, undefined]" in options
    assert "'config.process[0].datasets[x].do_i2v': [false, undefined]" in options
    assert "模型目录路径" in options
    assert "模型说明" in simple_job
    assert "模型目录路径" in settings
    assert "输入模型目录路径" in settings
    assert "Models Folder Path" not in options
    assert "Reference-to-video" not in options
    assert "label: '参考图呈现方式'" in options
    assert "label: '静态视频片段'" in options
    assert "Image Reference Presentation" not in options
    assert "Model notes" not in simple_job
    assert 'label="批次大小"' in simple_job
    assert 'docKey="datasets.batch_size"' in simple_job
    assert 'label="音频损失倍率"' in simple_job
    assert "Audio Loss Multiplier" not in simple_job
    assert "'train.audio_loss_multiplier':" in docs
    assert "'datasets.batch_size':" in docs


def test_minimax_h3_training_adapter_uses_compshare_defaults_and_localized_help():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")
    version = VERSION_SOURCE.read_text(encoding="utf-8")

    adapter_path = "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors"
    ref2va_adapter_path = (
        "ostris/minimax_h3_training_adapter/"
        "minimax_h3_ref2va_training_adapter_v1.safetensors"
    )
    assert options.count(adapter_path) == 3
    assert options.count(ref2va_adapter_path) == 3
    assert "/datasets/ComfyUI/models/loras/minimax_h3" not in options
    assert "label: '蒸馏保持方式'" in options
    assert "label: '对比引导'" in options
    assert "label: '训练适配器'" in options
    assert "label: '对比引导 + 训练适配器（默认）'" in options
    assert "'config.process[0].train.do_guidance_loss': [true, undefined]" in options
    assert "'config.process[0].train.guidance_loss_target': [4.0, undefined]" not in options
    assert "'config.process[0].model.assistant_lora_path': {" in docs
    assert "训练适配器路径" in docs
    assert 'VERSION = "1.18.4"' in version


def test_minimax_h3_ref2va_supports_video_references():
    model = MODEL_SOURCE.read_text(encoding="utf-8")

    assert "self.supports_video_control_images = True" in model
    assert "load_ref_video_latent" in model


def test_minimax_h3_remote_adapters_download_to_toolkit_models_folder():
    model = MODEL_SOURCE.read_text(encoding="utf-8")

    assert "from toolkit.paths import MODELS_PATH, TOOLKIT_ROOT" in model
    assert 'LEGACY_REF2VA_TRAINING_ADAPTER_PATH' in model
    assert 'if lora_path == LEGACY_REF2VA_TRAINING_ADAPTER_PATH:' in model
    assert 'lora_path = REF2VA_TRAINING_ADAPTER_REPO_PATH' in model
    assert 'TOOLKIT_ROOT, "models", "loras", "training_adapters"' in model
    assert "found = self._find_file_recursive(adapter_root, filename)" in model
    assert "local_dir=adapter_root" in model


def test_ltx25_uses_compshare_modelscope_path():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    model = LTX25_MODEL_SOURCE.read_text(encoding="utf-8")

    assert "'/model/ModelScope/Lightricks/LTX-2.5'" in options
    assert "if name_or_path and os.path.isdir(name_or_path):" in model
    assert "search_roots.append(name_or_path)" in model


def test_compshare_uses_the_project_models_folder_by_default():
    ui_paths = UI_PATHS_SOURCE.read_text(encoding="utf-8")
    cron_paths = CRON_PATHS_SOURCE.read_text(encoding="utf-8")
    settings_route = SETTINGS_ROUTE_SOURCE.read_text(encoding="utf-8")

    expected_default = "export const defaultModelsFolder = path.join(TOOLKIT_ROOT, 'models');"
    assert expected_default in ui_paths
    assert expected_default in cron_paths
    assert "let modelsPath = ''" in cron_paths
    assert "row.value !== defaultModelsFolder" in cron_paths
    assert "legacyDefaultModelsFolder" not in settings_route
    assert "/datasets/" not in ui_paths
    assert "/datasets/" not in cron_paths
