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
    assert 'ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"' in model
    assert "from .src.transformer import MiniMaxH3Transformer" in model
    assert "from .src.vae import MiniMaxH3VideoVAE" in model
    assert "def get_frame_count_snapper" in model


def test_minimax_h3_loads_original_metadata_from_project_models_folder():
    model = MODEL_SOURCE.read_text(encoding="utf-8")

    assert "from toolkit.paths import MODELS_PATH, TOOLKIT_ROOT" in model
    assert (
        'ORIGINAL_LOCAL_ROOT = os.path.join(TOOLKIT_ROOT, "models", "MiniMax-H3")'
        in model
    )
    for relative_path in (
        "FL2VA/tokenizer/merges.txt",
        "FL2VA/tokenizer/tokenizer.json",
        "FL2VA/tokenizer/tokenizer_config.json",
        "FL2VA/tokenizer/vocab.json",
        "FL2VA/processor/chat_template.json",
        "FL2VA/processor/preprocessor_config.json",
        "FL2VA/processor/video_preprocessor_config.json",
        "FL2VA/text_encoder/config.json",
    ):
        assert relative_path in model

    assert 'self._resolve_original_subfolder("tokenizer")' in model
    assert 'self._resolve_original_subfolder("processor")' in model
    assert 'self._resolve_original_subfolder("text_encoder")' in model
    assert "AutoTokenizer.from_pretrained(\n            tokenizer_path, local_files_only=True" in model
    assert "AutoProcessor.from_pretrained(\n            processor_path, local_files_only=True" in model
    assert "AutoConfig.from_pretrained(\n                text_encoder_config_path, local_files_only=True" in model
    assert "AutoTokenizer.from_pretrained(\n            ORIGINAL_REPO" not in model


def test_minimax_h3_ui_defaults_and_notes_are_localized():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    simple_job = SIMPLE_JOB_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")
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


def test_minimax_h3_training_adapter_uses_portable_default_and_localized_help():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")
    version = VERSION_SOURCE.read_text(encoding="utf-8")

    adapter_path = "./models/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors"
    ref2va_adapter_path = (
        "./models/minimax_h3_training_adapter/"
        "minimax_h3_ref2va_training_adapter_v1.safetensors"
    )
    assert options.count(adapter_path) == 3
    assert options.count(ref2va_adapter_path) == 3
    assert "ostris/minimax_h3_training_adapter/" not in options
    assert "/datasets/" not in options
    assert "/model/" not in options
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
    assert 'TOOLKIT_ROOT, "models", "loras", "training_adapters"' in model
    assert "found = self._find_file_recursive(adapter_root, filename)" in model
    assert "local_dir=adapter_root" in model
    assert "LEGACY_REF2VA_TRAINING_ADAPTER_PATH" not in model


def test_ltx25_uses_portable_transformer_path():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")

    assert (
        "'./models/diffusion_models/"
        "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors'"
        in options
    )
    assert "'Lightricks/LTX-2.5'" not in options


def test_main_uses_the_project_models_folder_by_default():
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
