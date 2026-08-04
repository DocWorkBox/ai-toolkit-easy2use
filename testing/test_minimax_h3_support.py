from pathlib import Path


REGISTRY_SOURCE = Path("extensions_built_in/diffusion_models/__init__.py")
MODEL_SOURCE = Path("extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py")
OPTIONS_SOURCE = Path("ui/src/app/jobs/new/options.tsx")
SIMPLE_JOB_SOURCE = Path("ui/src/app/jobs/new/SimpleJob.tsx")
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
    settings = SETTINGS_SOURCE.read_text(encoding="utf-8")

    assert "name: 'minimax_h3'" in options
    assert "'/model/ModelScope/Comfy-Org/MiniMax-H3'" in options
    assert "'config.process[0].datasets[x].do_audio': [true, undefined]" in options
    assert "'config.process[0].datasets[x].do_i2v': [false, undefined]" in options
    assert "模型目录路径" in options
    assert "模型说明" in simple_job
    assert "模型目录路径" in settings
    assert "输入模型目录路径" in settings
    assert "Models Folder Path" not in options
    assert "Model notes" not in simple_job


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
