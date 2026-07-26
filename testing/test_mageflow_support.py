from pathlib import Path


REGISTRY_SOURCE = Path("extensions_built_in/diffusion_models/__init__.py")
MAGEFLOW_SOURCE = Path("extensions_built_in/diffusion_models/mageflow/mageflow.py")
OPTIONS_SOURCE = Path("ui/src/app/jobs/new/options.ts")
SIMPLE_JOB_SOURCE = Path("ui/src/app/jobs/new/SimpleJob.tsx")


def test_mageflow_models_are_registered_once():
    registry = REGISTRY_SOURCE.read_text(encoding="utf-8")

    assert registry.count("from .mageflow import MageFlowModel, MageFlowEditModel") == 1
    assert registry.count("    MageFlowModel,") == 1
    assert registry.count("    MageFlowEditModel,") == 1
    assert registry.count("from .anima import AnimaModel") == 1
    assert registry.count("    AnimaModel,") == 1


def test_mageflow_uses_bundled_implementation_and_official_repos():
    source = MAGEFLOW_SOURCE.read_text(encoding="utf-8")
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")

    assert "from .src.transformer import MageFlow, MageFlowParams" in source
    assert "from .src.vae import MageVAE" in source
    assert "from .src.pipeline import MageFlowPipeline" in source
    assert "name: 'mageflow'" in options
    assert "name: 'mageflow_edit'" in options
    assert "microsoft/Mage-Flow-Base" in options
    assert "microsoft/Mage-Flow-Edit-Base" in options


def test_gated_model_help_is_localized():
    simple_job = SIMPLE_JOB_SOURCE.read_text(encoding="utf-8")

    assert "title: '受限模型'" in simple_job
    assert "只读访问令牌" in simple_job
    assert "设置页面" in simple_job
    assert "Gated model" not in simple_job
