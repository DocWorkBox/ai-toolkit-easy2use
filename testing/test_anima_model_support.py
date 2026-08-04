from pathlib import Path

import pytest


ANIMA_SOURCE = Path("extensions_built_in/diffusion_models/anima/anima.py")
ANIMA_LEGACY_SOURCE = Path("extensions_built_in/diffusion_models/anima/anima_model.py")
OPTIONS_SOURCE = Path("ui/src/app/jobs/new/options.tsx")
REQUIREMENTS_SOURCE = Path("requirements_base.txt")
PROMPT_UTILS_SOURCE = Path("toolkit/prompt_utils.py")


def test_anima_model_class_is_registered():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from toolkit.config_modules import ModelConfig
    from toolkit.util.get_model import get_model_class

    config = ModelConfig(name_or_path="circlestone-labs/Anima-Base-v1.0-Diffusers", arch="anima")

    model_class = get_model_class(config)

    assert model_class.__name__ == "AnimaModel"
    assert model_class.arch == "anima"


def test_anima_ui_uses_upstream_preset_with_repo_path():
    options = OPTIONS_SOURCE.read_text(encoding="utf-8")

    assert options.count("name: 'anima'") == 1
    assert (
        "'circlestone-labs/Anima-Base-v1.0-Diffusers', defaultNameOrPath"
        in options
    )
    assert "/datasets/studio/huggingface/models/Anima-Base-v1.0-Diffusers" not in options
    assert "/model/ModelScope/circlestone-labs/Anima-Base-v1.0-Diffusers" not in options
    assert "'config.process[0].model.quantize': [false, false]" in options
    assert "'config.process[0].model.qtype': ['', 'qfloat8']" in options
    assert "'config.process[0].model.qtype_te': ['', 'qfloat8']" in options
    assert "'config.process[0].train.timestep_type': ['weighted', 'sigmoid']" in options


def test_anima_uses_upstream_modular_pipeline():
    source = ANIMA_SOURCE.read_text(encoding="utf-8")

    assert ANIMA_SOURCE.is_file()
    assert not ANIMA_LEGACY_SOURCE.exists()
    assert "AnimaAutoBlocks" in source
    assert "AnimaModularPipeline" in source
    assert "AnimaTextConditioner" in source
    assert "AnimaEmbedsToImageBlocks" in source
    assert "AnimaAutoBlocks().init_pipeline" in source
    assert "pipe.load_components(**load_kwargs)" in source
    assert "return 16 * 2" in source
    assert '"shift": 3.0' in source


def test_anima_prompt_cache_and_text_conditioner_match_upstream():
    source = ANIMA_SOURCE.read_text(encoding="utf-8")
    prompt_utils = PROMPT_UTILS_SOURCE.read_text(encoding="utf-8")

    assert "class AnimaPromptEmbeds(PromptEmbeds)" in source
    assert '"qwen_prompt_embeds"' in source
    assert '"t5_input_ids"' in source
    assert "train_text_conditioner" in source
    assert 'self.target_lora_modules.append("AnimaTextConditioner")' in source
    assert 'metadata.get("class_name", "") == "AnimaPromptEmbeds"' in prompt_utils


def test_anima_uses_official_diffusers_pin():
    requirements = REQUIREMENTS_SOURCE.read_text(encoding="utf-8")

    assert (
        "git+https://github.com/huggingface/diffusers.git@"
        "c943837899b16cbae2f619b8dd4f7bb6f07dd81a"
    ) in requirements
    assert "DocWorkBox/diffusers.git" not in requirements


def test_anima_lora_conversion_supports_transformer_and_text_conditioner():
    source = ANIMA_SOURCE.read_text(encoding="utf-8")

    assert "def _strip_ai_toolkit_wrapper_prefix" in source
    assert "def _add_ai_toolkit_wrapper_prefix" in source
    assert 'key.startswith("text_conditioner.")' in source
    assert '"diffusion_model.llm_adapter."' in source
    assert "def convert_lora_weights_before_save" in source
    assert "def convert_lora_weights_before_load" in source
