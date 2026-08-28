from pathlib import Path


DEFAULT_CHECKPOINT = (
    "./models/text_encoders/"
    "qwen3_omni_30b_a3b_thinking_convrot8.safetensors"
)
HUIHUI_CHECKPOINT = (
    "./models/text_encoders/"
    "huihui_qwen3_omni_30b_a3b_thinking_abliterated_convrot8.safetensors"
)


def test_qwen3_omni_preserves_portable_local_defaults():
    options_source = Path("ui/src/helpers/captionOptions.ts").read_text(
        encoding="utf-8"
    )
    config_source = Path("ui/src/helpers/captionJobConfig.ts").read_text(
        encoding="utf-8"
    )
    backend_source = Path(
        "extensions_built_in/captioner/Qwen3OmniCaptioner.py"
    ).read_text(encoding="utf-8")

    section = options_source.split("name: 'Qwen3OmniCaptioner'", 1)[1].split(
        "name: 'Ideogram4Captioner'", 1
    )[0]
    assert DEFAULT_CHECKPOINT in section
    assert HUIHUI_CHECKPOINT in section
    assert f"model_name_or_path: '{DEFAULT_CHECKPOINT}'" in config_source
    assert "ai-toolkit/Qwen3-Omni-" not in section
    assert "/datasets/studio/huggingface/models" not in section
    assert "/datasets/studio/huggingface/models" not in backend_source
    assert '"base_repo": "./models/Qwen3-Omni-30B-A3B-Thinking"' in backend_source


def test_qwen3_omni_requires_complete_toolkit_local_base_assets():
    backend_source = Path(
        "extensions_built_in/captioner/Qwen3OmniCaptioner.py"
    ).read_text(encoding="utf-8")

    assert "def _resolve_base_repo(self) -> str:" in backend_source
    assert "from toolkit.paths import get_path" in backend_source
    assert 'base_repo = get_path(self._model_info["base_repo"])' in backend_source
    assert "Qwen3-Omni processor metadata is incomplete" in backend_source
    assert "config = AutoConfig.from_pretrained(base_repo, local_files_only=True)" in backend_source
    assert (
        "self.processor = AutoProcessor.from_pretrained(base_repo, local_files_only=True)"
        in backend_source
    )
    assert 'stale_cache = getattr(self.model, "_cache", None)' in backend_source
    assert "if stale_cache is not None and not stale_cache.is_initialized:" in backend_source
