from pathlib import Path


HUIHUI_MODEL_PATH = (
    "/datasets/studio/huggingface/models/"
    "Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated"
)


def test_qwen3_omni_uses_aigate_huihui_model_by_default():
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
    assert (
        f"'config.process[0].caption.model_name_or_path': "
        f"['{HUIHUI_MODEL_PATH}', defaultNameOrPath]"
    ) in section
    assert (
        f"value: '{HUIHUI_MODEL_PATH}', "
        "label: 'ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated'"
    ) in section
    assert f"model_name_or_path: '{HUIHUI_MODEL_PATH}'" in config_source
    assert f'"{HUIHUI_MODEL_PATH}":' in backend_source


def test_qwen3_omni_prefers_toolkit_local_base_assets():
    backend_source = Path(
        "extensions_built_in/captioner/Qwen3OmniCaptioner.py"
    ).read_text(encoding="utf-8")

    assert "def _resolve_base_assets(self, base_repo: str) -> str:" in backend_source
    assert 'TOOLKIT_ROOT, "models", os.path.basename(base_repo)' in backend_source
    assert "config = AutoConfig.from_pretrained(base_assets)" in backend_source
    assert "self.processor = AutoProcessor.from_pretrained(base_assets)" in backend_source
