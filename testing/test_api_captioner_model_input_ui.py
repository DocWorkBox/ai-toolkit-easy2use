from pathlib import Path


def test_api_captioners_force_model_name_to_plain_input():
    form_inputs_source = Path("ui/src/components/formInputs.tsx").read_text(encoding="utf-8")
    caption_source = Path("ui/src/components/CaptionSimpleJob.tsx").read_text(encoding="utf-8")

    assert "forceCustomInput?: boolean" in form_inputs_source
    assert "forceCustomInput ? (" in form_inputs_source
    assert "forceCustomInput={isRemoteApiCaptioner}" in caption_source


def test_api_captioner_settings_are_persisted_and_default_to_conservative_concurrency():
    caption_source = Path("ui/src/components/CaptionSimpleJob.tsx").read_text(encoding="utf-8")
    options_source = Path("ui/src/helpers/captionOptions.ts").read_text(encoding="utf-8")
    base_captioner_source = Path("extensions_built_in/captioner/BaseCaptioner.py").read_text(encoding="utf-8")

    assert "const DEFAULT_API_CONCURRENCY = 8" in caption_source
    assert "const API_CAPTIONER_STORAGE_KEY = 'AITK_CAPTION_API_SETTINGS'" in caption_source
    assert "restoreStoredCaptionApiSettings(setJobConfig)" in caption_source
    assert "writeStoredCaptionApiSettings" in caption_source
    assert "'config.process[0].caption.api_concurrency': [8, undefined]" in options_source
    assert 'self.api_concurrency = max(1, int(kwargs.get("api_concurrency", 8) or 8))' in base_captioner_source


def test_aigate_local_qwen_paths_do_not_leak_into_api_captioner():
    options_source = Path("ui/src/helpers/captionOptions.ts").read_text(encoding="utf-8")
    qwen_section = options_source.split("name: 'Qwen3VLCaptioner'", 1)[1].split(
        "name: 'RemoteAPICaptioner'", 1
    )[0]
    api_section = options_source.split("name: 'RemoteAPICaptioner'", 1)[1].split(
        "name: 'Ideogram4Captioner'", 1
    )[0]
    local_paths = [
        "/datasets/studio/huggingface/models/Qwen3.6-27B",
        "/datasets/studio/huggingface/models/Huihui-Qwen3.6-27B-abliterated",
        "/datasets/studio/huggingface/models/Huihui-Qwen3-VL-8B-Instruct-abliterated",
    ]

    for path in local_paths:
        assert path in qwen_section
        assert path not in api_section

    assert (
        "'config.process[0].caption.model_name_or_path': ['', defaultNameOrPath]"
        in api_section
    )
