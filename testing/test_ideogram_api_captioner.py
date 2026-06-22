import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ideogram_api_captioner_is_registered_and_reuses_remote_api():
    init_source = Path("extensions_built_in/captioner/__init__.py").read_text(encoding="utf-8")
    api_source = Path("extensions_built_in/captioner/Ideogram4APICaptioner.py").read_text(encoding="utf-8")

    assert 'uid = "Ideogram4APICaptioner"' in init_source
    assert "Ideogram4APICaptionerExtension" in init_source
    assert "RemoteAPICaptioner" in api_source
    assert "class Ideogram4APICaptioner(RemoteAPICaptioner)" in api_source


def test_ideogram_api_captioner_builds_json_prompt_and_normalizes_response():
    api_source = Path("extensions_built_in/captioner/Ideogram4APICaptioner.py").read_text(encoding="utf-8")

    assert "ideogram4_caption_prompt" in api_source
    assert "build_prompt_for_file" in api_source
    assert "compute_aspect_ratio" in api_source
    assert "_extract_json" in api_source
    assert "_normalize_caption" in api_source
    assert "_sanitize_palette" not in api_source
    assert "swap_bbox_xy_in_text(output_text)" in api_source
    assert "json.dumps(data, ensure_ascii=False, indent=2)" in api_source


def test_ideogram_api_captioner_ui_option_combines_api_and_ideogram_defaults():
    source = Path("ui/src/helpers/captionOptions.ts").read_text(encoding="utf-8")

    assert "name: 'Ideogram4APICaptioner'" in source
    assert "label: 'Ideogram 4 Captioner API'" in source
    assert "'config.process[0].caption.api_base_url': ['', undefined]" in source
    assert "'config.process[0].caption.api_key': ['', undefined]" in source
    assert "'config.process[0].caption.api_protocol': ['openai', undefined]" in source
    assert "'config.process[0].caption.max_new_tokens': [4096, undefined]" in source
    assert "'config.process[0].caption.api_concurrency': [8, undefined]" in source
    assert "'caption.api_base_url'" in source
    assert "'caption.api_key'" in source
    assert "'caption.api_protocol'" in source
    assert "'caption.api_concurrency'" in source
