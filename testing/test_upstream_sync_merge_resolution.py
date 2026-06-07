import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_base_captioner_source_keeps_remote_api_fields_and_newline_writes():
    source = Path("extensions_built_in/captioner/BaseCaptioner.py").read_text(
        encoding="utf-8"
    )

    assert 'self.api_base_url = kwargs.get("api_base_url", None)' in source
    assert 'self.api_key = kwargs.get("api_key", None)' in source
    assert 'self.api_protocol = kwargs.get("api_protocol", "openai")' in source
    assert "self.api_concurrency = max(1, int(kwargs.get(\"api_concurrency\", 8) or 8))" in source
    assert 'with open(caption_file_path, "w", encoding="utf-8", newline="\\n") as f:' in source


def test_config_modules_source_keeps_upstream_loss_and_in_context_fields():
    source = Path("toolkit/config_modules.py").read_text(encoding="utf-8")

    assert "pseudo_huber" in source
    assert 'self.in_context = kwargs.get("in_context", False)' in source
