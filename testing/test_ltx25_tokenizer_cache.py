from pathlib import Path


MODEL_SOURCE = Path("extensions_built_in/diffusion_models/ltx2/ltx2.py")


def test_ltx25_tokenizer_assets_use_persistent_toolkit_cache():
    source = MODEL_SOURCE.read_text(encoding="utf-8")

    assert "from toolkit.paths import MODELS_PATH, TOOLKIT_ROOT" in source
    assert 'TOOLKIT_ROOT, ".cache", "ltx2.5"' in source
    assert 'os.path.basename(te_path)' in source
    assert 'os.replace(tmp_path, out_path)' in source
    assert 'os.path.splitext(te_path)[0] + "_hf_assets"' not in source
