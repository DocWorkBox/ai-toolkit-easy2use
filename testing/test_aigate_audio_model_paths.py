from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_missing_chengyu_audio_models_keep_official_repo_defaults():
    sources = "\n".join(
        _source(path)
        for path in (
            "extensions_built_in/audio_models/ui.tsx",
            "extensions_built_in/audio_models/yue2/yue2_model.py",
            "extensions_built_in/audio_models/yue2/src/tokenizer.py",
            "extensions_built_in/captioner/Qwen25OmniCaptioner.py",
            "extensions_built_in/llm_models/qwen25_omni.py",
            "extensions_built_in/llm_models/ui.tsx",
            "toolkit/audio/moss_music/__init__.py",
            "toolkit/audio/melbandroformer/separate.py",
            "toolkit/models/registry.py",
            "ui/src/helpers/captionOptions.ts",
        )
    )
    expected_paths = (
        "Comfy-Org/YuE2",
        "Qwen/Qwen2.5-Omni-7B",
        "OpenMOSS-Team/MOSS-Music-8B-Instruct",
        "ai-toolkit/Qwen2.5-Omni-7B",
        "m-a-p/MERT-v2-FullSong",
        "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4",
        "ai-toolkit/melbandroformer",
    )
    for path in expected_paths:
        assert path in sources


def test_melbandroformer_checks_project_cache_before_download():
    source = _source("toolkit/audio/melbandroformer/separate.py")
    cache_check = source.index('ckpt_dir = os.path.join(MODELS_PATH, "checkpoints")')
    download = source.index("hf_hub_download(repo_id=HF_REPO")
    assert cache_check < download
    assert "LOCAL_MODEL_DIR" not in source
