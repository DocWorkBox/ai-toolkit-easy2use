from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_portable_audio_models_use_local_defaults_and_official_download_links():
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
            "portable_models.json",
        )
    )
    expected_local_paths = (
        "./models/checkpoints/yue2_3b_int8_convrot.safetensors",
        "./models/text_encoders/qwen2_5_omni_7b_convrot8.safetensors",
        "./models/Qwen2.5-Omni-7B",
        "./models/MOSS-Music-8B-Instruct",
        "./models/MERT-v2-FullSong",
        "./models/yue2-mothersuperior-realaudio-tokenizer-v4",
        "./models/SheetSage2",
        "./models/audio_encoders/sheetsage2_bf16.safetensors",
        "./models/checkpoints/melbandroformer_vocals_kj.safetensors",
    )
    for path in expected_local_paths:
        assert path in sources
    assert "https://huggingface.co/Comfy-Org/YuE2" in sources
    assert "https://huggingface.co/Qwen/Qwen2.5-Omni-7B" in sources
    assert "https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct" in sources
    assert "/datasets/" not in sources
    assert "/model/" not in sources


def test_melbandroformer_uses_project_cache_before_download():
    source = _source("toolkit/audio/melbandroformer/separate.py")
    cache_check = source.index('ckpt_dir = os.path.join(MODELS_PATH, "checkpoints")')
    download = source.index("hf_hub_download(repo_id=HF_REPO")
    assert cache_check < download
    assert "LOCAL_MODEL_DIR" not in source
