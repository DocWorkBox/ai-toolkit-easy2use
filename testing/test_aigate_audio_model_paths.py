from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_aigate_audio_models_use_server_local_paths():
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
        "/datasets/studio/huggingface/models/YuE2",
        "/datasets/studio/huggingface/models/Qwen2.5-Omni-7B",
        "/datasets/studio/huggingface/models/MOSS-Music-8B-Instruct",
        "/datasets/studio/huggingface/models/ai-tookit-Qwen2.5-Omni-7B",
        "/datasets/studio/huggingface/models/MERT-v2-FullSong",
        "/datasets/studio/huggingface/models/yue2-mothersuperior-realaudio-tokenizer-v4",
        "/datasets/studio/huggingface/models/melbandroformer",
    )
    for path in expected_paths:
        assert path in sources


def test_melbandroformer_checks_server_directory_before_download_cache():
    source = _source("toolkit/audio/melbandroformer/separate.py")
    local_check = source.index("local_path = os.path.join(LOCAL_MODEL_DIR, filename)")
    cache_check = source.index('ckpt_dir = os.path.join(MODELS_PATH, "checkpoints")')
    download = source.index("hf_hub_download(repo_id=HF_REPO")
    assert local_check < cache_check < download
