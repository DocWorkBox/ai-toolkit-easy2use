"""Keep Qwen3.8 configs and processor assets off read-only checkpoint mounts."""

import json
import os
import shutil
import tempfile

from huggingface_hub.utils import validate_repo_id

from toolkit.paths import TOOLKIT_ROOT


BASE_REPO = "Qwen/Qwen3.8-27B"
ASSET_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "processor_config.json", "chat_template.json", "chat_template.jinja",
)


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _has_assets(directory):
    def exists(filename):
        return _nonempty(os.path.join(directory, filename))

    if not all(exists(name) for name in (
        "config.json", "tokenizer_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json",
    )):
        return False
    if not (exists("tokenizer.json") or (exists("vocab.json") and exists("merges.txt"))):
        return False
    if exists("chat_template.json") or exists("chat_template.jinja"):
        return True
    try:
        with open(os.path.join(directory, "tokenizer_config.json"), encoding="utf-8") as f:
            return bool(json.load(f).get("chat_template"))
    except (OSError, ValueError):
        return False


def _copy_assets(source, target):
    for filename in ASSET_FILES:
        src, dst = os.path.join(source, filename), os.path.join(target, filename)
        if _nonempty(dst) or not _nonempty(src):
            continue
        # Parallel jobs must only see complete files in the shared project cache.
        with tempfile.NamedTemporaryFile(dir=target, suffix=".tmp", delete=False) as f:
            temporary = f.name
        try:
            shutil.copyfile(src, temporary)
            os.replace(temporary, dst)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


def resolve_qwen38_assets(source: str, status_fn=None) -> str:
    """Reuse local assets, otherwise cache only small Hub files under project models."""
    if os.path.isdir(source):
        return source
    validate_repo_id(source)
    relative = [os.path.basename(source)] if source == BASE_REPO else source.split("/")
    directory = os.path.join(TOOLKIT_ROOT, "models", *relative)
    if _has_assets(directory):
        return directory
    os.makedirs(directory, exist_ok=True)

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN", None)
    try:
        cached = snapshot_download(
            repo_id=source, allow_patterns=list(ASSET_FILES),
            token=token, local_files_only=True,
        )
    except OSError:
        cached = None
    if cached:
        _copy_assets(cached, directory)
    if _has_assets(directory):
        return directory

    missing = [name for name in ASSET_FILES if not _nonempty(os.path.join(directory, name))]
    if status_fn is not None:
        status_fn(f"Preparing Qwen3.8 config/tokenizer/processor from {source} in {directory}")
    kwargs = dict(repo_id=source, allow_patterns=missing, local_dir=directory, token=token)
    try:
        downloaded = snapshot_download(**kwargs)
    except RuntimeError as error:
        if "Cannot send a request, as the client has been closed." not in str(error):
            raise
        from huggingface_hub.utils import close_session

        close_session()
        if status_fn is not None:
            status_fn("Recreating closed Hugging Face HTTP client and retrying Qwen3.8 assets once")
        downloaded = snapshot_download(**kwargs)
    _copy_assets(downloaded, directory)
    if not _has_assets(directory):
        raise FileNotFoundError(f"Incomplete Qwen3.8 config/tokenizer/processor assets: {directory}")
    return directory
