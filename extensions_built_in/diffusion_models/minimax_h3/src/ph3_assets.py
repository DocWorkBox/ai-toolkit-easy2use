"""Cache the P-H3 tokenizer, processor and TE config in writable project storage."""

import os
import shutil
import tempfile

from huggingface_hub.utils import validate_repo_id

from toolkit.paths import TOOLKIT_ROOT


BASE_REPO = "PrunaAI/P-H3-Image"
REQUIRED_FILES = (
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja",
    "processor/processor_config.json", "processor/tokenizer.json", "processor/tokenizer_config.json",
    "processor/chat_template.jinja", "text_encoder/config.json",
)
ASSET_FILES = REQUIRED_FILES + ("text_encoder/generation_config.json",)


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _has_assets(directory):
    return all(_nonempty(os.path.join(directory, name)) for name in REQUIRED_FILES)


def _copy_assets(source, directory):
    for filename in ASSET_FILES:
        src, dst = os.path.join(source, filename), os.path.join(directory, filename)
        if _nonempty(dst) or not _nonempty(src):
            continue
        parent = os.path.dirname(dst)
        os.makedirs(parent, exist_ok=True)
        # Publish complete files so parallel jobs never see a partial JSON or tokenizer.
        with tempfile.NamedTemporaryFile(dir=parent, suffix=".tmp", delete=False) as f:
            temporary = f.name
        try:
            shutil.copyfile(src, temporary)
            os.replace(temporary, dst)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


def resolve_ph3_assets(source: str, status_fn=None) -> str:
    if os.path.isdir(source):
        return source
    validate_repo_id(source)
    relative = [os.path.basename(source)] if source == BASE_REPO else source.split("/")
    directory = os.path.join(TOOLKIT_ROOT, "models", *relative)
    if _has_assets(directory):
        return directory
    os.makedirs(directory, exist_ok=True)

    from toolkit.chengyu_model_storage import seed_project_assets

    seed_project_assets(source, directory)
    if _has_assets(directory):
        return directory

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN", None)
    try:
        cached = snapshot_download(
            repo_id=source, allow_patterns=list(ASSET_FILES), token=token, local_files_only=True,
        )
    except OSError:
        cached = None
    if cached:
        _copy_assets(cached, directory)
    if _has_assets(directory):
        return directory

    missing = [name for name in ASSET_FILES if not _nonempty(os.path.join(directory, name))]
    if status_fn is not None:
        status_fn(f"Preparing P-H3-Image config/tokenizer/processor from {source} in {directory}")
    kwargs = dict(repo_id=source, allow_patterns=missing, local_dir=directory, token=token)
    try:
        downloaded = snapshot_download(**kwargs)
    except RuntimeError as error:
        if "Cannot send a request, as the client has been closed." not in str(error):
            raise
        from huggingface_hub.utils import close_session

        close_session()
        if status_fn is not None:
            status_fn("Recreating closed Hugging Face HTTP client and retrying P-H3-Image assets once")
        downloaded = snapshot_download(**kwargs)
    _copy_assets(downloaded, directory)
    if not _has_assets(directory):
        raise FileNotFoundError(f"Incomplete P-H3-Image config/tokenizer/processor assets: {directory}")
    return directory
