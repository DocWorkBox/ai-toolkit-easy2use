"""Persist small MOSS assets under the toolkit, not the Comfy model mount."""

import os
import shutil
import tempfile

from huggingface_hub.utils import validate_repo_id

from toolkit.paths import TOOLKIT_ROOT


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
ASSET_FILES = {"config.json", "preprocessor_config.json"} | {
    f"tokenizer/{name}" for name in TOKENIZER_FILES
}


def _complete(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _copy_asset(source: str, target: str):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(target), prefix=".moss-asset-", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _project_asset(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    validate_repo_id(repo_id)
    if filename not in ASSET_FILES:
        raise ValueError(f"not a MOSS small asset: {filename}")
    root = os.path.join(TOOLKIT_ROOT, "models", *repo_id.split("/"))
    target = os.path.join(root, *filename.split("/"))
    if _complete(target):
        return target

    token = os.getenv("HF_TOKEN", None)
    try:
        cached = hf_hub_download(repo_id, filename, token=token, local_files_only=True)
    except Exception:
        cached = None
    if cached and _complete(cached):
        _copy_asset(cached, target)
        return target

    source = hf_hub_download(repo_id, filename, token=token, local_dir=root)
    if not _complete(source):
        raise FileNotFoundError(f"missing or empty MOSS asset: {repo_id}/{filename}")
    if os.path.abspath(source) != os.path.abspath(target):
        _copy_asset(source, target)
    return target


def hub_tokenizer_folder(repo_id: str) -> str:
    for filename in TOKENIZER_FILES:
        path = _project_asset(repo_id, f"tokenizer/{filename}")
    return os.path.dirname(path)


def hub_file_folder(repo_id: str, filename: str) -> str:
    return os.path.dirname(_project_asset(repo_id, filename))
