"""Creator-volume HF snapshots and project-local small-asset caches."""

import fnmatch
import json
import os
from pathlib import Path
import shutil
import tempfile

from toolkit.paths import MODELS_PATH


MODEL_STORE = Path(MODELS_PATH)
MANIFEST = Path(__file__).resolve().parents[1] / "scripts" / "chengyu_models.json"
GROUPS = json.loads(MANIFEST.read_text(encoding="utf-8"))["groups"]
SMALL_PATTERNS = ("*.json", "*.py", "*.txt", "*.model", "*.jinja", "*.yaml", "*.yml", "LICENSE*")
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _complete_snapshot(directory, repo_id):
    try:
        marker = json.loads((directory / ".ai-toolkit-complete.json").read_text(encoding="utf-8"))
        if marker.get("repo_id") != repo_id or not marker.get("files"):
            return False
        for item in marker["files"]:
            relative = Path(item["filename"])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = directory / relative
            if not path.is_file() or path.stat().st_size != item["size"]:
                return False
        return (directory / "config.json").is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def resolve_hf_source(source):
    """Use complete registered HF folders offline; repair missing files in place."""
    group = next((group for group in GROUPS
                  if group.get("kind") in {"snapshot", "code"}
                  and source in (group["repo_id"], str(MODEL_STORE / group["local_dir"]))), None)
    if group is None:
        return source
    directory = MODEL_STORE / group["local_dir"]
    if _complete_snapshot(directory, group["repo_id"]):
        return str(directory)
    from huggingface_hub import snapshot_download

    patterns = SMALL_PATTERNS + (WEIGHT_PATTERNS if group["kind"] == "snapshot" else ())
    snapshot_download(
        repo_id=group["repo_id"], local_dir=str(directory), allow_patterns=list(patterns),
        token=os.getenv("HF_TOKEN", None),
    )
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing model configuration after download: {directory}")
    files = [{"filename": str(path.relative_to(directory)).replace(os.sep, "/"), "size": path.stat().st_size}
             for path in directory.rglob("*") if path.is_file()
             and not any(part.startswith(".") for part in path.relative_to(directory).parts)
             and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)]
    marker = {"repo_id": group["repo_id"], "files": files}
    temporary = directory / f".ai-toolkit-complete.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(marker), encoding="utf-8")
    os.replace(temporary, directory / ".ai-toolkit-complete.json")
    return str(directory)


def seed_project_assets(repo_id, destination):
    """Copy only existing small assets from poddata; never overwrite user caches."""
    group = next((group for group in GROUPS if group.get("kind") == "assets" and group["repo_id"] == repo_id), None)
    if group is None:
        return
    source = MODEL_STORE / group["local_dir"]
    if not source.is_dir():
        return
    destination = Path(destination)
    for directory, dirs, names in os.walk(source):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in names:
            if name.startswith(".") or not any(fnmatch.fnmatch(name, pattern) for pattern in SMALL_PATTERNS):
                continue
            path = Path(directory) / name
            size = path.stat().st_size
            if not 0 < size <= 128 * 2**20:
                continue
            target = destination / path.relative_to(source)
            if target.is_file() and target.stat().st_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
            try:
                shutil.copyfile(path, temporary)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
