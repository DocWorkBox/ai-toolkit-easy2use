"""Update an archive-style Windows portable bundle from the protable branch."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from . import gitwin, util


REPOSITORY = "DocWorkBox/ai-toolkit-easy2use"
BRANCH = "protable"
MARKER_FILE = ".portable-branch.json"
LAUNCHER_FILE = "AI Toolkit Launcher.exe"
STATE_RELATIVE_PATH = Path(".cache/portable-update/state.json")
PENDING_LAUNCHER_RELATIVE_PATH = Path(
    ".cache/portable-update/AI Toolkit Launcher.update.exe"
)

_PROTECTED_ROOTS = {
    ".cache",
    ".git",
    "datasets",
    "models",
    "output",
    "runtime",
}
_PROTECTED_FILES = {
    "aitk_db.db",
    "aitk_db.db-shm",
    "aitk_db.db-wal",
}
_PROTECTED_UI_ROOTS = {
    "ui/.next",
    "ui/dist",
    "ui/node_modules",
}


class PortableUpdateError(RuntimeError):
    pass


def state_path(repo_root=util.REPO_ROOT):
    return Path(repo_root) / STATE_RELATIVE_PATH


def pending_launcher_path(repo_root=util.REPO_ROOT):
    return Path(repo_root) / PENDING_LAUNCHER_RELATIVE_PATH


def read_state(repo_root=util.REPO_ROOT):
    try:
        with state_path(repo_root).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(repo_root, commit, managed_files=None):
    target = state_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "commit": commit,
    }
    if managed_files is not None:
        payload["managed_files"] = sorted(set(managed_files))
    _write_json_atomic(target, payload)


def remote_status(repo_root=util.REPO_ROOT):
    state = read_state(repo_root)
    local_commit = state.get("commit") or "archive"
    try:
        remote_commit = _fetch_remote_commit()
    except (OSError, ValueError, PortableUpdateError) as error:
        return {
            "fetch_ok": False,
            "branch": BRANCH,
            "local_commit": local_commit,
            "remote_commit": None,
            "behind": None,
            "error": str(error),
        }
    return {
        "fetch_ok": True,
        "branch": BRANCH,
        "local_commit": local_commit,
        "remote_commit": remote_commit,
        "behind": 0 if local_commit == remote_commit else 1,
        "error": None,
    }


def update_from_remote(repo_root=util.REPO_ROOT, dry_run=False):
    status = remote_status(repo_root)
    if not status["fetch_ok"]:
        raise PortableUpdateError(
            status.get("error") or "Could not reach the portable update branch."
        )
    commit = status["remote_commit"]
    if status["behind"] == 0:
        util.ok("Portable code is already up to date.")
        return {
            "changed": False,
            "commit": commit,
            "files_updated": 0,
            "launcher_staged": False,
        }
    if dry_run:
        util.info("[dry-run] would update portable code to %s." % commit[:12])
        return {
            "changed": True,
            "commit": commit,
            "files_updated": 0,
            "launcher_staged": False,
        }

    cache_dir = state_path(repo_root).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=str(cache_dir)) as temp:
        archive_path = Path(temp) / "protable.zip"
        git_commit = _create_archive_from_git(archive_path, temp)
        if git_commit is not None:
            commit = git_commit
        else:
            url = "https://github.com/%s/archive/%s.zip" % (REPOSITORY, commit)
            util.download(url, str(archive_path), label="portable code")
        result = apply_archive(archive_path, commit, repo_root=repo_root)
    util.ok("Portable code updated to %s." % commit[:12])
    if result["launcher_staged"]:
        util.info(
            "A launcher update is staged and will be applied the next time "
            "AI Toolkit Launcher starts."
        )
    return result


def _create_archive_from_git(archive_path, temp_dir):
    git = gitwin.find_git()
    if not git:
        return None

    checkout = Path(temp_dir) / "checkout"
    repository_url = "https://github.com/%s.git" % REPOSITORY
    util.info("Fetching portable code with Git...")
    clone = subprocess.run(
        [
            git,
            "-c",
            "http.sslBackend=schannel",
            "clone",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            "--single-branch",
            "--no-tags",
            repository_url,
            str(checkout),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=util.clean_env(),
    )
    if clone.returncode != 0:
        detail = clone.stderr.decode("utf-8", errors="replace").strip()
        util.warn("Git fetch failed; retrying with the GitHub archive endpoint. %s" % detail)
        return None

    head = subprocess.run(
        [git, "-C", str(checkout), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=util.clean_env(),
    )
    commit = head.stdout.decode("ascii", errors="ignore").strip()
    if head.returncode != 0 or len(commit) < 12:
        util.warn("Could not read the fetched portable commit; retrying the archive endpoint.")
        return None

    prefix = "ai-toolkit-easy2use-%s/" % commit
    archived = subprocess.run(
        [
            git,
            "-C",
            str(checkout),
            "archive",
            "--format=zip",
            "--prefix=%s" % prefix,
            "--output=%s" % Path(archive_path).resolve(),
            "HEAD",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=util.clean_env(),
    )
    if archived.returncode != 0 or not Path(archive_path).is_file():
        detail = archived.stderr.decode("utf-8", errors="replace").strip()
        util.warn("Could not create the portable archive; retrying GitHub. %s" % detail)
        return None
    return commit


def apply_archive(archive_path, commit, repo_root=util.REPO_ROOT):
    repo_root = Path(repo_root).resolve()
    archive_path = Path(archive_path)
    extract_parent = archive_path.parent / ("extract-" + uuid.uuid4().hex)
    try:
        source_root = _extract_verified(archive_path, extract_parent)
        _validate_marker(source_root)

        previous_managed = _managed_files_from_state(read_state(repo_root))
        managed_files = []
        updated = 0
        launcher_staged = False
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            portable_path = relative.as_posix()
            if _is_protected(portable_path):
                continue
            if portable_path.casefold() == LAUNCHER_FILE.casefold():
                launcher_staged = _stage_launcher(source, repo_root)
                continue
            _copy_atomic(source, repo_root / relative)
            managed_files.append(portable_path)
            updated += 1

        removed = _remove_stale_managed_files(
            repo_root,
            previous_managed,
            set(managed_files),
        )
        save_state(repo_root, commit, managed_files=managed_files)
        return {
            "changed": True,
            "commit": commit,
            "files_updated": updated,
            "files_removed": removed,
            "launcher_staged": launcher_staged,
        }
    finally:
        shutil.rmtree(extract_parent, ignore_errors=True)


def _fetch_remote_commit():
    url = "https://api.github.com/repos/%s/commits/%s" % (REPOSITORY, BRANCH)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": util.USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except Exception as urllib_error:
        try:
            payload = _fetch_json_with_curl(url)
        except (OSError, ValueError, PortableUpdateError) as curl_error:
            raise PortableUpdateError(
                "Could not query origin/%s: %s; curl fallback failed: %s"
                % (BRANCH, urllib_error, curl_error)
            )
    commit = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(commit, str) or len(commit) < 12:
        raise PortableUpdateError("GitHub returned an invalid protable commit.")
    return commit


def _fetch_json_with_curl(url):
    curl = shutil.which("curl")
    if not curl:
        raise PortableUpdateError("curl is not available")
    result = subprocess.run(
        [
            curl,
            "-fsSL",
            "--retry",
            "3",
            "--max-time",
            "30",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "User-Agent: %s" % util.USER_AGENT,
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise PortableUpdateError(message or "curl exited with code %d" % result.returncode)
    return json.loads(result.stdout.decode("utf-8"))


def _extract_verified(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        roots = set()
        for entry in archive.infolist():
            if "\\" in entry.filename:
                raise PortableUpdateError("Unsafe path separator in portable update archive.")
            path = PurePosixPath(entry.filename)
            parts = path.parts
            if path.is_absolute() or ".." in parts or not parts:
                raise PortableUpdateError("Unsafe path in portable update archive.")
            mode = (entry.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise PortableUpdateError("Symlinks are not allowed in portable updates.")
            roots.add(parts[0])
        if len(roots) != 1:
            raise PortableUpdateError("Unexpected portable update archive layout.")
        archive.extractall(destination)
    source_root = destination / next(iter(roots))
    if not source_root.is_dir():
        raise PortableUpdateError("Portable update archive has no source root.")
    return source_root


def _validate_marker(source_root):
    try:
        with (source_root / MARKER_FILE).open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, ValueError) as error:
        raise PortableUpdateError("Portable branch marker is missing or invalid.") from error
    if (
        marker.get("schema") != 1
        or marker.get("repository") != REPOSITORY
        or marker.get("branch") != BRANCH
    ):
        raise PortableUpdateError("Portable branch marker does not match origin/protable.")


def _is_protected(relative_path):
    normalized = relative_path.replace("\\", "/").strip("/").casefold()
    if not normalized:
        return True
    first = normalized.split("/", 1)[0]
    if first in _PROTECTED_ROOTS or normalized in _PROTECTED_FILES:
        return True
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in _PROTECTED_UI_ROOTS
    )


def _stage_launcher(source, repo_root):
    if not _looks_like_windows_executable(source):
        raise PortableUpdateError("Portable branch launcher is not a Windows executable.")
    target = repo_root / LAUNCHER_FILE
    pending = pending_launcher_path(repo_root)
    if target.is_file() and _sha256(source) == _sha256(target):
        return False
    _copy_atomic(source, pending)
    return True


def _managed_files_from_state(state):
    managed = state.get("managed_files")
    if not isinstance(managed, list):
        return set()
    result = set()
    for value in managed:
        normalized = _normalize_managed_path(value)
        if normalized is not None:
            result.add(normalized)
    return result


def _normalize_managed_path(value):
    if not isinstance(value, str) or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    normalized = path.as_posix().strip("/")
    if (
        not normalized
        or _is_protected(normalized)
        or normalized.casefold() == LAUNCHER_FILE.casefold()
    ):
        return None
    return normalized


def _remove_stale_managed_files(repo_root, previous, current):
    removed = 0
    for relative in sorted(previous - current, reverse=True):
        normalized = _normalize_managed_path(relative)
        if normalized is None:
            continue
        target = repo_root.joinpath(*PurePosixPath(normalized).parts)
        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        parent = target.parent
        while parent != repo_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return removed


def _looks_like_windows_executable(path):
    try:
        with Path(path).open("rb") as handle:
            return handle.read(2) == b"MZ"
    except OSError:
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(target, payload):
    target = Path(target)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
