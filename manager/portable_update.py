"""Update an archive-style Windows portable bundle from the protable branch."""

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from . import portable_download, util


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

    result = _apply_incremental(commit, Path(repo_root).resolve())
    util.ok("Portable code updated to %s." % commit[:12])
    if result["launcher_staged"]:
        util.info(
            "A launcher update is staged and will be applied the next time "
            "AI Toolkit Launcher starts."
        )
    return result


def _fetch_tree(commit):
    payload = _fetch_json("https://api.github.com/repos/%s/git/trees/%s?recursive=1"
                          % (REPOSITORY, commit))
    if not isinstance(payload, dict) or payload.get("truncated") is not False:
        raise PortableUpdateError("GitHub file tree is invalid or truncated; update not applied.")
    if not isinstance(payload.get("tree"), list):
        raise PortableUpdateError("GitHub returned an invalid file tree.")
    return payload["tree"]


def _safe_relative(value):
    if not isinstance(value, str) or not value:
        raise PortableUpdateError("Unsafe update path: %r" % value)
    parts = value.split("/")
    for part in parts:
        if (not part or part in (".", "..") or part.endswith((".", " "))
                or re.search(r'[\\<>:"|?*\x00-\x1f]', part)
                or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise PortableUpdateError("Unsafe update path: %r" % value)
    return value


def _target(root, relative):
    target = root.joinpath(*_safe_relative(relative).split("/"))
    if root not in target.resolve().parents or target.is_symlink():
        raise PortableUpdateError("Unsafe linked update path: %s" % relative)
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0)
                                 & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise PortableUpdateError("Unsafe linked update path: %s" % relative)
    return target


def _manifest(entries):
    files = {}
    seen = set()
    for entry in entries:
        name = _safe_relative(entry.get("path"))
        key = name.casefold()
        if key in seen:
            raise PortableUpdateError("Windows path collision: %s" % name)
        seen.add(key)
        if _is_protected(name) or entry.get("type") in ("tree", "commit"):
            continue
        if (entry.get("type") != "blob" or entry.get("mode") not in ("100644", "100755")
                or not re.fullmatch("[0-9a-f]{40}", str(entry.get("sha", "")))
                or type(entry.get("size")) is not int or entry["size"] < 0):
            raise PortableUpdateError("Invalid or unsupported update file: %s" % name)
        files[name] = entry
    if MARKER_FILE not in files:
        raise PortableUpdateError("Portable branch marker is missing.")
    return files


def _blob_hash(path):
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path, entry, allow_crlf=False):
    if not path.is_file():
        return False
    if path.stat().st_size == entry["size"] and _blob_hash(path) == entry["sha"]:
        return True
    # Older Windows Git archives contain CRLF text. Do not redownload unchanged text.
    if allow_crlf and path.stat().st_size <= 8 * 1024 * 1024:
        data = path.read_bytes()
        if b"\0" not in data and b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
            return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == entry["sha"]
    return False


def _download_file(url, destination, **kwargs):
    portable_download.download(url, destination, **kwargs)


class _DownloadProgress:
    def __init__(self, files, total):
        self.files, self.total = files, total
        self.done = self.completed = 0
        self.current = 0
        self.started = self.last = time.monotonic()
        self.name = ""

    def report(self, count, force=False):
        self.current = count
        now = time.monotonic()
        if not force and now - self.last < 1:
            return
        received = self.completed + count
        speed = received / max(now - self.started, 0.001)
        pct = received * 100 / self.total if self.total else 100
        util.info("Download: %d/%d files | %.1f%% | %.2f/%.2f MiB | %.1f KiB/s | %s"
                  % (self.done, self.files, pct, received / 1048576,
                     self.total / 1048576, speed / 1024, self.name))
        self.last = now


def _apply_incremental(commit, root):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise PortableUpdateError("Invalid portable commit.")
    util.info("Reading file manifest for origin/%s @ %s..." % (BRANCH, commit[:12]))
    manifest = _manifest(_fetch_tree(commit))
    previous = _managed_files_from_state(read_state(root))
    managed = set(manifest) - {LAUNCHER_FILE}
    cache = _target(root, ".cache/portable-update/blobs")
    pending = _target(root, PENDING_LAUNCHER_RELATIVE_PATH.as_posix())
    plan = []
    refresh_pending = None
    launcher_staged = False
    util.info("Checking %d local files; unchanged files will not be downloaded..." % len(manifest))
    for name, entry in sorted(manifest.items()):
        target = _target(root, name)
        if name == LAUNCHER_FILE and _matches(pending, entry):
            launcher_staged = True
            continue
        if _matches(target, entry, allow_crlf=True):
            if name == LAUNCHER_FILE and pending.exists():
                refresh_pending = target
        else:
            plan.append((name, entry))
    for name in previous - managed:
        _target(root, name)
    needed = [(name, entry) for name, entry in plan if not _matches(cache / entry["sha"], entry)]
    total = sum(entry["size"] for _, entry in needed)
    util.info("Incremental update: %d files to apply, %d to download (%.2f MiB); %d unchanged."
              % (len(plan), len(needed), total / 1048576, len(manifest) - len(plan)))
    cache.mkdir(parents=True, exist_ok=True)
    progress = _DownloadProgress(len(needed), total)
    for name, entry in needed:
        progress.name = name
        progress.report(0, force=True)
        target = cache / entry["sha"]
        partial = target.with_suffix("." + uuid.uuid4().hex + ".part")
        url = "https://raw.githubusercontent.com/%s/%s/%s" % (
            REPOSITORY, commit, urllib.parse.quote(name, safe="/"))
        _download_file(url, partial, progress=progress.report)
        if not _matches(partial, entry):
            raise PortableUpdateError("Hash verification failed for %s; update not applied." % name)
        os.replace(partial, target)
        progress.done += 1
        progress.report(entry["size"], force=True)
        progress.completed += entry["size"]
    sources = {name: cache / entry["sha"] for name, entry in plan}
    if refresh_pending is not None:
        sources[LAUNCHER_FILE] = refresh_pending
    marker = sources.get(MARKER_FILE, root / MARKER_FILE)
    _validate_marker_file(marker)
    if LAUNCHER_FILE in sources and not _looks_like_windows_executable(sources[LAUNCHER_FILE]):
        raise PortableUpdateError("Portable branch launcher is not a Windows executable.")
    # Network failures cannot leave a partially applied installation.
    util.info("All downloads verified. Applying %d changed files..." % len(sources))
    updated = 0
    for name, source in sources.items():
        if name == LAUNCHER_FILE:
            launcher_staged = _stage_launcher(source, root)
        else:
            _copy_atomic(source, _target(root, name))
            updated += 1
    removed = _remove_stale_managed_files(root, previous, managed)
    save_state(root, commit, managed_files=managed)
    return {"changed": True, "commit": commit, "files_updated": updated,
            "files_removed": removed, "launcher_staged": launcher_staged,
            "files_downloaded": len(needed), "bytes_downloaded": total}


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
    payload = _fetch_json(url)
    commit = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(commit, str) or len(commit) < 12:
        raise PortableUpdateError("GitHub returned an invalid protable commit.")
    return commit


def _api_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": util.USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _fetch_json(url):
    request = urllib.request.Request(url, headers=_api_headers())
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
    return payload


def _fetch_json_with_curl(url):
    with tempfile.TemporaryDirectory(prefix="aitk-update-metadata-") as temp:
        target = Path(temp) / "response.json"
        portable_download.download(url, target, headers=_api_headers(), transport="curl")
        return json.loads(target.read_text(encoding="utf-8"))


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
    _validate_marker_file(source_root / MARKER_FILE)


def _validate_marker_file(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
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
        if not pending.is_file():
            return False
        if _sha256(source) == _sha256(pending):
            return True
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
