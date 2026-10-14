import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

from manager import portable_update as updater


COMMIT = "a" * 40


def blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def setup_remote(monkeypatch, files):
    monkeypatch.setattr(updater, "_create_archive_from_git",
                        lambda *args: pytest.fail("Full Git clone must not run"), raising=False)
    monkeypatch.setattr(updater.util, "download",
                        lambda *args, **kwargs: pytest.fail("Full archive must not download"))
    files = dict(files)
    files[updater.MARKER_FILE] = json.dumps({
        "schema": 1, "repository": updater.REPOSITORY, "branch": updater.BRANCH,
    }).encode()
    manifest = [{"path": name, "sha": blob(data), "size": len(data),
                 "type": "blob", "mode": "100644"} for name, data in files.items()]
    monkeypatch.setattr(updater, "_fetch_remote_commit", lambda: COMMIT)
    monkeypatch.setattr(updater, "_fetch_tree", lambda commit: manifest, raising=False)
    downloaded = []

    def download(url, destination, **kwargs):
        name = unquote(url.split(COMMIT + "/", 1)[1])
        downloaded.append(name)
        Path(destination).write_bytes(files[name])
        kwargs["progress"](len(files[name]))

    # Patch only the network boundary; planning, hashing, staging and applying are real.
    monkeypatch.setattr(updater, "_download_file", download, raising=False)
    return files, downloaded


def write_files(root, files):
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_only_changed_files_download_and_old_bundle_needs_no_git(monkeypatch, tmp_path):
    files, downloaded = setup_remote(monkeypatch, {
        "manager/new.py": b"new\n", "manager/unchanged.py": b"same\n",
        updater.LAUNCHER_FILE: b"MZ-existing-exe", "models/user.bin": b"do not install",
        "aitk_db.db": b"do not install", "ui/.next/BUILD_ID": b"do not install",
    })
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE],
                           "manager/unchanged.py": b"same\r\n",
                           updater.LAUNCHER_FILE: files[updater.LAUNCHER_FILE],
                           "models/user.bin": b"user model", "aitk_db.db": b"user db",
                           "manager/removed.py": b"old", "manager/user.py": b"private"})
    updater.save_state(tmp_path, "b" * 40, ["manager/removed.py", "manager/unchanged.py"])

    result = updater.update_from_remote(tmp_path)

    assert downloaded == ["manager/new.py"]
    assert (tmp_path / "manager/new.py").read_bytes() == b"new\n"
    assert (tmp_path / "manager/unchanged.py").read_bytes() == b"same\r\n"
    assert (tmp_path / "models/user.bin").read_bytes() == b"user model"
    assert (tmp_path / "aitk_db.db").read_bytes() == b"user db"
    assert (tmp_path / "manager/user.py").read_bytes() == b"private"
    assert not (tmp_path / "manager/removed.py").exists()
    assert result["files_updated"] == 1
    assert result["files_removed"] == 1
    assert not result["launcher_staged"]
    assert updater.read_state(tmp_path)["commit"] == COMMIT


def test_no_program_changes_until_all_downloads_are_verified(monkeypatch, tmp_path):
    files, _ = setup_remote(monkeypatch, {"a.py": b"new a", "b.py": b"new b"})
    write_files(tmp_path, {"a.py": b"old a", updater.MARKER_FILE: files[updater.MARKER_FILE]})
    updater.save_state(tmp_path, "b" * 40, ["a.py"])
    network = updater._download_file

    def broken(url, destination, **kwargs):
        if url.endswith("b.py"):
            raise OSError("connection interrupted")
        network(url, destination, **kwargs)

    monkeypatch.setattr(updater, "_download_file", broken)
    with pytest.raises((OSError, updater.PortableUpdateError), match="interrupted"):
        updater.update_from_remote(tmp_path)
    assert (tmp_path / "a.py").read_bytes() == b"old a"
    assert updater.read_state(tmp_path)["commit"] == "b" * 40

    calls = []
    def retry(url, destination, **kwargs):
        calls.append(url)
        network(url, destination, **kwargs)
    monkeypatch.setattr(updater, "_download_file", retry)
    updater.update_from_remote(tmp_path)
    assert len(calls) == 1 and calls[0].endswith("b.py")


def test_hash_mismatch_cannot_install_or_advance_state(monkeypatch, tmp_path):
    setup_remote(monkeypatch, {"a.py": b"expected"})
    monkeypatch.setattr(updater, "_download_file",
                        lambda url, dest, **kw: Path(dest).write_bytes(b"corrupted"))
    with pytest.raises(updater.PortableUpdateError, match="[Hh]ash|[Vv]erif"):
        updater.update_from_remote(tmp_path)
    assert not (tmp_path / "a.py").exists()
    assert updater.read_state(tmp_path) == {}


def test_changed_launcher_is_staged_and_pending_launcher_not_downloaded_twice(monkeypatch, tmp_path):
    files, downloaded = setup_remote(monkeypatch, {updater.LAUNCHER_FILE: b"MZ-new"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE],
                           updater.LAUNCHER_FILE: b"MZ-old"})
    result = updater.update_from_remote(tmp_path)
    assert result["launcher_staged"]
    assert (tmp_path / updater.LAUNCHER_FILE).read_bytes() == b"MZ-old"
    assert updater.pending_launcher_path(tmp_path).read_bytes() == b"MZ-new"
    updater.save_state(tmp_path, "b" * 40)
    downloaded.clear()
    result = updater.update_from_remote(tmp_path)
    assert downloaded == []
    assert result["launcher_staged"]


@pytest.mark.parametrize("path", ["../outside.py", "C:/outside.py", "dir\\bad.py",
                                      "file:stream", "a/../b", "a./b", "CON", "/root"])
def test_manifest_rejects_windows_unsafe_paths(monkeypatch, tmp_path, path):
    setup_remote(monkeypatch, {path: b"bad"})
    with pytest.raises(updater.PortableUpdateError, match="[Uu]nsafe"):
        updater.update_from_remote(tmp_path)


def test_manifest_rejects_case_collisions(monkeypatch, tmp_path):
    setup_remote(monkeypatch, {"a.py": b"first", "A.py": b"second"})
    with pytest.raises(updater.PortableUpdateError, match="[Cc]ollision"):
        updater.update_from_remote(tmp_path)


def test_truncated_tree_is_not_applied(monkeypatch):
    assert hasattr(updater, "_fetch_tree"), "Incremental tree reader is missing"
    monkeypatch.setattr(updater, "_fetch_json", lambda url: {"truncated": True, "tree": []})
    with pytest.raises(updater.PortableUpdateError, match="truncated"):
        updater._fetch_tree(COMMIT)


def test_dry_run_does_not_download(monkeypatch, tmp_path):
    _, downloaded = setup_remote(monkeypatch, {"new.py": b"new"})
    assert updater.update_from_remote(tmp_path, dry_run=True)["changed"]
    assert downloaded == []
    assert not updater.state_path(tmp_path).exists()


def test_corrupt_cache_is_redownloaded_and_empty_files_work(monkeypatch, tmp_path):
    files, downloaded = setup_remote(monkeypatch, {"empty.py": b"", "code.py": b"correct"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE],
                           ".cache/portable-update/blobs/" + blob(b"correct"): b"corrupt"})
    updater.update_from_remote(tmp_path)
    assert downloaded == ["code.py", "empty.py"]
    assert (tmp_path / "empty.py").read_bytes() == b""
    assert (tmp_path / "code.py").read_bytes() == b"correct"


def test_binary_line_endings_are_not_normalized(monkeypatch, tmp_path):
    files, downloaded = setup_remote(monkeypatch, {"binary.dat": b"\0x\n"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE], "binary.dat": b"\0x\r\n"})
    updater.update_from_remote(tmp_path)
    assert downloaded == ["binary.dat"]


def test_invalid_launcher_prevents_all_program_changes(monkeypatch, tmp_path):
    files, _ = setup_remote(monkeypatch, {"a.py": b"new", updater.LAUNCHER_FILE: b"not an exe"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE], "a.py": b"old"})
    with pytest.raises(updater.PortableUpdateError, match="not a Windows executable"):
        updater.update_from_remote(tmp_path)
    assert (tmp_path / "a.py").read_bytes() == b"old"
    assert updater.read_state(tmp_path) == {}


@pytest.mark.parametrize("mode", ["120000", "100600"])
def test_unsupported_modes_are_rejected(monkeypatch, tmp_path, mode):
    setup_remote(monkeypatch, {})
    tree = updater._fetch_tree(COMMIT)
    tree[0]["mode"] = mode
    with pytest.raises(updater.PortableUpdateError, match="unsupported"):
        updater.update_from_remote(tmp_path)


def test_renames_remove_only_old_managed_path(monkeypatch, tmp_path):
    files, _ = setup_remote(monkeypatch, {"new/file.py": b"renamed"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE],
                           "old/file.py": b"renamed", "old/user.py": b"user"})
    updater.save_state(tmp_path, "b" * 40, ["old/file.py", "models/model.bin"])
    write_files(tmp_path, {"models/model.bin": b"protected"})
    updater.update_from_remote(tmp_path)
    assert not (tmp_path / "old/file.py").exists()
    assert (tmp_path / "old/user.py").read_bytes() == b"user"
    assert (tmp_path / "models/model.bin").read_bytes() == b"protected"


def test_second_update_has_no_downloads(monkeypatch, tmp_path):
    _, downloaded = setup_remote(monkeypatch, {"a.py": b"new"})
    updater.update_from_remote(tmp_path)
    downloaded.clear()
    assert not updater.update_from_remote(tmp_path)["changed"]
    assert downloaded == []


def test_linked_code_directory_cannot_overwrite_models(monkeypatch, tmp_path):
    setup_remote(monkeypatch, {"manager/user.bin": b"code"})
    write_files(tmp_path, {"models/user.bin": b"user model"})
    try:
        (tmp_path / "manager").symlink_to(tmp_path / "models", target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip("Directory symlinks unavailable: %s" % error)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(tmp_path / "manager"),
                        str(tmp_path / "models")], check=True, capture_output=True)
    with pytest.raises(updater.PortableUpdateError, match="Unsafe"):
        updater.update_from_remote(tmp_path)
    assert (tmp_path / "models/user.bin").read_bytes() == b"user model"


def test_stale_pending_launcher_cannot_replace_current_version(monkeypatch, tmp_path):
    files, downloaded = setup_remote(monkeypatch, {updater.LAUNCHER_FILE: b"MZ-current"})
    write_files(tmp_path, {updater.MARKER_FILE: files[updater.MARKER_FILE],
                           updater.LAUNCHER_FILE: b"MZ-current",
                           updater.PENDING_LAUNCHER_RELATIVE_PATH.as_posix(): b"MZ-stale"})
    result = updater.update_from_remote(tmp_path)
    assert downloaded == []
    assert result["launcher_staged"]
    assert updater.pending_launcher_path(tmp_path).read_bytes() == b"MZ-current"


def test_progress_is_visible_to_existing_launchers(monkeypatch, tmp_path, capsys):
    setup_remote(monkeypatch, {"new.py": b"new"})
    updater.util.set_json_stream_mode(True)
    try:
        updater.update_from_remote(tmp_path)
    finally:
        updater.util.set_json_stream_mode(False)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(e["type"] == "message" and "%" in e["message"]
               and "/s" in e["message"] and "files" in e["message"] for e in events)
