import json
import zipfile
from pathlib import Path

import pytest

from manager import portable_update


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _archive(path, files):
    marker = {
        "schema": 1,
        "repository": portable_update.REPOSITORY,
        "branch": portable_update.BRANCH,
    }
    root = "ai-toolkit-easy2use-test/"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root + ".portable-branch.json", json.dumps(marker))
        for relative, content in files.items():
            archive.writestr(root + relative, content)


def test_remote_status_compares_saved_protable_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        portable_update,
        "_fetch_remote_commit",
        lambda: "1234567890abcdef",
    )

    first = portable_update.remote_status(tmp_path)
    assert first["fetch_ok"] is True
    assert first["branch"] == "protable"
    assert first["local_commit"] == "archive"
    assert first["remote_commit"] == "1234567890abcdef"
    assert first["behind"] == 1

    portable_update.save_state(tmp_path, "1234567890abcdef")
    current = portable_update.remote_status(tmp_path)
    assert current["behind"] == 0


def test_remote_commit_uses_curl_when_python_ssl_fails(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise OSError("SSL unexpected EOF")

    monkeypatch.setattr(portable_update.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(
        portable_update,
        "_fetch_json_with_curl",
        lambda _url: {"sha": "abcdef1234567890"},
    )

    assert portable_update._fetch_remote_commit() == "abcdef1234567890"


def test_archive_update_preserves_user_data_and_stages_launcher(tmp_path):
    protected = {
        "runtime/python/python.exe": "runtime",
        "models/model.bin": "model",
        "datasets/set/image.png": "dataset",
        "output/run/model.safetensors": "output",
        "ui/node_modules/package/index.js": "node modules",
        "ui/.next/BUILD_ID": "next build",
        "ui/dist/cron/worker.js": "worker build",
        "aitk_db.db": "database",
    }
    for relative, content in protected.items():
        _write(tmp_path / relative, content)
    _write(tmp_path / "manager/old.py", "old")
    (tmp_path / "AI Toolkit Launcher.exe").write_bytes(b"MZ-old-launcher")

    archive_path = tmp_path / "update.zip"
    _archive(
        archive_path,
        {
            "manager/new.py": b"new code",
            "ui/src/new.ts": b"new ui",
            "start.bat": b"new start",
            "runtime/python/python.exe": b"do not replace",
            "models/model.bin": b"do not replace",
            "datasets/set/image.png": b"do not replace",
            "output/run/model.safetensors": b"do not replace",
            "ui/node_modules/package/index.js": b"do not replace",
            "ui/.next/BUILD_ID": b"do not replace",
            "ui/dist/cron/worker.js": b"do not replace",
            "aitk_db.db": b"do not replace",
            "AI Toolkit Launcher.exe": b"MZ-new-launcher",
        },
    )

    result = portable_update.apply_archive(
        archive_path,
        "abcdef1234567890",
        repo_root=tmp_path,
    )

    assert (tmp_path / "manager/new.py").read_bytes() == b"new code"
    assert (tmp_path / "ui/src/new.ts").read_bytes() == b"new ui"
    assert (tmp_path / "start.bat").read_bytes() == b"new start"
    for relative, content in protected.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == content
    assert (tmp_path / "AI Toolkit Launcher.exe").read_bytes() == b"MZ-old-launcher"
    assert portable_update.pending_launcher_path(tmp_path).read_bytes() == b"MZ-new-launcher"
    assert portable_update.read_state(tmp_path)["commit"] == "abcdef1234567890"
    assert result["launcher_staged"] is True
    assert result["files_updated"] >= 4


def test_archive_update_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    _archive(archive_path, {"../../outside.txt": b"unsafe"})

    with pytest.raises(portable_update.PortableUpdateError):
        portable_update.apply_archive(
            archive_path,
            "abcdef1234567890",
            repo_root=tmp_path / "portable",
        )

    assert not (tmp_path / "outside.txt").exists()


def test_staged_sampling_update_deploys_backend_and_rebuilds_ui_once(monkeypatch, tmp_path):
    from manager import launch

    source = Path(__file__).resolve().parents[1]
    payload_paths = (
        "toolkit/staged_sampling.py",
        "toolkit/config_modules.py",
        "toolkit/models/base_model.py",
        "extensions_built_in/diffusion_models/ui.tsx",
        "extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py",
        "extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py",
        "extensions_built_in/diffusion_models/ming_image/ming_image.py",
        "extensions_built_in/diffusion_models/ming_image/src/pipeline.py",
        "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py",
        "extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py",
        "extensions_built_in/diffusion_models/ltx2/ltx2.py",
        "extensions_built_in/inference_engine/engine.py",
        "ui/src/app/generate/page.tsx",
        "ui/src/app/jobs/new/SimpleJob.tsx",
        "ui/src/app/jobs/new/options.tsx",
        "ui/src/app/jobs/new/jobConfig.ts",
        "ui/src/app/jobs/new/utils.ts",
        "ui/src/docs.tsx",
        "ui/src/types.ts",
    )
    payload = {relative: (source / relative).read_bytes() for relative in payload_paths}
    target = tmp_path / "portable"
    ui_dir = target / "ui"
    _write(ui_dir / "src/app/generate/page.tsx", "old generation page")
    for relative in launch.UI_BUILD_OUTPUTS:
        _write(ui_dir / relative, "old build")
    launch._mark_ui_build_current(ui_dir)
    assert launch.ui_build_is_current(ui_dir)
    archive_path = tmp_path / "staged-sampling.zip"
    _archive(archive_path, payload)

    result = portable_update.apply_archive(archive_path, "staged-sampling-test", repo_root=target)

    assert result["changed"] is True
    for relative, content in payload.items():
        assert (target / relative).read_bytes() == content
    assert not launch.ui_build_is_current(ui_dir)
    calls = []
    monkeypatch.setattr(launch.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert launch.ensure_ui_build("npm.cmd", {}, ui_dir) == 0
    assert launch.ensure_ui_build("npm.cmd", {}, ui_dir) == 0
    assert calls == [["npm.cmd", "run", "build"]]


def test_archive_update_removes_only_previously_managed_files(tmp_path):
    first_archive = tmp_path / "first.zip"
    _archive(
        first_archive,
        {
            "manager/removed.py": b"managed old code",
            "manager/kept.py": b"managed current code",
        },
    )
    portable_update.apply_archive(
        first_archive,
        "1111111111111111",
        repo_root=tmp_path,
    )
    _write(tmp_path / "manager/user-file.py", "user owned")
    _write(tmp_path / "models/user-model.bin", "model")

    second_archive = tmp_path / "second.zip"
    _archive(
        second_archive,
        {"manager/kept.py": b"updated current code"},
    )
    result = portable_update.apply_archive(
        second_archive,
        "2222222222222222",
        repo_root=tmp_path,
    )

    assert not (tmp_path / "manager/removed.py").exists()
    assert (tmp_path / "manager/kept.py").read_bytes() == b"updated current code"
    assert (tmp_path / "manager/user-file.py").read_text(encoding="utf-8") == "user owned"
    assert (tmp_path / "models/user-model.bin").read_text(encoding="utf-8") == "model"
    assert result["files_removed"] == 1


def test_archive_update_rejects_backslash_paths(tmp_path):
    archive_path = tmp_path / "unsafe-backslash.zip"
    _archive(archive_path, {"..\\outside.txt": b"unsafe"})

    with pytest.raises(portable_update.PortableUpdateError):
        portable_update.apply_archive(
            archive_path,
            "abcdef1234567890",
            repo_root=tmp_path / "portable",
        )
