import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from manager import env as manager_env
from manager import ffmpeg, gitwin, nodejs, util, uvbin


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_repo_root_can_be_explicitly_selected_for_a_launcher(tmp_path):
    selected = tmp_path / "portable-root"
    selected.mkdir()
    process_env = os.environ.copy()
    process_env["AITK_ROOT"] = str(selected)

    result = subprocess.run(
        [sys.executable, "-c", "from manager.util import REPO_ROOT; print(REPO_ROOT)"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=process_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(selected)


def test_auto_detects_windows_portable_runtime(monkeypatch, tmp_path):
    _touch(tmp_path / "runtime" / "python" / "python.exe")
    monkeypatch.setattr(util, "REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("AITK_RUNTIME_LAYOUT", raising=False)

    assert util.runtime_layout() == "portable"
    assert util.venv_dir() == str(tmp_path / "runtime" / "python")
    assert util.venv_python() == str(tmp_path / "runtime" / "python" / "python.exe")
    assert util.python_bin_dir() == str(tmp_path / "runtime" / "python")


def test_explicit_portable_layout_is_stable_before_runtime_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(util, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "portable")

    assert util.runtime_layout() == "portable"
    assert util.venv_dir() == str(tmp_path / "runtime" / "python")
    assert util.venv_python() == str(tmp_path / "runtime" / "python" / "python.exe")


def test_standard_layout_keeps_existing_venv_behavior(monkeypatch, tmp_path):
    _touch(tmp_path / "runtime" / "python" / "python.exe")
    _touch(tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(util, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "standard")

    assert util.runtime_layout() == "standard"
    assert util.venv_dir() == str(tmp_path / ".venv")
    assert util.venv_python() == str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert util.python_bin_dir() == str(tmp_path / ".venv" / "Scripts")


def test_component_directories_follow_selected_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(util, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "portable")

    assert nodejs.node_dir() == str(tmp_path / "runtime" / "node")
    assert ffmpeg.ffmpeg_dir() == str(tmp_path / "runtime" / "ffmpeg")
    assert uvbin.uv_dir() == str(tmp_path / "runtime" / "uv")
    assert gitwin.mingit_dir() == str(tmp_path / "runtime" / "mingit")

    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "standard")

    assert nodejs.node_dir() == str(tmp_path / ".node")
    assert ffmpeg.ffmpeg_dir() == str(tmp_path / ".ffmpeg")
    assert uvbin.uv_dir() == str(tmp_path / ".uv")
    assert gitwin.mingit_dir() == str(tmp_path / ".mingit")


def test_invalid_layout_override_is_rejected(monkeypatch):
    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "unexpected")

    try:
        util.runtime_layout()
    except ValueError as exc:
        assert "AITK_RUNTIME_LAYOUT" in str(exc)
    else:
        raise AssertionError("invalid layout override should fail")


def test_portable_runtime_is_never_deleted_for_architecture_switch(
    monkeypatch, tmp_path
):
    runtime_python = tmp_path / "runtime" / "python"
    _touch(runtime_python / "python.exe")
    sentinel = runtime_python / "keep-me.txt"
    sentinel.write_text("portable runtime", encoding="utf-8")
    monkeypatch.setattr(util, "REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("AITK_RUNTIME_LAYOUT", "portable")
    monkeypatch.setattr(manager_env, "venv_dir", lambda: str(runtime_python))
    monkeypatch.setattr(
        manager_env, "venv_python", lambda venv=None: str(runtime_python / "python.exe")
    )
    monkeypatch.setattr(manager_env, "venv_exists", lambda: True)
    monkeypatch.setattr(manager_env, "_venv_platform", lambda: "win-amd64")
    spec = SimpleNamespace(
        uv_python="cpython-3.12-windows-aarch64-none", python_version="3.12"
    )

    with pytest.raises(SystemExit):
        manager_env.ensure_venv(spec)

    assert sentinel.read_text(encoding="utf-8") == "portable runtime"
