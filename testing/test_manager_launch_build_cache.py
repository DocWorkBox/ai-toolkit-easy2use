from pathlib import Path

from manager import launch


def _make_ui_tree(root: Path) -> Path:
    ui_dir = root / "ui"
    for relative in (
        "src/app/page.tsx",
        "cron/worker.ts",
        "cron/fileServer.ts",
        "prisma/schema.prisma",
        "public/logo.png",
        "package.json",
        "package-lock.json",
        "next.config.ts",
    ):
        path = ui_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    for relative in (
        ".next/BUILD_ID",
        "dist/cron/worker.js",
        "dist/cron/fileServer.js",
    ):
        path = ui_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("built", encoding="utf-8")
    return ui_dir


def _write_current_marker(ui_dir: Path) -> None:
    marker = ui_dir / ".next" / ".aitk-build-fingerprint"
    marker.write_text(launch.ui_build_fingerprint(ui_dir), encoding="ascii")


def test_ui_build_cache_tracks_sources_but_ignores_generated_output(tmp_path):
    ui_dir = _make_ui_tree(tmp_path)
    _write_current_marker(ui_dir)

    assert launch.ui_build_is_current(ui_dir) is True

    generated = ui_dir / ".next" / "cache" / "runtime.bin"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"runtime output")
    assert launch.ui_build_is_current(ui_dir) is True

    (ui_dir / "src" / "app" / "page.tsx").write_text(
        "updated source", encoding="utf-8"
    )
    assert launch.ui_build_is_current(ui_dir) is False


def test_ui_build_cache_requires_next_and_worker_outputs(tmp_path):
    ui_dir = _make_ui_tree(tmp_path)
    _write_current_marker(ui_dir)

    (ui_dir / "dist" / "cron" / "worker.js").unlink()

    assert launch.ui_build_is_current(ui_dir) is False


def test_ensure_ui_build_rebuilds_once_after_source_change(monkeypatch, tmp_path):
    ui_dir = _make_ui_tree(tmp_path)
    _write_current_marker(ui_dir)
    (ui_dir / "cron" / "worker.ts").write_text("updated", encoding="utf-8")
    calls = []

    def fake_call(command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(launch.subprocess, "call", fake_call)

    assert launch.ensure_ui_build("npm.cmd", {"PATH": "test"}, ui_dir) == 0
    assert launch.ensure_ui_build("npm.cmd", {"PATH": "test"}, ui_dir) == 0
    assert [call[0] for call in calls] == [["npm.cmd", "run", "build"]]


def test_launch_runs_database_sync_and_cached_start_instead_of_db_build_start(
    monkeypatch, tmp_path
):
    commands = []
    ensured = []

    class FakeProcess:
        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(launch, "UI_DIR", str(tmp_path / "ui"))
    monkeypatch.setattr(launch, "venv_python", lambda: str(tmp_path / "python.exe"))
    monkeypatch.setattr(launch.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(launch, "build_env", lambda: {"PATH": "test"})
    monkeypatch.setattr(launch.nodejs, "find_npm", lambda env=None: "npm.cmd")
    monkeypatch.setattr(launch.nodejs, "have_usable_node", lambda: ("node.exe", 24))
    monkeypatch.setattr(launch.nodejs, "ensure_ui_deps", lambda env=None: None)
    monkeypatch.setattr(
        launch,
        "ensure_ui_build",
        lambda npm, env, ui_dir=None: ensured.append((npm, env, ui_dir)) or 0,
    )
    monkeypatch.setattr(
        launch.subprocess,
        "call",
        lambda command, **kwargs: commands.append(command) or 0,
    )
    monkeypatch.setattr(
        launch.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or FakeProcess(),
    )

    assert launch.launch_ui(open_browser=False) == 0
    assert commands == [
        ["npm.cmd", "run", "update_db"],
        ["npm.cmd", "run", "start"],
    ]
    assert ensured == [("npm.cmd", {"PATH": "test"}, str(tmp_path / "ui"))]
