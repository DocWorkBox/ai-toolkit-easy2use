"""Prepare changed UI assets once, then launch the AI Toolkit web UI."""

import hashlib
import os
import subprocess
import threading
from pathlib import Path

from . import ffmpeg, nodejs
from .util import (
    IS_LINUX,
    REPO_ROOT,
    clean_env,
    die,
    info,
    ok,
    python_bin_dir,
    venv_python,
    warn,
)

UI_PORT = 8675
UI_URL = "http://localhost:%d" % UI_PORT
BROWSER_POLL_SECONDS = 300
UI_DIR = os.path.join(REPO_ROOT, "ui")
UI_BUILD_MARKER = ".aitk-build-fingerprint"
UI_BUILD_EXCLUDED_DIRS = {".next", "dist", "node_modules"}
UI_BUILD_OUTPUTS = (
    ".next/BUILD_ID",
    "dist/cron/worker.js",
    "dist/cron/fileServer.js",
)


def build_env():
    """Scrubbed env with local node, ffmpeg, and the venv on PATH.

    Everything the UI worker spawns (training jobs) inherits this, so the
    local ffmpeg/node are visible to the whole process tree.
    """
    env = clean_env()
    path_dirs = []
    if os.path.isdir(nodejs.node_bin_dir()):
        path_dirs.append(nodejs.node_bin_dir())
    ff_paths, ff_libs = ffmpeg.env_additions()
    path_dirs += ff_paths
    vbin = python_bin_dir()
    if os.path.isdir(vbin):
        path_dirs.append(vbin)
    if path_dirs:
        env["PATH"] = os.pathsep.join(path_dirs) + os.pathsep + env.get("PATH", "")
    if ff_libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            ff_libs + [env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    return env


def _headless():
    return IS_LINUX and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def _open_browser_when_ready(stop_event):
    """Poll the UI port in the background; open the browser once it responds."""
    import time
    import urllib.request
    import webbrowser

    waited = 0
    while not stop_event.is_set() and waited < BROWSER_POLL_SECONDS:
        try:
            urllib.request.urlopen(UI_URL, timeout=2).close()
            webbrowser.open(UI_URL)
            return
        except OSError:
            time.sleep(2)
            waited += 2


def _ui_build_input_paths(ui_dir):
    ui_dir = Path(ui_dir)
    paths = []
    for current, directory_names, file_names in os.walk(ui_dir):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in UI_BUILD_EXCLUDED_DIRS
        )
        current_path = Path(current)
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.is_file():
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(ui_dir).as_posix())


def ui_build_fingerprint(ui_dir=None):
    """Hash UI build inputs while excluding generated and installed files."""
    ui_dir = Path(ui_dir or UI_DIR)
    digest = hashlib.sha256()
    digest.update(b"aitk-ui-build-v1\0")
    for path in _ui_build_input_paths(ui_dir):
        relative = path.relative_to(ui_dir).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _ui_build_marker_path(ui_dir):
    return Path(ui_dir) / ".next" / UI_BUILD_MARKER


def ui_build_is_current(ui_dir=None):
    ui_dir = Path(ui_dir or UI_DIR)
    if any(not (ui_dir / relative).is_file() for relative in UI_BUILD_OUTPUTS):
        return False
    try:
        recorded = _ui_build_marker_path(ui_dir).read_text(encoding="ascii").strip()
    except OSError:
        return False
    return recorded == ui_build_fingerprint(ui_dir)


def _mark_ui_build_current(ui_dir):
    marker = _ui_build_marker_path(ui_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(ui_build_fingerprint(ui_dir), encoding="ascii")
    os.replace(temporary, marker)


def ensure_ui_build(npm, env, ui_dir=None):
    ui_dir = str(ui_dir or UI_DIR)
    if ui_build_is_current(ui_dir):
        ok("UI build is current; skipping rebuild.")
        return 0

    info("UI sources or build outputs changed; building UI once...")
    code = subprocess.call([npm, "run", "build"], cwd=ui_dir, env=env)
    if code != 0:
        warn("UI build failed with code %d." % code)
        return code
    if any(not (Path(ui_dir) / relative).is_file() for relative in UI_BUILD_OUTPUTS):
        warn("UI build completed but required output files are missing.")
        return 1
    try:
        _mark_ui_build_current(ui_dir)
    except OSError as error:
        warn("Could not save the UI build cache: %s" % error)
        return 1
    ok("UI build is ready; future launches will reuse it until sources change.")
    return 0


def launch_ui(open_browser=True):
    if not os.path.isfile(venv_python()):
        die("No Python environment found. Run: python3 -m manager install")

    env = build_env()
    npm = nodejs.find_npm(env)
    if not npm:
        die(
            "Node.js was not found. Run `python3 -m manager sync` to install a "
            "local copy, or install Node.js >= %d from https://nodejs.org."
            % nodejs.MIN_NODE_MAJOR
        )
    _, major = nodejs.have_usable_node()
    if major is not None and major < nodejs.MIN_NODE_MAJOR:
        die(
            "Node.js v%d found, but >= %d is required. Run `python3 -m manager sync`."
            % (major, nodejs.MIN_NODE_MAJOR)
        )

    # deps are installed here rather than by the npm script so the lockfile is
    # never rewritten (see nodejs.ensure_ui_deps); a no-op once they're in sync
    nodejs.ensure_ui_deps(env=env)

    info("Synchronizing the UI database schema...")
    code = subprocess.call([npm, "run", "update_db"], cwd=UI_DIR, env=env)
    if code != 0:
        return code

    code = ensure_ui_build(npm, env, UI_DIR)
    if code != 0:
        return code

    info("Starting AI Toolkit UI (%s) ..." % UI_URL)
    stop_event = threading.Event()
    if open_browser and not _headless():
        threading.Thread(
            target=_open_browser_when_ready, args=(stop_event,), daemon=True
        ).start()

    proc = subprocess.Popen(
        [npm, "run", "start"], cwd=UI_DIR, env=env
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130
    finally:
        stop_event.set()
