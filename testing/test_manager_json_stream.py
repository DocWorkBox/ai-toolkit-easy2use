import json
import os
import subprocess
import sys

from manager import util


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_manager_messages_become_ndjson_events(capsys):
    util.set_json_stream_mode(True)
    try:
        util.info("checking")
        util.warn("attention")
        util.ok("ready")
    finally:
        util.set_json_stream_mode(False)

    events = _json_lines(capsys.readouterr().out)
    assert events == [
        {"type": "message", "level": "info", "message": "checking"},
        {"type": "message", "level": "warning", "message": "attention"},
        {"type": "message", "level": "success", "message": "ready"},
    ]


def test_streamed_child_output_is_wrapped_as_log_events(capsys):
    util.set_json_stream_mode(True)
    try:
        code, output = util.run(
            [sys.executable, "-c", "print('child output')"], stream=True
        )
    finally:
        util.set_json_stream_mode(False)

    assert code == 0
    assert output is None
    assert _json_lines(capsys.readouterr().out) == [
        {"type": "log", "stream": "combined", "message": "child output"}
    ]


def test_sync_dry_run_outputs_only_ndjson_and_finishes_with_result():
    env = os.environ.copy()
    env["AITK_RUNTIME_LAYOUT"] = "standard"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "manager",
            "--json-stream",
            "sync",
            "--dry-run",
        ],
        cwd=util.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    events = _json_lines(completed.stdout)
    assert events
    assert events[-1] == {
        "type": "result",
        "command": "sync",
        "ok": True,
        "exit_code": 0,
    }
    assert any(event.get("type") == "message" for event in events)
