import json
from types import SimpleNamespace

from manager import __main__ as manager_main
from manager import env, gitops, portable_update, util


def test_archive_checkout_uses_portable_git_metadata(monkeypatch):
    monkeypatch.setattr(
        gitops,
        "_git",
        lambda args, capture=True, check=True: (128, None),
    )

    assert gitops.is_checkout() is False
    assert gitops.current_branch() == "portable"
    assert gitops.current_commit() == "archive"
    assert gitops.is_dirty() is False


def test_portable_check_uses_protable_branch_status(monkeypatch, capsys):
    monkeypatch.setattr(gitops, "is_checkout", lambda: False)
    monkeypatch.setattr(
        portable_update,
        "remote_status",
        lambda: {
            "fetch_ok": True,
            "branch": "protable",
            "local_commit": "archive",
            "remote_commit": "abcdef1234567890",
            "behind": 1,
            "error": None,
        },
    )
    monkeypatch.setattr(
        manager_main,
        "_resolve_spec",
        lambda args: ({"os": "windows"}, SimpleNamespace(backend="cu130")),
    )
    monkeypatch.setattr(env, "venv_exists", lambda: True)
    monkeypatch.setattr(env, "torch_matches", lambda spec: True)
    monkeypatch.setattr(env, "requirements_in_sync", lambda spec: True)

    manager_main.cmd_check(SimpleNamespace(json=True, cpu=False))
    payload = json.loads(capsys.readouterr().out)

    assert payload["branch"] == "protable"
    assert payload["commit"] == "archive"
    assert payload["remote_commit"] == "abcdef1234567890"
    assert payload["behind"] == 1
    assert payload["update_available"] is True


def test_portable_update_syncs_dependencies_when_code_is_current(monkeypatch):
    observed = []
    monkeypatch.setattr(gitops, "is_checkout", lambda: False)
    monkeypatch.setattr(
        portable_update,
        "update_from_remote",
        lambda dry_run=False: {
            "changed": False,
            "commit": "abcdef1234567890",
            "files_updated": 0,
            "launcher_staged": False,
        },
    )
    monkeypatch.setattr(
        manager_main,
        "_resolve_spec",
        lambda args: ({"os": "windows"}, "portable-spec"),
    )
    monkeypatch.setattr(
        env,
        "sync",
        lambda spec, detection, dry_run=False: observed.append(
            (spec, detection, dry_run)
        ),
    )

    manager_main.cmd_update(
        SimpleNamespace(force=False, auto=False, dry_run=False)
    )

    assert observed == [("portable-spec", {"os": "windows"}, False)]


def test_portable_update_reexecutes_after_code_overlay(monkeypatch):
    observed = []
    monkeypatch.setattr(gitops, "is_checkout", lambda: False)
    monkeypatch.setattr(
        portable_update,
        "update_from_remote",
        lambda dry_run=False: {
            "changed": True,
            "commit": "abcdef1234567890",
            "files_updated": 42,
            "launcher_staged": True,
        },
    )
    monkeypatch.setattr(
        manager_main,
        "_reexec_sync",
        lambda args: observed.append((args.dry_run, util.json_stream_mode())),
    )

    manager_main.cmd_update(
        SimpleNamespace(force=False, auto=False, dry_run=False)
    )

    assert observed == [(False, False)]
