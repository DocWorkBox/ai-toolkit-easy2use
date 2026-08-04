import json
from types import SimpleNamespace

from manager import __main__ as manager_main
from manager import env, gitops, portable_update


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
    assert payload["incoming"] == ["Portable branch update abcdef123456"]
    assert payload["update_available"] is True
    assert payload["dependency_update_available"] is False


def test_portable_check_keeps_dependency_advice_separate_from_code(
    monkeypatch, capsys
):
    monkeypatch.setattr(gitops, "is_checkout", lambda: False)
    monkeypatch.setattr(
        portable_update,
        "remote_status",
        lambda: {
            "fetch_ok": True,
            "branch": "protable",
            "local_commit": "abcdef1234567890",
            "remote_commit": "abcdef1234567890",
            "behind": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(
        manager_main,
        "_resolve_spec",
        lambda args: ({"os": "windows"}, SimpleNamespace(backend="cu130")),
    )
    monkeypatch.setattr(env, "venv_exists", lambda: True)
    monkeypatch.setattr(env, "torch_matches", lambda spec: False)
    monkeypatch.setattr(env, "requirements_in_sync", lambda spec: True)

    manager_main.cmd_check(SimpleNamespace(json=True, cpu=False))
    payload = json.loads(capsys.readouterr().out)

    assert payload["behind"] == 0
    assert payload["deps_in_sync"] is False
    assert payload["dependency_update_available"] is True
    assert payload["update_available"] is False


def test_portable_update_does_not_sync_dependencies_when_code_is_current(
    monkeypatch,
):
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
        lambda args: (_ for _ in ()).throw(
            AssertionError("portable code update resolved the environment")
        ),
    )
    monkeypatch.setattr(
        env,
        "sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("portable code update synced dependencies")
        ),
    )

    manager_main.cmd_update(
        SimpleNamespace(force=False, auto=False, dry_run=False)
    )

def test_portable_update_does_not_sync_dependencies_after_code_overlay(monkeypatch):
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
        lambda args: (_ for _ in ()).throw(
            AssertionError("portable code update re-executed dependency sync")
        ),
    )

    manager_main.cmd_update(
        SimpleNamespace(force=False, auto=False, dry_run=False)
    )
