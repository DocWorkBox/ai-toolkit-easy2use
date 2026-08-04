from types import SimpleNamespace

from manager import __main__ as manager_main
from manager import env, gitops


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


def test_portable_update_skips_git_and_syncs_dependencies(monkeypatch):
    observed = []
    monkeypatch.setattr(gitops, "is_checkout", lambda: False)
    monkeypatch.setattr(
        gitops,
        "fetch",
        lambda: (_ for _ in ()).throw(AssertionError("portable update fetched git")),
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
