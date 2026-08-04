from types import SimpleNamespace

from manager import nodejs


def test_ui_dependency_check_ignores_platform_optional_packages(monkeypatch, tmp_path):
    (tmp_path / "node_modules").mkdir()
    observed = []

    def fake_run(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(nodejs, "UI_DIR", str(tmp_path))
    monkeypatch.setattr(nodejs, "find_npm", lambda env=None: "npm.cmd")
    monkeypatch.setattr(nodejs.subprocess, "run", fake_run)

    healthy, _ = nodejs.check_ui_dependencies()

    assert healthy is True
    assert observed == [["npm.cmd", "ls", "--depth=0", "--omit=optional", "--json"]]
