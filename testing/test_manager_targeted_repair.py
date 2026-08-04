from types import SimpleNamespace

from manager import env, repair


def test_repair_dispatches_only_failed_components(monkeypatch):
    calls = []
    report = {
        "repairable_failures": ["requirements"],
    }
    monkeypatch.setattr(repair.doctor, "collect_environment_report", lambda: report)
    monkeypatch.setattr(
        repair.env,
        "repair_requirements",
        lambda spec, dry_run=False: calls.append(("requirements", dry_run)),
    )
    monkeypatch.setattr(
        repair.env,
        "ensure_torch",
        lambda *args, **kwargs: calls.append(("torch", kwargs.get("dry_run"))),
    )
    monkeypatch.setattr(
        repair.env,
        "write_sitecustomize",
        lambda dry_run=False, spec=None: calls.append(("sitecustomize", dry_run)),
    )

    repaired = repair.repair_environment(
        SimpleNamespace(), SimpleNamespace(), dry_run=True
    )

    assert repaired is True
    assert calls == [("requirements", True)]


def test_requirement_repair_installs_only_mismatched_entries(monkeypatch):
    spec = SimpleNamespace(
        backend="cu130",
        find_links=[],
    )
    report = {
        "ok": False,
        "problems": [
            {
                "requirement": "huggingface_hub==1.23.0",
                "name": "huggingface_hub",
                "reason": "version",
                "installed": "1.10.1",
                "expected": "==1.23.0",
            }
        ],
    }
    installs = []
    monkeypatch.setattr(env, "_requirements_report", lambda spec: (report, None))
    monkeypatch.setattr(env, "_torch_pin_args", lambda spec, dry_run=False: [])
    monkeypatch.setattr(
        env,
        "_pip_install",
        lambda args, dry_run=False, **kwargs: installs.append((args, dry_run)),
    )
    monkeypatch.setattr(env, "_verify_torch", lambda spec, dry_run=False: False)

    repaired = env.repair_requirements(spec, dry_run=True)

    assert repaired is True
    assert installs == [(["huggingface_hub==1.23.0"], True)]
