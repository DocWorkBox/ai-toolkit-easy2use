from types import SimpleNamespace

import pytest

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
    existing_torch = {
        "torch": "2.9.1+cu128",
        "torchvision": "0.24.1+cu128",
        "torchaudio": "2.9.1+cu128",
    }
    monkeypatch.setattr(env, "_requirements_report", lambda spec: (report, None))
    monkeypatch.setattr(env, "torch_stack", lambda: existing_torch)
    monkeypatch.setattr(
        env,
        "_torch_preservation_args",
        lambda stack, dry_run=False: ["--constraint", "preserved-torch.txt"],
    )
    monkeypatch.setattr(
        env,
        "_pip_install",
        lambda args, dry_run=False, **kwargs: installs.append((args, dry_run)),
    )

    repaired = env.repair_requirements(spec, dry_run=True)

    assert repaired is True
    assert installs == [
        (
            [
                "huggingface_hub==1.23.0",
                "--constraint",
                "preserved-torch.txt",
            ],
            True,
        )
    ]


def test_requirement_repair_keeps_preexisting_torch_versions(monkeypatch, tmp_path):
    spec = SimpleNamespace(backend="cu130", find_links=[])
    report = {
        "ok": False,
        "problems": [
            {
                "requirement": "huggingface_hub==1.23.0",
                "name": "huggingface_hub",
                "reason": "version",
            }
        ],
    }
    existing_torch = {
        "torch": "2.9.1+cu128",
        "torchvision": "0.24.1+cu128",
        "torchaudio": "2.9.1+cu128",
    }
    installs = []
    monkeypatch.setattr(env, "_requirements_report", lambda spec: (report, None))
    monkeypatch.setattr(env, "torch_stack", lambda: dict(existing_torch))
    monkeypatch.setattr(env, "venv_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        env,
        "_pip_install",
        lambda args, dry_run=False, **kwargs: installs.append(list(args)),
    )
    monkeypatch.setattr(env, "requirements_healthy", lambda spec: (False, "test"))

    assert env.repair_requirements(spec) is True
    assert len(installs) == 1
    constraints_path = installs[0][2]
    with open(constraints_path, encoding="utf-8") as constraints_file:
        constraints = constraints_file.read()
    assert "torch==2.9.1+cu128" in constraints
    assert "torchvision==0.24.1+cu128" in constraints
    assert "torchaudio==2.9.1+cu128" in constraints
    assert "2.13.0" not in constraints


def test_requirement_repair_never_auto_downloads_torch_after_change(
    monkeypatch, tmp_path
):
    spec = SimpleNamespace(backend="cu130", find_links=[])
    report = {
        "ok": False,
        "problems": [
            {
                "requirement": "huggingface_hub==1.23.0",
                "name": "huggingface_hub",
                "reason": "version",
            }
        ],
    }
    before = {
        "torch": "2.9.1+cu128",
        "torchvision": "0.24.1+cu128",
        "torchaudio": "2.9.1+cu128",
    }
    after = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "torchaudio": "2.11.0+cu130",
    }
    stacks = iter((before, after))
    installs = []
    monkeypatch.setattr(env, "_requirements_report", lambda spec: (report, None))
    monkeypatch.setattr(env, "torch_stack", lambda: next(stacks))
    monkeypatch.setattr(env, "venv_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        env,
        "_pip_install",
        lambda args, dry_run=False, **kwargs: installs.append(list(args)),
    )

    with pytest.raises(SystemExit):
        env.repair_requirements(spec)

    assert len(installs) == 1
    assert installs[0][0] == "huggingface_hub==1.23.0"
