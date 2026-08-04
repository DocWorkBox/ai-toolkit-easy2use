import json
from types import SimpleNamespace

from manager import doctor, env
from manager import __main__ as manager_main


def test_summarize_checks_distinguishes_required_and_repairable_failures():
    checks = [
        {
            "key": "python",
            "label": "python",
            "passed": True,
            "required": True,
            "repairable": True,
            "detail": "3.12",
        },
        {
            "key": "requirements",
            "label": "requirements",
            "passed": False,
            "required": True,
            "repairable": True,
            "detail": "out of sync",
        },
        {
            "key": "disk_space",
            "label": "disk space",
            "passed": False,
            "required": False,
            "repairable": False,
            "detail": "20 GB free",
        },
    ]

    report = doctor.summarize_checks(checks, environment_exists=True)

    assert report["ok"] is False
    assert report["environment_exists"] is True
    assert report["required_passed"] == 1
    assert report["required_total"] == 2
    assert report["failed_required"] == ["requirements"]
    assert report["repairable_failures"] == ["requirements"]
    assert report["warnings"] == ["disk_space"]


def test_doctor_json_emits_one_machine_readable_document(monkeypatch, capsys):
    expected = {
        "ok": True,
        "environment_exists": True,
        "required_passed": 1,
        "required_total": 1,
        "failed_required": [],
        "repairable_failures": [],
        "warnings": [],
        "checks": [],
    }
    monkeypatch.setattr(doctor, "collect_environment_report", lambda: expected)

    result = doctor.run_doctor(json_output=True)

    assert result == expected
    assert json.loads(capsys.readouterr().out) == expected


def test_doctor_cli_forwards_json_flag(monkeypatch):
    observed = []
    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda json_output=False: observed.append(json_output),
    )

    manager_main.cmd_doctor(SimpleNamespace(json=True))

    assert observed == [True]


def test_requirements_health_checks_actual_installed_packages(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest\n", encoding="utf-8")
    spec = SimpleNamespace(requirements_path=lambda: str(requirements))
    monkeypatch.setattr(env, "venv_exists", lambda: True)
    monkeypatch.setattr(env, "venv_python", lambda: __import__("sys").executable)

    healthy, detail = env.requirements_healthy(spec)

    assert healthy is True
    assert "1 requirements" in detail

    requirements.write_text("aitk-package-that-does-not-exist==99.0\n", encoding="utf-8")
    healthy, detail = env.requirements_healthy(spec)

    assert healthy is False
    assert "aitk-package-that-does-not-exist" in detail
