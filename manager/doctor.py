"""Environment diagnostics: `python -m manager doctor`."""

import os
import shutil
import subprocess
import sys

from . import detect as detect_mod
from . import env, ffmpeg, gitops, nodejs
from .util import REPO_ROOT, clean_env, find_uv, print_json, venv_dir, venv_python


def _result(key, label, passed, detail="", required=True, repairable=True):
    return {
        "key": key,
        "label": label,
        "passed": bool(passed),
        "required": bool(required),
        "repairable": bool(repairable),
        "detail": detail or "",
    }


def summarize_checks(checks, environment_exists):
    required = [item for item in checks if item["required"]]
    return {
        "ok": all(item["passed"] for item in required),
        "environment_exists": bool(environment_exists),
        "required_passed": sum(1 for item in required if item["passed"]),
        "required_total": len(required),
        "failed_required": [
            item["key"] for item in required if not item["passed"]
        ],
        "repairable_failures": [
            item["key"]
            for item in required
            if not item["passed"] and item["repairable"]
        ],
        "warnings": [
            item["key"]
            for item in checks
            if not item["required"] and not item["passed"]
        ],
        "checks": checks,
    }


def _pip_check():
    try:
        result = subprocess.run(
            [venv_python(), "-m", "pip", "check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            env=clean_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "could not run pip check"

    output = (result.stdout + result.stderr).decode(errors="replace").strip()
    if result.returncode == 0:
        return True, output or "all installed packages are compatible"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return False, "; ".join(lines[:3]) or "pip check failed"


def collect_environment_report():
    checks = []
    d = detect_mod.detect()

    from . import gitwin
    from . import spec as spec_mod

    try:
        wanted = spec_mod.build_spec(d, allow_cpu=True)
        spec_error = None
    except RuntimeError as error:
        wanted = None
        spec_error = str(error)

    arch_detail = "%s %s" % (d["os"], d["arch"])
    if wanted and d["os"] == "windows" and d["arch"] == "aarch64":
        if wanted.backend == "cu134":
            arch_detail += " (RTX Spark: native win_arm64 CUDA stack)"
        else:
            arch_detail += " (Windows-on-ARM: x64 stack via emulation)"
    checks.append(
        _result("os_arch", "os / arch", True, arch_detail, repairable=False)
    )
    checks.append(
        _result(
            "environment_spec",
            "environment spec",
            wanted is not None,
            "%s / Python %s" % (wanted.backend, wanted.python_version)
            if wanted
            else spec_error,
            repairable=False,
        )
    )

    git = gitwin.find_git()
    checks.append(
        _result(
            "git",
            "git",
            git is not None,
            git or "not found (manager sync installs a local copy on Windows)",
        )
    )

    uv = find_uv()
    checks.append(
        _result(
            "uv",
            "uv",
            uv is not None,
            uv or "not found (optional, recommended)",
            required=False,
            repairable=True,
        )
    )

    if d["nvidia"]:
        names = ", ".join(g["name"] for g in d["nvidia"]["gpus"])
        gpu_passed = True
        gpu_detail = "%s (driver %s, CUDA %s)" % (
            names,
            d["nvidia"]["driver"],
            d["nvidia"]["cuda_version"],
        )
    elif d["rocm"]:
        gpu_passed, gpu_detail = True, "AMD ROCm (experimental)"
    elif d["backend"] == "mps":
        gpu_passed, gpu_detail = True, "Apple Silicon (MPS)"
    else:
        gpu_passed, gpu_detail = False, "no supported GPU detected"
    checks.append(
        _result("gpu", "gpu", gpu_passed, gpu_detail, repairable=False)
    )

    has_environment = env.venv_exists()
    checks.append(
        _result(
            "venv",
            "AI Toolkit environment",
            has_environment,
            venv_dir() if has_environment else "not installed",
        )
    )

    if has_environment:
        python_version = env.venv_python_version()
        expected_python = wanted.python_version if wanted else None
        checks.append(
            _result(
                "python",
                "python",
                bool(expected_python) and python_version == expected_python,
                "%s (expected %s)"
                % (python_version or "unavailable", expected_python or "unknown"),
            )
        )

        stack = env.torch_stack()
        for name in ("torch", "torchvision", "torchaudio"):
            found = stack.get(name)
            checks.append(
                _result(
                    name,
                    name,
                    bool(found) and not found.startswith("ERROR"),
                    found or "not installed",
                )
            )

        checks.append(
            _result(
                "torch_stack_pins",
                "manager torch target",
                bool(wanted) and env.torch_stack_matches(wanted, stack),
                "expected %s (%s)"
                % (
                    ", ".join(
                        "%s %s" % (key, value)
                        for key, value in sorted(wanted.torch_packages.items())
                    ),
                    wanted.backend,
                )
                if wanted
                else spec_error,
                required=False,
            )
        )

        pip_ok, pip_detail = _pip_check()
        checks.append(
            _result("pip_check", "python packages", pip_ok, pip_detail)
        )
        requirements_ok, requirements_detail = (
            env.requirements_healthy(wanted)
            if wanted
            else (False, spec_error or "environment spec is unavailable")
        )
        checks.append(
            _result(
                "requirements",
                "AI Toolkit requirements",
                requirements_ok,
                requirements_detail,
            )
        )

        if d["backend"] == "cuda":
            try:
                result = subprocess.run(
                    [venv_python(), "-c", "import torch; print(torch.cuda.is_available())"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    env=clean_env(),
                )
                torch_gpu = result.stdout.decode().strip() == "True"
                torch_gpu_detail = (
                    "available" if torch_gpu else "torch.cuda.is_available() is False"
                )
            except (OSError, subprocess.TimeoutExpired):
                torch_gpu, torch_gpu_detail = False, "could not query"
            checks.append(
                _result(
                    "torch_gpu",
                    "torch sees gpu",
                    torch_gpu,
                    torch_gpu_detail,
                )
            )

    node_exe, node_major = nodejs.have_usable_node()
    checks.append(
        _result(
            "node",
            "node",
            node_exe is not None,
            "%s (v%s)" % (node_exe, node_major)
            if node_exe
            else "none >= %d found" % nodejs.MIN_NODE_MAJOR,
        )
    )
    ui_dependencies_ok, ui_dependencies_detail = nodejs.check_ui_dependencies()
    checks.append(
        _result(
            "ui_dependencies",
            "UI dependencies",
            ui_dependencies_ok,
            ui_dependencies_detail,
        )
    )

    if os.path.isfile(ffmpeg.ffmpeg_exe()):
        from . import launch

        try:
            result = subprocess.run(
                [ffmpeg.ffmpeg_exe(), "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                env=launch.build_env(),
            )
            ffmpeg_ok = result.returncode == 0
            ffmpeg_detail = (
                result.stdout.decode(errors="replace").splitlines()[0]
                if ffmpeg_ok
                else "installed but fails to run"
            )
        except (OSError, subprocess.TimeoutExpired):
            ffmpeg_ok, ffmpeg_detail = False, "installed but fails to run"
    else:
        ffmpeg_ok, ffmpeg_detail = False, "not installed"
    checks.append(_result("ffmpeg", "ffmpeg (local)", ffmpeg_ok, ffmpeg_detail))

    try:
        free_gb = shutil.disk_usage(REPO_ROOT).free / (1024**3)
        checks.append(
            _result(
                "disk_space",
                "disk space",
                free_gb > 30,
                "%.0f GB free" % free_gb,
                required=False,
                repairable=False,
            )
        )
    except OSError:
        pass

    branch = gitops.current_branch()
    checks.append(
        _result(
            "git_checkout",
            "git checkout",
            True,
            "%s @ %s%s"
            % (
                branch,
                gitops.current_commit(),
                " (dirty)" if gitops.is_dirty() else "",
            ),
            required=False,
            repairable=False,
        )
    )

    return summarize_checks(checks, has_environment)


def _print_check(item):
    if sys.stdout.isatty():
        mark = "\033[32mOK\033[0m " if item["passed"] else "\033[31mFAIL\033[0m"
    else:
        mark = "OK  " if item["passed"] else "FAIL"
    print("  [%s] %-22s %s" % (mark, item["label"], item["detail"]))


def run_doctor(json_output=False):
    report = collect_environment_report()
    if json_output:
        print_json(report)
        return report

    print("AI Toolkit doctor\n")
    for item in report["checks"]:
        _print_check(item)
    if report["ok"]:
        print("\nEnvironment meets AI Toolkit requirements.")
    else:
        print(
            "\nEnvironment does not meet AI Toolkit requirements "
            "(%d/%d required checks passed)."
            % (report["required_passed"], report["required_total"])
        )
    return report
