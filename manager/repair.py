"""Targeted environment repair driven by the doctor report."""

from . import doctor, env, ffmpeg, gitwin, nodejs
from .util import ok, warn


_TORCH_FAILURES = {"torch", "torchvision", "torchaudio", "torch_gpu"}


def repair_environment(spec, detection, dry_run=False):
    report = doctor.collect_environment_report()
    failures = set(report["repairable_failures"])
    if not failures:
        ok("No automatically repairable environment problems found.")
        return False

    handled = set()
    if "git" in failures:
        gitwin.ensure_git(dry_run=dry_run)
        handled.add("git")

    torch_failures = failures & _TORCH_FAILURES
    if torch_failures:
        env.ensure_torch(spec, dry_run=dry_run)
        handled.update(torch_failures)

    if "requirements" in failures:
        env.repair_requirements(spec, dry_run=dry_run)
        handled.add("requirements")

    if "node" in failures:
        nodejs.ensure_node(detection, dry_run=dry_run)
        handled.add("node")
    if "ui_dependencies" in failures:
        nodejs.ensure_ui_deps(dry_run=dry_run, force=True)
        handled.add("ui_dependencies")

    if "ffmpeg" in failures:
        ffmpeg.ensure_ffmpeg(detection, dry_run=dry_run, spec=spec, force=True)
        handled.add("ffmpeg")

    unhandled = sorted(failures - handled)
    if unhandled:
        warn("No safe automatic repair is available for: %s" % ", ".join(unhandled))

    if handled & (_TORCH_FAILURES | {"ffmpeg"}):
        env.write_sitecustomize(dry_run=dry_run, spec=spec)
    if handled:
        ok("Targeted environment repair completed.")
    return bool(handled)
