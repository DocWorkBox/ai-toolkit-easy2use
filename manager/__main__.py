"""AI Toolkit manager CLI.

Runs with any Python >= 3.8 and no dependencies, so it works before the
training environment exists. This is the single entry point every installer
frontend (shell scripts, the desktop launcher, the web UI) shells out to.

    python3 -m manager install          first-time environment setup
    python3 -m manager check [--json]   is an update / dep sync needed?
    python3 -m manager update           git pull + dependency sync + migrations
    python3 -m manager sync             dependency sync only (no git pull)
    python3 -m manager launch           start the web UI
    python3 -m manager detect [--json]  show detected hardware
    python3 -m manager doctor           full environment diagnostics
"""

import argparse
import os
import subprocess
import sys

# allow `python manager/__main__.py` as well as `python -m manager`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manager import detect as detect_mod
from manager import env, gitops, launch, portable_update, spec as spec_mod, util
from manager.util import die, info, ok, print_json, warn


def _resolve_spec(args):
    detection = detect_mod.detect()
    try:
        return detection, spec_mod.build_spec(
            detection, allow_cpu=getattr(args, "cpu", False)
        )
    except RuntimeError as e:
        die(str(e))


def cmd_detect(args):
    detection = detect_mod.detect()
    try:
        s = spec_mod.build_spec(detection, allow_cpu=True)
        detection["spec"] = s.as_dict()
    except RuntimeError as e:
        detection["spec_error"] = str(e)
    if args.json:
        print_json(detection)
    else:
        backend = detection.get("spec", {}).get("backend", "unknown")
        info("os=%s arch=%s backend=%s" % (detection["os"], detection["arch"], backend))
        if detection["nvidia"]:
            for gpu in detection["nvidia"]["gpus"]:
                info("gpu: %s (%s)" % (gpu["name"], gpu["memory"]))


def cmd_install(args):
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run, force=args.force)
    if not args.dry_run:
        ok("Install complete. Start the UI with: python3 -m manager launch")


def cmd_sync(args):
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run, force=args.force)


def cmd_repair(args):
    from manager import repair

    detection, s = _resolve_spec(args)
    repair.repair_environment(s, detection, dry_run=args.dry_run)


def cmd_check(args):
    _, s = _resolve_spec(args)
    git_checkout = gitops.is_checkout()
    if git_checkout:
        fetched = gitops.fetch()
        branch = gitops.current_branch()
        commit = gitops.current_commit()
        remote_commit = gitops.remote_commit()
        behind = gitops.behind_count()
        incoming = gitops.incoming_log()
        dirty = gitops.is_dirty()
    else:
        portable = portable_update.remote_status()
        fetched = portable["fetch_ok"]
        branch = portable["branch"]
        commit = portable["local_commit"]
        remote_commit = portable["remote_commit"]
        behind = portable["behind"]
        incoming = (
            ["Portable branch update %s" % remote_commit[:12]]
            if behind and remote_commit
            else []
        )
        dirty = False
    data = {
        "version": _toolkit_version(),
        "branch": branch,
        "commit": commit,
        "remote_commit": remote_commit,
        "dirty": dirty,
        "fetch_ok": fetched,
        "git_checkout": git_checkout,
        "behind": behind,
        "incoming": incoming,
        "venv": env.venv_exists(),
        "deps_in_sync": env.venv_exists()
        and env.torch_matches(s)
        and env.requirements_in_sync(s),
        "backend": s.backend,
    }
    data["dependency_update_available"] = not data["deps_in_sync"]
    data["update_available"] = bool(behind) or (
        git_checkout and data["dependency_update_available"]
    )
    if args.json:
        print_json(data)
        return
    info("AI Toolkit %s (%s @ %s)" % (data["version"], data["branch"], data["commit"]))
    if not fetched:
        warn("Could not reach the remote (offline?) — update status may be stale.")
    if behind:
        info("Update available: %d new commit(s)." % behind)
        for line in data["incoming"]:
            print("    " + line)
    elif behind == 0:
        ok("Code is up to date.")
    if not data["deps_in_sync"]:
        warn("Dependencies are out of sync. Run: python3 -m manager sync")
    elif behind == 0:
        ok("Dependencies are in sync.")


def cmd_update(args):
    """Update code without destructive local changes.

    Archive-style portable bundles update code only; environment changes stay
    behind the explicit diagnose/repair controls. Git checkouts retain the
    pull-then-sync behavior. Local work is sacred: a dirty checkout either
    aborts (default), or with --auto is skipped with a warning. We never
    reset/clean; even a forced pull is --ff-only.
    """
    if not gitops.is_checkout():
        try:
            portable_update.update_from_remote(dry_run=args.dry_run)
        except portable_update.PortableUpdateError as error:
            if getattr(args, "auto", False):
                warn("Could not update portable code — %s" % error)
            else:
                die(str(error))
        return

    auto = getattr(args, "auto", False)
    skip_pull = False

    if gitops.is_dirty() and not args.force:
        if auto:
            warn(
                "Local changes detected — skipping the code update to protect "
                "your work. Commit or stash your changes to receive updates."
            )
            skip_pull = True
        else:
            die(
                "You have local changes to tracked files. Commit or stash them, "
                "or re-run with --force to attempt the update anyway (git will "
                "still refuse rather than overwrite your changes)."
            )

    if not skip_pull and not gitops.fetch():
        if auto:
            warn("Could not reach the git remote — skipping the update check.")
            skip_pull = True
        else:
            die("Could not reach the git remote. Check your network and try again.")

    if not skip_pull:
        behind = gitops.behind_count()
        if behind is None:
            warn("Current branch has no upstream; skipping git pull.")
        elif behind == 0:
            ok("Code already up to date.")
        else:
            info("Pulling %d new commit(s)..." % behind)
            gitops.pull_ff()
            ok("Code updated to %s." % gitops.current_commit())
            # Re-exec so the freshly pulled manager code runs its own dependency
            # sync and migrations (the in-memory copy of this module is stale now).
            _reexec_sync(args)
            return
    # nothing was pulled — safe to sync with the code already loaded
    detection, s = _resolve_spec(args)
    env.sync(s, detection, dry_run=args.dry_run)


def _reexec_sync(args):
    cmd = [sys.executable, "-m", "manager"]
    if util.json_stream_mode():
        cmd.append("--json-stream")
    cmd.append("sync")
    if args.dry_run:
        cmd.append("--dry-run")
    sys.exit(subprocess.call(cmd, cwd=util.REPO_ROOT))


def cmd_launch(args):
    sys.exit(launch.launch_ui(open_browser=not args.no_browser))


def cmd_doctor(args):
    from manager import doctor

    doctor.run_doctor(json_output=args.json)


def cmd_version(args):
    print(_toolkit_version())


def _toolkit_version():
    version = {}
    try:
        with open(os.path.join(util.REPO_ROOT, "version.py")) as f:
            exec(f.read(), version)
        return version.get("VERSION", "unknown")
    except OSError:
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="manager", description="AI Toolkit install / update manager"
    )
    parser.add_argument(
        "--json-stream",
        action="store_true",
        help="emit NDJSON progress events for install, sync, or update",
    )
    sub = parser.add_subparsers(dest="command")

    def add(name, fn, **kwargs):
        p = sub.add_parser(name, **kwargs)
        p.set_defaults(fn=fn)
        return p

    p = add("detect", cmd_detect, help="show detected hardware and env spec")
    p.add_argument("--json", action="store_true")

    for name, fn, help_text in (
        ("install", cmd_install, "first-time environment setup"),
        ("sync", cmd_sync, "sync dependencies for the current checkout"),
    ):
        p = add(name, fn, help=help_text)
        p.add_argument("--cpu", action="store_true", help="allow CPU-only install")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--force",
            action="store_true",
            help="reinstall requirements even if in sync",
        )

    p = add("repair", cmd_repair, help="repair only failed environment checks")
    p.add_argument("--cpu", action="store_true", help="allow CPU-only repair")
    p.add_argument("--dry-run", action="store_true")

    p = add("check", cmd_check, help="check for updates (use --json for machines)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)

    p = add("update", cmd_update, help="git pull + dependency sync + migrations")
    p.add_argument("--cpu", action="store_true", help="allow CPU-only install")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="update even with local changes"
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="unattended mode (run scripts): on local changes or an unreachable "
        "remote, warn and skip the code update instead of failing; deps still sync",
    )

    p = add("launch", cmd_launch, help="start the web UI")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser when the UI is ready",
    )
    p = add("doctor", cmd_doctor, help="diagnose the environment")
    p.add_argument("--json", action="store_true")
    add("version", cmd_version, help="print the toolkit version")

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    if args.json_stream and args.command not in (
        "install",
        "sync",
        "repair",
        "update",
    ):
        parser.error("--json-stream is supported by install, sync, repair, and update")
    util.set_json_stream_mode(args.json_stream)
    util.set_json_mode(bool(getattr(args, "json", False)) or args.json_stream)
    try:
        result = args.fn(args)
    except KeyboardInterrupt:
        return 130
    exit_code = 0 if result is None else int(result)
    if args.json_stream:
        util.emit_event(
            "result",
            command=args.command,
            ok=exit_code == 0,
            exit_code=exit_code,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
