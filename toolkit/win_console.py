"""Keep child processes from popping console windows on Windows.

The UI worker launches training jobs with pythonw.exe and DETACHED_PROCESS so
they survive the UI shutting down (see ui/cron/actions/startJob.ts). That
leaves the job with no console at all, and Windows hands a brand new console
-- with a visible window -- to any console program launched from a process
that has none. MSVC during a torch/triton compile, git during a HF download
and ffmpeg would each flash a window on the user's desktop.

Defaulting those spawns to CREATE_NO_WINDOW suppresses the flash when the
parent has no console. Native Windows tools may also emit non-UTF-8 output,
so text pipe decoding needs a fallback even when the parent has a console.
"""

import subprocess
import sys

CREATE_NEW_CONSOLE = 0x00000010
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

# creationflags is the 14th positional parameter of Popen.__init__ after self.
_CREATIONFLAGS_POSITION = 14

_patched = False


def _ensure_text_subprocess_decode_fallback(kwargs):
    """Prevent native Windows output from crashing UTF-8 reader threads."""
    uses_text_mode = (
        kwargs.get("text") is True
        or kwargs.get("universal_newlines") is True
        or kwargs.get("encoding") is not None
    )
    if uses_text_mode and kwargs.get("errors") is None:
        kwargs["errors"] = "replace"


def _has_console():
    import ctypes

    return bool(ctypes.windll.kernel32.GetConsoleWindow())


def suppress_child_consoles():
    """Protect Windows text pipes and hide children of console-free jobs."""
    global _patched
    if _patched or sys.platform != "win32":
        return
    hide_child_windows = False
    try:
        hide_child_windows = not _has_console()
    except Exception:
        # Console detection must not disable the independent decode protection.
        pass

    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        _ensure_text_subprocess_decode_fallback(kwargs)
        if not hide_child_windows or len(args) >= _CREATIONFLAGS_POSITION:
            # Passed positionally; leave the caller's choice alone.
            return original_init(self, *args, **kwargs)
        flags = kwargs.get("creationflags", 0)
        if not flags & (CREATE_NEW_CONSOLE | DETACHED_PROCESS | CREATE_NO_WINDOW):
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init
    _patched = True
