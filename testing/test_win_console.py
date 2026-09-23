import subprocess
import sys
import unittest
from unittest.mock import patch

from toolkit import win_console
from toolkit.win_console import _ensure_text_subprocess_decode_fallback


class WinConsoleTests(unittest.TestCase):
    def test_text_subprocess_tolerates_native_windows_bytes(self):
        kwargs = {"capture_output": True, "text": True, "encoding": "utf-8"}
        _ensure_text_subprocess_decode_fallback(kwargs)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write('测试'.encode('cp936'))",
            ],
            **kwargs,
            check=True,
        )

        self.assertIn("\ufffd", result.stdout)

    def test_explicit_subprocess_error_policy_is_preserved(self):
        kwargs = {"text": True, "encoding": "utf-8", "errors": "strict"}

        _ensure_text_subprocess_decode_fallback(kwargs)

        self.assertEqual("strict", kwargs["errors"])

    def test_binary_subprocess_is_not_changed(self):
        kwargs = {"stdout": subprocess.PIPE}

        _ensure_text_subprocess_decode_fallback(kwargs)

        self.assertNotIn("errors", kwargs)

    def test_console_process_still_gets_decode_protection(self):
        with (
            patch.object(win_console.sys, "platform", "win32"),
            patch.object(win_console, "_has_console", return_value=True),
            patch.object(win_console, "_patched", False),
            patch.object(subprocess.Popen, "__init__", autospec=True, return_value=None) as original,
        ):
            win_console.suppress_child_consoles()
            subprocess.Popen(["native-tool"], text=True, encoding="utf-8")

        self.assertEqual("replace", original.call_args.kwargs.get("errors"))
        self.assertNotIn("creationflags", original.call_args.kwargs)

    def test_hidden_process_gets_decode_protection_and_no_window(self):
        with (
            patch.object(win_console.sys, "platform", "win32"),
            patch.object(win_console, "_has_console", return_value=False),
            patch.object(win_console, "_patched", False),
            patch.object(subprocess.Popen, "__init__", autospec=True, return_value=None) as original,
        ):
            win_console.suppress_child_consoles()
            subprocess.Popen(["native-tool"], text=True)

        self.assertEqual("replace", original.call_args.kwargs.get("errors"))
        self.assertEqual(win_console.CREATE_NO_WINDOW, original.call_args.kwargs["creationflags"])

    def test_console_detection_failure_does_not_disable_decode_protection(self):
        with (
            patch.object(win_console.sys, "platform", "win32"),
            patch.object(win_console, "_has_console", side_effect=OSError("unavailable")),
            patch.object(win_console, "_patched", False),
            patch.object(subprocess.Popen, "__init__", autospec=True, return_value=None) as original,
        ):
            win_console.suppress_child_consoles()
            subprocess.Popen(["native-tool"], text=True)

        self.assertEqual("replace", original.call_args.kwargs.get("errors"))
        self.assertNotIn("creationflags", original.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
