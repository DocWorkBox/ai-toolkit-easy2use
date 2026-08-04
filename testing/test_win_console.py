import subprocess
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
