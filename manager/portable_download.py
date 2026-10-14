"""Portable updater downloads with streaming progress and inactivity timeouts."""

import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from . import util


IDLE_TIMEOUT = 60


def download(url, destination, *, progress=None, headers=None, idle_timeout=IDLE_TIMEOUT,
             transport=None):
    headers = {"User-Agent": util.USER_AGENT, **(headers or {})}
    destination = Path(destination)
    if transport == "curl":
        return _curl(url, destination, headers, progress, idle_timeout)
    try:
        request = urllib.request.Request(url, headers=headers)
        # Socket timeouts apply to inactivity, not the duration of the whole file.
        with urllib.request.urlopen(request, timeout=idle_timeout) as response:
            with destination.open("wb") as output:
                count = 0
                while True:
                    # read() can wait to fill a buffer, hiding a slow but live transfer.
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    count += len(chunk)
                    if progress:
                        progress(count)
                length = response.headers.get("Content-Length")
                if length is not None and count != int(length):
                    raise OSError("Incomplete download: received %d of %s bytes" % (count, length))
    except (OSError, ValueError) as error:
        if transport == "python" or not shutil.which("curl"):
            raise
        util.warn("Download interrupted (%s); retrying with curl." % error)
        if progress:
            progress(0)
        _curl(url, destination, headers, progress, idle_timeout)


def _curl(url, destination, headers, progress, idle_timeout):
    curl = shutil.which("curl")
    if not curl:
        raise OSError("curl is not available")
    command = [curl, "--fail", "--location", "--silent", "--show-error", "--no-buffer",
               "--connect-timeout", str(idle_timeout)]
    for key, value in headers.items():
        command += ["--header", "%s: %s" % (key, value)]
    command.append(url)
    chunks = queue.Queue()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, env=util.clean_env(),
            creationflags=subprocess.CREATE_NO_WINDOW if util.IS_WINDOWS else 0,
        )

        def read():
            try:
                while True:
                    chunk = process.stdout.read1(64 * 1024)
                    if not chunk:
                        break
                    chunks.put(chunk)
            finally:
                chunks.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            with destination.open("wb") as output:
                count = 0
                while True:
                    try:
                        chunk = chunks.get(timeout=idle_timeout)
                    except queue.Empty as error:
                        raise TimeoutError("No data received for %s seconds" % idle_timeout) from error
                    if chunk is None:
                        break
                    output.write(chunk)
                    count += len(chunk)
                    if progress:
                        progress(count)
            try:
                code = process.wait(timeout=idle_timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("No data received for %s seconds" % idle_timeout) from error
            if code:
                errors.seek(0)
                raise OSError(errors.read().decode("utf-8", errors="replace").strip()
                              or "curl exited with code %d" % code)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join()
            process.stdout.close()
