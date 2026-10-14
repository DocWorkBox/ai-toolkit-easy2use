import importlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def downloader():
    assert importlib.util.find_spec("manager.portable_download"), "Idle-aware streaming downloader is missing"
    return importlib.import_module("manager.portable_download")


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "20")
            self.end_headers()
            try:
                for index in range(20):
                    if self.path == "/stall" and index == 1:
                        time.sleep(0.8)
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % httpd.server_port
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize("transport", ["python", "curl"])
def test_slow_continuous_transfer_outlives_idle_timeout(server, tmp_path, transport):
    module = downloader()
    counts = []
    started = time.monotonic()
    module.download(server + "/slow", tmp_path / "file", progress=counts.append,
                    idle_timeout=0.3, transport=transport)
    assert time.monotonic() - started > 0.6
    assert (tmp_path / "file").read_bytes() == b"x" * 20
    assert len(set(counts)) > 3
    assert counts[-1] == 20


@pytest.mark.parametrize("transport", ["python", "curl"])
def test_no_data_transfer_times_out(server, tmp_path, transport):
    module = downloader()
    started = time.monotonic()
    with pytest.raises(OSError, match="[Tt]imed out|[Nn]o data"):
        module.download(server + "/stall", tmp_path / "file", idle_timeout=0.3,
                        transport=transport)
    assert time.monotonic() - started < 2


def test_python_ssl_failure_falls_back_to_streaming_curl(monkeypatch, server, tmp_path):
    module = downloader()
    def fail(*args, **kwargs):
        raise OSError("SSL handshake failed")
    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    counts = []
    module.download(server + "/slow", tmp_path / "file", progress=counts.append, idle_timeout=1)
    assert (tmp_path / "file").read_bytes() == b"x" * 20
    assert counts[-1] == 20
