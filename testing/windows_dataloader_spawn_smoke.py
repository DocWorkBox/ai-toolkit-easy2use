"""Run with the bundled pythonw.exe to verify Windows UI-style workers."""

import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.win_console import suppress_child_consoles

suppress_child_consoles()

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info


class SingleImage(Dataset):
    def __init__(self, image_path, reports):
        self.image_path = image_path
        self.reports = reports

    def __len__(self):
        return 1

    def __getitem__(self, index):
        with Image.open(self.image_path) as image:
            pixel = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
        return torch.tensor(pixel), os.getpid()


def inspect_worker(worker_id):
    # Exercise both reader threads with the same non-UTF-8 bytes as native tools.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(bytes([178,226,202,212])); "
            "sys.stderr.buffer.write(bytes([178,226,202,212]))",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=20,
    )
    get_worker_info().dataset.reports.put({
        "worker": worker_id,
        "pid": os.getpid(),
        "executable": sys.executable,
        "console": int(ctypes.windll.kernel32.GetConsoleWindow()),
        "stdout_read": bool(result.stdout),
        "stderr_read": bool(result.stderr),
    })


def main():
    assert sys.platform == "win32"
    assert sys.executable.lower().endswith("pythonw.exe"), sys.executable
    assert not ctypes.windll.kernel32.GetConsoleWindow()
    context = multiprocessing.get_context("spawn")
    reports = context.Queue()
    loader = DataLoader(
        SingleImage(sys.argv[1], reports),
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context=context,
        worker_init_fn=inspect_worker,
    )
    batches = []
    try:
        for epoch in range(5):
            epoch_batches = list(loader)
            assert len(epoch_batches) == 1
            batches.append({"epoch": epoch, "pid": int(epoch_batches[0][1][0])})
        workers = [reports.get(timeout=20) for _ in range(2)]
        assert len({worker["pid"] for worker in workers}) == 2
        assert all(worker["console"] == 0 for worker in workers), workers
        assert all(worker["executable"].lower().endswith("pythonw.exe") for worker in workers)
        assert all(worker["stdout_read"] and worker["stderr_read"] for worker in workers)
        print(json.dumps({"workers": workers, "epochs": batches, "result": "passed"}))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        reports.close()
        reports.join_thread()


if __name__ == "__main__":
    main()
