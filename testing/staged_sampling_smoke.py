"""Run an isolated four-step portable training/sampling smoke test."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-text-embeddings", action="store_true")
    parser.add_argument("--layer-offloading", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.config or root / "output/my_first_lora_v1/.job_config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["config"]["name"] = "staged_sampling_smoke"
    process = config["config"]["process"][0]
    process["training_folder"] = str(out)
    process["sqlite_db_path"] = str(out / "unused.db")
    process["model"]["sample_staged_offloading"] = True
    process["model"]["layer_offloading"] = args.layer_offloading
    if args.layer_offloading:
        process["model"]["layer_offloading_transformer_percent"] = 0.5
        process["model"]["layer_offloading_text_encoder_percent"] = 0.5
    process["model"]["compile"] = False
    process["train"].update(steps=4, cache_text_embeddings=args.cache_text_embeddings, skip_first_sample=False)
    process["sample"]["sample_every"] = 2
    if args.sample_count > 1:
        base = process["sample"]["samples"][0]
        variations = ["at sunrise", "at sunset", "in the rain", "on a sunny day", "under cloudy skies",
                      "at night", "in winter", "in autumn", "in spring", "in summer"]
        process["sample"]["samples"] = [dict(base, prompt=base["prompt"] + ", " + variations[i % len(variations)])
                                         for i in range(args.sample_count)]
    process["save"].update(save_every=100, push_to_hub=False)
    # Isolate caches as well as generated images/checkpoints from the user's dataset.
    for i, dataset in enumerate(process["datasets"]):
        src = Path(dataset["folder_path"])
        if not src.is_absolute():
            src = root / src
        dst = out / f"dataset_{i}"
        dst.mkdir()
        for path in src.iterdir():
            if path.is_file() and not path.name.startswith("."):
                shutil.copy2(path, dst / path.name)
        dataset["folder_path"] = str(dst)
        dataset["cache_latents_to_disk"] = True
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    for key in ("AITK_JOB_ID", "IS_AI_TOOLKIT_UI"):
        env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with sqlite3.connect((root / "aitk_db.db").as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute('SELECT value FROM Settings WHERE key = ?', ("MODELS_PATH",)).fetchone()
    env["MODELS_PATH"] = row[0] if row and row[0] else str(root / "models")
    python = root / "runtime/python/python.exe"
    with (out / "run.log").open("w", encoding="utf-8") as log, (out / "gpu.csv").open("w", encoding="utf-8", newline="") as telemetry:
        writer = csv.writer(telemetry)
        writer.writerow(["elapsed_s", "gpu_memory_mib", "gpu_util_percent", "power_w"])
        child = subprocess.Popen(
            [str(python), "-X", "utf8", "-u", str(root / "run.py"), str(config_path)],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        print(f"SMOKE_PID={child.pid} OUTPUT={out}", flush=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f"Smoke test exceeded {args.timeout} seconds")
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                writer.writerow([round(time.monotonic() - started, 2), *gpu.stdout.strip().split(", ")])
                telemetry.flush()
                time.sleep(1)
        finally:
            if child.poll() is None:
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait()
        print(f"EXIT_CODE={child.returncode} ELAPSED={time.monotonic() - started:.2f}s", flush=True)
        return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
