"""Resumable, revision-pinned model downloads for the chengyu creator volume."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import json
import os
from pathlib import Path
import threading
import time


SMALL_PATTERNS = ("*.json", "*.py", "*.txt", "*.model", "*.jinja", "*.yaml", "*.yml", "LICENSE*")
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
DONE_STATUSES = {"existing", "reused", "downloaded"}


def target_path(root, item):
    root = Path(root).absolute()
    relative = Path(item["local_dir"]) / item["filename"]
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid model destination: {relative}")
    target = root / relative
    parent = target.parent.resolve()
    if not parent.is_relative_to(root.resolve()):
        raise ValueError(f"Model destination escapes root: {target}")
    return target


def reusable_source(item, index):
    filename = Path(item["filename"]).name
    repo_name = item["repo_id"].split("/")[-1].lower()
    for source in index.get(filename, []):
        if not source.is_file() or source.stat().st_size != item["size"]:
            continue
        if item["kind"] in {"snapshot", "assets", "code"}:
            # Common shard/config names must come from the same model repository.
            markers = {part.lower() for part in source.parts}
            cache_name = "models--" + item["repo_id"].replace("/", "--").lower()
            if repo_name not in markers and cache_name not in markers:
                continue
        yield source


def fetch_file(item, root, index, download):
    target = target_path(root, item)
    if target.exists():
        if target.is_file() and target.stat().st_size == item["size"]:
            return {"status": "existing", "path": str(target)}
        raise FileExistsError(f"Existing destination differs; not overwriting: {target}")
    if target.is_symlink():
        raise FileExistsError(f"Broken destination symlink; not replacing: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for source in reusable_source(item, index):
        if source.absolute() == target.absolute():
            continue
        try:
            try:
                os.link(source, target)
            except OSError:
                target.symlink_to(source.absolute())
        except FileExistsError:
            if target.is_file() and target.stat().st_size == item["size"]:
                return {"status": "existing", "path": str(target)}
            raise
        return {"status": "reused", "path": str(target), "source": str(source)}
    downloaded = download(
        repo_id=item["repo_id"], revision=item["revision"], filename=item["filename"],
        local_dir=str(Path(root) / item["local_dir"]), token=False,
    )
    if Path(downloaded).stat().st_size != item["size"]:
        raise OSError(f"Downloaded size does not match metadata: {downloaded}")
    return {"status": "downloaded", "path": str(target)}


def prepare_items(manifest, api):
    items = []
    for group in manifest["groups"]:
        print(f"METADATA {group['repo_id']}", flush=True)
        info = api.model_info(group["repo_id"], files_metadata=True)
        siblings = {file.rfilename: file for file in info.siblings}
        filenames = group.get("files")
        kind = group.get("kind", "files")
        if filenames is None:
            patterns = SMALL_PATTERNS + (WEIGHT_PATTERNS if kind == "snapshot" else ())
            filenames = [name for name in siblings if not name.startswith(("assets/", "examples/", "."))
                         and any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
        for filename in filenames:
            if filename not in siblings:
                raise FileNotFoundError(f"Missing Hub file: {group['repo_id']}/{filename}")
            metadata = siblings[filename]
            size = metadata.size
            if not isinstance(size, int) or size < 0:
                raise ValueError(f"Missing size metadata: {group['repo_id']}/{filename}")
            items.append({"repo_id": group["repo_id"], "revision": info.sha, "filename": filename,
                          "local_dir": group["local_dir"], "kind": kind, "size": size,
                          "sha256": getattr(getattr(metadata, "lfs", None), "sha256", None)})
    return items


def index_existing(roots, items):
    wanted = {Path(item["filename"]).name for item in items}
    index = {}
    for root in roots:
        if not Path(root).is_dir():
            continue
        print(f"SEARCH {root}", flush=True)
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {".git", ".cache", "__pycache__", "node_modules"}]
            for name in wanted.intersection(files):
                path = Path(directory) / name
                if path.is_file():
                    index.setdefault(name, []).append(path)
    return index


def write_json(path, data):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def get_progress(root):
    root = Path(root)
    state_path = root / "download-state.json"
    if not state_path.is_file():
        return {"status": "preparing"}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    files = list(state["files"].values())
    received = sum(file["size"] for file in files if file["status"] in DONE_STATUSES)
    for path in root.rglob("*.incomplete"):
        try:
            received += path.stat().st_size
        except FileNotFoundError:
            pass
    return {"status": state["status"], "pid": state.get("pid"),
            "total_bytes": state["total_bytes"], "received_bytes": received,
            "completed_files": sum(file["status"] in DONE_STATUSES for file in files),
            "failed_files": sum(file["status"] == "failed" for file in files)}


def run(manifest, root, workers, dry_run=False):
    from huggingface_hub import HfApi, hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    items = prepare_items(manifest, HfApi(token=False))
    index = index_existing([str(root)] + manifest["reuse_roots"], items)
    total = sum(item["size"] for item in items)
    print(f"PLAN {len(items)} files {total / 2**30:.3f} GiB", flush=True)
    if dry_run:
        for item in items:
            print(json.dumps({**item, "target": str(target_path(root, item))}), flush=True)
        return 0
    status = {"status": "running", "pid": os.getpid(), "started_at": time.time(),
              "total_bytes": total, "files": {}}
    status_path = root / "download-state.json"
    lock = threading.Lock()

    def record(key, value):
        with lock:
            status["files"][key] = value
            status["updated_at"] = time.time()
            write_json(status_path, status)

    def process(item):
        key = str(Path(item["local_dir"]) / item["filename"])
        record(key, {**item, "status": "downloading"})
        for attempt in range(3):
            try:
                result = fetch_file(item, root, index, hf_hub_download)
                record(key, {**item, **result})
                print(f"DONE {result['status']} {key}", flush=True)
                return
            except FileExistsError as error:
                record(key, {**item, "status": "failed", "error": str(error)})
                return
            except Exception as error:
                if attempt == 2:
                    record(key, {**item, "status": "failed", "error": str(error)})
                    print(f"FAILED {key}: {error}", flush=True)
                    return
                print(f"RETRY {attempt + 1} {key}: {error}", flush=True)
                time.sleep(5 * (attempt + 1))

    write_json(status_path, status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, item) for item in items]
        for future in as_completed(futures):
            future.result()
    for group in manifest["groups"]:
        if group.get("kind") not in {"snapshot", "assets", "code"}:
            continue
        selected = [value for value in status["files"].values()
                    if value["repo_id"] == group["repo_id"] and value["local_dir"] == group["local_dir"]]
        if selected and all(value["status"] in DONE_STATUSES for value in selected):
            write_json(root / group["local_dir"] / ".ai-toolkit-complete.json",
                       {"repo_id": group["repo_id"], "revision": selected[0]["revision"],
                        "files": [{"filename": value["filename"], "size": value["size"]} for value in selected]})
    failed = [key for key, value in status["files"].items() if value["status"] == "failed"]
    status.update(status="failed" if failed else "complete", finished_at=time.time(), failed_files=failed)
    write_json(status_path, status)
    print(f"FINISHED status={status['status']} failed={len(failed)}", flush=True)
    return bool(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("chengyu_models.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root or Path(manifest["root"])
    if args.status:
        print(json.dumps(get_progress(root), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_XET_CACHE", str(root / ".cache" / "xet"))
    import fcntl

    with (root / "download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(manifest, root, max(1, args.workers), args.dry_run))


if __name__ == "__main__":
    main()
