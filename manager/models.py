"""Read-only validation for portable model files and directories."""

import json
import os
from pathlib import Path, PurePosixPath

from manager import util


CATALOG_FILENAME = "portable_models.json"
IGNORED_TOP_LEVEL_NAMES = {".cache", ".locks"}
GENERIC_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.gguf",
)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ModelCatalogError(RuntimeError):
    pass


def load_catalog(catalog_path=None):
    path = Path(catalog_path or Path(util.REPO_ROOT) / CATALOG_FILENAME)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCatalogError("Unable to read model catalog %s: %s" % (path, error))

    if data.get("schema") != 1 or not isinstance(data.get("models"), list):
        raise ModelCatalogError("Unsupported model catalog schema in %s" % path)
    return data


def scan_models(repo_root=None, catalog_path=None):
    root = Path(repo_root or util.REPO_ROOT).resolve()
    catalog_file = Path(catalog_path or root / CATALOG_FILENAME)
    catalog = load_catalog(catalog_file)
    models_root = _portable_path(root, catalog.get("models_root", "./models"))
    misplaced_index = _build_misplaced_index(models_root)
    expected_top_level = set()
    recognized_actual_top_level = set()
    results = []

    for entry in catalog["models"]:
        portable_path = entry.get("path", "")
        target = _portable_path(root, portable_path)
        relative = target.relative_to(root)
        if relative.parts:
            expected_top_level.add(relative.parts[1] if relative.parts[0] == "models" else relative.parts[0])
        result = _scan_entry(root, entry, target, misplaced_index)
        results.append(result)
        actual_path = Path(result["absolute_path"])
        if result["status"] in {"ready", "incomplete", "misplaced", "name_mismatch"}:
            try:
                recognized_actual_top_level.add(actual_path.relative_to(models_root).parts[0])
            except (ValueError, IndexError):
                pass

    if models_root.is_dir():
        for child in sorted(models_root.iterdir(), key=lambda item: item.name.casefold()):
            if child.name in IGNORED_TOP_LEVEL_NAMES or child.name.startswith("."):
                continue
            if child.name.casefold() in {
                name.casefold() for name in expected_top_level | recognized_actual_top_level
            }:
                continue
            results.append(
                {
                    "id": "unrecognized:%s" % child.name,
                    "name": child.name,
                    "category": "未识别",
                    "status": "unrecognized",
                    "path": "./models/%s" % child.name.replace("\\", "/"),
                    "absolute_path": str(child.resolve()),
                    "detail": "不在便携版默认模型清单中；仍可在任务配置中手动指定。",
                    "download_url": "",
                    "special": False,
                }
            )

    order = {"incomplete": 0, "misplaced": 1, "name_mismatch": 2, "ready": 3, "missing": 4, "unrecognized": 5}
    results.sort(
        key=lambda item: (
            order.get(item["status"], 9),
            item["category"],
            item["name"].casefold(),
        )
    )
    summary = {
        "ready": sum(item["status"] == "ready" for item in results),
        "issues": sum(item["status"] in {"incomplete", "misplaced", "name_mismatch"} for item in results),
        "missing": sum(item["status"] == "missing" for item in results),
        "unrecognized": sum(item["status"] == "unrecognized" for item in results),
        "total": len(results),
    }
    return {
        "schema": 1,
        "models_root": str(models_root),
        "catalog_path": str(catalog_file.resolve()),
        "summary": summary,
        "models": results,
    }


def _scan_entry(root, entry, target, misplaced_index):
    result = {
        "id": str(entry["id"]),
        "name": str(entry["name"]),
        "category": str(entry.get("category", "模型")),
        "status": "missing",
        "path": str(entry["path"]),
        "absolute_path": str(target),
        "detail": "未安装。请下载后放到 %s" % entry["path"],
        "download_url": str(entry.get("download_url", "")),
        "special": bool(entry.get("special", False)),
    }

    actual, case_mismatch = _resolve_existing_case(root, target.relative_to(root).parts)
    if actual is None:
        misplaced = _find_misplaced(misplaced_index, target, entry.get("kind"))
        if misplaced is not None:
            result["status"] = "misplaced"
            result["absolute_path"] = str(misplaced.resolve())
            result["detail"] = "发现于 %s；应移动到 %s" % (misplaced, entry["path"])
        return result

    result["absolute_path"] = str(actual.resolve())
    expected_kind = entry.get("kind")
    if expected_kind == "file" and not actual.is_file():
        result["status"] = "incomplete"
        result["detail"] = "目标应为文件：%s" % entry["path"]
        return result
    if expected_kind == "directory" and not actual.is_dir():
        result["status"] = "incomplete"
        result["detail"] = "目标应为目录：%s" % entry["path"]
        return result

    problems = _validate_content(actual, entry)
    if problems:
        result["status"] = "incomplete"
        result["detail"] = "；".join(problems)
        return result

    if case_mismatch:
        result["status"] = "name_mismatch"
        result["detail"] = "当前名称为 %s；建议改为 %s" % (actual, entry["path"])
        return result

    result["status"] = "ready"
    result["detail"] = "路径和文件结构可被训练器读取。"
    return result


def _portable_path(root, value):
    if not isinstance(value, str) or not value.startswith("./models"):
        raise ModelCatalogError("Model path must start with ./models: %r" % value)
    pure = PurePosixPath(value[2:])
    if pure.is_absolute() or ".." in pure.parts:
        raise ModelCatalogError("Unsafe model path: %r" % value)
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ModelCatalogError("Model path escapes repository root: %r" % value)
    return target


def _resolve_existing_case(root, parts):
    current = root
    mismatch = False
    for part in parts:
        if not current.is_dir():
            return None, mismatch
        exact = None
        folded = None
        try:
            children = list(current.iterdir())
        except OSError:
            return None, mismatch
        for child in children:
            if child.name == part:
                exact = child
                break
            if child.name.casefold() == part.casefold():
                folded = child
        if exact is not None:
            current = exact
        elif folded is not None:
            current = folded
            mismatch = True
        else:
            return None, mismatch
    return current, mismatch


def _validate_content(target, entry):
    if target.is_file():
        problem = _weight_file_problem(target)
        return [problem] if problem else []

    problems = []
    for pattern in entry.get("required_all", []):
        matches = _matches(target, pattern)
        if not matches:
            problems.append("缺少 %s" % pattern)
        else:
            problems.extend(filter(None, (_weight_file_problem(path) for path in matches)))

    required_any = entry.get("required_any", [])
    if required_any:
        matches = []
        for pattern in required_any:
            matches.extend(_matches(target, pattern))
        if not matches:
            problems.append("至少需要以下一种文件：%s" % "、".join(required_any))
        else:
            problems.extend(filter(None, (_weight_file_problem(path) for path in matches)))

    if not entry.get("required_all") and not required_any:
        weights = []
        for pattern in GENERIC_WEIGHT_PATTERNS:
            weights.extend(_limited_glob(target, pattern, max_depth=3))
        if not weights:
            problems.append("目录中未找到模型权重文件")
        else:
            problems.extend(filter(None, (_weight_file_problem(path) for path in weights)))
    return list(dict.fromkeys(problems))


def _matches(root, pattern):
    return [path for path in root.glob(pattern) if path.is_file()]


def _limited_glob(root, pattern, max_depth):
    matches = []
    for path in root.rglob(pattern):
        try:
            depth = len(path.relative_to(root).parts) - 1
        except ValueError:
            continue
        if depth <= max_depth and path.is_file():
            matches.append(path)
    return matches


def _weight_file_problem(path):
    if path.suffix.lower() not in {".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".gguf"}:
        return None
    try:
        if path.stat().st_size == 0:
            return "%s 是空文件" % path.name
        with path.open("rb") as stream:
            if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                return "%s 仍是 Git LFS 指针，模型权重尚未下载" % path.name
    except OSError as error:
        return "无法读取 %s：%s" % (path.name, error)
    return None


def _find_misplaced(index, target, kind):
    if kind == "file" and target.name != "config.json":
        candidates = [
            candidate
            for candidate in index["files"].get(target.name.casefold(), [])
            if candidate.resolve() != target
        ]
        if len(candidates) == 1:
            return candidates[0]
    if kind == "directory":
        candidates = [
            candidate
            for candidate in index["directories"].get(target.name.casefold(), [])
            if candidate.resolve() != target
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _build_misplaced_index(models_root):
    index = {"files": {}, "directories": {}}
    if not models_root.is_dir():
        return index

    for current_value, directory_names, file_names in os.walk(models_root):
        current = Path(current_value)
        try:
            depth = len(current.relative_to(models_root).parts)
        except ValueError:
            continue

        if depth <= 2:
            for name in directory_names:
                index["directories"].setdefault(name.casefold(), []).append(current / name)
        if depth <= 3:
            for name in file_names:
                index["files"].setdefault(name.casefold(), []).append(current / name)
        if depth >= 3:
            directory_names[:] = []
    return index
