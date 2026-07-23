import ast
import glob
import os
import re
from pathlib import Path
from types import SimpleNamespace

import yaml


SOURCE = Path("jobs/process/BaseSDTrainProcess.py")


def _load_checkpoint_selector(metadata_by_path=None):
    source_tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    process_class = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess"
    )
    method_names = {"_get_checkpoint_step", "get_latest_save_path"}
    methods = [
        node
        for node in process_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names

    selector_class = ast.ClassDef(
        name="CheckpointSelector",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[selector_class], type_ignores=[])
    )
    metadata_by_path = metadata_by_path or {}
    namespace = {
        "glob": glob,
        "load_metadata_from_safetensors": lambda path: metadata_by_path.get(
            path, {}
        ),
        "os": os,
        "print_acc": lambda message: None,
        "re": re,
        "yaml": yaml,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["CheckpointSelector"]


def _make_selector(tmp_path, metadata_by_path=None):
    selector_class = _load_checkpoint_selector(metadata_by_path)
    selector = selector_class()
    selector.save_root = str(tmp_path)
    selector.job = SimpleNamespace(name="job")
    selector.network_config = None
    return selector


def test_latest_checkpoint_uses_highest_step_instead_of_newest_ctime(
    tmp_path, monkeypatch
):
    low_step = tmp_path / "job_000001500.safetensors"
    high_step = tmp_path / "job_000027000.safetensors"
    low_step.touch()
    high_step.touch()
    ctimes = {
        str(low_step): 200,
        str(high_step): 100,
    }
    monkeypatch.setattr(os.path, "getctime", lambda path: ctimes[path])

    selector = _make_selector(tmp_path)

    assert selector.get_latest_save_path() == str(high_step)


def test_final_checkpoint_metadata_step_beats_numbered_checkpoint(tmp_path):
    numbered = tmp_path / "job_000027000.safetensors"
    final = tmp_path / "job.safetensors"
    numbered.touch()
    final.touch()
    metadata = {
        str(final): {
            "training_info": {
                "step": 27500,
            }
        }
    }

    selector = _make_selector(tmp_path, metadata)

    assert selector.get_latest_save_path() == str(final)
