"""Qwen3.8 small assets stay writable; checkpoint resolution is unchanged."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3.8-27B"
ASSET_MODULE = "_qwen38_asset_cache_tests"
FILES = {
    "config.json": {"model_type": "qwen3_5"},
    "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
    "tokenizer.json": {},
    "preprocessor_config.json": {},
    "video_preprocessor_config.json": {},
    "chat_template.jinja": "{{ messages }}",
}


def load_assets():
    spec = importlib.util.spec_from_file_location(
        ASSET_MODULE, ROOT / "extensions_built_in/llm_models/src/qwen38_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen38LoadIntegrationTests(unittest.TestCase):
    def test_config_and_processor_use_same_local_assets_without_moving_weights(self):
        path = ROOT / "extensions_built_in/llm_models/qwen38.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "load_model")
        config, processor = Mock(), SimpleNamespace(tokenizer=Mock())
        config_loader = SimpleNamespace(from_pretrained=Mock(return_value=config))
        processor_loader = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        assets = Mock(return_value="/toolkit/models/Qwen3.8-27B")
        weights = "/readonly/text_encoders/qwen3.8_27b_w4a8.safetensors"
        checkpoint = Mock(return_value=weights)
        backend = ModuleType("extensions_built_in.llm_models.src.qwen38_load")
        backend.load_qwen38 = Mock(return_value=(Mock(), True))
        backend.dequantize_vision_tower = Mock()
        namespace = {
            "__package__": "extensions_built_in.llm_models", "os": os,
            "AutoConfig": config_loader, "AutoProcessor": processor_loader,
            "resolve_qwen38_assets": assets, "resolve_component_file": checkpoint,
            "DEFAULT_CHECKPOINT": "Comfy-Org/Qwen3.8-27B/text_encoders/qwen3.8_27b_w4a8.safetensors",
            "Qwen38Generator": Mock(), "flush": Mock(), "print_acc": Mock(),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        for source in (namespace["DEFAULT_CHECKPOINT"], "/readonly/hf-model"):
            with self.subTest(source=source):
                config_loader.from_pretrained.reset_mock()
                processor_loader.from_pretrained.reset_mock()
                assets.reset_mock()
                backend.load_qwen38.reset_mock()
                fake = SimpleNamespace(
                    torch_dtype="bf16", device_torch="cuda", vision_bf16=False, cuda_graph=True,
                    _config_source=lambda: REPO, _build_template=Mock(),
                    print_and_status_update=Mock(), model_config=SimpleNamespace(
                        name_or_path=source, extras_name_or_path=None, quantize=True,
                        qtype="convrotw4a8", layer_offloading=False,
                    ),
                )
                with patch.dict(sys.modules, {backend.__name__: backend}):
                    namespace["load_model"](fake)
                assets.assert_called_once_with(
                    REPO if source.endswith(".safetensors") else source,
                    status_fn=fake.print_and_status_update,
                )
                config_loader.from_pretrained.assert_called_once_with(assets.return_value, local_files_only=True)
                processor_loader.from_pretrained.assert_called_once_with(assets.return_value, local_files_only=True)
                self.assertEqual(backend.load_qwen38.call_args.args[0], weights if source.endswith(".safetensors") else source)
                self.assertEqual(fake.tokenizer, processor.tokenizer)


class Qwen38AssetCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = load_assets()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="qwen38-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "toolkit"
        self.cache = self.project / "models/Qwen3.8-27B"
        self.hub = self.root / "hub"
        self.patch(patch.object(self.assets, "TOOLKIT_ROOT", str(self.project)))
        self.download = self.patch(patch("huggingface_hub.snapshot_download", side_effect=self.download_assets))

    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def write_assets(self, directory, files=FILES):
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            data = value if isinstance(value, str) else json.dumps(value)
            (directory / name).write_text(data, encoding="utf-8")

    def download_assets(self, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("no cached snapshot")
        target = Path(kwargs["local_dir"])
        selected = {name: value for name, value in FILES.items() if name in kwargs["allow_patterns"]}
        self.write_assets(target, selected)
        return str(target)

    def test_downloads_only_small_files_to_project_models_and_then_works_offline(self):
        self.assertEqual(self.assets.resolve_qwen38_assets(REPO), str(self.cache))
        online = [call.kwargs for call in self.download.call_args_list if not call.kwargs.get("local_files_only")]
        self.assertEqual(len(online), 1)
        self.assertEqual(online[0]["repo_id"], REPO)
        self.assertEqual(online[0]["local_dir"], str(self.cache))
        self.assertIn("config.json", online[0]["allow_patterns"])
        self.assertIn("chat_template.jinja", online[0]["allow_patterns"])
        self.assertTrue(all("*" not in name and not name.endswith((".bin", ".safetensors"))
                            for name in online[0]["allow_patterns"]))
        self.download.reset_mock()
        self.download.side_effect = AssertionError("network used despite complete project cache")
        self.assertEqual(self.assets.resolve_qwen38_assets(REPO), str(self.cache))
        self.download.assert_not_called()

    def test_copies_cached_hub_assets_without_touching_source_or_weights(self):
        self.write_assets(self.hub)
        weight = self.hub / "model.safetensors"
        weight.write_bytes(b"do not copy")
        before = {p.name: p.read_bytes() for p in self.hub.iterdir()}
        self.download.side_effect = lambda **_: str(self.hub)
        self.assertEqual(self.assets.resolve_qwen38_assets(REPO), str(self.cache))
        self.download.assert_called_once()
        self.assertTrue(self.download.call_args.kwargs["local_files_only"])
        self.assertFalse((self.cache / weight.name).exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.hub.iterdir()})

    def test_partial_project_cache_keeps_existing_files_and_repairs_empty_files(self):
        self.write_assets(self.cache, {"config.json": {"keep": True}, "tokenizer.json": ""})
        self.assets.resolve_qwen38_assets(REPO)
        online = [call for call in self.download.call_args_list if not call.kwargs.get("local_files_only")]
        self.assertNotIn("config.json", online[0].kwargs["allow_patterns"])
        self.assertIn("tokenizer.json", online[0].kwargs["allow_patterns"])
        self.assertEqual(json.loads((self.cache / "config.json").read_text()), {"keep": True})
        self.assertGreater((self.cache / "tokenizer.json").stat().st_size, 0)

    def test_explicit_local_config_directory_is_not_changed(self):
        source = self.root / "readonly-custom-model"
        self.write_assets(source)
        (source / "model.safetensors").write_bytes(b"weight")
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        self.assertEqual(self.assets.resolve_qwen38_assets(str(source)), str(source))
        self.download.assert_not_called()
        self.assertFalse(self.project.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_custom_repositories_are_isolated(self):
        for repo in ("org-a/Qwen3.8-27B", "org-b/Qwen3.8-27B"):
            with self.subTest(repo=repo):
                expected = self.project / "models" / repo
                self.assertEqual(self.assets.resolve_qwen38_assets(repo), str(expected))

    def test_closed_client_is_recreated_once_and_download_is_retried(self):
        calls = 0

        def download(**kwargs):
            nonlocal calls
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("no cache")
            calls += 1
            if calls == 1:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return self.download_assets(**kwargs)

        self.download.side_effect = download
        with patch("huggingface_hub.utils.close_session", create=True) as reset:
            self.assertEqual(self.assets.resolve_qwen38_assets(REPO), str(self.cache))
        reset.assert_called_once_with()
        self.assertEqual(calls, 2)

    def test_client_retry_is_bounded_and_unrelated_errors_are_not_retried(self):
        for message, resets, attempts in (
            ("Cannot send a request, as the client has been closed.", 1, 2),
            ("unrelated runtime failure", 0, 1),
        ):
            with self.subTest(message=message):
                self.download.reset_mock()
                def download(**kwargs):
                    if kwargs.get("local_files_only"):
                        raise FileNotFoundError("no cache")
                    raise RuntimeError(message)
                self.download.side_effect = download
                with patch("huggingface_hub.utils.close_session", create=True) as reset:
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.assets.resolve_qwen38_assets(REPO)
                self.assertEqual(reset.call_count, resets)
                self.assertEqual(sum(not call.kwargs.get("local_files_only")
                                     for call in self.download.call_args_list), attempts)

    def test_incomplete_download_cannot_be_reported_ready(self):
        self.download.side_effect = lambda **_: str(self.root / "missing")
        with self.assertRaisesRegex(FileNotFoundError, "Qwen3.8"):
            self.assets.resolve_qwen38_assets(REPO)

    def test_tokenizer_alternatives_and_embedded_chat_template_work_offline(self):
        files = {name: value for name, value in FILES.items()
                 if name not in ("tokenizer.json", "chat_template.jinja")}
        files.update({"vocab.json": {}, "merges.txt": "#version: 0.2",
                      "tokenizer_config.json": {"chat_template": "{{ messages }}"}})
        self.write_assets(self.cache, files)
        self.assertEqual(self.assets.resolve_qwen38_assets(REPO), str(self.cache))
        self.download.assert_not_called()

    def test_missing_local_path_and_traversal_repo_do_not_download(self):
        for source in (str(self.root / "missing-local"), "../escape"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.assets.resolve_qwen38_assets(source)
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
