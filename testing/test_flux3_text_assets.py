"""FLUX.3 text assets stay writable without changing Comfy weight selection."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from testing.test_flux3_staged_sampling import load_flux3


class Flux3TextAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = load_flux3()
        cls.assets = sys.modules[cls.flux.Flux3TextEncoder.__module__]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".", prefix="flux3-text-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / "models/Qwen3-VL-4B-Instruct"
        self.source = self.root / "readonly-models/Qwen3-VL-4B-Instruct"
        for name, value in (
            ("TOOLKIT_ROOT", str(self.root)),
        ):
            patcher = patch.object(self.assets, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_assets(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "preprocessor_config.json", "video_preprocessor_config.json",
            "chat_template.json",
        ):
            (directory / filename).write_text(json.dumps({"test": filename}), encoding="utf-8")

    def make_model(self, te_path=None, model_kwargs=None):
        model = self.flux.Flux3Model.__new__(self.flux.Flux3Model)
        model.model_config = SimpleNamespace(
            name_or_path=self.flux.BASE_REPO,
            te_name_or_path=te_path, model_kwargs=model_kwargs or {},
        )
        model.component_load_kwargs = lambda _: {"dtype": "bf16", "qtype": "float8"}
        model.print_and_status_update = Mock()
        return model

    def test_complete_cache_is_reused_offline(self):
        self.write_assets(self.cache)
        with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")) as download:
            self.assertEqual(self.flux.Flux3TextEncoder.resolve_assets(), str(self.cache))
        download.assert_not_called()

    def test_copies_only_small_assets_from_existing_readonly_model(self):
        self.write_assets(self.source)
        (self.source / "model.safetensors").write_bytes(b"not a real weight")
        (self.source / "pytorch_model.bin").write_bytes(b"not a real weight")
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch("huggingface_hub.snapshot_download") as download:
            result = self.flux.Flux3TextEncoder.resolve_assets(local_source=str(self.source))
        self.assertEqual(result, str(self.cache))
        self.assertTrue((self.cache / "tokenizer.json").is_file())
        self.assertFalse((self.cache / "model.safetensors").exists())
        self.assertFalse((self.cache / "pytorch_model.bin").exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        download.assert_not_called()

    def test_config_only_cache_is_completed_from_source(self):
        self.write_assets(self.source)
        self.cache.mkdir(parents=True)
        (self.cache / "config.json").write_text('{"keep": true}', encoding="utf-8")
        with patch("huggingface_hub.snapshot_download") as download:
            self.flux.Flux3TextEncoder.resolve_assets(local_source=str(self.source))
        self.assertEqual(json.loads((self.cache / "config.json").read_text()), {"keep": True})
        self.assertTrue((self.cache / "chat_template.json").is_file())
        download.assert_not_called()

    def test_downloads_only_assets_into_toolkit_models(self):
        def download_assets(**kwargs):
            self.write_assets(Path(kwargs["local_dir"]))
            return kwargs["local_dir"]

        with patch("huggingface_hub.snapshot_download", side_effect=download_assets) as download:
            self.assertEqual(self.flux.Flux3TextEncoder.resolve_assets(), str(self.cache))
        kwargs = download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], self.flux.QWEN3_VL_4B_REPO)
        self.assertEqual(kwargs["local_dir"], str(self.cache))
        self.assertIn("tokenizer.json", kwargs["allow_patterns"])
        self.assertNotIn("*", kwargs["allow_patterns"])
        self.assertFalse(any(name.endswith((".safetensors", ".bin")) for name in kwargs["allow_patterns"]))

    def test_partial_source_is_supplemented_by_download(self):
        self.source.mkdir(parents=True)
        (self.source / "config.json").write_text("{}", encoding="utf-8")

        def download_assets(**kwargs):
            self.assertTrue((self.cache / "config.json").is_file())
            self.assertNotIn("config.json", kwargs["allow_patterns"])
            self.write_assets(Path(kwargs["local_dir"]))

        with patch("huggingface_hub.snapshot_download", side_effect=download_assets) as download:
            self.flux.Flux3TextEncoder.resolve_assets(local_source=str(self.source))
        download.assert_called_once()

    def test_default_assets_use_official_repo_without_a_server_specific_source(self):
        self.write_assets(self.source)

        def download_assets(**kwargs):
            self.write_assets(Path(kwargs["local_dir"]))

        with patch("huggingface_hub.snapshot_download", side_effect=download_assets) as download:
            self.flux.Flux3TextEncoder.resolve_assets()
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["repo_id"], self.flux.QWEN3_VL_4B_REPO)

    def test_incomplete_download_does_not_report_ready(self):
        with patch("huggingface_hub.snapshot_download", return_value=str(self.cache)):
            with self.assertRaisesRegex(FileNotFoundError, "Qwen3-VL"):
                self.flux.Flux3TextEncoder.resolve_assets()

    def test_default_loader_keeps_comfy_weights_and_uses_cached_config_and_processor(self):
        self.write_assets(self.cache)
        encoder = Mock()
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
        cls = self.flux.Flux3TextEncoder
        comfy_file = str(self.root / "text_encoders/qwen3vl_4b_fp8_scaled.safetensors")
        with patch.object(cls, "resolve_comfy_weights", return_value=comfy_file) as resolve:
            with patch.object(cls, "_resolve_single_file", return_value=comfy_file):
                with patch.object(cls, "_load_single_file", return_value=encoder) as load:
                    with patch.object(cls, "load_processor", return_value=processor) as load_processor:
                        with patch.object(self.flux, "flush"):
                            result = self.make_model()._load_text_encoder()
        resolve.assert_called_once_with(self.flux.QWEN3_VL_4B_REPO, subfolder=None, qtype="float8")
        self.assertEqual(load.call_args.kwargs["config_path"], str(self.cache))
        load_processor.assert_called_once_with(str(self.cache))
        self.assertEqual(result, (processor, encoder))
        self.assertEqual(processor.tokenizer.padding_side, "right")
        encoder.drop_vision_tower.assert_called_once()
        encoder.quantize_.assert_not_called()

    def test_explicit_local_model_directory_still_takes_priority(self):
        self.write_assets(self.source)
        for te_path, model_kwargs in (
            (str(self.source), {}),
            (None, {"text_encoder_path": str(self.source)}),
        ):
            with self.subTest(te_path=te_path):
                cls = self.flux.Flux3TextEncoder
                processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
                with patch.object(cls, "load_model", return_value=Mock()) as load:
                    with patch.object(cls, "load_processor", return_value=processor) as load_processor:
                        with patch("huggingface_hub.snapshot_download") as download:
                            with patch.object(self.flux, "flush"):
                                self.make_model(te_path, model_kwargs)._load_text_encoder()
                self.assertEqual(load.call_args.args[0], str(self.source))
                load_processor.assert_called_once_with(str(self.source))
                download.assert_not_called()

    def test_explicit_single_file_keeps_weight_path_and_caches_assets(self):
        self.write_assets(self.cache)
        weight = str(self.source / "custom.safetensors")
        cls = self.flux.Flux3TextEncoder
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
        with patch.object(cls, "load_model", return_value=Mock()) as load:
            with patch.object(cls, "load_processor", return_value=processor) as load_processor:
                with patch.object(self.flux, "flush"):
                    self.make_model(weight)._load_text_encoder()
        self.assertEqual(load.call_args.args[0], weight)
        self.assertEqual(load.call_args.kwargs["config_path"], str(self.cache))
        load_processor.assert_called_once_with(str(self.cache))


if __name__ == "__main__":
    unittest.main()
