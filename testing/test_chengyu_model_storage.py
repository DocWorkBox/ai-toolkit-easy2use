import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
STORE = "/poddata/ai-toolkit-models"
REPO = "m-a-p/MERT-v2-FullSong"


class ChengyuStorageDefaultsTests(unittest.TestCase):
    def test_python_defaults_to_creator_models_and_keeps_environment_override(self):
        with patch.dict("os.environ", {"MODELS_PATH": ""}):
            paths = runpy.run_path(str(ROOT / "toolkit/paths.py"))
        self.assertEqual(paths["MODELS_PATH"], STORE)
        with patch.dict("os.environ", {"MODELS_PATH": "/custom/model-store"}):
            paths = runpy.run_path(str(ROOT / "toolkit/paths.py"))
        self.assertEqual(paths["MODELS_PATH"], "/custom/model-store")

    def test_ui_and_worker_share_creator_default_and_recognize_old_project_default(self):
        for filename in ("ui/src/paths.ts", "ui/cron/paths.ts"):
            text = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(f"defaultModelsFolder = '{STORE}'", text)
            self.assertIn("legacyDefaultModelsFolder", text)
        self.assertIn("row.value !== legacyDefaultModelsFolder", (ROOT / "ui/cron/paths.ts").read_text(encoding="utf-8"))

    def test_all_assistant_adapters_resolve_through_creator_model_store(self):
        for filename in (
            "minimax_h3/minimax_h3.py", "qwen_image_2/qwen_image_2.py", "ming_image/ming_image.py",
            "krea2/krea2.py", "ideogram4/ideogram4.py",
        ):
            text = (ROOT / "extensions_built_in/diffusion_models" / filename).read_text(encoding="utf-8")
            self.assertIn("resolve_lora_file(", text, filename)
            self.assertNotIn('TOOLKIT_ROOT, "models", "loras", "training_adapters"', text, filename)


class ChengyuSnapshotTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_chengyu_storage_test", ROOT / "toolkit/chengyu_model_storage.py")
        self.storage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.storage)
        temp = tempfile.TemporaryDirectory(dir=".", prefix="chengyu-storage-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.storage.MODEL_STORE = self.root
        self.download = Mock(side_effect=self.download_snapshot)
        patcher = patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=self.download)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def download_snapshot(self, **kwargs):
        folder = Path(kwargs["local_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "pytorch_model.bin").write_bytes(b"weights")
        return str(folder)

    def marker(self, folder, repo, files):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".ai-toolkit-complete.json").write_text(json.dumps({
            "repo_id": repo, "files": [{"filename": name, "size": (folder / name).stat().st_size} for name in files],
        }), encoding="utf-8")

    def test_completed_snapshot_is_loaded_offline_without_download(self):
        folder = self.root / "huggingface" / REPO
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "pytorch_model.bin").write_bytes(b"weights")
        self.marker(folder, REPO, ["config.json", "pytorch_model.bin"])
        self.assertEqual(self.storage.resolve_hf_source(REPO), str(folder))
        self.download.assert_not_called()

    def test_missing_snapshot_is_downloaded_to_creator_volume_and_reused(self):
        expected = self.root / "huggingface" / REPO
        self.assertEqual(self.storage.resolve_hf_source(REPO), str(expected))
        self.assertEqual(Path(self.download.call_args.kwargs["local_dir"]), expected)
        self.assertEqual(self.storage.resolve_hf_source(REPO), str(expected))
        self.download.assert_called_once()

    def test_partial_local_snapshot_is_completed_before_reading(self):
        folder = self.root / "huggingface" / REPO
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.storage.resolve_hf_source(str(folder)), str(folder))
        self.download.assert_called_once()

    def test_unregistered_custom_source_is_not_changed(self):
        self.assertEqual(self.storage.resolve_hf_source("custom/model"), "custom/model")
        self.assertEqual(self.storage.resolve_hf_source("/custom/local/model"), "/custom/local/model")
        self.download.assert_not_called()

    def test_small_assets_seed_project_cache_without_copying_weights_or_overwriting(self):
        source = self.root / "configs" / "Qwen3.8-27B"
        source.mkdir(parents=True)
        (source / "config.json").write_text('{"from": "creator"}', encoding="utf-8")
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        (source / "weights.safetensors").write_bytes(b"do not copy")
        target = self.root / "project" / "models" / "Qwen3.8-27B"
        target.mkdir(parents=True)
        (target / "config.json").write_text('{"keep": true}', encoding="utf-8")
        self.storage.seed_project_assets("Qwen/Qwen3.8-27B", str(target))
        self.assertEqual((target / "config.json").read_text(), '{"keep": true}')
        self.assertTrue((target / "tokenizer.json").is_file())
        self.assertFalse((target / "weights.safetensors").exists())
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
