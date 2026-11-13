import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


class ChengyuDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("chengyu_downloads", ROOT / "scripts/download_chengyu_models.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="chengyu-download-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.destination = self.root / "models"
        self.destination.mkdir()

    def entry(self, filename="text_encoders/qwen38.safetensors"):
        return {"repo_id": "org/repo", "revision": "commit", "filename": filename,
                "local_dir": "", "size": 4, "sha256": None, "kind": "files"}

    def test_h3_image_manifest_keeps_shared_files_and_creator_adapter_destination(self):
        manifest = json.loads((ROOT / "scripts/chengyu_models.json").read_text(encoding="utf-8"))
        image = next((g for g in manifest["groups"] if g["repo_id"] == "Comfy-Org/MiniMax-H3-Image"), None)
        self.assertIsNotNone(image)
        self.assertEqual(image["local_dir"], "")
        self.assertEqual(set(image["files"]), {
            "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "vae/minimax_h3_image_vae_fp16.safetensors",
        })
        adapter = next(g for g in manifest["groups"] if g["repo_id"] == "ostris/minimax_h3_training_adapter")
        self.assertEqual(adapter["local_dir"], "loras/training_adapters")
        self.assertIn("minimax_h3_image_training_adapter_v1.safetensors", adapter["files"])

    def test_complete_existing_file_is_not_downloaded(self):
        item = self.entry()
        target = self.destination / item["filename"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"data")
        download = Mock()
        result = self.module.fetch_file(item, self.destination, {}, download)
        self.assertEqual(result["status"], "existing")
        download.assert_not_called()

    def test_persistent_shared_weights_are_reused_without_copy_or_download(self):
        item = self.entry()
        shared = self.root / "shared" / "qwen38.safetensors"
        shared.parent.mkdir()
        shared.write_bytes(b"data")
        download = Mock()
        result = self.module.fetch_file(item, self.destination, {shared.name: [shared]}, download)
        self.assertEqual(result["status"], "reused")
        target = self.destination / item["filename"]
        self.assertTrue(target.samefile(shared))
        download.assert_not_called()

    def test_unrelated_hf_shard_with_same_name_and_size_is_not_reused(self):
        item = self.entry("model-00001-of-00002.safetensors")
        item["kind"] = "snapshot"
        item["local_dir"] = "huggingface/org/repo"
        wrong = self.root / "different-repo" / item["filename"]
        wrong.parent.mkdir()
        wrong.write_bytes(b"data")
        def fetch(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"new!")
            return str(target)
        download = Mock(side_effect=fetch)
        result = self.module.fetch_file(item, self.destination, {wrong.name: [wrong]}, download)
        self.assertEqual(result["status"], "downloaded")
        download.assert_called_once()

    def test_existing_different_size_is_not_overwritten(self):
        item = self.entry()
        target = self.destination / item["filename"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep")
        item["size"] = 20
        download = Mock()
        with self.assertRaises(FileExistsError):
            self.module.fetch_file(item, self.destination, {}, download)
        self.assertEqual(target.read_bytes(), b"keep")
        download.assert_not_called()

    def test_new_download_uses_pinned_revision_and_expected_directory(self):
        item = self.entry()
        def fetch(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
            return str(target)
        download = Mock(side_effect=fetch)
        self.module.fetch_file(item, self.destination, {}, download)
        self.assertEqual(download.call_args.kwargs["revision"], "commit")
        self.assertEqual(Path(download.call_args.kwargs["local_dir"]), self.destination)
        self.assertIs(download.call_args.kwargs["token"], False)

    def test_download_paths_cannot_escape_destination(self):
        item = self.entry("../../outside.safetensors")
        with self.assertRaises(ValueError):
            self.module.fetch_file(item, self.destination, {}, Mock())

    def test_live_status_counts_completed_and_partial_bytes(self):
        (self.destination / "download-state.json").write_text(json.dumps({
            "status": "running", "total_bytes": 20,
            "files": {"done": {"status": "existing", "size": 4},
                      "active": {"status": "downloading", "size": 16}},
        }), encoding="utf-8")
        (self.destination / "weights.incomplete").write_bytes(b"partial!")
        progress = self.module.get_progress(self.destination)
        self.assertEqual(progress["completed_files"], 1)
        self.assertEqual(progress["received_bytes"], 12)
        self.assertEqual(progress["failed_files"], 0)


if __name__ == "__main__":
    unittest.main()
