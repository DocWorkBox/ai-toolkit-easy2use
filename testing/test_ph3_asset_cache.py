"""P-H3 small configuration assets are persistent and independent of model mounts."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
REPO = "PrunaAI/P-H3-Image"
FILES = (
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja",
    "processor/processor_config.json", "processor/tokenizer.json", "processor/tokenizer_config.json",
    "processor/chat_template.jinja", "text_encoder/config.json",
)


class PH3AssetCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_ph3_asset_cache_tests", ROOT / "extensions_built_in/diffusion_models/minimax_h3/src/ph3_assets.py"
        )
        cls.assets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.assets)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="ph3-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "toolkit"
        self.cache = self.project / "models/P-H3-Image"
        self.hub = self.root / "hub"
        self.patch(patch.object(self.assets, "TOOLKIT_ROOT", str(self.project)))
        self.download = self.patch(patch("huggingface_hub.snapshot_download", side_effect=self.download_assets))

    def patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def write(self, directory, filenames=FILES):
        for name in filenames:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    def download_assets(self, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not in Hub cache")
        root = Path(kwargs["local_dir"])
        self.write(root, [name for name in FILES if name in kwargs["allow_patterns"]])
        return str(root)

    def test_downloads_only_small_assets_into_project_and_reuses_cache_offline(self):
        self.assertEqual(self.assets.resolve_ph3_assets(REPO), str(self.cache))
        online = [call.kwargs for call in self.download.call_args_list if not call.kwargs.get("local_files_only")]
        self.assertEqual(len(online), 1)
        self.assertEqual(online[0]["local_dir"], str(self.cache))
        self.assertEqual(online[0]["repo_id"], REPO)
        self.assertTrue(set(FILES) <= set(online[0]["allow_patterns"]))
        self.assertTrue(all("*" not in name and not name.endswith((".safetensors", ".bin"))
                            for name in online[0]["allow_patterns"]))
        self.download.reset_mock()
        self.download.side_effect = AssertionError("cached assets must not contact the Hub")
        self.assertEqual(self.assets.resolve_ph3_assets(REPO), str(self.cache))
        self.download.assert_not_called()

    def test_reuses_hub_cache_without_copying_weights_or_changing_source(self):
        self.write(self.hub)
        weight = self.hub / "text_encoder/model.safetensors"
        weight.write_bytes(b"do not copy weights")
        self.download.side_effect = lambda **_: str(self.hub)
        self.assertEqual(self.assets.resolve_ph3_assets(REPO), str(self.cache))
        self.download.assert_called_once()
        self.assertTrue(self.download.call_args.kwargs["local_files_only"])
        self.assertFalse((self.cache / "text_encoder/model.safetensors").exists())
        self.assertEqual(weight.read_bytes(), b"do not copy weights")

    def test_preserves_existing_cache_and_repairs_missing_or_empty_files(self):
        self.write(self.cache, FILES[:2])
        keep = self.cache / FILES[0]
        keep.write_text('{"keep": true}', encoding="utf-8")
        empty = self.cache / FILES[1]
        empty.write_bytes(b"")
        self.assets.resolve_ph3_assets(REPO)
        online = [call.kwargs for call in self.download.call_args_list if not call.kwargs.get("local_files_only")]
        self.assertNotIn(FILES[0], online[0]["allow_patterns"])
        self.assertIn(FILES[1], online[0]["allow_patterns"])
        self.assertEqual(keep.read_text(encoding="utf-8"), '{"keep": true}')
        self.assertGreater(empty.stat().st_size, 0)

    def test_closed_http_client_is_recreated_once(self):
        attempts = 0

        def download(**kwargs):
            nonlocal attempts
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("no cache")
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return self.download_assets(**kwargs)

        self.download.side_effect = download
        with patch("huggingface_hub.utils.close_session", create=True) as reset:
            self.assertEqual(self.assets.resolve_ph3_assets(REPO), str(self.cache))
        reset.assert_called_once_with()
        self.assertEqual(attempts, 2)

    def test_incomplete_download_is_not_reported_ready(self):
        self.download.side_effect = lambda **_: str(self.hub)
        with self.assertRaisesRegex(FileNotFoundError, "P-H3-Image"):
            self.assets.resolve_ph3_assets(REPO)

    def test_network_failure_reports_target_before_download_and_propagates(self):
        messages = []
        self.download.side_effect = OSError("Network is unreachable")
        with self.assertRaisesRegex(OSError, "Network is unreachable"):
            self.assets.resolve_ph3_assets(REPO, status_fn=messages.append)
        self.assertTrue(any(str(self.cache) in message for message in messages))

    def test_explicit_local_directory_does_not_write_or_download(self):
        local = self.root / "read-only-assets"
        self.write(local)
        self.assertEqual(self.assets.resolve_ph3_assets(str(local)), str(local))
        self.download.assert_not_called()
        self.assertFalse(self.project.exists())

    def test_invalid_repo_cannot_escape_project_cache(self):
        with self.assertRaises(ValueError):
            self.assets.resolve_ph3_assets("../outside")
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
