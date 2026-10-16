"""Keep the vendor's FLUX.3 config in the writable toolkit models folder."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from testing.test_flux3_staged_sampling import load_flux3


class Flux3ConfigCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = load_flux3()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".", prefix="flux3-config-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        root_patch = patch.object(self.flux, "TOOLKIT_ROOT", str(self.root), create=True)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.model = self.flux.Flux3Model.__new__(self.flux.Flux3Model)
        self.model.model_config = SimpleNamespace(name_or_path=self.flux.BASE_REPO, model_kwargs={})
        self.model.print_and_status_update = lambda *_: None
        self.model.package_config = None
        self.model._package_config_checked = False
        self.config_path = self.root / "models/flux-3-image-klein-base/config.json"

    def write_config(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {"dit": {"hidden": 8}, "cache_test": True}
        path.write_text(json.dumps(config), encoding="utf-8")
        return config

    def test_download_targets_toolkit_models_not_comfy_directory(self):
        with patch("huggingface_hub.hf_hub_download", return_value=str(self.config_path)) as download:
            result = self.model._package_file("config.json")
        self.assertEqual(result, str(self.config_path))
        self.assertEqual(download.call_args.kwargs["repo_id"], self.flux.BASE_REPO)
        self.assertEqual(download.call_args.kwargs.get("local_dir"), str(self.config_path.parent))

    def test_existing_config_is_loaded_without_network(self):
        expected = self.write_config(self.config_path)
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")) as download:
            self.assertEqual(self.model._load_package_config(), expected)
        download.assert_not_called()

    def test_explicit_config_path_remains_first_priority(self):
        override = self.root / "custom/config.json"
        expected = self.write_config(override)
        self.model.model_config.model_kwargs["config_path"] = str(override)
        with patch("huggingface_hub.hf_hub_download") as download:
            self.assertEqual(self.model._load_package_config(), expected)
        download.assert_not_called()
        self.assertFalse(self.config_path.exists())

    def test_weights_still_use_comfy_resolution(self):
        with patch.object(self.model, "_comfy_weight", return_value="existing-comfy-weight") as comfy:
            with patch("huggingface_hub.hf_hub_download") as download:
                for filename in ("model.safetensors", "ae.safetensors"):
                    self.assertEqual(self.model._package_file(filename), "existing-comfy-weight")
                    comfy.assert_called_with(filename, self.flux.BASE_REPO)
        download.assert_not_called()

    def test_custom_repo_keeps_its_existing_download_policy(self):
        self.model.model_config.name_or_path = "example/custom-flux3"
        with patch("huggingface_hub.hf_hub_download", return_value="custom-config.json") as download:
            self.assertEqual(self.model._package_file("config.json"), "custom-config.json")
        self.assertEqual(download.call_args.kwargs["repo_id"], "example/custom-flux3")
        self.assertNotIn("local_dir", download.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
