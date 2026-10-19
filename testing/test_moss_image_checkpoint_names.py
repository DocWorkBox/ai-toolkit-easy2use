"""MOSS renamed INT8 checkpoints must remain compatible with legacy files."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from testing.test_moss_image_components import load_components
from toolkit.models.v2 import resolver


class MossImageCheckpointNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.checkpoints = load_components()

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=".", prefix="moss-name-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for target, name in (
            (self.checkpoints, "MODELS_PATH"),
            (self.checkpoints, "MOSS_COMFY_LOCAL_PATH"),
            (resolver, "MODELS_PATH"),
        ):
            mock = patch.object(target, name, str(self.root), create=True)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Unexpected download"))
        self.download = mock.start()
        self.addCleanup(mock.stop)

    def weight(self, variant, suffix):
        target = self.root / "diffusion_models" / f"moss_image_{variant}_{suffix}.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing weight")
        return target

    def variants(self):
        return (
            (self.checkpoints.SFT_REPO, "sft"),
            (self.checkpoints.BASE_REPO, "base"),
            (self.checkpoints.COMFY_REPO, "sft"),
        )

    def test_current_int8_names_are_reused_without_downloads(self):
        for variant in ("sft", "base"):
            self.weight(variant, "int8_convrot")
        for repo, variant in self.variants():
            with self.subTest(repo=repo):
                expected = self.root / "diffusion_models" / f"moss_image_{variant}_int8_convrot.safetensors"
                actual = self.checkpoints.comfy_transformer_file(repo, qtype="convrot8")
                self.assertEqual(Path(actual), expected)
        self.download.assert_not_called()

    def test_legacy_attn_names_are_still_reused_without_downloads(self):
        for variant in ("sft", "base"):
            self.weight(variant, "int8_convrot_attn")
        for repo, variant in self.variants():
            with self.subTest(repo=repo):
                expected = self.root / "diffusion_models" / f"moss_image_{variant}_int8_convrot_attn.safetensors"
                actual = self.checkpoints.comfy_transformer_file(repo, qtype="convrot8")
                self.assertEqual(Path(actual), expected)
        self.download.assert_not_called()

    def test_precision_preference_is_unchanged_with_both_names_present(self):
        for variant in ("sft", "base"):
            for suffix in ("bf16", "int8_convrot", "int8_convrot_attn"):
                self.weight(variant, suffix)
        for repo, variant in self.variants():
            for qtype, expected_precision in (
                (None, "bf16"), ("convrot8", "int8_convrot"),
                ("float8", "bf16"), ("nvfp4", "bf16"),
            ):
                with self.subTest(repo=repo, qtype=qtype):
                    actual = Path(self.checkpoints.comfy_transformer_file(repo, qtype=qtype))
                    self.assertTrue(actual.name.startswith(f"moss_image_{variant}_{expected_precision}"))
        self.download.assert_not_called()

    def test_obsolete_names_are_local_only_downloads_use_current_names(self):
        for variant, (repo_files, local_extra) in self.checkpoints.COMFY_TRANSFORMER_FILES.items():
            with self.subTest(variant=variant):
                current = f"diffusion_models/moss_image_{variant}_int8_convrot.safetensors"
                legacy = f"diffusion_models/moss_image_{variant}_int8_convrot_attn.safetensors"
                self.assertIn(current, repo_files)
                self.assertIn(legacy, local_extra)
                self.assertFalse(any("_attn" in name for name in repo_files))

    def test_comfy_download_requests_current_sft_name(self):
        filename = "diffusion_models/moss_image_sft_int8_convrot.safetensors"
        expected = self.root / filename
        with patch("huggingface_hub.hf_hub_download", side_effect=lambda **kw: str(self.root / kw["filename"])) as download:
            actual = self.checkpoints.comfy_transformer_file(self.checkpoints.COMFY_REPO, qtype="convrot8")
        self.assertEqual(Path(actual), expected)
        download.assert_called_once_with(
            repo_id=self.checkpoints.COMFY_REPO, filename=filename, token=None, local_dir=str(self.root)
        )

    def test_base_bf16_download_fallback_is_unchanged(self):
        filename = "diffusion_models/moss_image_base_bf16.safetensors"
        expected = self.root / filename
        with patch("huggingface_hub.hf_hub_download", return_value=str(expected)) as download:
            actual = self.checkpoints.comfy_transformer_file(self.checkpoints.BASE_REPO, qtype="convrot8")
        self.assertEqual(Path(actual), expected)
        download.assert_called_once_with(
            repo_id=self.checkpoints.COMFY_REPO, filename=filename, token=None, local_dir=str(self.root)
        )


if __name__ == "__main__":
    unittest.main()
