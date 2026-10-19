"""Resolve MOSS components from compshare's read-only Comfy bundle."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from testing.test_moss_image_components import load_components
from toolkit.models.v2 import resolver


class MossImageCompsharePathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.checkpoints = load_components()

    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=".", prefix="moss-compshare-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.bundle = self.root / "bundle"
        self.cache = self.root / "project-models"
        self.bundle.mkdir()
        self.cache.mkdir()
        for target, name, value in (
            (self.checkpoints, "MOSS_COMFY_LOCAL_PATH", str(self.bundle)),
            (self.checkpoints, "MODELS_PATH", str(self.cache)),
            (resolver, "MODELS_PATH", str(self.cache)),
        ):
            mock = patch.object(target, name, value, create=True)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Unexpected download"))
        self.download = mock.start()
        self.addCleanup(mock.stop)

    def weight(self, relative, root=None):
        path = (root or self.bundle) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing weight")
        return path

    def test_branch_default_points_to_server_bundle(self):
        source = Path(self.checkpoints.__file__).read_text(encoding="utf-8")
        self.assertIn('MOSS_COMFY_LOCAL_PATH = "/model/ModelScope/Comfy-Org/MOSS-Image"', source)

    def test_sft_and_base_keep_their_own_precision_candidates(self):
        for variant in ("sft", "base"):
            for suffix in ("bf16", "int8_convrot_attn"):
                self.weight(f"diffusion_models/moss_image_{variant}_{suffix}.safetensors")
        for repo, variant in ((self.checkpoints.SFT_REPO, "sft"), (self.checkpoints.BASE_REPO, "base")):
            for qtype, suffix in ((None, "bf16"), ("convrot8", "int8_convrot_attn")):
                with self.subTest(variant=variant, qtype=qtype):
                    actual = self.checkpoints.comfy_transformer_file(repo, qtype=qtype)
                    self.assertEqual(Path(actual), self.bundle / f"diffusion_models/moss_image_{variant}_{suffix}.safetensors")
        self.download.assert_not_called()

    def test_text_encoder_and_vae_are_read_without_changing_bundle(self):
        bf16 = self.weight("text_encoders/qwen3.5_9b_bf16.safetensors")
        int8 = self.weight("text_encoders/qwen3.5_9b_int8_convrot.safetensors")
        vae = self.weight("vae/flux2-vae.safetensors")
        before = {path: path.read_bytes() for path in self.bundle.rglob("*") if path.is_file()}
        self.assertEqual(Path(self.checkpoints.comfy_text_encoder_file()), bf16)
        self.assertEqual(Path(self.checkpoints.comfy_text_encoder_file(qtype="convrot8")), int8)
        self.assertEqual(Path(self.checkpoints.comfy_vae_file()), vae)
        after = {path: path.read_bytes() for path in self.bundle.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.download.assert_not_called()

    def test_bundle_wins_over_same_file_in_project_models(self):
        rel = "diffusion_models/moss_image_base_bf16.safetensors"
        expected = self.weight(rel)
        self.weight(rel, root=self.cache)
        actual = self.checkpoints.comfy_transformer_file(self.checkpoints.BASE_REPO)
        self.assertEqual(Path(actual), expected)

    def test_nested_and_flat_bundle_files_are_supported(self):
        te = self.weight("text_encoders/custom/qwen3.5_9b_bf16.safetensors")
        vae = self.weight("flux2-vae.safetensors")
        self.assertEqual(Path(self.checkpoints.comfy_text_encoder_file()), te)
        self.assertEqual(Path(self.checkpoints.comfy_vae_file()), vae)

    def test_existing_shared_model_directory_remains_a_fallback(self):
        expected = self.weight("diffusion_models/moss_image_base_bf16.safetensors", root=self.cache)
        actual = self.checkpoints.comfy_transformer_file(self.checkpoints.BASE_REPO)
        self.assertEqual(Path(actual), expected)

    def test_explicit_local_file_is_not_replaced_by_bundle(self):
        explicit = self.weight("custom.safetensors", root=self.root)
        self.weight("diffusion_models/moss_image_base_bf16.safetensors")
        self.assertEqual(Path(self.checkpoints.resolve_hub_file(str(explicit))), explicit)

    def test_custom_finetune_is_not_replaced_by_official_weights(self):
        self.weight("diffusion_models/moss_image_sft_bf16.safetensors")
        expected = self.weight("diffusion_models/custom_moss_finetune_bf16.safetensors", root=self.cache)
        actual = self.checkpoints.comfy_transformer_file("example/custom-moss-finetune")
        self.assertEqual(Path(actual), expected)

    def test_missing_base_never_loads_sft_weights(self):
        self.weight("diffusion_models/moss_image_sft_bf16.safetensors")
        with self.assertRaisesRegex(AssertionError, "Unexpected download"):
            self.checkpoints.comfy_transformer_file(self.checkpoints.BASE_REPO)
        self.download.assert_called_once()
        self.assertEqual(self.download.call_args.kwargs["filename"], "diffusion_models/moss_image_base_bf16.safetensors")


if __name__ == "__main__":
    unittest.main()
