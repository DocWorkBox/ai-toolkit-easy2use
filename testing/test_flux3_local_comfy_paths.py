"""Read FLUX.3 Comfy weights in place from a local model folder."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from testing.test_flux3_staged_sampling import load_flux3


class Flux3LocalComfyPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = load_flux3()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".", prefix="flux3-local-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.flux.Flux3Model.__new__(self.flux.Flux3Model)
        self.model.model_config = SimpleNamespace(
            name_or_path=str(self.root), model_kwargs={}, te_name_or_path=None,
            quantize=True, qtype="convrot8",
        )
        self.model.torch_dtype = torch.bfloat16
        self.model.print_and_status_update = Mock()
        self.model._package_config_checked = False
        self.model.package_config = None
        self.model.component_load_kwargs = lambda _: {"dtype": torch.bfloat16, "qtype": "convrot8"}

    def weight(self, rel):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test checkpoint placeholder")
        return str(path)

    def test_local_comfy_transformer_is_selected_by_requested_precision(self):
        files = {
            key: self.weight(f"diffusion_models/flux3_image_klein_base_{key}.safetensors")
            for key in ("bf16", "int8_convrot", "nvfp4")
        }
        self.weight("diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors")
        for quantize, qtype, expected in (
            (True, "convrot8", "int8_convrot"),
            (True, "nvfp4", "nvfp4"),
            (False, "convrot8", "bf16"),
        ):
            with self.subTest(qtype=qtype, quantize=quantize):
                self.model.model_config.quantize = quantize
                self.model.model_config.qtype = qtype
                with patch.object(self.flux.Flux3Transformer, "load_model", return_value=Mock()) as load:
                    with patch("huggingface_hub.hf_hub_download") as download:
                        self.model._load_transformer()
                self.assertEqual(Path(load.call_args.args[0]), Path(files[expected]))
                download.assert_not_called()

    def test_legacy_comfy_transformer_filename_is_still_supported(self):
        expected = self.weight("diffusion_models/flux3_image_klein_int8_convrot.safetensors")
        with patch.object(self.flux.Flux3Transformer, "load_model", return_value=Mock()) as load:
            self.model._load_transformer()
        self.assertEqual(Path(load.call_args.args[0]), Path(expected))

    def test_hyphenated_base_names_keep_requested_precision(self):
        files = {
            precision: self.weight(f"diffusion_models/flux3-image-klein-base-{precision}.safetensors")
            for precision in ("bf16", "int8-convrot", "nvfp4")
        }
        for precision in files:
            self.weight(f"diffusion_models/flux3-image-klein-{precision}.safetensors")
        for quantize, qtype, expected in (
            (True, "convrot8", "int8-convrot"),
            (True, "nvfp4", "nvfp4"),
            (False, "convrot8", "bf16"),
        ):
            with self.subTest(qtype=qtype, quantize=quantize):
                self.model.model_config.quantize = quantize
                self.model.model_config.qtype = qtype
                with patch.object(self.flux.Flux3Transformer, "load_model", return_value=Mock()) as load:
                    with patch("huggingface_hub.hf_hub_download") as download:
                        self.model._load_transformer()
                self.assertEqual(Path(load.call_args.args[0]), Path(files[expected]))
                download.assert_not_called()

    def test_registered_comfy_repo_resolves_hyphenated_base_weights(self):
        expected = self.weight("diffusion_models/flux3-image-klein-base-int8-convrot.safetensors")
        with patch("toolkit.models.v2.resolver.MODELS_PATH", str(self.root)):
            with patch("huggingface_hub.hf_hub_download") as download:
                result = self.flux.Flux3Transformer.resolve_comfy_weights(
                    self.flux.COMFY_REPO, qtype="convrot8", local_only=True
                )
        self.assertIsNotNone(result)
        self.assertEqual(Path(result), Path(expected))
        download.assert_not_called()

    def test_non_base_hyphenated_weights_are_not_used_for_training(self):
        self.weight("diffusion_models/flux3-image-klein-int8-convrot.safetensors")
        with patch.object(self.flux.Flux3Transformer, "load_model") as load:
            with patch("huggingface_hub.hf_hub_download") as download:
                with self.assertRaisesRegex(FileNotFoundError, "FLUX.3 Base checkpoint.*diffusion_models"):
                    self.model._load_transformer()
        load.assert_not_called()
        download.assert_not_called()

    def test_vae_is_read_from_local_comfy_layout_without_downloading(self):
        expected = self.weight("vae/flux2-vae.safetensors")
        with patch("huggingface_hub.hf_hub_download") as download:
            self.assertEqual(Path(self.model._package_file("ae.safetensors")), Path(expected))
        download.assert_not_called()

    def test_original_package_and_explicit_overrides_keep_priority(self):
        package = self.weight("model.safetensors")
        self.weight("diffusion_models/flux3_image_klein_base_int8_convrot.safetensors")
        override = self.weight("custom.safetensors")
        for kwargs, expected in (({}, package), ({"model_path": override}, override)):
            with self.subTest(kwargs=kwargs):
                self.model.model_config.model_kwargs = kwargs
                with patch.object(self.flux.Flux3Transformer, "load_model", return_value=Mock()) as load:
                    self.model._load_transformer()
                self.assertEqual(Path(load.call_args.args[0]), Path(expected))

    def test_text_encoder_uses_local_comfy_weight_and_project_assets(self):
        self.weight("text_encoders/qwen3vl_4b_bf16.safetensors")
        expected = self.weight("text_encoders/qwen3vl_4b_int8_convrot.safetensors")
        cache = str(self.root / "project/models/Qwen3-VL-4B-Instruct")
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
        cls = self.flux.Flux3TextEncoder
        with patch.object(cls, "resolve_assets", return_value=cache):
            with patch.object(cls, "load_model", return_value=Mock()) as load:
                with patch.object(cls, "load_processor", return_value=processor) as load_processor:
                    with patch.object(self.flux, "flush"):
                        self.model._load_text_encoder()
        self.assertEqual(Path(load.call_args.args[0]), Path(expected))
        self.assertEqual(load.call_args.kwargs["config_path"], cache)
        load_processor.assert_called_once_with(cache)

    def test_explicit_text_encoder_override_is_not_replaced_by_local_weight(self):
        self.weight("text_encoders/qwen3vl_4b_int8_convrot.safetensors")
        override = self.weight("custom-te.safetensors")
        self.model.model_config.model_kwargs["text_encoder_path"] = override
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))
        cls = self.flux.Flux3TextEncoder
        with patch.object(cls, "resolve_assets", return_value="cached-assets"):
            with patch.object(cls, "load_model", return_value=Mock()) as load:
                with patch.object(cls, "load_processor", return_value=processor):
                    with patch.object(self.flux, "flush"):
                        self.model._load_text_encoder()
        self.assertEqual(Path(load.call_args.args[0]), Path(override))

    def test_missing_local_weight_does_not_download_into_readonly_directory(self):
        with patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaises(FileNotFoundError):
                self.model._load_transformer()
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
