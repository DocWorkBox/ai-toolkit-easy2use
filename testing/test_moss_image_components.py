"""Exercise MOSS custom components without checkpoints or a Qwen text encoder."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.models.v2 import resolver


ROOT = Path(__file__).resolve().parents[1]


def load_components():
    package = ModuleType("_moss_component_tests")
    package.__path__ = [str(ROOT / "extensions_built_in/diffusion_models/moss_image/src")]
    sys.modules[package.__name__] = package
    modules = []
    for name in ("transformer", "pipeline", "checkpoints"):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", Path(package.__path__[0]) / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


class MossImageComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transformer, cls.pipeline, cls.checkpoints = load_components()

    def tiny_transformer(self):
        torch.manual_seed(42)
        params = self.transformer.MossImageParams(
            dim=32, ffn_dim=64, num_heads=4, num_layers=2,
            text_dim=16, freq_dim=16, latent_channels=4,
        )
        return self.transformer.MossImageTransformer(params)

    def test_transformer_supports_variable_text_lengths_and_backward(self):
        model = self.tiny_transformer()
        hidden = torch.randn(2, 4, 2, 2)
        context = torch.randn(2, 3, 16)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        output = model(hidden, torch.tensor([900, 500]), context, mask)
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.video_embedder.weight.grad).all())

    @torch.no_grad()
    def test_reference_kv_cache_matches_uncached_at_a_later_timestep(self):
        model = self.tiny_transformer()
        hidden = torch.randn(2, 4, 2, 2)
        context = torch.randn(2, 3, 16)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        refs = [[torch.randn(4, 1, 2), torch.randn(4, 1, 1)], [torch.randn(4, 1, 1)]]
        cache = {}
        kwargs = dict(ref_latents=refs, isolate_refs=True, skip_layers=(1,))
        model(hidden, torch.tensor([900, 900]), context, mask, ref_kv_cache=cache, **kwargs)
        self.assertIn("kv", cache)
        expected = model(hidden, torch.tensor([500, 500]), context, mask, **kwargs)
        actual = model(hidden, torch.tensor([500, 500]), context, mask, ref_kv_cache=cache, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @torch.no_grad()
    def test_sampler_produces_deterministic_image_with_reference_cache(self):
        transformer = self.tiny_transformer()
        model = SimpleNamespace(
            transformer=transformer, device_torch=torch.device("cpu"), torch_dtype=torch.float32,
            vae_device_torch=torch.device("cpu"), sample_stage=lambda _: None,
            decode_latents=lambda x, **_: F.interpolate(x[:, :3], scale_factor=16),
        )
        pipeline = self.pipeline.MossImagePipeline(model)
        kwargs = dict(
            conditional_embeds=SimpleNamespace(text_embeds=[torch.randn(3, 16)]),
            unconditional_embeds=SimpleNamespace(text_embeds=[torch.randn(2, 16)]),
            height=32, width=32, num_inference_steps=2, uncond_skip_layers=(1,),
            ref_latents=[[torch.randn(4, 1, 1)]], isolate_refs=True,
        )
        first = pipeline(generator=torch.Generator().manual_seed(123), **kwargs)[0]
        second = pipeline(generator=torch.Generator().manual_seed(123), **kwargs)[0]
        self.assertEqual(first.size, (32, 32))
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertGreater(len(np.unique(np.asarray(first))), 1)

    def test_shard_merge_fuses_qkv_and_casts_to_bf16(self):
        with tempfile.TemporaryDirectory(dir=".", prefix="moss-merge-test-") as folder:
            root = Path(folder)
            first, second, output = (root / name for name in ("first.safetensors", "second.safetensors", "out.safetensors"))
            q, k, v = (torch.full((2, 2), float(n)) for n in (1, 2, 3))
            save_file({"layers.0.proj.q.weight": q}, str(first))
            save_file({"layers.0.proj.k.weight": k, "layers.0.proj.v.weight": v}, str(second))
            self.checkpoints.merge_safetensors([str(first), str(second)], str(output), fuse_qkv=True)
            merged = load_file(str(output))
            self.assertEqual(set(merged), {"layers.0.proj.qkv.weight"})
            torch.testing.assert_close(merged["layers.0.proj.qkv.weight"], torch.cat((q, k, v)).bfloat16())
            self.assertFalse(Path(str(output) + ".part").exists())

    def test_local_comfy_weights_are_reused_without_downloads(self):
        with tempfile.TemporaryDirectory(dir=".", prefix="moss-local-test-") as folder:
            root = Path(folder).resolve()
            relative = "diffusion_models/moss_image_base_int8_convrot_attn.safetensors"
            target = root / relative
            target.parent.mkdir(parents=True)
            target.touch()
            with patch.object(resolver, "MODELS_PATH", str(root)):
                with patch("huggingface_hub.hf_hub_download") as download:
                    result = self.checkpoints.comfy_transformer_file(
                        self.checkpoints.BASE_REPO, qtype="convrot8"
                    )
            self.assertEqual(Path(result), target)
            download.assert_not_called()


class LocalFirstValidationTests(unittest.TestCase):
    def test_incomplete_cache_retries_online(self):
        partial, complete = SimpleNamespace(), SimpleNamespace()
        calls = []

        def load(*args, **kwargs):
            calls.append(kwargs)
            return partial if kwargs.get("local_files_only") else complete

        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}):
            result = OstrisModelMixin._local_first(load, "example/repo", validate=lambda x: x is complete)
        self.assertIs(result, complete)
        self.assertEqual(calls, [{"local_files_only": True}, {}])

    def test_complete_cache_never_retries_online(self):
        calls = []

        def load(*args, **kwargs):
            calls.append(kwargs)
            return "complete"

        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}):
            result = OstrisModelMixin._local_first(load, "example/repo", validate=lambda _: True)
        self.assertEqual(result, "complete")
        self.assertEqual(calls, [{"local_files_only": True}])

    def test_offline_mode_does_not_retry_an_incomplete_cache(self):
        calls = []

        def load(*args, **kwargs):
            calls.append(kwargs)
            return "partial"

        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}):
            result = OstrisModelMixin._local_first(load, "example/repo", validate=lambda _: False)
        self.assertEqual(result, "partial")
        self.assertEqual(calls, [{}])


if __name__ == "__main__":
    unittest.main()
