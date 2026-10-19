"""Exercise MOSS phase-major sampling without loading large model weights."""

from pathlib import Path
import importlib
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from testing.test_moss_image_asset_cache import load_model_module
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.staged_sampling import PreparedSample, StagedSampleBatch, StagedSampling
from toolkit.unloader import FakeTextEncoder


ROOT = Path(__file__).resolve().parents[1]


class Component(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.multiplier = 0.75
        self.is_active = True

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        image_tokens = [9] * text.count("<|image_pad|>")
        tokens = [3, 1, 4] if "positive" in text else [1, 0, 2]
        return {"input_ids": torch.tensor([image_tokens + tokens])}


class ImageProcessor:
    merge_size = 2

    def __call__(self, images, **kwargs):
        return {"pixel_values": torch.ones(len(images), 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images))}


class MossStagedSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moss = load_model_module()
        cls.transformer_module = importlib.import_module(cls.moss.MossImageTransformer.__module__)

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        temp = tempfile.TemporaryDirectory(dir=".", prefix="moss-staged-test-")
        self.addCleanup(temp.cleanup)
        self.paths = []
        for index, size in enumerate(((64, 32), (32, 64))):
            path = Path(temp.name) / f"reference-{index}.png"
            Image.new("RGB", size, (64 + index * 80, 96, 128)).save(path)
            self.paths.append(str(path))

    def model(self, cached=False, edit=False, kv_cache=True, device=None):
        test = self
        device = device or self.device
        model = self.moss.MossImageModel.__new__(self.moss.MossImageModel)
        model.device_torch = model.te_device_torch = model.vae_device_torch = device
        model.torch_dtype = model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(
            compile=False, layer_offloading=False, model_kwargs={"match_target_res": True}
        )
        model.is_edit, model.kv_cache = edit, kv_cache
        model.sample_shift, model.uncond_skip_layers = 5.0, (1,)
        model.max_text_length = 128
        model.events, model.saved, model.caches = [], [], []
        model.network = Component().to(device)
        model.assistant_lora = model.accuracy_recovery_adapter = None
        model.sample_step_hook = None
        model._after_sample_image = lambda *_: None
        model.tokenizer, model.image_processor = Tokenizer(), ImageProcessor()

        def phase(expected):
            session = getattr(model, "_staged_sampling", None)
            if session is not None:
                test.assertEqual(session.phase, expected)
                if device.type == "cuda":
                    for name, component in (("text_encoder", model.text_encoder),
                                            ("denoise", model.model), ("vae", model.vae)):
                        if name != expected and not isinstance(component, FakeTextEncoder):
                            test.assertEqual(component.device.type, "cpu")

        class TextEncoder(Component):
            config = SimpleNamespace(image_token_id=9)

            def forward(self, input_ids, **kwargs):
                phase("text_encoder")
                model.events.append("text")
                return SimpleNamespace(last_hidden_state=input_ids.float()[:, :, None].repeat(1, 1, 4))

        class Transformer(Component):
            params = SimpleNamespace(latent_channels=4)

            def forward(self, hidden, timestep, context, mask, ref_kv_cache=None, **kwargs):
                phase("denoise")
                model.events.append("denoise")
                if ref_kv_cache is not None and not any(c is ref_kv_cache for c in model.caches):
                    model.caches.append(ref_kv_cache)
                reference = sum(t.mean() for refs in (kwargs.get("ref_latents") or []) for t in refs)
                value = context[mask].mean() * 0.01 + reference * 0.01
                return hidden * 0.1 * self.weight * model.network.multiplier + value

        class VAE(Component):
            def encode(self, images):
                phase("vae")
                model.events.append("encode")
                return F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (4, 4)).repeat(1, 4, 1, 1)

            def decode(self, latents):
                phase("vae")
                model.events.append("decode")
                return F.interpolate(latents[:, :3] * self.weight, size=(64, 64), mode="bilinear")

        model.model = Transformer().to(device)
        model.vae = VAE()
        model.text_encoder = FakeTextEncoder(device, torch.float32) if cached else TextEncoder()
        if not cached:
            model.text_encoder.requires_grad_(False)
        return model

    def config(self, index=0, references=0, cfg=3.0):
        return SimpleNamespace(
            seed=42 + index, prompt="positive", negative_prompt="negative", width=64, height=64,
            num_inference_steps=3, guidance_scale=cfg, latents=None, network_multiplier=0.5 + index,
            ctrl_img=self.paths[0] if references else None,
            ctrl_img_1=self.paths[0] if references else None,
            ctrl_img_2=self.paths[1] if references > 1 else None, ctrl_img_3=None,
            save_image_atomic=lambda *_: None, log_image=lambda *_: None,
        )

    def embeds(self, model, config, cached):
        if cached:
            return (AdvancedPromptEmbeds(text_embeds=[torch.ones(3, 4)]),
                    AdvancedPromptEmbeds(text_embeds=[torch.zeros(2, 4)]))
        controls = [torch.ones(1, 3, 32, 64, device=model.device_torch)] if model.is_edit else None
        return (model.get_prompt_embeds(config.prompt, control_images=controls),
                model.get_prompt_embeds(config.negative_prompt, control_images=controls))

    def test_all_four_moss_options_expose_existing_staging_control(self):
        self.assertTrue(self.moss.MossImageModel.supports_staged_sampling)
        source = (ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text(encoding="utf-8")
        for name in ("moss_image", "moss_image:base", "moss_image:edit", "moss_image:base_edit"):
            entry = next(block for block in re.split(r'\n    name: ', source) if block.startswith(f'"{name}"'))
            self.assertIn('"model.sample_staged_offloading"', entry, name)
        docs = (ROOT / "ui/src/docs.tsx").read_text(encoding="utf-8")
        self.assertIn("'model.sample_staged_offloading':", docs)

    def test_preparation_queues_cpu_data_without_denoising_or_decoding(self):
        model = self.model(cached=True, edit=True)
        pipeline = self.moss.MossImagePipeline(model)
        session = StagedSampling(model)
        model._staged_sampling = session
        try:
            with session:
                session.preparing = True
                task = model.generate_single_image(
                    pipeline, self.config(references=2), *self.embeds(model, self.config(), True),
                    torch.Generator().manual_seed(42), {},
                )
                self.assertIsInstance(task, PreparedSample)
                self.assertEqual(model.events, ["encode", "encode"])
                self.assertEqual(task.kwargs["latents"].device.type, "cpu")
                for refs in task.kwargs["ref_latents"]:
                    self.assertTrue(all(t.device.type == "cpu" for t in refs))
        finally:
            del model._staged_sampling

    def test_supplied_latents_do_not_consume_generator(self):
        model = self.model(cached=True)
        pipeline = self.moss.MossImagePipeline(model)
        self.assertTrue(hasattr(pipeline, "prepare_staged_sample"))
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        conditional, unconditional = self.embeds(model, self.config(), True)
        latents = torch.ones(1, 4, 4, 4)
        task = pipeline.prepare_staged_sample(
            conditional_embeds=conditional, unconditional_embeds=unconditional,
            height=64, width=64, latents=latents, generator=generator,
        )
        self.assertTrue(torch.equal(generator.get_state(), before))
        torch.testing.assert_close(task.kwargs["latents"], latents)
        self.assertFalse(model.events)

    def test_staging_does_not_bypass_residency_manager_with_raw_module_moves(self):
        model = self.model(edit=True, device=torch.device("cpu"))
        session = SimpleNamespace(phase="idle", preparing=True)
        session.stage = lambda phase: setattr(session, "phase", phase)
        model._staged_sampling = session
        try:
            with patch.object(model.model, "to", side_effect=AssertionError("raw transformer move")):
                with patch.object(model.text_encoder, "to", side_effect=AssertionError("raw encoder move")):
                    with patch.object(model.vae, "to", side_effect=AssertionError("raw VAE move")):
                        conditional, unconditional = self.embeds(model, self.config(), False)
                        task = model.generate_single_image(
                            self.moss.MossImagePipeline(model), self.config(references=2),
                            conditional, unconditional, torch.Generator().manual_seed(42), {},
                        )
                        self.assertIsInstance(task, PreparedSample)
                        model.decode_latents(torch.ones(1, 4, 4, 4))
        finally:
            del model._staged_sampling

    def test_phase_major_matches_serial_pixels_and_restores_training_state(self):
        for cached in (False, True):
            for references in (0, 2):
                for cfg in (1.0, 3.0):
                    with self.subTest(cached=cached, references=references, cfg=cfg):
                        baseline = self.model(cached=cached, edit=bool(references))
                        expected = []
                        for index in range(2):
                            config = self.config(index, references, cfg)
                            baseline.network.multiplier = config.network_multiplier
                            image = baseline.generate_single_image(
                                self.moss.MossImagePipeline(baseline), config,
                                *self.embeds(baseline, config, cached),
                                torch.Generator().manual_seed(config.seed), {},
                            )
                            expected.append(np.asarray(image))
                        model = self.model(cached=cached, edit=bool(references))
                        session = StagedSampling(model)
                        states = [(c, c.device, c.training, [(p, p.requires_grad) for p in c.parameters()])
                                  for c in (model.model, model.vae, model.network, model.text_encoder)]
                        rng = torch.get_rng_state().clone()
                        model._staged_sampling = session
                        try:
                            with session:
                                batch = StagedSampleBatch(session, self.moss.MossImagePipeline(model), model.network)
                                for index in range(2):
                                    config = self.config(index, references, cfg)
                                    config.save_image_atomic = lambda image, i: model.saved.append((i, np.array(image)))
                                    torch.manual_seed(config.seed)
                                    conditional, unconditional = self.embeds(model, config, cached)
                                    original = conditional.text_embeds[0].clone()
                                    batch.add(config, conditional, unconditional, {})
                                    torch.testing.assert_close(conditional.text_embeds[0], original)
                                batch.run()
                        finally:
                            del model._staged_sampling
                        self.assertEqual(len(model.saved), 2)
                        for (index, actual), serial in zip(model.saved, expected):
                            np.testing.assert_array_equal(actual, serial)
                        steps = 3 * (1 if cfg == 1.0 else 2) * 2
                        events = [e for e in model.events if e != "text"]
                        self.assertEqual(events, ["encode"] * (references * 2) + ["denoise"] * steps + ["decode"] * 2)
                        for component, device, training, parameters in states:
                            self.assertEqual(component.device, device)
                            self.assertEqual(component.training, training)
                            for parameter, requires_grad in parameters:
                                self.assertTrue(any(p is parameter for p in component.parameters()))
                                self.assertEqual(parameter.requires_grad, requires_grad)
                        self.assertEqual(model.network.multiplier, 0.75)
                        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                        self.assertEqual(len(model.caches), 4 if references and cfg != 1.0 else 2 if references else 0)

    def test_failures_in_each_phase_restore_state_and_clear_queue(self):
        for failure in ("encode", "denoise", "decode"):
            with self.subTest(failure=failure):
                model = self.model(cached=True, edit=True)
                component = model.model if failure == "denoise" else model.vae
                method = "forward" if failure == "denoise" else failure
                states = [(c, c.device, c.training, c.weight) for c in (model.model, model.vae, model.network)]
                session = StagedSampling(model)
                batch = StagedSampleBatch(session, self.moss.MossImagePipeline(model), model.network)
                model._staged_sampling = session
                rng = torch.get_rng_state().clone()
                try:
                    with self.assertRaisesRegex(RuntimeError, f"failed {failure}"):
                        with session:
                            batch.add(self.config(references=2), *self.embeds(model, self.config(), True), {})
                            with patch.object(component, method, side_effect=RuntimeError(f"failed {failure}")):
                                batch.run()
                finally:
                    del model._staged_sampling
                self.assertFalse(batch.jobs)
                self.assertFalse(session.preparing)
                self.assertFalse(model.saved)
                for c, device, training, parameter in states:
                    self.assertEqual(c.device, device)
                    self.assertEqual(c.training, training)
                    self.assertIs(c.weight, parameter)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_real_tiny_transformer_matches_regular_sampling_with_edit_kv_cache(self):
        model = self.model(cached=True, edit=True)
        torch.manual_seed(123)
        params = self.transformer_module.MossImageParams(
            dim=32, ffn_dim=64, num_heads=4, num_layers=2,
            text_dim=16, freq_dim=16, latent_channels=4,
        )
        model.model = self.moss.MossImageTransformer(params).to(self.device)
        conditional = AdvancedPromptEmbeds(text_embeds=[torch.ones(3, 16)])
        unconditional = AdvancedPromptEmbeds(text_embeds=[torch.zeros(2, 16)])
        config = self.config(references=2)
        expected = model.generate_single_image(
            self.moss.MossImagePipeline(model), config, conditional, unconditional,
            torch.Generator().manual_seed(config.seed), {},
        )
        session = StagedSampling(model)
        model._staged_sampling = session
        try:
            with session:
                session.preparing = True
                task = model.generate_single_image(
                    self.moss.MossImagePipeline(model), config, conditional, unconditional,
                    torch.Generator().manual_seed(config.seed), {},
                )
                self.assertIsInstance(task, PreparedSample)
                session.preparing = False
                session.stage("denoise")
                task.denoise(self.device)
                self.assertEqual(session.phase, "denoise")
                actual = task.decode(model.vae_device_torch)
        finally:
            del model._staged_sampling
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()
