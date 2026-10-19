"""Run the production FLUX.3 sampler with small components and no checkpoints."""

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.staged_sampling import StagedSampleBatch, StagedSampling
from toolkit.unloader import FakeTextEncoder


ROOT = Path(__file__).resolve().parents[1]


def load_flux3():
    # Avoid importing every model plugin, which requires unrelated backends.
    root = ROOT / "extensions_built_in/diffusion_models/flux3"
    package = ModuleType("_flux3_staged_tests")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".flux3", root / "flux3.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class Flux3StagedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux = load_flux3()

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.temp = tempfile.TemporaryDirectory(dir=".", prefix="flux3-staged-test-")
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for index, size in enumerate(((64, 32), (32, 64))):
            path = Path(self.temp.name) / f"reference-{index}.png"
            Image.new("RGB", size, (64 + index * 80, 96, 128)).save(path)
            self.paths.append(str(path))

    def model(self, cached=False):
        test = self
        model = self.flux.Flux3Model.__new__(self.flux.Flux3Model)
        model.device_torch = model.te_device_torch = model.vae_device_torch = self.device
        model.torch_dtype = model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(compile=False, layer_offloading=False, model_kwargs={})
        model.match_target_res = True
        model.events, model.saved = [], []
        model.network = Component().to(self.device)
        model.assistant_lora = model.accuracy_recovery_adapter = None
        model.text_encoder = FakeTextEncoder(self.device, torch.float32) if cached else Component()
        model.sample_step_hook = None
        model._after_sample_image = lambda *_: None

        def phase(expected):
            session = getattr(model, "_staged_sampling", None)
            if session is not None:
                test.assertEqual(session.phase, expected)

        class Transformer(Component):
            def forward(self, **kwargs):
                phase("denoise")
                model.events.append("denoise")
                context = kwargs["ctx"].mean(dim=(1, 2))[:, None, None]
                reference = kwargs.get("x_image_cond", torch.zeros(1, device=test.device)).mean()
                return {"x_image": kwargs["x_image"] * 0.1 * self.weight * model.network.multiplier
                        + context * 0.01 + reference * 0.01}

        class VAE(Component):
            def encode(self, images):
                phase("vae")
                model.events.append("encode")
                return F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (4, 4)).repeat(1, 128, 1, 1)

            def decode(self, latents):
                phase("vae")
                model.events.append("decode")
                return F.interpolate(latents[:, :3] * self.weight, size=(64, 64), mode="bilinear")

        model.model = Transformer().to(self.device)
        model.vae = VAE()

        def encode_at_length(prompt, length):
            phase("text_encoder")
            model.events.append("align")
            return AdvancedPromptEmbeds(text_embeds=[torch.full((length, 4), 1.0 if prompt == "positive" else 0.0)])

        model._encode_at_length = encode_at_length
        return model

    def config(self, index=0, references=0, cfg=3.5):
        return SimpleNamespace(
            seed=42 + index, prompt="positive", negative_prompt="negative", width=64, height=64,
            num_inference_steps=3, guidance_scale=cfg, latents=None, network_multiplier=0.5 + index,
            ctrl_img=self.paths[0] if references else None,
            ctrl_img_1=self.paths[0] if references else None,
            ctrl_img_2=self.paths[1] if references > 1 else None, ctrl_img_3=None,
            save_image_atomic=lambda *_: None, log_image=lambda *_: None,
        )

    def embeds(self):
        return (AdvancedPromptEmbeds(text_embeds=[torch.ones(80, 4)]),
                AdvancedPromptEmbeds(text_embeds=[torch.zeros(160, 4)]))

    def test_model_and_both_ui_pages_expose_staging(self):
        self.assertTrue(self.flux.Flux3Model.supports_staged_sampling)
        source = (ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text(encoding="utf-8")
        entry = next(block for block in re.split(r'\n    name: ', source) if block.startswith('"flux3"'))
        self.assertIn('"model.sample_staged_offloading"', entry)

    def test_training_preserves_each_reference_in_stacked_control_tensors(self):
        for references in (1, 2):
            with self.subTest(references=references):
                model = self.model(cached=True)
                controls = torch.rand(1, references, 3, 32, 64, device=self.device)
                if references == 1:
                    controls = controls[:, 0]
                batch = SimpleNamespace(control_tensor_list=None, control_tensor=controls)
                latents = torch.ones(1, 128, 4, 4, device=self.device)
                output = model.get_noise_prediction(
                    latents, torch.tensor([500], device=self.device), self.embeds()[0], batch
                )
                self.assertEqual(output.shape, latents.shape)
                self.assertEqual(model.events.count("encode"), references)

    def test_pipeline_can_prepare_without_denoising_or_decoding(self):
        model = self.model(cached=True)
        pipeline = self.flux.Flux3Pipeline(model)
        self.assertTrue(hasattr(pipeline, "prepare_staged_sample"))
        conditional, unconditional = self.embeds()
        session = StagedSampling(model)
        model._staged_sampling = session
        try:
            with session:
                task = pipeline.prepare_staged_sample(
                    conditional_embeds=conditional, unconditional_embeds=unconditional,
                    width=64, height=64, num_inference_steps=3,
                    ctrl_imgs=[torch.ones(1, 3, 64, 64)], generator=torch.Generator().manual_seed(42),
                )
                self.assertEqual(model.events, ["encode"])
                self.assertTrue(all(value.device.type == "cpu" for value in task.kwargs["reference_latents"]))
                self.assertEqual(task.kwargs["latents"].device.type, "cpu")
        finally:
            del model._staged_sampling

    def test_prompt_alignment_runs_before_queueing_and_preserves_cache(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                model = self.model(cached=cached)
                self.assertTrue(hasattr(model, "prepare_staged_prompt_embeds"))
                conditional, unconditional = self.embeds()
                session = StagedSampling(model)
                model._staged_sampling = session
                try:
                    with session:
                        batch = StagedSampleBatch(session, self.flux.Flux3Pipeline(model), model.network)
                        batch.add(self.config(), conditional, unconditional, {})
                        queued_cond, queued_uncond, _ = batch.jobs[0]["inputs"]
                        self.assertEqual(queued_cond.text_embeds[0].shape[0], 80 if cached else 160)
                        self.assertEqual(queued_uncond.text_embeds[0].shape[0], 160)
                        self.assertEqual(model.events, [] if cached else ["align"])
                        self.assertEqual(conditional.text_embeds[0].shape[0], 80)
                finally:
                    del model._staged_sampling

    def test_short_negative_prompt_aligns_without_mutating_original(self):
        model = self.model()
        conditional, unconditional = reversed(self.embeds())
        session = StagedSampling(model)
        model._staged_sampling = session
        try:
            with session:
                batch = StagedSampleBatch(session, self.flux.Flux3Pipeline(model), model.network)
                batch.add(self.config(), conditional, unconditional, {})
                queued_cond, queued_uncond, _ = batch.jobs[0]["inputs"]
                self.assertEqual(queued_cond.text_embeds[0].shape[0], 160)
                self.assertEqual(queued_uncond.text_embeds[0].shape[0], 160)
                self.assertEqual(unconditional.text_embeds[0].shape[0], 80)
                self.assertEqual(model.events, ["align"])
        finally:
            del model._staged_sampling
        conditional, unconditional = model.prepare_staged_prompt_embeds(self.config(), conditional, None)
        self.assertIsNone(unconditional)

    def test_supplied_latents_do_not_consume_generator(self):
        model = self.model(cached=True)
        pipeline = self.flux.Flux3Pipeline(model)
        generator = torch.Generator().manual_seed(42)
        state = generator.get_state().clone()
        conditional, unconditional = self.embeds()
        task = pipeline.prepare_staged_sample(
            conditional_embeds=conditional, unconditional_embeds=unconditional,
            height=64, width=64, latents=torch.ones(1, 128, 4, 4), generator=generator,
        )
        self.assertTrue(torch.equal(generator.get_state(), state))
        self.assertTrue(torch.equal(task.kwargs["latents"], torch.ones(1, 128, 4, 4)))

    def test_denoising_failure_clears_queue_and_restores_model(self):
        model = self.model(cached=True)
        states = [(component, component.device, component.training, component.weight)
                  for component in (model.model, model.vae, model.network)]

        def fail(**_):
            raise RuntimeError("injected denoiser failure")

        model.model.forward = fail
        session = StagedSampling(model)
        model._staged_sampling = session
        batch = StagedSampleBatch(session, self.flux.Flux3Pipeline(model), model.network)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected denoiser failure"):
                with session:
                    batch.add(self.config(references=2), *self.embeds(), {})
                    batch.run()
        finally:
            del model._staged_sampling
        self.assertFalse(batch.jobs)
        self.assertFalse(session.preparing)
        self.assertEqual(model.saved, [])
        for component, device, training, parameter in states:
            self.assertEqual(component.device, device)
            self.assertEqual(component.training, training)
            self.assertIs(component.weight, parameter)

    def test_phase_major_batch_matches_serial_pixels_and_restores_state(self):
        for cached in (False, True):
            for references in (0, 2):
                for cfg in (1.0, 3.5):
                    with self.subTest(cached=cached, references=references, cfg=cfg):
                        baseline = self.model(cached=cached)
                        expected = []
                        for index in range(2):
                            config = self.config(index, references, cfg)
                            baseline.network.multiplier = config.network_multiplier
                            image = baseline.generate_single_image(self.flux.Flux3Pipeline(baseline), config,
                                                                   *self.embeds(), torch.Generator().manual_seed(config.seed), {})
                            expected.append(np.array(image))
                        model = self.model(cached=cached)
                        session = StagedSampling(model)
                        states = [(component, component.device, component.training)
                                  for component in (model.model, model.vae, model.network)]
                        rng = torch.get_rng_state().clone()
                        model._staged_sampling = session
                        try:
                            with session:
                                batch = StagedSampleBatch(session, self.flux.Flux3Pipeline(model), model.network)
                                for index in range(2):
                                    config = self.config(index, references, cfg)
                                    config.save_image_atomic = lambda image, i: model.saved.append((i, np.array(image)))
                                    torch.manual_seed(config.seed)
                                    batch.add(config, *self.embeds(), {})
                                batch.run()
                        finally:
                            del model._staged_sampling
                        for (index, pixels), serial in zip(model.saved, expected):
                            np.testing.assert_array_equal(pixels, serial)
                        self.assertEqual(len(model.saved), 2)
                        events = [event for event in model.events if event != "align"]
                        self.assertEqual(events, ["encode"] * (references * 2) + ["denoise"] * 6 + ["decode"] * 2)
                        for component, device, training in states:
                            self.assertEqual(component.device, device)
                            self.assertEqual(component.training, training)
                        self.assertEqual(model.network.multiplier, 0.75)
                        self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
