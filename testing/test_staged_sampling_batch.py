"""Phase-major sampling must remain serial, seed-isolated and CPU queued."""

import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

import toolkit.staged_sampling as sampling
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


class BatchTests(unittest.TestCase):
    def test_generation_page_exposes_only_supported_models_and_engine_keeps_progress(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        page = (root / "ui/src/app/generate/page.tsx").read_text(encoding="utf-8")
        self.assertIn("entry?.supportsStagedSampling", page)
        self.assertIn("sample_staged_offloading", page)
        engine = (root / "extensions_built_in/inference_engine/engine.py").read_text(encoding="utf-8")
        self.assertIn("step_hook.requires_vae = False", engine)

    def test_ten_samples_run_phase_major_with_independent_rng_and_settings(self):
        self.assertTrue(hasattr(sampling, "StagedSampleBatch"), "Missing phase-major batch executor")
        events, saved, progress = [], [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class Pipeline:
            def __call__(self, value, generator, **kwargs):
                events.append(("denoise", int(value.item())))
                return value + torch.rand((), generator=generator).to(device) + torch.rand((), device=device)

            def decode_staged_sample(self, latent, offset):
                events.append(("decode", offset))
                return latent.cpu() + offset + torch.rand(())

        model = SimpleNamespace(device_torch=device, vae_device_torch=device, torch_dtype=torch.float32,
                                unet=SimpleNamespace(dtype=torch.float32),
                                network=SimpleNamespace(multiplier=1), sample_step_hook=None)
        session = SimpleNamespace(model=model, stage=lambda phase: None,
                                  progress=lambda *args: progress.append(args), preparing=False)
        pipeline = Pipeline()

        def prepare(pipe, config, cond, uncond, generator, extra):
            events.append(("prepare", config.seed))
            self.assertEqual(cond.text_embeds[0].device.type, "cpu")
            torch.rand(())
            return sampling.PreparedSample(pipe, {"value": cond.text_embeds[0], "generator": generator},
                                           {"offset": config.seed}).map(lambda result: result * 2)

        model.generate_single_image = prepare
        model._after_sample_image = lambda i, count: self.assertEqual(count, 10)
        batch = sampling.StagedSampleBatch(session, pipeline, model.network)
        expected = []
        for i in range(10):
            torch.manual_seed(i)
            # Consume prompt-encoding RNG before queuing, as the real caller does.
            torch.rand(())
            config = SimpleNamespace(seed=i, network_multiplier=i / 10,
                                     save_image_atomic=lambda image, index: saved.append((index, image)),
                                     log_image=lambda *args: None)
            embeds = AdvancedPromptEmbeds(text_embeds=[torch.tensor(float(i), device=device)])
            batch.add(config, embeds, embeds, {})
            gen = torch.default_generator
            torch.rand(())
            result = torch.tensor(float(i), device=device) + torch.rand((), generator=gen).to(device) + torch.rand((), device=device)
            expected.append((result.cpu() + i + torch.rand(())) * 2)
        batch.run()
        self.assertEqual([phase for phase, _ in events], ["prepare"] * 10 + ["denoise"] * 10 + ["decode"] * 10)
        for i, (index, image) in enumerate(saved):
            self.assertEqual(index, i)
            torch.testing.assert_close(image, expected[i], rtol=0, atol=0)
        self.assertFalse(session.preparing)
        self.assertFalse(batch.jobs)
        for phase in ("prepare", "denoise", "decode"):
            self.assertIn((phase, 0, 10), progress)
            self.assertIn((phase, 10, 10), progress)

    def test_tree_copy_does_not_move_or_mutate_prompt_cache(self):
        self.assertTrue(hasattr(sampling, "sample_to_device"), "Missing CPU queue transfer")
        from toolkit.prompt_utils import PromptEmbeds
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        original = PromptEmbeds(torch.ones(1, 2, 3, device=device))
        copied = sampling.sample_to_device({"prompt": original}, "cpu")["prompt"]
        self.assertIsNot(copied, original)
        self.assertEqual(copied.text_embeds.device.type, "cpu")
        self.assertEqual(str(original.text_embeds.device), device)

    def test_batch_cancellation_clears_queued_tensors_and_restores_preview(self):
        hook = lambda *args: None
        model = SimpleNamespace(sample_step_hook=hook)
        session = SimpleNamespace(model=model, preparing=False)

        def cancelled(*args):
            raise InterruptedError("cancelled")

        session.progress = cancelled
        batch = sampling.StagedSampleBatch(session, None, None)
        batch.add(None, torch.ones(2), None, {})
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            batch.run()
        self.assertFalse(batch.jobs)
        self.assertFalse(session.preparing)
        self.assertIs(model.sample_step_hook, hook)

    def test_generation_engine_streams_phases_and_cleans_up_on_cancel(self):
        from PIL import Image
        from extensions_built_in.inference_engine.engine import Engine, GenerationJob

        for cancel_at in (None, "phase", "step"):
            with self.subTest(cancel_at=cancel_at), tempfile.TemporaryDirectory(dir=".", prefix="staged-engine-test-") as folder:
                engine = Engine("cpu", folder, assets_folder=folder)
                job = GenerationJob({"arch": "qwen_image_2", "name_or_path": "unused-test-model", "sample_staged_offloading": True},
                                    {"prompt": "test", "width": 32, "height": 32, "num_inference_steps": 2},
                                    {"latents": "none"})
                model_config = engine._build_model_config(job.model)
                self.assertTrue(model_config.sample_staged_offloading)
                holder = SimpleNamespace()

                def generate(configs):
                    self.assertFalse(holder.sample_step_hook.requires_vae)
                    holder.sample_progress_hook("text_encoder", 0, 1)
                    holder.sample_progress_hook("denoise", 0, 1)
                    holder.sample_step_hook(0, 2, torch.zeros(1))
                    if cancel_at is not None:
                        job.cancel.set()
                    if cancel_at == "phase":
                        holder.sample_progress_hook("decode", 0, 1)
                    holder.sample_step_hook(1, 2, torch.zeros(1))
                    holder.sample_progress_hook("decode", 1, 1)
                    configs[0].save_image_atomic(Image.new("RGB", (32, 32)), 0)

                holder.generate_images = generate
                with patch.object(engine, "_ensure_holder", return_value=holder), patch.object(job, "emit", wraps=job.emit) as emit:
                    engine._run(job)
                self.assertEqual(job.status, "cancelled" if cancel_at else "done", job.error)
                self.assertTrue(job.finished)
                self.assertIsNone(holder.sample_step_hook)
                self.assertIsNone(holder.sample_progress_hook)
                self.assertTrue(any(call.args[0] == "sample_phase" for call in emit.call_args_list))
                self.assertTrue(any(call.args[0] == "progress" for call in emit.call_args_list))


if __name__ == "__main__":
    unittest.main()
