"""Exercise production sampler control flow with tiny, deterministic components."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from PIL import Image

from toolkit.staged_sampling import StagedSampling


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(patch_size=1, patch_size_t=1, in_channels=3,
                                      sample_rate=16000, mel_hop_length=160)
        self.spatial_compression_ratio = 32
        self.temporal_compression_ratio = 8
        self.mel_compression_ratio = 4

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        return x * self.weight

    def encode(self, x):
        return self(x)

    def decode(self, x):
        return self(x)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from extensions_built_in.diffusion_models.ming_image.src.pipeline import MingImagePipeline
        from extensions_built_in.diffusion_models.ming_image.ming_image import MingImageModel
        from extensions_built_in.diffusion_models.minimax_h3.src.pipeline import MiniMaxH3Pipeline
        from extensions_built_in.diffusion_models.ltx2.ltx2 import (
            LTX2Model, LTX25Model, StagedLTX2ImageToVideoPipeline,
        )
        cls.ming_pipeline = MingImagePipeline
        cls.ming_model = MingImageModel
        cls.h3_pipeline = MiniMaxH3Pipeline
        cls.ltx_model = LTX2Model
        cls.ltx25_model = LTX25Model
        cls.ltx_i2v = StagedLTX2ImageToVideoPipeline

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(compile=False, layer_offloading=False),
            device_torch=self.device, te_device_torch=self.device, vae_device_torch=self.device,
            torch_dtype=torch.float32, text_encoder=[TinyModule().to(self.device)],
            unet=TinyModule().to(self.device), vae=TinyModule().to(self.device),
            network=None, assistant_lora=None,
        )
        self.model.sample_stage = lambda phase: getattr(self.model, "_staged_sampling", None) and self.model._staged_sampling.stage(phase)

    def check_phase(self, phase):
        session = getattr(self.model, "_staged_sampling", None)
        if session is not None:
            self.assertEqual(session.phase, phase)

    def staged(self, call):
        session = StagedSampling(self.model)
        self.model._staged_sampling = session
        try:
            with session:
                return call()
        finally:
            del self.model._staged_sampling

    def test_ming_actual_sampler_matches_and_uses_phase_boundaries(self):
        test = self

        class Transformer(TinyModule):
            def forward(self, images, *args, **kwargs):
                test.check_phase("denoise")
                return ([image * self.weight * 0.1 for image in images],)

        model = self.model
        model.unet = model.transformer = Transformer().to(self.device)
        model.get_train_scheduler = self.ming_model.get_train_scheduler

        def decode(latents):
            self.check_phase("vae")
            return latents * model.vae.weight

        model.decode_to_images = decode
        embeds = SimpleNamespace(text_embeds=[torch.ones(2, 3)], direct_embeds=[torch.ones(2, 3)])
        call = lambda: self.ming_pipeline(model)(
            embeds, height=32, width=32, num_inference_steps=3, guidance_scale=2,
            generator=torch.Generator().manual_seed(42), reference_latents=[torch.zeros(3, 4, 4, device=self.device)],
        )
        expected = call()
        actual = self.staged(call)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        pipe = self.ming_pipeline(model)
        self.assertTrue(hasattr(pipe, "prepare_staged_sample"), "Ming needs a deferred decode path")
        task = pipe.prepare_staged_sample(
            conditional_embeds=embeds, height=32, width=32, num_inference_steps=3, guidance_scale=2,
            generator=torch.Generator().manual_seed(42), reference_latents=[torch.zeros(3, 4, 4, device=self.device)])
        task.denoise(self.device)
        self.assertEqual(task.latents.device.type, "cpu")
        torch.testing.assert_close(task.decode(self.device), expected, rtol=0, atol=0)

    def test_h3_actual_sampler_image_video_audio_and_reference_paths(self):
        test = self

        class Transformer(TinyModule):
            def forward(self, hidden_states, audio_hidden_states, **kwargs):
                test.check_phase("denoise")
                return hidden_states * self.weight * 0.1, audio_hidden_states * self.weight * 0.1

        model = self.model
        model.unet = model.transformer = Transformer().to(self.device)
        model.vae = torch.nn.ModuleList([TinyModule(), TinyModule()]).to(self.device)
        embeds = SimpleNamespace(text_embeds=[torch.ones(2, 3)], text_token_tags=[torch.ones(2, dtype=torch.long)])

        def encode(frames):
            self.check_phase("vae")
            return torch.ones(1, 24, 1, 2, 2, device=self.device) * model.vae[0].weight

        def decode(latents):
            self.check_phase("vae")
            return latents[:, :3] * model.vae[0].weight

        def audio(latents):
            self.check_phase("vae")
            return latents[:, :1] * model.vae[1].weight

        model.encode_keyframe_latents = encode
        model.decode_latents = decode
        model.decode_audio_latents = audio
        image = Image.new("RGB", (32, 32))
        cases = ({"num_frames": 1}, {"num_frames": 5}, {"num_frames": 5, "ctrl_img": image},
                 {"num_frames": 5, "ref_images": [image]},
                 {"num_frames": 5, "ref_images": [torch.zeros(24, 2, 2, 2)]})
        for case in cases:
            with self.subTest(case=list(case)):
                call = lambda: self.h3_pipeline(model)(
                    embeds, height=32, width=32, num_inference_steps=2,
                    generator=torch.Generator().manual_seed(42), **case,
                )
                expected = call()
                actual = self.staged(call)
                if isinstance(actual, dict):
                    torch.testing.assert_close(actual["video"], expected["video"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["audio"], expected["audio"], rtol=0, atol=0)
                else:
                    self.assertEqual(actual[0].tobytes(), expected[0].tobytes())
                pipe = self.h3_pipeline(model)
                self.assertTrue(hasattr(pipe, "prepare_staged_sample"), "H3 needs batched conditioning/decode")
                task = pipe.prepare_staged_sample(
                    conditional_embeds=embeds, height=32, width=32, num_inference_steps=2,
                    generator=torch.Generator().manual_seed(42), **case)
                task.denoise(self.device)
                actual = task.decode(self.device)
                if isinstance(expected, dict):
                    torch.testing.assert_close(actual["video"], expected["video"], rtol=0, atol=0)
                    torch.testing.assert_close(actual["audio"], expected["audio"], rtol=0, atol=0)
                else:
                    self.assertEqual(actual[0].tobytes(), expected[0].tobytes())

    def test_ltx_production_adapter_covers_i2v_connectors_dual_vae_vocoder(self):
        model = self.model
        video, audio = TinyModule().to(self.device), TinyModule()
        model.vae = torch.nn.ModuleList([video, audio])
        model.model = model.unet
        model.pipeline = SimpleNamespace(vae=video, audio_vae=audio, tokenizer=None,
                                         connectors=TinyModule().to(self.device), vocoder=TinyModule().to(self.device))
        model.get_staged_sampling_extra_modules = lambda: self.ltx25_model.get_staged_sampling_extra_modules(model)
        model.get_staged_sampling_vae_modules = lambda: self.ltx25_model.get_staged_sampling_vae_modules(model)
        model.configure_staged_sampling = lambda session: self.ltx25_model.configure_staged_sampling(model, session)
        x = torch.ones(1, device=self.device)
        for fail in (False, True):
            with self.subTest(cancel=fail):
                def sample():
                    pipeline = self.ltx_model.get_generation_pipeline(model)
                    self.assertEqual(pipeline._execution_device, self.device)
                    # The actual T2V/I2V constructors share the scoped modules.
                    i2v = self.ltx_i2v(**pipeline.components)
                    i2v._staged_execution_device = self.device
                    self.assertEqual(i2v._execution_device, self.device)
                    pipeline.connectors(x)
                    self.check_phase("denoise")
                    i2v.vae.encode(x)
                    self.check_phase("vae")
                    i2v.transformer(x)
                    self.check_phase("denoise")
                    pipeline.vae.decode(x)
                    pipeline.audio_vae.decode(x)
                    pipeline.vocoder(x)
                    self.check_phase("vae")
                    if fail:
                        raise RuntimeError("cancel")
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "cancel"):
                        self.staged(sample)
                else:
                    self.staged(sample)
                for vae in (video, audio):
                    self.assertNotIn("encode", vae.__dict__)
                    self.assertNotIn("decode", vae.__dict__)
                    self.assertEqual(vae.device, self.device if vae is video else torch.device("cpu"))
                self.assertFalse(model.unet._forward_pre_hooks)
                self.assertFalse(model.pipeline.connectors._forward_pre_hooks)

    def test_ltx_actual_pipeline_deferred_matches_image_and_video_conditioning(self):
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
        from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
        from extensions_built_in.diffusion_models.ltx2.ltx2 import StagedLTX2Pipeline

        class Transformer(TinyModule):
            def __init__(self):
                super().__init__()
                self.rope = SimpleNamespace(prepare_video_coords=lambda *a, **k: torch.zeros(1, 1, 1))
                self.audio_rope = SimpleNamespace(prepare_audio_coords=lambda *a, **k: torch.zeros(1, 1, 1))

            def cache_context(self, *args):
                return nullcontext()

            def forward(self, hidden_states, audio_hidden_states, **kwargs):
                return hidden_states * 0.1, audio_hidden_states * 0.1

        class Connectors(TinyModule):
            def forward(self, embeds, mask, **kwargs):
                return embeds, embeds, mask

        class VAE(TinyModule):
            def __init__(self):
                super().__init__()
                self.config.scaling_factor = 1.5
                self.config.timestep_conditioning = True
                self.config.mel_bins = 8
                self.config.latent_channels = 3
                self.latents_mean = torch.tensor([0.2, 0.3, 0.4])
                self.latents_std = torch.tensor([0.5, 0.6, 0.7])

            def encode(self, image):
                return SimpleNamespace(latents=image[:, :, :, ::32, ::32])

            def decode(self, latent, *args, **kwargs):
                return (latent,)

        for i2v in (False, True):
            with self.subTest(i2v=i2v):
                audio_vae = VAE()
                audio_vae.latents_mean = torch.full((6,), 0.2)
                audio_vae.latents_std = torch.full((6,), 0.5)
                components = dict(scheduler=self.ltx_model.get_train_scheduler(), vae=VAE(), audio_vae=audio_vae,
                                  text_encoder=None, tokenizer=None, connectors=Connectors(),
                                  transformer=Transformer(), vocoder=TinyModule())
                original = (LTX2ImageToVideoPipeline if i2v else LTX2Pipeline)(**components)
                staged = (self.ltx_i2v if i2v else StagedLTX2Pipeline)(**components)
                staged._staged_execution_device = torch.device("cpu")
                self.assertTrue(hasattr(staged, "prepare_staged_sample"), "LTX needs deferred video/audio decode")
                kwargs = dict(prompt_embeds=torch.ones(1, 2, 3), prompt_attention_mask=torch.ones(1, 2),
                              height=32, width=32, num_frames=9, num_inference_steps=2,
                              guidance_scale=1, output_type="np", return_dict=False,
                              decode_timestep=0.2, decode_noise_scale=0.15)
                if i2v:
                    kwargs["image"] = Image.new("RGB", (32, 32))
                expected = original(**kwargs, generator=torch.Generator().manual_seed(42))
                task = staged.prepare_staged_sample(**kwargs, generator=torch.Generator().manual_seed(42))
                task.denoise("cpu")
                actual = task.decode("cpu")
                import numpy as np
                np.testing.assert_array_equal(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
