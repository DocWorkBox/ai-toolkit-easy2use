"""Exercise P-H3 and FastH3 Omni phase-major sampling with tiny components."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3
from extensions_built_in.diffusion_models.minimax_h3 import p_h3_image as ph3_module
from extensions_built_in.diffusion_models.minimax_h3.p_h3_image import PH3ImageModel, PH3_SAMPLE_SIGMAS
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.staged_sampling import StagedSampleBatch, StagedSampling


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (PH3ImageModel, h3.MinimaxH3OmniModel)


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


class H3DerivativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="h3-derivative-staged-")
        self.addCleanup(temporary.cleanup)
        self.reference = str(Path(temporary.name) / "reference.png")
        Image.new("RGB", (32, 32), (128, 64, 96)).save(self.reference)

    def test_training_adapters_use_inherited_creator_lora_cache(self):
        for cls, filename in (
            (PH3ImageModel, "p_h3_image_training_adapter_v1.safetensors"),
            (h3.MinimaxH3OmniModel, "fastvideo_fasth3_omni_training_adapter_v1.safetensors"),
        ):
            with self.subTest(arch=cls.arch):
                self.assertIs(cls.load_training_adapter, h3.MinimaxH3Model.load_training_adapter)
                model = cls.__new__(cls)
                reference = f"ostris/minimax_h3_training_adapter/{filename}"
                model.model_config = SimpleNamespace(assistant_lora_path=reference)
                model.print_and_status_update = Mock()
                project = str(Path(self.reference).parent)
                store = str(Path(project) / "creator-models")
                cached = str(Path(store) / "loras" / "training_adapters" / filename)
                with patch.object(h3, "MODELS_PATH", store), patch.object(
                    h3, "resolve_lora_file", return_value=cached
                ) as resolve, patch.object(h3, "load_file", side_effect=RuntimeError("resolved adapter")) as load:
                    with self.assertRaisesRegex(RuntimeError, "resolved adapter"):
                        model.load_training_adapter(Mock())
                resolve.assert_called_once_with(
                    reference, status_fn=model.print_and_status_update,
                    download_dir=str(Path(store) / "loras" / "training_adapters"),
                )
                load.assert_called_once_with(cached)
                self.assertEqual(model.model_config.assistant_lora_path, cached)

    def model(self, cls, device=None):
        test = self
        model = cls.__new__(cls)
        device = device or self.device
        model.device_torch = model.te_device_torch = model.vae_device_torch = device
        model.torch_dtype = model.vae_torch_dtype = torch.float32
        model.model_config = SimpleNamespace(compile=False, layer_offloading=False, model_kwargs={})
        model.max_text_length = 128
        model.edit_prompt_prefix = True
        model.sample_step_hook = None
        model.sample_progress_hook = None
        model.network, model.assistant_lora = Component(), Component()
        model.accuracy_recovery_adapter = None
        model.saved, model.events = [], []
        model.tokenizer = model.processor = None
        model._after_sample_image = lambda *_: None

        def phase(expected):
            session = getattr(model, "_staged_sampling", None)
            if session is None:
                return
            test.assertEqual(session.phase, expected)
            if device.type == "cuda":
                for name, modules in (
                    ("text_encoder", [model.text_encoder]),
                    ("denoise", [model.model, model.network, model.assistant_lora]),
                    ("vae", [model.video_vae, model.audio_vae]),
                ):
                    if name != expected:
                        for module in modules:
                            if module is not None:
                                test.assertEqual(module.device.type, "cpu")

        class TextEncoder(Component):
            text_connector = torch.nn.Identity()

            def forward(self):
                phase("text_encoder")
                model.events.append("text")
                return self.weight.expand(2, 3)

        class Transformer(Component):
            params = SimpleNamespace(pdd_steps=32 if cls is h3.MinimaxH3OmniModel else 1)

            def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, pdd_weights, **kwargs):
                phase("denoise")
                model.events.append("denoise")
                context = encoder_hidden_states.mean() * 0.01
                if self.params.pdd_steps > 1:
                    test.assertEqual(tuple(pdd_weights[0].shape), (1, 32))
                    test.assertTrue(torch.isfinite(pdd_weights[0]).all())
                    context = context + pdd_weights[0].to(self.device).mean() * 0.01
                else:
                    test.assertIsNone(pdd_weights)
                scale = self.weight * model.network.multiplier * 0.1
                return hidden_states * scale + context, audio_hidden_states * scale + context

        class VideoVAE(Component):
            def encode(self, frames, generator=None, **kwargs):
                phase("vae")
                model.events.append("reference")
                shape = (1 if frames.shape[2] == 1 else 2, 2, 2)
                return F.adaptive_avg_pool3d(frames.mean(1, keepdim=True), shape).repeat(1, 24, 1, 1, 1)

            def decode(self, latents):
                phase("vae")
                model.events.append("decode")
                return F.interpolate(latents[:, :3] * self.weight, size=(latents.shape[2], 32, 32))

        class AudioVAE(Component):
            def encode(self, waveform):
                phase("vae")
                return F.adaptive_avg_pool1d(waveform, 2).repeat(1, 32, 1) * self.weight

            def decode(self, latents):
                phase("vae")
                model.events.append("audio")
                return latents[:, :1] * self.weight

        model.model = Transformer()
        model.text_encoder = TextEncoder().requires_grad_(False)
        audio = AudioVAE().requires_grad_(False) if cls is h3.MinimaxH3OmniModel else None
        model.vae = h3.MiniMaxH3VaeBundle(VideoVAE().requires_grad_(False), audio)
        return model

    def config(self, model, index=0, references=False):
        return SimpleNamespace(
            width=32, height=32, num_frames=5, fps=24, seed=42 + index,
            num_inference_steps=8, guidance_scale=1.0, latents=None,
            network_multiplier=0.5 + index * 0.1,
            ctrl_img=self.reference if references else None,
            ctrl_img_1=None, ctrl_img_2=None, ctrl_img_3=None,
            save_image_atomic=lambda image, _: model.saved.append(image), log_image=lambda *_: None,
        )

    def embeds(self, model, index=0):
        return AdvancedPromptEmbeds(
            text_embeds=[torch.full((2, 3), 1.0 + index)],
            text_token_tags=[torch.ones(2, dtype=torch.long)],
        )

    def test_both_ui_entries_expose_existing_opt_in_control(self):
        source = (ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text(encoding="utf-8")
        for cls in CLASSES:
            with self.subTest(arch=cls.arch):
                entry = source.split(f'name: "{cls.arch}"', 1)[1].split('\n  },', 1)[0]
                sections = entry.split("additionalSections: [", 1)[1].split("]", 1)[0]
                self.assertIn('"model.sample_staged_offloading"', sections)
                self.assertNotIn('"config.process[0].model.sample_staged_offloading": [true', entry)
                self.assertTrue(cls.supports_staged_sampling)

    def test_ph3_single_file_te_uses_local_project_config_tokenizer_and_processor(self):
        model = self.model(PH3ImageModel, device=torch.device("cpu"))
        model.model_config.te_name_or_path = None
        model.te_torch_dtype = torch.float32
        model.print_and_status_update = Mock()
        weight = "/read-only/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors"
        model._resolve_comfy_file = Mock(return_value=weight)
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4), tie_word_embeddings=True)
        encoder = Mock(config=config)
        encoder.named_modules.return_value = []
        encoder.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        assets = "/toolkit/models/P-H3-Image"
        with patch.object(ph3_module, "resolve_ph3_assets", return_value=assets, create=True) as resolve, \
                patch("transformers.AutoTokenizer.from_pretrained", return_value="tokenizer") as tokenizer, \
                patch("transformers.AutoProcessor.from_pretrained", return_value="processor") as processor, \
                patch("transformers.AutoConfig.from_pretrained", return_value=config) as load_config, \
                patch.object(ph3_module, "Qwen3VLTextEncoder", return_value=encoder), \
                patch.object(ph3_module, "PH3TextConnector", return_value=torch.nn.Identity()), \
                patch.object(ph3_module, "load_file", return_value={}) as load_weights, \
                patch.object(ph3_module, "import_comfy_quantized_layers", return_value=({}, 0)), \
                patch.object(ph3_module, "flush"):
            self.assertEqual(model._load_text_encoder(), ("tokenizer", "processor", encoder))
        resolve.assert_called_once_with(ph3_module.PH3_ORIGINAL_REPO, status_fn=model.print_and_status_update)
        tokenizer.assert_called_once_with(assets, subfolder="tokenizer", local_files_only=True)
        processor.assert_called_once_with(assets, subfolder="processor", local_files_only=True)
        load_config.assert_called_once_with(assets, subfolder="text_encoder", local_files_only=True)
        load_weights.assert_called_once_with(weight)

    def test_phase_major_matches_serial_images_video_audio_and_restores_state(self):
        for cls in CLASSES:
            for references in (False, True):
                with self.subTest(arch=cls.arch, references=references):
                    baseline = self.model(cls)
                    expected = []
                    for i in range(2):
                        config = self.config(baseline, i, references)
                        baseline.network.multiplier = config.network_multiplier
                        expected.append(baseline.generate_single_image(
                            baseline.get_generation_pipeline(), config, self.embeds(baseline, i), None,
                            torch.Generator().manual_seed(config.seed), {},
                        ))
                    model = self.model(cls)
                    modules = [model.model, model.text_encoder, model.network, model.assistant_lora,
                               model.video_vae] + ([model.audio_vae] if model.audio_vae is not None else [])
                    states = [(m, m.device, m.training, [(p, p.requires_grad) for p in m.parameters()]) for m in modules]
                    rng = torch.get_rng_state().clone()
                    session = StagedSampling(model)
                    batch = StagedSampleBatch(session, model.get_generation_pipeline(), model.network)
                    model._staged_sampling = session
                    try:
                        with session:
                            for i in range(2):
                                config = self.config(model, i, references)
                                torch.manual_seed(config.seed)
                                batch.add(config, self.embeds(model, i), None, {})
                            batch.run()
                    finally:
                        del model._staged_sampling
                    for actual, wanted in zip(model.saved, expected):
                        if cls is PH3ImageModel:
                            np.testing.assert_array_equal(np.asarray(actual), np.asarray(wanted))
                        else:
                            torch.testing.assert_close(actual["video"], wanted["video"], rtol=0, atol=0)
                            torch.testing.assert_close(actual["audio"], wanted["audio"], rtol=0, atol=0)
                            self.assertEqual(actual["fps"], wanted["fps"])
                    events = model.events
                    expected_events = ["reference"] * (2 if references else 0) + ["denoise"] * 16
                    expected_events += ["decode"] * 2 if cls is PH3ImageModel else ["decode", "audio"] * 2
                    self.assertEqual(events, expected_events)
                    self.assertEqual(len(model.saved), 2)
                    for module, device, training, parameters in states:
                        self.assertEqual(module.device, device)
                        self.assertEqual(module.training, training)
                        for parameter, grad in parameters:
                            self.assertTrue(any(p is parameter for p in module.parameters()))
                            self.assertEqual(parameter.requires_grad, grad)
                    self.assertEqual(model.network.multiplier, 0.75)
                    self.assertTrue(model.assistant_lora.is_active)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_direct_component_calls_keep_stage_manager_in_control(self):
        def encode_prompt(encoder, *args, **kwargs):
            embeds = encoder()
            return embeds, torch.ones(embeds.shape[0], dtype=torch.long, device=embeds.device)

        for cls in CLASSES:
            model = self.model(cls, device=torch.device("cpu"))
            session = SimpleNamespace(phase="idle", stage=lambda phase: setattr(session, "phase", phase))
            model._staged_sampling = session
            with self.subTest(arch=cls.arch), patch.object(model.text_encoder, "to", side_effect=AssertionError("raw TE move")), \
                    patch.object(model.vae, "to", side_effect=AssertionError("raw VAE move")), \
                    patch.object(h3, "encode_minimax_h3_prompt", side_effect=encode_prompt):
                for name, call in (
                    ("text", lambda: model.get_prompt_embeds("test prompt")),
                    ("encode", lambda: model.encode_images([torch.ones(3, 32, 32)])),
                    ("reference", lambda: model.encode_keyframe_latents(torch.ones(1, 3, 1, 32, 32))),
                    ("decode", lambda: model.decode_latents(torch.ones(1, 24, 1, 2, 2))),
                ):
                    with self.subTest(operation=name):
                        call()
                if model.audio_vae is not None:
                    with self.subTest(operation="audio"):
                        model.decode_audio_latents(torch.ones(2, 32, 2))
                        model.encode_audio([{"waveform": torch.ones(2, 1600), "sample_rate": 32000}])

    def test_derivative_schedules_and_image_only_vae_remain_intact(self):
        ph3 = self.model(PH3ImageModel)
        omni = self.model(h3.MinimaxH3OmniModel)
        torch.testing.assert_close(ph3.build_sample_sigmas(8), torch.tensor(PH3_SAMPLE_SIGMAS))
        self.assertIsNone(ph3.audio_vae)
        torch.testing.assert_close(omni.build_sample_sigmas(8), h3.packing.build_sigma_schedule(8, 12.0, True))
        self.assertEqual(omni.model.params.pdd_steps, 32)

    def test_uncached_prompts_use_text_phase_with_other_components_on_cpu(self):
        for cls in CLASSES:
            with self.subTest(arch=cls.arch):
                model = self.model(cls)
                session = StagedSampling(model)
                model._staged_sampling = session

                def encode_prompt(encoder, *args, **kwargs):
                    embeds = encoder()
                    return embeds, torch.ones(embeds.shape[0], dtype=torch.long, device=embeds.device)

                try:
                    with session, patch.object(h3, "encode_minimax_h3_prompt", side_effect=encode_prompt):
                        controls = [torch.ones(1, 3, 32, 32, device=self.device)]
                        for _ in range(2):
                            result = model.get_prompt_embeds("edit the picture", control_images=controls)
                            self.assertEqual(result.text_embeds[0].device, self.device)
                    self.assertEqual(model.events, ["text", "text"])
                finally:
                    del model._staged_sampling

    def test_phase_failures_clear_batch_and_restore_modules_and_rng(self):
        for cls in CLASSES:
            for failure in ("reference", "denoise", "decode", "audio"):
                if failure == "audio" and cls is PH3ImageModel:
                    continue
                with self.subTest(arch=cls.arch, failure=failure):
                    model = self.model(cls)
                    component, method = {
                        "reference": (model.video_vae, "encode"),
                        "denoise": (model.model, "forward"),
                        "decode": (model.video_vae, "decode"),
                        "audio": (model.audio_vae, "decode"),
                    }[failure]
                    modules = [model.model, model.network, model.assistant_lora, model.video_vae]
                    states = [(module, module.device, module.training, module.weight) for module in modules]
                    rng = torch.get_rng_state().clone()
                    session = StagedSampling(model)
                    batch = StagedSampleBatch(session, model.get_generation_pipeline(), model.network)
                    model._staged_sampling = session
                    try:
                        with self.assertRaisesRegex(RuntimeError, f"failed {failure}"):
                            with session:
                                batch.add(self.config(model, references=True), self.embeds(model), None, {})
                                with patch.object(component, method, side_effect=RuntimeError(f"failed {failure}")):
                                    batch.run()
                    finally:
                        del model._staged_sampling
                    self.assertFalse(batch.jobs)
                    self.assertFalse(session.preparing)
                    self.assertFalse(model.saved)
                    self.assertEqual(model.network.multiplier, 0.75)
                    for module, device, training, parameter in states:
                        self.assertEqual(module.device, device)
                        self.assertEqual(module.training, training)
                        self.assertIs(module.weight, parameter)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
