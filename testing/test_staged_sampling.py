import importlib.util
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


class Network(torch.nn.Linear):
    def __init__(self, device):
        super().__init__(3, 3, device=device)
        self.is_active = True
        self.multiplier = 0.75


class StagedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("toolkit.staged_sampling"),
            "The opt-in staged sampling implementation is missing",
        )
        from toolkit.staged_sampling import StagedSampling, staged_sampling
        self.context_type = StagedSampling
        self.decorate = staged_sampling
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(sample_staged_offloading=True, layer_offloading=False),
            supports_staged_sampling=True,
            device_torch=self.device,
            te_device_torch=self.device,
            vae_device_torch=self.device,
            unet=torch.nn.Linear(3, 3, device=self.device),
            text_encoder=[torch.nn.Linear(3, 3)],
            vae=torch.nn.Linear(3, 3),
            network=Network(self.device),
            assistant_lora=Network(self.device),
            accuracy_recovery_adapter=None,
        )
        self.model.unet.weight.requires_grad_(False)
        self.model.text_encoder[0].eval()

    def test_off_switch_calls_original_without_moving_anything(self):
        self.model.model_config.sample_staged_offloading = False
        self.model.supports_staged_sampling = False

        @self.decorate
        def sample(model):
            self.assertFalse(hasattr(model, "_staged_sampling"))
            return 42

        self.assertEqual(sample(self.model), 42)

    def test_non_vae_progress_hook_is_not_wrapped_or_disabled(self):
        called = []
        def hook(*args):
            called.append(args)
        hook.requires_vae = False
        self.model.sample_step_hook = hook
        with self.context_type(self.model) as staging:
            staging.stage("denoise")
            self.assertIs(self.model.sample_step_hook, hook)
            self.model.sample_step_hook(0, 10, None)
            self.assertEqual(staging.phase, "denoise")
        self.assertEqual(len(called), 1)

    def test_phase_residency_and_exact_training_state_restoration(self):
        model = self.model
        parameter = model.network.weight
        original = parameter.detach().clone()
        grad = torch.ones_like(parameter)
        parameter.grad = grad
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        with self.context_type(model) as staging:
            for phase, active in (
                ("text_encoder", model.text_encoder[0]),
                ("denoise", model.unet),
                ("vae", model.vae),
                ("text_encoder", model.text_encoder[0]),
            ):
                staging.stage(phase)
                self.assertEqual(next(active.parameters()).device, self.device)
                for root in (model.unet, model.vae, model.text_encoder[0]):
                    if root is not active:
                        self.assertEqual(next(root.parameters()).device.type, "cpu")
            model.network.is_active = False
            model.network.multiplier = 0.0
            model.assistant_lora.is_active = False
            model.network.eval()
            model.network.requires_grad_(False)
            torch.manual_seed(901)
        self.assertIs(parameter, model.network.weight)
        torch.testing.assert_close(parameter, original)
        torch.testing.assert_close(parameter.grad, grad)
        self.assertTrue(model.network.training)
        self.assertTrue(model.network.weight.requires_grad)
        self.assertFalse(model.unet.weight.requires_grad)
        self.assertFalse(model.text_encoder[0].training)
        self.assertTrue(model.network.is_active)
        self.assertTrue(model.assistant_lora.is_active)
        self.assertEqual(model.network.multiplier, 0.75)
        self.assertEqual(next(model.text_encoder[0].parameters()).device.type, "cpu")
        self.assertEqual(next(model.unet.parameters()).device, self.device)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []):
            self.assertTrue(torch.equal(before, after))

    def test_exception_and_cancel_restore_and_remove_hooks(self):
        for error in (RuntimeError("decode failed"), KeyboardInterrupt()):
            cleanup = []

            @self.decorate
            def sample(model):
                stage = model._staged_sampling
                stage.track_cleanup(lambda: cleanup.append(True))
                stage.stage("vae")
                model.assistant_lora.is_active = False
                raise error

            with self.assertRaises(type(error)):
                sample(self.model)
            self.assertEqual(cleanup, [True])
            self.assertTrue(self.model.assistant_lora.is_active)
            self.assertFalse(hasattr(self.model, "_staged_sampling"))
            self.assertEqual(self.model.network.weight.device, self.device)

    def test_train_sample_train_keeps_optimizer_references_and_values(self):
        model = self.model
        optimizer = torch.optim.AdamW(model.network.parameters(), lr=0.01)
        x = torch.ones(1, 3, device=self.device)

        def step():
            optimizer.zero_grad()
            loss = model.network(x).square().mean()
            loss.backward()
            optimizer.step()
            return loss.item()

        step()
        parameter = model.network.weight
        state = optimizer.state[parameter]
        momentum = state["exp_avg"].clone()
        with self.context_type(model) as staging, torch.no_grad():
            staging.stage("denoise")
            model.network(x)
            staging.stage("vae")
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        torch.testing.assert_close(state["exp_avg"], momentum)
        before = parameter.detach().clone()
        self.assertTrue(torch.isfinite(torch.tensor(step())))
        self.assertFalse(torch.equal(before, parameter))

    def test_layer_offloading_can_be_enabled_without_attached_layers(self):
        self.model.model_config.layer_offloading = True
        with self.context_type(self.model) as staging:
            staging.stage("denoise")

    def test_layer_offloading_keeps_cpu_weights_and_restores_mixed_placement(self):
        if self.device.type != "cuda":
            self.skipTest("Layer offloading requires CUDA")
        from toolkit.memory_management import MemoryManager
        from toolkit.memory_management.manager_modules import sync_grad_transfers

        model = self.model
        model.model_config.layer_offloading = True
        model.unet = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.LayerNorm(3), torch.nn.Linear(3, 3)
        )
        model.unet.register_buffer("resident_buffer", torch.ones(3))
        MemoryManager.attach(model.unet, self.device, ignore_modules=[model.unet[2]])
        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=0.01)
        x = torch.ones(2, 3, device=self.device)

        def train_step():
            optimizer.zero_grad()
            loss = model.unet(x).square().mean()
            loss.backward()
            sync_grad_transfers()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))

        train_step()
        before = {name: (p, p.device, p.detach().clone(), p.grad.clone())
                  for name, p in model.unet.named_parameters()}
        cpu_pointer = model.unet[0].weight.data_ptr()
        with torch.no_grad():
            expected = model.unet(x)
        for fail in (False, True):
            try:
                with self.context_type(model) as staging, torch.no_grad():
                    self.assertEqual(model.unet[1].weight.device.type, "cpu")
                    self.assertEqual(model.unet.resident_buffer.device.type, "cpu")
                    staging.stage("denoise")
                    self.assertEqual(model.unet[0].weight.device.type, "cpu")
                    self.assertEqual(model.unet[0].weight.data_ptr(), cpu_pointer)
                    self.assertEqual(model.unet[1].weight.device, self.device)
                    self.assertEqual(model.unet.resident_buffer.device, self.device)
                    torch.testing.assert_close(model.unet(x), expected)
                    staging.stage("vae")
                    self.assertEqual(model.unet[2].weight.device.type, "cpu")
                    if fail:
                        raise RuntimeError("sample failed")
            except RuntimeError as error:
                self.assertEqual(str(error), "sample failed")
            for name, p in model.unet.named_parameters():
                original, device, weight, grad = before[name]
                self.assertIs(p, original)
                self.assertEqual(p.device, device)
                torch.testing.assert_close(p, weight)
                torch.testing.assert_close(p.grad, grad)
            self.assertTrue(hasattr(model.unet[0], "_layer_memory_manager"))
            self.assertEqual(model.unet.resident_buffer.device, self.device)
        train_step()

    def test_unsupported_model_rejects_opt_in(self):
        self.model.supports_staged_sampling = False

        @self.decorate
        def sample(model):
            self.fail("Unsupported models must not silently use the old path")

        with self.assertRaisesRegex(ValueError, "does not support"):
            sample(self.model)

    def test_compiled_model_is_rejected_before_moves(self):
        self.model.model_config.compile = True
        with self.assertRaisesRegex(ValueError, "compile"):
            with self.context_type(self.model):
                self.fail("Compiled components may replace .to() with a no-op")

    def test_lora_multiplier_follows_weights_and_restores_after_sampling(self):
        from toolkit.network_mixins import ToolkitNetworkMixin

        class Lora(ToolkitNetworkMixin, torch.nn.Module):
            def __init__(self, device):
                torch.nn.Module.__init__(self)
                self.lora_down = torch.nn.Linear(3, 3, device=device)
                self._multiplier = None
                self.multiplier = 0.75
                self.is_active = True

            def get_all_modules(self):
                return [self]

            def forward(self, x):
                return self.lora_down(x) * self.torch_multiplier

        model = self.model
        model.network = Lora(self.device)
        multiplier = model.network.torch_multiplier
        with self.context_type(model) as staging:
            model.network.multiplier = 1.0
            staging.stage("denoise")
            self.assertEqual(model.network.torch_multiplier.device, self.device)
            model.network(torch.ones(1, 3, device=self.device))
            staging.stage("vae")
            self.assertEqual(model.network.torch_multiplier.device.type, "cpu")
        self.assertEqual(model.network.multiplier, 0.75)
        torch.testing.assert_close(model.network.torch_multiplier, multiplier)
        model.network(torch.ones(1, 3, device=self.device))

    def test_numerics_match_without_staging(self):
        model = self.model
        x = torch.ones(1, 3, device=self.device)
        with torch.no_grad():
            model.text_encoder[0].to(self.device)
            model.vae.to(self.device)
            expected = model.vae(model.unet(model.text_encoder[0](x)))
            with self.context_type(model) as staging:
                staging.stage("text_encoder")
                encoded = model.text_encoder[0](x)
                staging.stage("denoise")
                latent = model.unet(encoded)
                staging.stage("vae")
                actual = model.vae(latent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_extra_components_and_scoped_pipeline_methods(self):
        model = self.model
        connector = torch.nn.Linear(3, 3, device=self.device)
        vocoder = torch.nn.Linear(3, 3, device=self.device)
        model.get_staged_sampling_extra_modules = lambda: {
            "denoise": [connector], "vae": [vocoder],
        }
        original_forward = model.vae.forward
        for fail in (False, True):
            try:
                with self.context_type(model) as session:
                    session.wrap_method(model.vae, "forward", "vae")
                    session.before_forward(model.unet, "denoise")
                    x = torch.ones(1, 3, device=self.device)
                    model.unet(x)
                    self.assertEqual(session.phase, "denoise")
                    self.assertEqual(connector.weight.device, self.device)
                    model.vae(x)
                    self.assertEqual(session.phase, "vae")
                    self.assertEqual(vocoder.weight.device, self.device)
                    self.assertEqual(connector.weight.device.type, "cpu")
                    model.unet(x)
                    self.assertEqual(session.phase, "denoise")
                    if fail:
                        raise RuntimeError("sample cancelled")
            except RuntimeError as error:
                self.assertEqual(str(error), "sample cancelled")
            self.assertEqual(model.vae.forward, original_forward)
            self.assertNotIn("forward", model.vae.__dict__)
            self.assertFalse(model.unet._forward_pre_hooks)
            self.assertEqual(vocoder.weight.device, self.device)

    def test_external_offload_hooks_are_rejected_before_moving_weights(self):
        for attribute, value in (("_hf_hook", SimpleNamespace(offload=True)),
                                 ("hf_device_map", {"a": "cpu", "b": "cuda:0"}),
                                 ("hf_device_map", {"a": "disk"})):
            with self.subTest(attribute=attribute):
                setattr(self.model.unet, attribute, value)
                try:
                    with self.assertRaises(ValueError):
                        self.context_type(self.model)
                    self.assertEqual(self.model.unet.weight.device, self.device)
                finally:
                    delattr(self.model.unet, attribute)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA layer offloading")
    def test_layer_manager_bare_resident_tensor_follows_phase(self):
        from toolkit.memory_management import MemoryManager
        model = self.model
        model.unet = torch.nn.Sequential(torch.nn.Linear(3, 3)).to(self.device)
        resident = torch.ones(3, device=self.device)
        MemoryManager.attach(model.unet, self.device, ignore_modules=[resident])
        with self.context_type(model) as session:
            self.assertEqual(resident.device.type, "cpu")
            session.stage("denoise")
            self.assertEqual(resident.device, self.device)
        self.assertEqual(resident.device, self.device)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA layer offloading")
    def test_nested_adapter_layer_managers_restore_cpu_weights(self):
        from toolkit.memory_management import MemoryManager
        model = self.model
        model.assistant_lora = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3)),
            torch.nn.Linear(3, 3),
        ).to(self.device)
        MemoryManager.attach(model.assistant_lora[0], self.device)
        cpu_weight = model.assistant_lora[0][0].weight
        cpu_ptr = cpu_weight.data_ptr()
        with self.context_type(model) as session:
            session.stage("denoise")
            self.assertEqual(cpu_weight.device.type, "cpu")
            self.assertEqual(cpu_weight.data_ptr(), cpu_ptr)
            model.assistant_lora(torch.ones(1, 3, device=self.device))
            session.stage("vae")
            self.assertEqual(model.assistant_lora[1].weight.device.type, "cpu")
        self.assertEqual(model.assistant_lora[1].weight.device, self.device)
        self.assertEqual(cpu_weight.data_ptr(), cpu_ptr)

    def test_preview_temporarily_loads_vae_and_restores_denoising(self):
        model = self.model
        def preview(*_):
            self.assertEqual(model.vae.weight.device, self.device)
            self.assertEqual(model.unet.weight.device.type, "cpu")
            raise RuntimeError("preview failed")
        model.sample_step_hook = preview
        with self.context_type(model) as session:
            session.stage("denoise")
            with self.assertRaisesRegex(RuntimeError, "preview failed"):
                model.sample_step_hook(0, 1, None)
            self.assertEqual(session.phase, "denoise")
            self.assertEqual(model.unet.weight.device, self.device)
        self.assertIs(model.sample_step_hook, preview)


class StagedSamplingWiringTests(unittest.TestCase):
    def test_configured_models_have_explicit_phase_support(self):
        ui = (ROOT / "extensions_built_in/diffusion_models/ui.tsx").read_text(encoding="utf-8")
        for arch in ("qwen_image_2", "qwen_image_2:turbo", "ming_image",
                     "minimax_h3", "minimax_h3_ref2va", "minimax_h3_vsa", "ltx2.5"):
            # Top-level architecture records have four-space indented names.
            blocks = re.split(r'\n    name: ', ui)
            block = next((b for b in blocks if b.startswith('"' + arch + '"')), None)
            self.assertIsNotNone(block, arch)
            self.assertIn('"model.sample_staged_offloading"', block, arch)
        for path, class_name in (("ming_image/ming_image.py", "MingImageModel"),
                                 ("minimax_h3/minimax_h3.py", "MinimaxH3Model"),
                                 ("ltx2/ltx2.py", "LTX25Model")):
            tree = ast.parse((ROOT / "extensions_built_in/diffusion_models" / path).read_text(encoding="utf-8"))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            flags = {t.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name) and t.id == "supports_staged_sampling"}
            self.assertTrue(flags.get("supports_staged_sampling"), class_name)

    def test_conflicting_config_rejected_before_model_loading(self):
        from toolkit.config_modules import ModelConfig
        with self.assertRaisesRegex(ValueError, "compile"):
            ModelConfig(name_or_path="unused", sample_staged_offloading=True, compile=True)
        config = ModelConfig(name_or_path="unused", sample_staged_offloading=True, layer_offloading=True)
        self.assertTrue(config.layer_offloading and config.sample_staged_offloading)

    def test_config_is_opt_in(self):
        from toolkit.config_modules import ModelConfig
        self.assertFalse(getattr(ModelConfig(name_or_path="unused"), "sample_staged_offloading", None))
        self.assertTrue(getattr(ModelConfig(name_or_path="unused", sample_staged_offloading=True), "sample_staged_offloading", False))

    def test_switch_and_qwen_phase_hooks_are_wired(self):
        ui = (ROOT / "ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")
        self.assertIn('label="采样分阶段装卸（实验）"', ui)
        self.assertNotIn('layer_offloading: value ? false', ui)
        self.assertNotIn('sample_staged_offloading: value ? false', ui)
        self.assertIn("config.process[0].model.sample_staged_offloading", ui)
        base = (ROOT / "toolkit/models/base_model.py").read_text(encoding="utf-8")
        self.assertIn("@staged_sampling", base)
        pipeline = (ROOT / "extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py").read_text(encoding="utf-8")
        self.assertIn('sample_stage("denoise")', pipeline)
        self.assertIn('sample_stage("vae")', pipeline)


if __name__ == "__main__":
    unittest.main()
