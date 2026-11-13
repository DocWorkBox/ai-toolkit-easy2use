"""Keep custom sampling contracts while integrating the November H3 update."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
H3 = "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py"


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def h3_method(name):
    tree = ast.parse(source(H3))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MinimaxH3Model")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), H3, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("name", ["_stage_for_sampling", "_stage_for_decode", "_park_vae_for_denoise"])
def test_h3_low_vram_helpers_do_not_bypass_an_active_stage_manager(name):
    # Any access to the raw components would bypass the stage manager.
    model = SimpleNamespace(_staged_sampling=object())
    h3_method(name)(model)


def test_phase_major_queue_selects_each_samples_own_sigmas():
    import toolkit.staged_sampling as sampling

    seen, saved = [], []
    holder = SimpleNamespace(
        device_torch="cpu", vae_device_torch="cpu", unet=SimpleNamespace(dtype=torch.float32),
        sample_step_hook=None, sample_sigmas=[0.123], _after_sample_image=lambda *_: None,
    )
    network = SimpleNamespace(multiplier=1.0)
    session = SimpleNamespace(model=holder, stage=lambda _: None, progress=lambda *_: None, preparing=False)

    class Pipeline:
        def __call__(self, **kwargs):
            seen.append(("denoise", holder.sample_sigmas))
            return torch.ones(1)

        def decode_staged_sample(self, latent):
            return latent

    pipe = Pipeline()

    def prepare(pipeline, config, *_):
        seen.append(("prepare", holder.sample_sigmas))
        return sampling.PreparedSample(pipeline, {})

    holder.generate_single_image = prepare
    batch = sampling.StagedSampleBatch(session, pipe, network)
    schedules = ([1.0, 0.4], None, [0.9, 0.2, 0.1])
    for sigmas in schedules:
        config = SimpleNamespace(sigmas=sigmas, network_multiplier=1.0,
                                 save_image_atomic=lambda image, _: saved.append(image), log_image=lambda *_: None)
        batch.add(config, None, None, {})
    with patch.object(sampling, "flush") as flush:
        batch.run()
    assert seen == [(phase, sigmas) for phase in ("prepare", "denoise") for sigmas in schedules]
    assert len(saved) == 3
    assert all(call.kwargs == {"garbage_collect": False} for call in flush.call_args_list)


def test_sigma_hooks_are_restored_by_the_staged_sampling_context():
    text = source("toolkit/models/base_model.py")
    assert "unwrap_sigma_override = staging.track_cleanup(unwrap_sigma_override)" in text
    assert "self.sample_sigmas = gen_config.sigmas" in text


@pytest.mark.parametrize("cancel", [False, True])
def test_sigma_scheduler_wrapper_is_removed_after_sampling_or_cancellation(cancel):
    from toolkit.sample_sigmas import install_sample_sigma_override
    from toolkit.staged_sampling import StagedSampling

    class Scheduler:
        config = SimpleNamespace(use_flow_sigmas=True, num_train_timesteps=1000)

        def set_timesteps(self, count):
            self.timesteps = torch.linspace(1000, 100, count)
            self.sigmas = torch.linspace(1, 0, count + 1)

    scheduler = Scheduler()
    original = Scheduler.set_timesteps
    model = SimpleNamespace(
        model_config=SimpleNamespace(compile=False), text_encoder=None, unet=None,
        network=None, assistant_lora=None, vae=None,
        device_torch="cpu", te_device_torch="cpu", vae_device_torch="cpu",
    )
    session = StagedSampling(model)
    expected = pytest.raises(RuntimeError, match="cancel") if cancel else nullcontext()
    with expected:
        with session:
            unwrap = install_sample_sigma_override(
                model, SimpleNamespace(scheduler=scheduler), [SimpleNamespace(sigmas=[1.0, 0.3])],
            )
            session.track_cleanup(unwrap)
            model.sample_sigmas = [1.0, 0.3]
            scheduler.set_timesteps(5)
            torch.testing.assert_close(scheduler.sigmas, torch.tensor([1.0, 0.3, 0.0]))
            if cancel:
                raise RuntimeError("cancel")
    assert Scheduler.set_timesteps is original
    assert model.sample_sigmas is None


def test_h3_image_keeps_model_name_localized_notes_and_opt_in_staging():
    ui = source("extensions_built_in/diffusion_models/ui.tsx")
    entry = ui.split('name: "minimax_h3_image"', 1)[1].split('\n  },', 1)[0]
    assert 'label: "MiniMax-H3-Image"' in entry
    assert '"model.sample_staged_offloading"' in entry
    assert "图像编辑" in entry and "训练适配器" in entry
    assert "Weights load from the" not in entry
    assert '"Comfy-Org/MiniMax-H3-Image"' in entry
    assert "minimax_h3_image_training_adapter_v1.safetensors" in entry
    assert '"config.process[0].model.sample_staged_offloading": [true' not in entry
    assert source("version.py").strip() == 'VERSION = "1.19.1"'


def test_h3_image_inherits_project_adapter_cache_and_h3_vae_stage_group():
    image = source("extensions_built_in/diffusion_models/minimax_h3/minimax_h3_image.py")
    assert "class MinimaxH3ImageModel(MinimaxH3Ref2VAModel)" in image
    base = source(H3)
    assert 'TOOLKIT_ROOT, "models", "loras", "training_adapters"' in base
    assert 'ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"' in base
    assert "return [self.video_vae, self.audio_vae]" in base


def test_shared_audio_contract_reuses_noise_and_stream_override():
    from toolkit.dto import DTO

    text = ast.parse(source("toolkit/models/base_model.py"))
    cls = next(n for n in text.body if isinstance(n, ast.ClassDef) and n.name == "BaseModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "audio_stream_input")
    namespace = {"torch": torch, "DTO": DTO}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "audio_contract", "exec"), namespace)
    audio = namespace[method.name]
    raw = torch.ones(2, 3, 4)
    sigma = torch.tensor([0.25, 0.75])
    batch = SimpleNamespace(latents=DTO(torch.zeros(2, 1)))
    first = audio(None, torch.zeros(2, 1), batch, raw, sigma, torch.zeros_like(raw))
    second = audio(None, torch.zeros(2, 1), batch, raw, sigma, torch.zeros_like(raw))
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    override = torch.full_like(raw, 3)
    rows, target, noisy, stream = audio(None, DTO(torch.zeros(2, 1), audio=override), batch, raw, sigma, torch.zeros_like(raw))
    torch.testing.assert_close(rows, override)
    assert target is None and stream
    torch.testing.assert_close(noisy, override)
