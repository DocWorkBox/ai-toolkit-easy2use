"""Opt-in component residency for model implementations with sampling phase hooks."""

from functools import wraps
from copy import copy
from itertools import chain
from time import perf_counter

import torch
from tqdm import tqdm

from toolkit.basic import flush
from toolkit.print import print_acc


def sample_to_device(value, device):
    """Copy sample data, never module weights or mutable prompt-cache containers."""
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
    from toolkit.prompt_utils import PromptEmbeds

    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, (PromptEmbeds, AdvancedPromptEmbeds)):
        return copy(value).to(device)
    if isinstance(value, dict):
        return {key: sample_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(sample_to_device(item, device) for item in value)
    return value


class SampleRNG:
    def __init__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def restore(self):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


class PreparedSample:
    """One serial denoise request and its deferred decode, queued in host memory."""
    def __init__(self, pipeline, kwargs, decode_kwargs=None):
        self.pipeline = pipeline
        self.kwargs = sample_to_device(kwargs, "cpu")
        self.decode_kwargs = decode_kwargs or {}
        self.transforms = []
        self.latents = None

    def map(self, transform):
        self.transforms.append(transform)
        return self

    def denoise(self, device):
        kwargs = sample_to_device(self.kwargs, device)
        try:
            self.latents = sample_to_device(self.pipeline(**kwargs), "cpu")
        finally:
            self.kwargs = None

    def decode(self, device):
        try:
            result = self.pipeline.decode_staged_sample(
                sample_to_device(self.latents, device), **self.decode_kwargs)
            for transform in self.transforms:
                result = transform(result)
            return result
        finally:
            self.latents = None


def call_sample_pipeline(model, pipeline, **kwargs):
    session = getattr(model, "_staged_sampling", None)
    if session is not None and session.preparing:
        return pipeline.prepare_staged_sample(**kwargs)
    return pipeline(**kwargs)


def map_sample(result, transform):
    return result.map(transform) if isinstance(result, PreparedSample) else transform(result)


class StagedSampleBatch:
    """Encode in BaseModel, then prepare all references, denoise all, decode all."""
    def __init__(self, session, pipeline, network):
        self.session, self.pipeline, self.network = session, pipeline, network
        self.model = session.model
        self.jobs = []

    def add(self, config, conditional, unconditional, extra):
        prepare = getattr(self.model, "prepare_staged_prompt_embeds", None)
        if prepare is not None:
            conditional, unconditional = prepare(config, conditional, unconditional)
        self.jobs.append(dict(config=config, rng=SampleRNG(),
                              inputs=sample_to_device((conditional, unconditional, extra), "cpu")))

    @torch.no_grad()
    def run(self):
        hook = getattr(self.model, "sample_step_hook", None)
        if hook is not None and getattr(hook, "requires_vae", True):
            print_acc("[Sampling stages] Per-step VAE previews paused during phase-major sampling")
            self.model.sample_step_hook = None
        try:
            for phase in ("prepare", "denoise", "decode"):
                self.session.preparing = phase == "prepare"
                if phase != "prepare" and self.jobs:
                    self.session.stage("vae" if phase == "decode" else "denoise")
                for i, job in enumerate(tqdm(self.jobs, desc=f"Sampling {phase}", leave=True)):
                    self.session.progress(phase, i, len(self.jobs))
                    config = job["config"]
                    job["rng"].restore()
                    self.network.multiplier = config.network_multiplier
                    self.model._sample_step_index = 0
                    if phase == "prepare":
                        cond, uncond, extra = job.pop("inputs")
                        cond = cond.to("cpu", dtype=self.model.unet.dtype) if cond is not None else None
                        uncond = uncond.to("cpu", dtype=self.model.unet.dtype) if uncond is not None else None
                        task = self.model.generate_single_image(
                            self.pipeline, config, cond, uncond, torch.default_generator, extra)
                        if not isinstance(task, PreparedSample):
                            raise TypeError("Staged sampler must return a PreparedSample")
                        job["task"] = task
                        del cond, uncond, extra, task
                    elif phase == "denoise":
                        job["task"].denoise(self.model.device_torch)
                    else:
                        image = job.pop("task").decode(self.model.vae_device_torch)
                        config.save_image_atomic(image, i)
                        config.log_image(image, i)
                        self.model._after_sample_image(i, len(self.jobs))
                        del image
                    job["rng"] = SampleRNG()
                    self.session.progress(phase, i + 1, len(self.jobs))
                    flush()
        finally:
            self.session.preparing = False
            self.model.sample_step_hook = hook
            self.jobs.clear()


class LayerOffloadResidency:
    """Move resident tensors without detaching the layer manager or its pinned weights."""

    def __init__(self, root):
        from toolkit.memory_management.manager_modules import _is_quantized_tensor, _storage_device

        self.managers = {id(m._memory_manager): m._memory_manager for m in root.modules()
                         if hasattr(m, "_memory_manager")}.values()
        self.tensors = []
        for module in root.modules():
            if hasattr(module, "_layer_memory_manager"):
                continue
            for name, tensor in chain(module.named_parameters(recurse=False),
                                      module.named_buffers(recurse=False)):
                if isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad and _is_quantized_tensor(tensor):
                    raise ValueError("Staged sampling does not support trainable quantized tensor subclasses")
                self.tensors.append((module, name, _storage_device(tensor)))
        registered = {id(t) for t in chain(root.parameters(), root.buffers())}
        self.bare = []
        for tensor in chain.from_iterable(m.unmanaged_modules for m in self.managers):
            if isinstance(tensor, torch.Tensor) and id(tensor) not in registered:
                if _is_quantized_tensor(tensor):
                    raise ValueError("Staged sampling does not support unregistered quantized resident tensors")
                self.bare.append((tensor, tensor.device))

    def move(self, device=None):
        from toolkit.memory_management.manager_modules import _is_quantized_tensor, _move_own_param, _storage_device

        for module, name, original_device in self.tensors:
            tensor = getattr(module, name)
            target = torch.device(device) if device is not None else original_device
            if _storage_device(tensor) == target:
                continue
            if isinstance(tensor, torch.nn.Parameter):
                grad = tensor.grad
                _move_own_param(module, name, tensor, target)
                if grad is not None:
                    getattr(module, name).grad = grad.to(target)
            elif _is_quantized_tensor(tensor):
                setattr(module, name, tensor.to(target))
            else:
                tensor.data = tensor.data.to(target)
            moved = getattr(module, name)
            if moved is not tensor:
                for manager in self.managers:
                    manager.unmanaged_modules[:] = [
                        moved if entry is tensor else entry for entry in manager.unmanaged_modules
                    ]
        for tensor, original_device in self.bare:
            target = torch.device(device) if device is not None else original_device
            tensor.data = tensor.data.to(target)
            if tensor.is_leaf and tensor.grad is not None:
                tensor.grad = tensor.grad.to(target)


class StagedSampling:
    def __init__(self, model):
        if getattr(model.model_config, "compile", False):
            raise ValueError("sample_staged_offloading cannot be combined with compile")
        self.model = model
        encoders = model.text_encoder
        if not isinstance(encoders, list):
            encoders = [encoders]
        self.groups = {
            "text_encoder": encoders,
            "denoise": [model.unet, model.network, model.assistant_lora,
                        getattr(model, "accuracy_recovery_adapter", None)],
            "vae": getattr(model, "get_staged_sampling_vae_modules", lambda: [model.vae])(),
        }
        extra_groups = getattr(model, "get_staged_sampling_extra_modules", lambda: {})()
        for phase, modules in extra_groups.items():
            self.groups[phase].extend(modules)
        self.devices = {
            "text_encoder": model.te_device_torch,
            "denoise": model.device_torch,
            "vae": model.vae_device_torch,
        }
        roots = {id(m): m for group in self.groups.values() for m in group
                 if isinstance(m, torch.nn.Module)}
        self.states = []
        self.layer_residency = {}
        for module in roots.values():
            for child in module.modules():
                if getattr(getattr(child, "_hf_hook", None), "offload", False):
                    raise ValueError("Staged sampling cannot be combined with Accelerate CPU/disk offload hooks")
                device_map = getattr(child, "hf_device_map", {}) or {}
                if "disk" in device_map.values() or len(set(map(str, device_map.values()))) > 1:
                    raise ValueError("Staged sampling cannot be combined with a split device_map")
            if any(hasattr(child, "_memory_manager") for child in module.modules()):
                self.layer_residency[module] = LayerOffloadResidency(module)
                self.states.append((module, None))
                continue
            devices = {t.device for t in chain(module.parameters(), module.buffers())}
            if len(devices) > 1:
                raise ValueError(f"Staged sampling requires single-device components; unmanaged split placement in {type(module).__name__} is unsupported")
            self.states.append((module, next(iter(devices), torch.device("cpu"))))
        self.training = {m: m.training for root in roots.values() for m in root.modules()}
        self.requires_grad = {p: p.requires_grad for root in roots.values() for p in root.parameters()}
        self.network_state = [
            (network, {key: getattr(network, key) for key in ("is_active", "multiplier", "torch_multiplier")
                       if hasattr(network, key)})
            for network in (model.network, model.assistant_lora)
            if network is not None
        ]
        self.cleanups = []
        self.preparing = False
        self.phase = None
        self.phase_started = None
        self.rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def __enter__(self):
        try:
            self.stage("idle")
            for module, _ in self.states:
                module.eval()
            configure = getattr(self.model, "configure_staged_sampling", None)
            if configure is not None:
                configure(self)
            hook = getattr(self.model, "sample_step_hook", None)
            if hook is not None and getattr(hook, "requires_vae", True):
                def preview(*args, **kwargs):
                    previous_phase = self.phase
                    try:
                        self.stage("vae")
                        return hook(*args, **kwargs)
                    finally:
                        self.stage(previous_phase)

                self.model.sample_step_hook = preview
                self.track_cleanup(lambda: setattr(self.model, "sample_step_hook", hook))
        except BaseException:
            self.restore()
            raise
        return self

    def progress(self, phase, completed, total):
        hook = getattr(self.model, "sample_progress_hook", None)
        if hook is not None:
            hook(phase, completed, total)

    def track_cleanup(self, callback):
        called = False

        def once():
            nonlocal called
            if not called:
                called = True
                callback()

        self.cleanups.append(once)
        return once

    def wrap_method(self, module, name, phase):
        original = getattr(module, name)
        had_override = name in module.__dict__

        @wraps(original)
        def call(*args, **kwargs):
            self.stage(phase)
            return original(*args, **kwargs)

        setattr(module, name, call)

        def restore_method():
            if had_override:
                setattr(module, name, original)
            else:
                delattr(module, name)

        self.track_cleanup(restore_method)

    def before_forward(self, module, phase):
        handle = module.register_forward_pre_hook(lambda *_: self.stage(phase), prepend=True)
        self.track_cleanup(handle.remove)

    def move(self, module, device):
        if module in self.layer_residency:
            self.layer_residency[module].move(device)
        else:
            module.to(device)
        # LoRA keeps its batch multiplier outside registered parameters/buffers.
        multiplier = getattr(module, "torch_multiplier", None)
        if isinstance(multiplier, torch.Tensor) and device is not None:
            module.torch_multiplier = multiplier.to(device)

    def stage(self, phase):
        if phase == self.phase:
            return
        if phase != "idle" and phase not in self.groups:
            raise ValueError(f"Unknown sampling phase: {phase}")
        if self.phase_started is not None:
            print_acc(f"[Sampling stages] {self.phase}: {perf_counter() - self.phase_started:.2f}s")
        active = {id(m) for m in self.groups.get(phase, []) if m is not None}
        if self.layer_residency and torch.cuda.is_initialized():
            # Layer offloading has asynchronous weight/gradient transfers.
            torch.cuda.synchronize(self.model.device_torch)
        # Release the outgoing component before allocating the incoming one.
        for module, _ in self.states:
            if id(module) not in active:
                self.move(module, "cpu")
        flush()
        for module, _ in self.states:
            if id(module) in active:
                self.move(module, self.devices[phase])
        self.phase = phase
        self.phase_started = perf_counter() if phase != "idle" else None
        if phase != "idle":
            print_acc(f"[Sampling stages] {phase}")

    def restore(self):
        try:
            for cleanup in reversed(self.cleanups):
                cleanup()
            self.stage("idle")
            for module, device in self.states:
                self.move(module, device)
        finally:
            for parameter, requires_grad in self.requires_grad.items():
                parameter.requires_grad_(requires_grad)
            # Set each flag directly so parent.train() cannot overwrite mixed states.
            for module, training in self.training.items():
                module.training = training
            for network, state in self.network_state:
                for key, value in state.items():
                    setattr(network, key, value)
            torch.set_rng_state(self.rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state_all(self.cuda_rng)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def staged_sampling(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        if not getattr(model.model_config, "sample_staged_offloading", False):
            return function(model, *args, **kwargs)
        if not getattr(model, "supports_staged_sampling", False):
            raise ValueError("This model does not support sample_staged_offloading")
        previous = getattr(model, "_staged_sampling", None)
        if previous is not None:
            raise RuntimeError("Nested staged sampling is not supported")
        session = StagedSampling(model)
        model._staged_sampling = session
        try:
            with session:
                return function(model, *args, **kwargs)
        finally:
            del model._staged_sampling

    return wrapped
