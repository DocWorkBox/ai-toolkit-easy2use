import math
import os
import sys
import types

import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from toolkit.optimizer import get_optimizer
from toolkit.optimizers.singularity import Singularity


def _tiny_model(dtype=torch.float32):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.LayerNorm(3),
    ).to(dtype=dtype)


def _train_one_step(model, optimizer):
    x = torch.randn(2, 4, dtype=next(model.parameters()).dtype)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()


def test_singularity_default_lr_matches_lora_starting_point():
    param = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Singularity([param])

    assert optimizer.get_avg_learning_rate() == 1e-4


def test_train_config_uses_singularity_lr_default_only_when_lr_is_omitted():
    sys.modules.pop("toolkit.config_modules", None)
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    album_module = types.ModuleType("toolkit.audio.album_artwork")
    album_module.add_album_artwork = lambda *args, **kwargs: None
    sys.modules.setdefault("toolkit.audio.album_artwork", album_module)
    prompt_module = types.ModuleType("toolkit.prompt_utils")
    prompt_module.PromptEmbeds = object
    sys.modules.setdefault("toolkit.prompt_utils", prompt_module)
    torchao_module = types.ModuleType("torchao")
    torchao_quantization_module = types.ModuleType("torchao.quantization")
    torchao_primitives_module = types.ModuleType("torchao.quantization.quant_primitives")
    torchao_primitives_module._DTYPE_TO_BIT_WIDTH = {}
    sys.modules.setdefault("torchao", torchao_module)
    sys.modules.setdefault("torchao.quantization", torchao_quantization_module)
    sys.modules.setdefault("torchao.quantization.quant_primitives", torchao_primitives_module)

    from toolkit.config_modules import TrainConfig

    assert TrainConfig(optimizer="singularity").lr == 1e-4
    assert TrainConfig(optimizer="singularity", lr=1e-6).lr == 1e-6
    assert TrainConfig().lr == 1e-6


def test_singularity_is_registered_and_updates_params():
    model = _tiny_model()
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type="singularity",
        learning_rate=1e-4,
        optimizer_params={"fused": False},
    )

    before = [p.detach().clone() for p in model.parameters()]
    _train_one_step(model, optimizer)

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )
    lrs = optimizer.get_learning_rates()
    assert lrs
    assert all(math.isfinite(lr) and lr > 0 for lr in lrs)

    initialized_state = next(state for state in optimizer.state.values() if state)
    assert "prev_update" in initialized_state
    assert "pol_hist" not in initialized_state


def test_singularity_group_lr_experiment_uses_one_lr_per_tensor():
    model = _tiny_model()
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type="singularity_group",
        learning_rate=1e-4,
        optimizer_params={"fused": False},
    )

    _train_one_step(model, optimizer)

    matrix_state = next(
        optimizer.state[p]
        for p in model.parameters()
        if p.dim() >= 2 and "lr" in optimizer.state[p]
    )
    assert matrix_state["lr"].ndim == 0
    assert optimizer.get_learning_rates()


def test_singularity_default_keeps_per_row_learning_rates():
    param = torch.nn.Parameter(torch.ones(3, 4))
    optimizer = Singularity([param], fused=False)

    param.sum().backward()
    optimizer.step()

    assert optimizer.state[param]["lr"].shape == (3,)


def test_singularity_keeps_compact_low_precision_state_across_roundtrip():
    model = _tiny_model()
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type="singularity",
        learning_rate=1e-4,
        optimizer_params={"fused": False, "state_dtype": "float16"},
    )
    _train_one_step(model, optimizer)

    restored_model = _tiny_model()
    restored = get_optimizer(
        restored_model.parameters(),
        optimizer_type="singularity",
        learning_rate=1e-4,
        optimizer_params={"fused": False, "state_dtype": "float16"},
    )
    restored.load_state_dict(optimizer.state_dict())

    assert restored.get_learning_rates()
    for state in restored.state.values():
        if "prev_update" in state:
            assert state["prev_update"].dtype == torch.float16


def test_singularity_fused_mode_updates_during_backward_and_releases_grad():
    param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, -0.25]]))
    optimizer = get_optimizer(
        [param],
        optimizer_type="singularity",
        learning_rate=1e-3,
        optimizer_params={"fused": True},
    )

    before = param.detach().clone()
    param.pow(2).sum().backward()

    assert param.grad is None
    assert not torch.equal(before, param.detach())
    optimizer.step()
