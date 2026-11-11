"""Exercise component routing without importing the GPU training stack."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "toolkit/models/wan21/wan21.py"


class ReachedComponent(Exception):
    pass


def load_method(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Wan21")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def trainer(source, **overrides):
    config = dict(
        name_or_path=str(source), extras_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        te_name_or_path=None, vae_path=None, low_vram=True, split_model_over_gpus=False,
        assistant_lora_path=None, inference_lora_path=None, lora_path=None,
    )
    config.update(overrides)
    return SimpleNamespace(
        torch_dtype="bf16", model_config=SimpleNamespace(**config), _wan_vae_path=None,
        print_and_status_update=Mock(), component_load_kwargs=Mock(return_value={}),
        load_wan_transformer=Mock(return_value=SimpleNamespace(to=Mock())), device_torch="cuda",
    )


@pytest.mark.parametrize("single_file", [False, True])
def test_transformer_keeps_single_file_but_appends_subfolder_to_directory(tmp_path, single_file):
    source = tmp_path / ("wan.safetensors" if single_file else "checkpoint")
    source.touch() if single_file else source.mkdir()
    sd = trainer(source)
    sd.load_wan_transformer.side_effect = ReachedComponent
    method = load_method("load_model", {"os": os})
    with pytest.raises(ReachedComponent):
        method(sd)
    expected = source if single_file else source / "transformer"
    assert sd.load_wan_transformer.call_args.args[0] == str(expected)


def test_single_file_transformer_uses_the_explicit_config_source(tmp_path):
    source = tmp_path / "wan.safetensors"
    source.touch()
    sd = trainer(source)
    component = SimpleNamespace(load=Mock())
    method = load_method("load_wan_transformer", {"WanTransformer3DModel": component, "flush": Mock()})
    method(sd, str(source), subfolder="transformer")
    assert component.load.call_args.kwargs["config_path"] == sd.model_config.extras_name_or_path


def test_existing_directory_transformer_does_not_get_new_config_override():
    sd = trainer("checkpoint")
    component = SimpleNamespace(load=Mock())
    method = load_method("load_wan_transformer", {"WanTransformer3DModel": component, "flush": Mock()})
    method(sd, "checkpoint/transformer")
    assert "config_path" not in component.load.call_args.kwargs


def test_explicit_text_encoder_is_not_replaced_by_the_base_checkpoint(tmp_path):
    source = tmp_path / "checkpoint"
    (source / "text_encoder").mkdir(parents=True)
    sd = trainer(source, te_name_or_path="existing-umt5")
    encoder = SimpleNamespace(load_tokenizer=Mock(side_effect=ReachedComponent))
    method = load_method("load_model", {"os": os, "UMT5TextEncoder": encoder, "flush": Mock()})
    with pytest.raises(ReachedComponent):
        method(sd)
    encoder.load_tokenizer.assert_called_once_with("existing-umt5")


def test_single_file_vae_keeps_the_matching_config_source(tmp_path):
    source = tmp_path / "wan.safetensors"
    source.touch()
    sd = trainer(source, vae_path="wan2.2_vae.safetensors")
    encoder = SimpleNamespace(load_tokenizer=Mock(), load=Mock())
    vae = SimpleNamespace(load_model=Mock(side_effect=ReachedComponent))
    method = load_method("load_model", {
        "os": os, "UMT5TextEncoder": encoder, "flush": Mock(), "WanVAE": vae,
        "Wan21": SimpleNamespace(get_train_scheduler=Mock()),
    })
    with pytest.raises(ReachedComponent):
        method(sd)
    assert vae.load_model.call_args.args[0] == "wan2.2_vae.safetensors"
    assert vae.load_model.call_args.kwargs["config_path"] == sd.model_config.extras_name_or_path


def vae_class():
    source = ROOT / "toolkit/models/v2/vae/wan.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = {"AutoencoderKLWan": type("VAE", (), {}), "OstrisModelMixin": type("Mixin", (), {})}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[cls.name]


def test_wan_vae_converts_original_keys_but_leaves_diffusers_keys_unchanged():
    cls = vae_class()
    converted = {"encoder.conv_in.weight": object()}
    conversion = Mock(return_value=converted)
    module = SimpleNamespace(convert_wan_vae_to_diffusers=conversion)
    with patch.dict("sys.modules", {"diffusers.loaders.single_file_utils": module}):
        original = {"encoder.conv1.weight": object()}
        assert cls.convert_state_dict_on_load(original) is converted
        conversion.assert_called_once()
        native = {"encoder.conv_in.weight": object()}
        assert cls.convert_state_dict_on_load(native) is native


def test_wan22_vae_preserves_nested_blocks_without_wan21_flat_block_mapping():
    cls = vae_class()
    value = object()
    state = {
        "encoder.downsamples.1.downsamples.0.residual.0.gamma": value,
        "encoder.downsamples.1.downsamples.2.time_conv.weight": value,
        "decoder.upsamples.2.upsamples.0.shortcut.weight": value,
        "decoder.upsamples.2.upsamples.1.residual.2.weight": value,
        "decoder.upsamples.2.upsamples.3.resample.1.weight": value,
        "conv1.weight": value,
    }
    conversion = Mock(return_value={"quant_conv.weight": value})
    module = SimpleNamespace(convert_wan_vae_to_diffusers=conversion)
    with patch.dict("sys.modules", {"diffusers.loaders.single_file_utils": module}):
        assert cls.convert_state_dict_on_load(state) == {
            "encoder.down_blocks.1.resnets.0.norm1.gamma": value,
            "encoder.down_blocks.1.downsampler.time_conv.weight": value,
            "decoder.up_blocks.2.resnets.0.conv_shortcut.weight": value,
            "decoder.up_blocks.2.resnets.1.conv1.weight": value,
            "decoder.up_blocks.2.upsampler.resample.1.weight": value,
            "quant_conv.weight": value,
        }
        conversion.assert_called_once_with({"conv1.weight": value})
