from pathlib import Path


def test_aigate_ideogram4_uses_local_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )

    assert "'config.process[0].model.name_or_path': ['/datasets/studio/huggingface/models/ideogram-4-fp8', defaultNameOrPath]" in options_source
    assert "'/datasets/studio/huggingface/models/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors'" in options_source
    assert 'QWEN3_VL_PATH = "/datasets/ComfyUI/models/prompt_generator/Qwen3-VL-8B-Instruct"' in ideogram_source


def test_aigate_boogu_uses_local_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")

    assert "'config.process[0].model.name_or_path': ['/datasets/studio/huggingface/models/Boogu-Image-0.1-Base', defaultNameOrPath]" in options_source
    assert "'config.process[0].model.name_or_path': ['/datasets/studio/huggingface/models/Boogu-Image-0.1-Edit', defaultNameOrPath]" in options_source


def test_aigate_krea2_uses_local_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    simple_job_source = Path("ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")

    assert "'config.process[0].model.name_or_path': ['/datasets/studio/huggingface/models/Krea-2-Raw', defaultNameOrPath]" in options_source
    assert "'config.process[0].model.name_or_path': ['/datasets/studio/huggingface/models/Krea-2-Turbo', defaultNameOrPath]" in options_source
    assert "'/datasets/studio/huggingface/models/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors'" in options_source
    assert "Krea 2 Turbo（训练适配器）" in options_source
    assert "训练适配器路径" in simple_job_source
