from pathlib import Path

UI_SOURCE = Path("extensions_built_in/diffusion_models/ui.tsx")


def test_aigate_ideogram4_uses_local_model_paths():
    options_source = UI_SOURCE.read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )

    assert '"/datasets/studio/huggingface/models/ideogram-4-fp8"' in options_source
    assert '"/datasets/studio/huggingface/models/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors"' in options_source
    assert 'QWEN3_VL_PATH = "/datasets/ComfyUI/models/prompt_generator/Qwen3-VL-8B-Instruct"' in ideogram_source


def test_aigate_boogu_uses_local_model_paths():
    options_source = UI_SOURCE.read_text(encoding="utf-8")

    assert '"/datasets/studio/huggingface/models/Boogu-Image-0.1-Base"' in options_source
    assert '"/datasets/studio/huggingface/models/Boogu-Image-0.1-Edit"' in options_source


def test_aigate_krea2_uses_local_model_paths():
    options_source = UI_SOURCE.read_text(encoding="utf-8")
    simple_job_source = Path("ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")
    krea2_source = Path("extensions_built_in/diffusion_models/krea2/krea2.py").read_text(encoding="utf-8")

    assert options_source.count('"/datasets/studio/huggingface/models/Krea-2-Raw"') >= 2
    assert options_source.count('"/datasets/studio/huggingface/models/Krea-2-Turbo"') >= 2
    assert options_source.count('"/datasets/studio/huggingface/models/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors"') >= 2
    assert 'name: "krea2:o_edit"' in options_source
    assert 'name: "krea2:o_edit_turbo"' in options_source
    assert '"krea/Krea-2-Raw"' not in options_source
    assert '"krea/Krea-2-Turbo"' not in options_source
    assert '"ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors"' not in options_source
    assert 'QWEN3_VL_PATH = "/datasets/ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct"' in krea2_source
    assert 'QWEN_IMAGE_VAE_PATH = "/datasets/ai-toolkit/models/Qwen-Image"' in krea2_source
    assert 'QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"' not in krea2_source
    assert 'QWEN_IMAGE_VAE_PATH = "Qwen/Qwen-Image"' not in krea2_source
    assert "Krea 2 Raw" in options_source
    assert "Krea 2 Turbo（训练适配器）" in options_source
    assert "训练适配器路径" in simple_job_source
