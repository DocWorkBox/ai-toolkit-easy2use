from pathlib import Path


def test_main_ideogram4_uses_repo_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )
    upsample_source = Path("ui_scripts/upsample_ideogram4_caption.py").read_text(encoding="utf-8")

    assert "'config.process[0].model.name_or_path': ['ideogram-ai/ideogram-4-fp8', defaultNameOrPath]" in options_source
    assert (
        "'ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors'"
        in options_source
    )
    assert 'QWEN3_VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"' in ideogram_source
    assert 'default="Qwen/Qwen3-VL-8B-Instruct"' in upsample_source


def test_main_boogu_keeps_repo_defaults():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")

    assert "'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Base', defaultNameOrPath]" in options_source
    assert "'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Edit', defaultNameOrPath]" in options_source


def test_main_model_defaults_do_not_use_branch_local_roots():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )
    upsample_source = Path("ui_scripts/upsample_ideogram4_caption.py").read_text(encoding="utf-8")

    assert "/datasets/studio/huggingface/models" not in options_source
    assert "/datasets/ComfyUI/models/prompt_generator" not in options_source
    assert "/model/ModelScope" not in options_source
    assert "/model/HuggingFace" not in options_source
    assert "/datasets/ComfyUI/models/prompt_generator" not in ideogram_source
    assert "/model/ModelScope" not in ideogram_source
    assert "/datasets/ComfyUI/models/prompt_generator" not in upsample_source
    assert "/model/ModelScope" not in upsample_source


def test_main_krea2_keeps_repo_defaults():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    simple_job_source = Path("ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")
    krea2_source = Path("extensions_built_in/diffusion_models/krea2/krea2.py").read_text(
        encoding="utf-8"
    )

    assert "'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', defaultNameOrPath]" in options_source
    assert "'config.process[0].model.name_or_path': ['krea/Krea-2-Turbo', defaultNameOrPath]" in options_source
    assert "'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors'" in options_source
    assert 'QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"' in krea2_source
    assert 'QWEN_IMAGE_VAE_PATH = "Qwen/Qwen-Image"' in krea2_source
    assert "Krea 2 Raw" in options_source
    assert "Krea 2 Turbo" in options_source
    assert "model.assistant_lora_path" in simple_job_source
