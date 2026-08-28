from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_protable_branch_keeps_local_model_defaults():
    krea = _read("extensions_built_in/diffusion_models/krea2/krea2.py")
    options = _read("ui/src/app/jobs/new/options.tsx")
    captioners = _read("ui/src/helpers/captionOptions.ts")
    upsampler = _read("ui_scripts/upsample_ideogram4_caption.py")

    assert 'QWEN3_VL_PATH = "./models/Qwen3-VL-4B-Instruct"' in krea
    assert 'QWEN_IMAGE_VAE_PATH = "./models/Qwen-Image"' in krea
    assert "'./models/Krea-2-Raw'" in options
    assert "'./models/Krea-2-Turbo'" in options
    assert "'./models/krea2_turbo_training_adapter/" in options
    assert (
        "'./models/minimax_h3_training_adapter/"
        "minimax_h3_training_adapter_v1.safetensors'"
    ) in options
    assert "'ostris/minimax_h3_training_adapter/" not in options
    assert "'./models/Qwen3-VL-8B-Instruct'" in captioners
    assert "'./models/Qwen3.6-27B'" in captioners
    assert "'./models/Qwen2.5-Omni-7B-H3-Prompt-Rewriter'" in captioners
    assert "'./models/text_encoders/qwen3_omni_30b_a3b_thinking_convrot8.safetensors'" in captioners
    assert (
        "'./models/diffusion_models/"
        "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors'"
        in options
    )
    assert 'default="./models/Qwen3-VL-8B-Instruct"' in upsampler

    assert "'./models/Boogu-Image-0.1-Base'" in options
    assert "'./models/Boogu-Image-0.1-Edit'" in options


def test_protable_branch_contains_portable_launch_assets():
    launcher = ROOT / "AI Toolkit Launcher.exe"
    assert launcher.is_file()
    assert launcher.read_bytes()[:2] == b"MZ"
    assert launcher.stat().st_size < 100 * 1024 * 1024
    assert (ROOT / "start.bat").is_file()
    assert (ROOT / "scripts/portable/nvidia_smi.cmd").is_file()
    assert (ROOT / "scripts/portable/run_portable_supervisor.ps1").is_file()


def test_protable_ui_build_does_not_require_google_fonts():
    layout = _read("ui/src/app/layout.tsx")

    assert "next/font/google" not in layout
    assert "Inter(" not in layout
