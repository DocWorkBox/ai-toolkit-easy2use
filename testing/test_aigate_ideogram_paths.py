from pathlib import Path


def test_aigate_ideogram4_uses_local_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )

    assert "'config.process[0].model.name_or_path': ['/model/ModelScope/ideogram-ai/ideogram-4-fp8', defaultNameOrPath]" in options_source
    assert 'QWEN3_VL_PATH = "/model/ModelScope/Qwen/Qwen3-VL-8B-Instruct"' in ideogram_source


def test_compshare_model_paths_do_not_use_aigate_dataset_roots():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    flux2_klein_source = Path(
        "extensions_built_in/diffusion_models/flux2/flux2_klein_model.py"
    ).read_text(encoding="utf-8")
    wan22_source = Path("extensions_built_in/diffusion_models/wan22/wan22_14b_model.py").read_text(
        encoding="utf-8"
    )

    assert "/datasets/studio/huggingface/models" not in options_source
    assert 'flux2_vae_path: str = "ai-toolkit/flux2_vae"' in flux2_klein_source
    assert '_wan_vae_path = "ai-toolkit/wan2.1-vae"' in wan22_source
