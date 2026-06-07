from pathlib import Path


def test_aigate_ideogram4_uses_local_model_paths():
    options_source = Path("ui/src/app/jobs/new/options.ts").read_text(encoding="utf-8")
    ideogram_source = Path("extensions_built_in/diffusion_models/ideogram4/ideogram4.py").read_text(
        encoding="utf-8"
    )

    assert "'config.process[0].model.name_or_path': ['/model/ModelScope/ideogram-ai/ideogram-4-fp8', defaultNameOrPath]" in options_source
    assert 'QWEN3_VL_PATH = "/model/ModelScope/Qwen/Qwen3-VL-8B-Instruct"' in ideogram_source
