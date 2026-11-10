"""Creator models use final local paths and writable small-asset caches."""

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
QWEN38_CHECKPOINT = (
    "/poddata/ai-toolkit-models/"
    "text_encoders/qwen3.8_27b_w4a8.safetensors"
)


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def model_block(path, name):
    text = source(path)
    match = re.search(r'\bname:\s*[\'\"]' + re.escape(name) + r'[\'\"]', text)
    if match is None:
        raise AssertionError(f"Missing model option: {name}")
    return re.split(r'\bname:\s*[\'\"]', text[match.end():], maxsplit=1)[0]


def selected_default(block, key):
    match = re.search(
        r'[\'\"]' + re.escape(key) + r'[\'\"]:\s*\[\s*[\'\"]([^\'\"]+)[\'\"]',
        block,
    )
    if match is None:
        raise AssertionError(f"Missing default: {key}")
    return match.group(1)


class Chengyu119ModelPathTests(unittest.TestCase):
    def test_ph3_defaults_to_creator_model_root(self):
        block = model_block("extensions_built_in/diffusion_models/ui.tsx", "p_h3_image")
        self.assertEqual(
            selected_default(block, "config.process[0].model.name_or_path"),
            "/poddata/ai-toolkit-models",
        )
        self.assertIn("p_h3_image_training_adapter_v1.safetensors", block)

    def test_fasth3_omni_defaults_to_creator_model_root(self):
        block = model_block("extensions_built_in/diffusion_models/ui.tsx", "minimax_h3_omni")
        self.assertEqual(
            selected_default(block, "config.process[0].model.name_or_path"),
            "/poddata/ai-toolkit-models",
        )
        self.assertIn("fastvideo_fasth3_omni_training_adapter_v1.safetensors", block)

    def test_qwen38_training_captioning_and_backend_share_local_w4a8_file(self):
        train = model_block("extensions_built_in/llm_models/ui.tsx", "qwen38")
        caption = model_block("ui/src/helpers/captionOptions.ts", "Qwen38Captioner")
        self.assertEqual(selected_default(train, "config.process[0].model.name_or_path"), QWEN38_CHECKPOINT)
        self.assertEqual(selected_default(caption, "config.process[0].caption.model_name_or_path"), QWEN38_CHECKPOINT)
        self.assertIn(f"value: '{QWEN38_CHECKPOINT}'", caption)
        tree = ast.parse(source("extensions_built_in/llm_models/qwen38.py"))
        defaults = {target.id: node.value.value for node in tree.body
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    for target in node.targets if isinstance(target, ast.Name)}
        self.assertEqual(defaults["DEFAULT_CHECKPOINT"], QWEN38_CHECKPOINT)
        self.assertEqual(defaults["BASE_REPO"], "Qwen/Qwen3.8-27B")
        self.assertIn("value: 'Qwen/Qwen3.8-27B'", caption)

    def test_original_config_repos_and_project_asset_caches_remain_separate(self):
        ph3 = source("extensions_built_in/diffusion_models/minimax_h3/p_h3_image.py")
        self.assertIn('PH3_ORIGINAL_REPO = "PrunaAI/P-H3-Image"', ph3)
        self.assertIn("resolve_ph3_assets(PH3_ORIGINAL_REPO", ph3)
        for path in (
            "extensions_built_in/diffusion_models/minimax_h3/src/ph3_assets.py",
            "extensions_built_in/llm_models/src/qwen38_assets.py",
        ):
            text = source(path)
            self.assertIn("TOOLKIT_ROOT", text)
            self.assertNotIn("/model/ModelScope/", text)
            self.assertNotIn("/model/HuggingFace/", text)


if __name__ == "__main__":
    unittest.main()
