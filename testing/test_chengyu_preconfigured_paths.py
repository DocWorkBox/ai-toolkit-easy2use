import ast
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
STORE = "/poddata/ai-toolkit-models"


def constant(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing constant: {name}")


def test_backend_defaults_match_prefetched_creator_files():
    expected = {
        ("extensions_built_in/llm_models/qwen38.py", "DEFAULT_CHECKPOINT"):
            "text_encoders/qwen3.8_27b_w4a8.safetensors",
        ("extensions_built_in/audio_models/yue2/yue2_model.py", "DEFAULT_CHECKPOINT"):
            "checkpoints/yue2_3b_int8_convrot.safetensors",
        ("extensions_built_in/audio_models/yue2/src/tokenizer.py", "MERT_REPO"):
            "huggingface/m-a-p/MERT-v2-FullSong",
        ("extensions_built_in/audio_models/yue2/src/tokenizer.py", "SHEETSAGE_REPO"):
            "huggingface/m-a-p/SheetSage2",
        ("extensions_built_in/audio_models/yue2/src/tokenizer.py", "SHEETSAGE_FILE"):
            "audio_encoders/sheetsage2_bf16.safetensors",
        ("toolkit/audio/moss_music/__init__.py", "HF_REPO"):
            "huggingface/OpenMOSS-Team/MOSS-Music-8B-Instruct",
    }
    for (path, name), relative in expected.items():
        assert constant(path, name) == f"{STORE}/{relative}"


def test_inference_registry_uses_same_creator_files_as_training_ui():
    get_entry = runpy.run_path(str(ROOT / "toolkit/models/registry.py"))["get_arch_entry"]
    for arch, relative in {
        "zimage_l2p": "diffusion_models/model-1k-merge.safetensors",
        "ace_step_15": "checkpoints/ace_step_1.5_base_aio.safetensors",
        "yue2": "checkpoints/yue2_3b_int8_convrot.safetensors",
    }.items():
        assert get_entry(arch)["model"]["name_or_path"] == f"{STORE}/{relative}"
    assert get_entry("zimage_l2p")["model"]["extras_name_or_path"] == "/chenyudata/Other/Z-Image-Turbo"


class OmniCreatorCheckpointTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "extensions_built_in/captioner/Qwen3OmniCaptioner.py").read_text(encoding="utf-8"))
        constants = {"CONVROT_MODELS", "DEFAULT_CONVROT_MODEL"}
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniCaptioner")
        nodes.append(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_checkpoint"))
        namespace = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "omni-checkpoint", "exec"), namespace)
        self.resolve = namespace["_resolve_checkpoint"]
        self.models = namespace["CONVROT_MODELS"]
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="chengyu-omni-paths-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.download = Mock(side_effect=AssertionError("Explicit local paths must not call the Hub"))
        modules = {"toolkit.paths": SimpleNamespace(MODELS_PATH=str(self.root)),
                   "huggingface_hub": SimpleNamespace(hf_hub_download=self.download)}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_completed_local_huihui_file_keeps_thinking_configuration(self):
        info = self.models["ai-toolkit/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated"]
        path = self.root / info["filename"]
        path.write_bytes(b"weights")
        captioner = SimpleNamespace(caption_config=SimpleNamespace(model_name_or_path=str(path)))
        self.assertEqual(self.resolve(captioner), str(path))
        self.assertEqual(captioner._model_info, info)
        self.assertTrue(captioner._model_info["thinking"])
        self.download.assert_not_called()

    def test_pending_local_file_reports_download_state_without_using_path_as_repo(self):
        path = (self.root / "pending.safetensors").as_posix()
        captioner = SimpleNamespace(caption_config=SimpleNamespace(model_name_or_path=path),
                                    print_and_status_update=Mock())
        with self.assertRaisesRegex(FileNotFoundError, "missing|not found|downloading"):
            self.resolve(captioner)
        self.download.assert_not_called()


class L2PCreatorCheckpointTests(unittest.TestCase):
    def test_pending_local_file_is_not_used_as_a_download_repo(self):
        source = ROOT / "extensions_built_in/diffusion_models/z_image/z_image_l2p_model.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZImageL2PModel")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
        with tempfile.TemporaryDirectory(dir=".", prefix="chengyu-l2p-paths-") as temporary:
            root = Path(temporary).resolve()
            download = Mock(side_effect=AssertionError("Local pending weights must not request a Hub repo"))
            namespace = {"os": os, "MODELS_PATH": str(root), "HF_TOKEN": None,
                         "huggingface_hub": SimpleNamespace(hf_hub_download=download)}
            exec(compile(ast.Module(body=[method], type_ignores=[]), "l2p-local-path", "exec"), namespace)
            model = SimpleNamespace(torch_dtype=None, print_and_status_update=Mock(),
                                    model_config=SimpleNamespace(name_or_path=(root / "pending.safetensors").as_posix(),
                                                                 extras_name_or_path="/chenyudata/Other/Z-Image-Turbo"))
            with self.assertRaisesRegex(FileNotFoundError, "missing|not found|downloading"):
                namespace["load_model"](model)
            download.assert_not_called()

