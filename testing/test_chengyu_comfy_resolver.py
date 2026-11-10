"""The shared chengyu model mount is searched but never written to."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class ChengyuComfyResolverTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "_chengyu_resolver_test", ROOT / "toolkit/models/v2/resolver.py"
        )
        self.resolver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.resolver)
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="chengyu-models-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.project, self.shared, self.extra = (root / name for name in ("models", "shared", "bundle"))
        for path in (self.project, self.shared, self.extra):
            path.mkdir()
        for name, value in (("MODELS_PATH", self.project), ("CHENGYU_COMFY_MODELS_PATH", self.shared)):
            patcher = patch.object(self.resolver, name, str(value), create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.download = Mock(side_effect=self.download_file)
        patcher = patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=self.download)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def weight(self, root, relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weight")
        return str(path)

    def download_file(self, *, filename, local_dir, **kwargs):
        self.assertEqual(Path(local_dir), self.project)
        return self.weight(Path(local_dir), filename)

    def test_nested_shared_weight_is_read_without_download_or_writes(self):
        expected = self.weight(self.shared, "diffusion_models/Fibo/fibo2_dev_bf16.safetensors")
        result = self.resolver.resolve_comfy_file("diffusion_models/fibo2_dev_bf16.safetensors", "Comfy-Org/Fibo-2.0")
        self.assertEqual(Path(result), Path(expected))
        self.download.assert_not_called()
        self.assertEqual(list(self.project.iterdir()), [])

    def test_shared_precision_candidates_keep_requested_priority(self):
        files = {precision: self.weight(self.shared, f"text_encoders/qwen/qwen3vl_4b_{precision}.safetensors")
                 for precision in ("bf16", "int8_convrot")}
        candidates = [f"text_encoders/qwen3vl_4b_{precision}.safetensors" for precision in files]
        for qtype, expected in ((None, "bf16"), ("convrot8", "int8_convrot")):
            with self.subTest(qtype=qtype):
                result = self.resolver.resolve_comfy_candidates(candidates, "Comfy-Org/Qwen3-VL", qtype=qtype)
                self.assertEqual(Path(result), Path(files[expected]))
        self.download.assert_not_called()

    def test_explicit_bundle_and_project_cache_precede_shared_mount(self):
        relative = "vae/flux2-vae.safetensors"
        self.weight(self.shared, relative)
        bundle = self.weight(self.extra, relative)
        self.assertEqual(Path(self.resolver.resolve_comfy_file(relative, "Comfy-Org/Fibo-2.0", extra_roots=[str(self.extra)])), Path(bundle))
        cached = self.weight(self.project, relative)
        self.assertEqual(Path(self.resolver.resolve_comfy_file(relative, "Comfy-Org/Fibo-2.0", extra_roots=[str(self.extra)])), Path(cached))
        self.download.assert_not_called()

    def test_missing_weights_download_only_into_project_models(self):
        result = self.resolver.resolve_comfy_file("text_encoders/missing.safetensors", "org/repo")
        self.assertEqual(result, str(self.project / "text_encoders/missing.safetensors"))
        self.download.assert_called_once()
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_nested_project_cache_precedes_shared_exact_match(self):
        relative = "vae/flux2-vae.safetensors"
        self.weight(self.shared, relative)
        expected = self.weight(self.project, "vae/custom/flux2-vae.safetensors")
        result = self.resolver.resolve_comfy_file(relative, "Comfy-Org/Fibo-2.0")
        self.assertEqual(Path(result), Path(expected))
        self.download.assert_not_called()

    def test_missing_absolute_directory_is_not_treated_as_a_hub_repo(self):
        missing = (self.project / "not-downloaded-yet").as_posix()
        self.assertEqual(self.resolver.repo_id_from_name_or_path(missing, "org/repo"), "org/repo")

    def test_missing_explicit_local_weights_never_trigger_invalid_hub_requests(self):
        missing = (self.project / "pending.safetensors").as_posix()
        for name in ("resolve_component_file", "resolve_named_file", "resolve_lora_file"):
            with self.subTest(resolver=name), self.assertRaisesRegex(FileNotFoundError, "missing|not found|downloading"):
                kwargs = {"folder": "text_encoders"} if name == "resolve_component_file" else {}
                getattr(self.resolver, name)(missing, **kwargs)
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
