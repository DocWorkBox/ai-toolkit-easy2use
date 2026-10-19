"""MOSS small assets must not be written into read-only model mounts."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_moss_asset_cache_tests"


def load_model_module():
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "extensions_built_in/diffusion_models/moss_image")]
    sys.modules[PACKAGE] = package
    backend = ModuleType("toolkit.models.v2.text_encoders.qwen35")
    backend.Qwen35TextOnlyEncoder = type("TextEncoder", (), {"load_model": Mock()})
    backend.Qwen35VLEncoder = type("VisionEncoder", (), {"load_model": Mock()})
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.moss_image", Path(package.__path__[0]) / "moss_image.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The installed transformers is older than Qwen3.5; only mock weight loading.
    previous = sys.modules.get(backend.__name__)
    sys.modules[backend.__name__] = backend
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(backend.__name__, None)
        else:
            sys.modules[backend.__name__] = previous
    return module


class MossImageAssetCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_module = load_model_module()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=".", prefix="moss-assets-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "toolkit"
        self.hub = self.root / "hub"
        self.calls = []
        self.patch(patch.dict(
            self.model_module._hub_file_folder.__globals__,
            {"TOOLKIT_ROOT": str(self.project)},
        ))
        self.download = self.patch(patch("huggingface_hub.hf_hub_download", side_effect=self.hub_download))
        self.snapshot = self.patch(patch("huggingface_hub.snapshot_download", side_effect=self.snapshot_download))

    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def write(self, path, content=b"asset"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def project_repo(self, repo):
        return self.project / "models" / repo

    def hub_download(self, repo, filename, **kwargs):
        self.calls.append((repo, filename, kwargs))
        cached = self.hub / repo / filename
        if kwargs.get("local_files_only"):
            if not cached.is_file():
                raise FileNotFoundError(cached)
            return str(cached)
        root = Path(kwargs.get("local_dir", self.hub / repo))
        return self.write(root / filename)

    def snapshot_download(self, repo, **kwargs):
        root = self.hub / repo
        if kwargs.get("local_files_only") and not root.is_dir():
            raise FileNotFoundError(root)
        if not kwargs.get("local_files_only"):
            for filename in kwargs["allow_patterns"]:
                self.write(root / filename)
        return str(root)

    def test_config_is_downloaded_under_project_models(self):
        repo = "Qwen/Qwen3.5-9B"
        folder = self.model_module._hub_file_folder(repo, "config.json")
        self.assertEqual(Path(folder), self.project_repo(repo))
        self.assertEqual((Path(folder) / "config.json").read_bytes(), b"asset")
        online = [call for call in self.calls if not call[2].get("local_files_only")]
        self.assertEqual(len(online), 1)
        self.assertEqual(Path(online[0][2]["local_dir"]), self.project_repo(repo))

    def test_project_cache_is_reused_without_hub_access(self):
        repo = "Qwen/Qwen3.5-9B"
        self.write(self.project_repo(repo) / "config.json", b"offline config")
        folder = self.model_module._hub_file_folder(repo, "config.json")
        self.assertEqual(Path(folder), self.project_repo(repo))
        self.download.assert_not_called()
        self.snapshot.assert_not_called()

    def test_hub_cache_is_copied_without_online_download_or_source_changes(self):
        repo = "Qwen/Qwen3.5-9B"
        source = self.hub / repo / "config.json"
        self.write(source, b"cached config")
        before = source.stat().st_mtime_ns
        folder = self.model_module._hub_file_folder(repo, "config.json")
        self.assertEqual(Path(folder), self.project_repo(repo))
        self.assertEqual((Path(folder) / "config.json").read_bytes(), b"cached config")
        self.assertEqual(source.read_bytes(), b"cached config")
        self.assertEqual(source.stat().st_mtime_ns, before)
        self.assertTrue(all(call[2].get("local_files_only") for call in self.calls))

    def test_tokenizer_only_downloads_missing_files_and_isolates_repos(self):
        repos = ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base")
        for repo in repos:
            expected = self.project_repo(repo) / "tokenizer"
            self.write(expected / "tokenizer.json", b"existing tokenizer")
            self.write(expected / "tokenizer_config.json", b"existing config")
            folder = self.model_module._hub_tokenizer_folder(repo)
            self.assertEqual(Path(folder), expected)
            self.assertEqual((expected / "tokenizer.json").read_bytes(), b"existing tokenizer")
            self.assertTrue((expected / "chat_template.jinja").is_file())
        online = [(repo, name) for repo, name, kw in self.calls if not kw.get("local_files_only")]
        self.assertEqual(online, [(repo, "tokenizer/chat_template.jinja") for repo in repos])

    def test_fresh_tokenizer_downloads_only_the_three_required_assets(self):
        repo = "OpenMOSS-Team/MOSS-Image-SFT"
        folder = Path(self.model_module._hub_tokenizer_folder(repo))
        files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        self.assertEqual(folder, self.project_repo(repo) / "tokenizer")
        self.assertEqual({path.name for path in folder.iterdir()}, set(files))
        self.assertEqual(
            {name for _, name, _ in self.calls},
            {f"tokenizer/{name}" for name in files},
        )

    def test_empty_project_file_is_repaired(self):
        repo = "Qwen/Qwen3.5-9B"
        target = self.project_repo(repo) / "preprocessor_config.json"
        self.write(target, b"")
        self.model_module._hub_file_folder(repo, "preprocessor_config.json")
        self.assertEqual(target.read_bytes(), b"asset")

    def test_download_failure_does_not_return_an_incomplete_cache(self):
        self.download.side_effect = OSError("offline")
        with self.assertRaisesRegex(OSError, "offline"):
            self.model_module._hub_file_folder("Qwen/Qwen3.5-9B", "config.json")

    def test_empty_download_is_rejected(self):
        source = self.root / "empty.json"
        self.write(source, b"")
        self.download.side_effect = lambda *args, **kwargs: str(source)
        with self.assertRaisesRegex(FileNotFoundError, "missing or empty"):
            self.model_module._hub_file_folder("Qwen/Qwen3.5-9B", "config.json")

    def test_small_asset_loader_rejects_weight_downloads_and_invalid_repo_ids(self):
        for repo, filename in (
            ("../outside", "config.json"),
            ("Qwen/Qwen3.5-9B", "model.safetensors"),
        ):
            with self.subTest(repo=repo, filename=filename), self.assertRaises(ValueError):
                self.model_module._hub_file_folder(repo, filename)
        self.download.assert_not_called()

    def test_single_file_encoder_uses_project_config_and_keeps_weight_path(self):
        module = self.model_module
        weight = self.root / "read-only-models" / "qwen3.5_9b.safetensors"
        self.write(weight, b"weight placeholder")
        for edit in (False, True):
            with self.subTest(edit=edit):
                model = module.MossImageModel.__new__(module.MossImageModel)
                model.model_config = SimpleNamespace(te_name_or_path=None, model_kwargs={})
                model.is_edit = edit
                model.component_load_kwargs = Mock(return_value={
                    "use_comfy_weights": True, "qtype": "convrot8", "dtype": "bf16",
                })
                model.print_and_status_update = Mock()
                encoder = Mock()
                backend = module.Qwen35VLEncoder if edit else module.Qwen35TextOnlyEncoder
                with patch.object(module, "comfy_text_encoder_file", return_value=str(weight)):
                    with patch.object(backend, "load_model", return_value=encoder) as load:
                        with patch.object(module.AutoImageProcessor, "from_pretrained") as processor:
                            self.assertIs(model._load_text_encoder(), encoder)
                self.assertEqual(load.call_args.args[0], str(weight))
                self.assertEqual(
                    Path(load.call_args.kwargs["config_path"]),
                    self.project_repo("Qwen/Qwen3.5-9B"),
                )
                if edit:
                    processor.assert_called_once_with(str(self.project_repo("Qwen/Qwen3.5-9B")))
                    self.assertTrue((self.project_repo("Qwen/Qwen3.5-9B") / "preprocessor_config.json").is_file())
                else:
                    processor.assert_not_called()

    def test_explicit_local_tokenizer_keeps_priority(self):
        module = self.model_module
        local = self.root / "user-model"
        (local / "tokenizer").mkdir(parents=True)
        model = module.MossImageModel.__new__(module.MossImageModel)
        model.model_config = SimpleNamespace(name_or_path=str(local))
        tokenizer = MagicMock(chat_template="template")
        tokenizer.__len__.return_value = 150000
        with patch.object(module.Qwen2Tokenizer, "from_pretrained", return_value=tokenizer) as load:
            self.assertIs(model._load_tokenizer(), tokenizer)
        load.assert_called_once_with(str(local / "tokenizer"))
        self.download.assert_not_called()
        self.snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
