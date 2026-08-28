import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "portable_models.json"


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_uses_local_paths_and_official_download_links():
    catalog = _catalog()

    assert catalog["schema"] == 1
    assert len(catalog["models"]) >= 50
    assert len({item["id"] for item in catalog["models"]}) == len(catalog["models"])

    for item in catalog["models"]:
        if item.get("root") == "configured_models":
            assert not item["path"].startswith(("/", "./models/")), item
        else:
            assert item["path"].startswith("./models/"), item
        assert "\\" not in item["path"], item
        assert item["kind"] in {"directory", "file"}, item
        assert item["download_url"].startswith(
            ("https://huggingface.co/", "https://modelscope.cn/models/")
        ), item


def test_catalog_defines_special_component_layouts():
    by_id = {item["id"]: item for item in _catalog()["models"]}

    assert by_id["flux2-dev"]["path"] == "./models/FLUX.2-dev"
    assert by_id["flux2-dev"]["required_all"] == ["flux2-dev.safetensors"]
    assert by_id["flux2-mistral"]["path"] == "./models/Mistral-Small-3.1-24B-Instruct-2503"
    assert by_id["flux2-vae"]["path"] == "./models/flux2_vae/ae.safetensors"
    assert by_id["flux2-klein-4b-text-encoder"]["path"] == "./models/Qwen3-4B"
    assert by_id["flux2-klein-9b-text-encoder"]["path"] == "./models/Qwen3-8B"
    assert by_id["ltx2-text-encoder"]["path"] == "./models/gemma-3-12b-it-qat-q4_0-unquantized"
    assert by_id["ltx25-transformer"]["path"] == (
        "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
    )
    assert by_id["ltx25-text-encoder"]["path"] == (
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    )
    assert by_id["ltx25-video-vae"]["path"] == (
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    )
    assert by_id["ltx25-audio-vae"]["path"] == (
        "vae/ltx-2.5-audio-vae-bf16.safetensors"
    )
    assert by_id["wan-umt5"]["path"] == "./models/umt5_xxl_encoder"
    assert by_id["wan-vae"]["path"] == "./models/wan2.1-vae"
    assert by_id["qwen-image"]["path"] == "./models/Qwen-Image"


def test_catalog_groups_related_models_and_scans_minimax_from_models_path():
    by_id = {item["id"]: item for item in _catalog()["models"]}

    assert by_id["flux2-dev"]["family"] == "FLUX.2"
    assert by_id["flux2-mistral"]["family"] == "FLUX.2"
    assert by_id["flux2-vae"]["family"] == "FLUX.2"
    assert by_id["krea2-raw"]["family"] == "Krea 2"
    assert by_id["qwen-image-vae"]["family"] == "Krea 2"
    assert by_id["krea2-turbo-adapter"]["family"] == "Krea 2"

    minimax_weight_ids = {
        "minimax-h3-fl2va",
        "minimax-h3-ref2va",
        "minimax-h3-text-encoder",
        "minimax-h3-video-vae",
        "minimax-h3-audio-vae",
    }
    minimax_weights = [by_id[item_id] for item_id in minimax_weight_ids]
    assert {item["family"] for item in minimax_weights} == {"MiniMax-H3"}
    assert {item["root"] for item in minimax_weights} == {"configured_models"}
    assert {
        item["path"] for item in minimax_weights
    } == {
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    }

    metadata = {
        item_id: by_id[item_id]
        for item_id in {
            "minimax-h3-tokenizer",
            "minimax-h3-processor",
            "minimax-h3-text-encoder-config",
        }
    }
    assert {item["family"] for item in metadata.values()} == {"MiniMax-H3"}
    assert all("root" not in item for item in metadata.values())
    assert metadata["minimax-h3-tokenizer"]["path"] == (
        "./models/MiniMax-H3/FL2VA/tokenizer"
    )
    assert metadata["minimax-h3-tokenizer"]["required_all"] == [
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    assert metadata["minimax-h3-processor"]["path"] == (
        "./models/MiniMax-H3/FL2VA/processor"
    )
    assert metadata["minimax-h3-processor"]["required_all"] == [
        "chat_template.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ]
    assert metadata["minimax-h3-text-encoder-config"]["path"] == (
        "./models/MiniMax-H3/FL2VA/text_encoder"
    )
    assert metadata["minimax-h3-text-encoder-config"]["required_all"] == [
        "config.json"
    ]
    adapter = by_id["minimax-h3-training-adapter"]
    assert adapter["family"] == "MiniMax-H3"
    assert adapter["path"] == (
        "./models/minimax_h3_training_adapter/"
        "minimax_h3_training_adapter_alpha.safetensors"
    )
    adapter_v1 = by_id["minimax-h3-training-adapter-v1"]
    assert adapter_v1["family"] == "MiniMax-H3"
    assert adapter_v1["path"] == (
        "./models/minimax_h3_training_adapter/"
        "minimax_h3_training_adapter_v1.safetensors"
    )
    ref2va_adapter_v1 = by_id["minimax-h3-ref2va-training-adapter-v1"]
    assert ref2va_adapter_v1["family"] == "MiniMax-H3"
    assert ref2va_adapter_v1["path"] == (
        "./models/minimax_h3_training_adapter/"
        "minimax_h3_ref2va_training_adapter_v1.safetensors"
    )


def test_catalog_covers_qwen3_omni_checkpoints_and_local_metadata():
    by_id = {item["id"]: item for item in _catalog()["models"]}

    qwen25 = by_id["qwen25-omni-h3-prompt-rewriter"]
    assert qwen25["family"] == "Qwen2.5-Omni"
    assert qwen25["path"] == "./models/Qwen2.5-Omni-7B-H3-Prompt-Rewriter"
    assert qwen25["download_url"] == (
        "https://modelscope.cn/models/"
        "zhaoke1006/Qwen2.5-Omni-7B-H3-Prompt-Rewriter"
    )

    checkpoint_paths = {
        by_id["qwen3-omni-instruct"]["path"],
        by_id["qwen3-omni-thinking"]["path"],
        by_id["qwen3-omni-thinking-abliterated"]["path"],
    }
    assert checkpoint_paths == {
        "./models/text_encoders/qwen3_omni_30b_a3b_instruct_thinker_convrot8.safetensors",
        "./models/text_encoders/qwen3_omni_30b_a3b_thinking_convrot8.safetensors",
        "./models/text_encoders/huihui_qwen3_omni_30b_a3b_thinking_abliterated_convrot8.safetensors",
    }
    assert {
        by_id["qwen3-omni-instruct-metadata"]["path"],
        by_id["qwen3-omni-thinking-metadata"]["path"],
    } == {
        "./models/Qwen3-Omni-30B-A3B-Instruct",
        "./models/Qwen3-Omni-30B-A3B-Thinking",
    }
    assert {
        by_id["qwen3-omni-instruct"]["family"],
        by_id["qwen3-omni-thinking"]["family"],
        by_id["qwen3-omni-thinking-abliterated"]["family"],
        by_id["qwen3-omni-instruct-metadata"]["family"],
        by_id["qwen3-omni-thinking-metadata"]["family"],
    } == {"Qwen3-Omni"}


def test_minimax_h3_is_the_only_remote_training_default():
    options = _read("ui/src/app/jobs/new/options.tsx")
    values = re.findall(
        r"'config\.process\[0\]\.model\.(?:name_or_path|extras_name_or_path|assistant_lora_path|unconditional_lora_path)'\s*:\s*\[\s*'([^']+)'",
        options,
    )

    remote_values = [value for value in values if not value.startswith("./models/")]
    assert remote_values
    assert set(remote_values) == {"Comfy-Org/MiniMax-H3"}

    minimax = _read("extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py")
    assert 'COMFY_REPO = "Comfy-Org/MiniMax-H3"' in minimax
    assert 'ORIGINAL_REPO = "MiniMaxAI/MiniMax-H3"' in minimax


def test_caption_and_runtime_component_defaults_are_local():
    caption_options = _read("ui/src/helpers/captionOptions.ts")
    caption_job = _read("ui/src/helpers/captionJobConfig.ts")
    job_config = _read("ui/src/app/jobs/new/jobConfig.ts")
    flux2 = _read("extensions_built_in/diffusion_models/flux2/flux2_model.py")
    klein = _read("extensions_built_in/diffusion_models/flux2/flux2_klein_model.py")
    ltx2 = _read("extensions_built_in/diffusion_models/ltx2/ltx2.py")
    wan = _read("toolkit/models/wan21/wan21.py")
    wan22 = _read("extensions_built_in/diffusion_models/wan22/wan22_14b_model.py")
    qwen_image = _read("extensions_built_in/diffusion_models/qwen_image/qwen_image.py")
    ideogram = _read("extensions_built_in/diffusion_models/ideogram4/ideogram4.py")

    assert "ACE-Step/acestep-" not in caption_options
    assert "ACE-Step/acestep-" not in caption_job
    assert "Qwen/Qwen3-VL-" not in caption_options
    assert "ai-toolkit/Qwen3-Omni-" not in caption_options
    assert "zhaoke1006/Qwen2.5-Omni-7B-H3-Prompt-Rewriter" not in caption_options
    assert "./models/Qwen2.5-Omni-7B-H3-Prompt-Rewriter" in caption_options
    assert "./models/text_encoders/qwen3_omni_30b_a3b_thinking_convrot8.safetensors" in caption_job
    assert "name_or_path: './models/Flex.1-alpha'" in job_config
    assert 'MISTRAL_PATH = "./models/Mistral-Small-3.1-24B-Instruct-2503"' in flux2
    assert 'flux2_vae_path: str = "./models/flux2_vae/ae.safetensors"' in flux2
    assert 'flux2_vae_path: str = "./models/flux2_vae/ae.safetensors"' in klein
    assert 'flux2_klein_te_path: str = "./models/Qwen3-4B"' in klein
    assert 'flux2_klein_te_path: str = "./models/Qwen3-8B"' in klein
    assert 'base_te_path = "./models/gemma-3-12b-it-qat-q4_0-unquantized"' in ltx2
    assert 'te_path = "./models/umt5_xxl_encoder"' in wan
    assert '_wan_vae_path = "./models/wan2.1-vae"' in wan22
    assert 'base_model_path = "./models/Qwen-Image"' in qwen_image
    assert 'config="./models/Qwen-Image"' in qwen_image
    assert 'QWEN3_VL_PATH = "./models/Qwen3-VL-8B-Instruct"' in ideogram


def test_qwen3_omni_uses_local_checkpoint_and_processor_metadata():
    source = _read("extensions_built_in/captioner/Qwen3OmniCaptioner.py")

    assert '"base_repo": "./models/Qwen3-Omni-30B-A3B-Instruct"' in source
    assert '"base_repo": "./models/Qwen3-Omni-30B-A3B-Thinking"' in source
    assert 'if name_or_path.lower().endswith(".safetensors"):' in source
    assert "Qwen3-Omni checkpoint not found" in source
    assert "Qwen3-Omni processor metadata is incomplete" in source
    assert "AutoProcessor.from_pretrained(base_repo, local_files_only=True)" in source


def test_catalog_covers_every_portable_model_literal():
    catalog_paths = set()
    for item in _catalog()["models"]:
        catalog_path = item["path"].rstrip("/")
        catalog_paths.add(catalog_path)
        if item.get("root") == "configured_models":
            catalog_paths.add("./models/" + catalog_path)
    source_roots = [ROOT / "extensions_built_in", ROOT / "toolkit", ROOT / "ui" / "src"]
    model_literals = set()

    for source_root in source_roots:
        for source_path in source_root.rglob("*"):
            if source_path.suffix not in {".py", ".ts", ".tsx"}:
                continue
            for line in source_path.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith(("#", "//")):
                    continue
                model_literals.update(
                        value.rstrip("/")
                        for value in re.findall(r"[\"'](\./models/[^\"']+)[\"']", line)
                        if value != "./models/"
                        and "{" not in value
                        and not value.endswith("/dummy-ltx2")
                )

    uncovered = sorted(
        literal
        for literal in model_literals
        if not any(
            literal == catalog_path
            or literal.startswith(catalog_path + "/")
            or catalog_path.startswith(literal + "/")
            for catalog_path in catalog_paths
        )
    )
    assert uncovered == []


def test_windows_launcher_packaging_copies_model_catalog():
    build_script = _read("scripts/build_windows_launcher.ps1")

    assert "portable_models.json" in build_script
    assert "model catalog" in build_script
    assert "$repositoryLauncher = Join-Path $repoRoot 'AI Toolkit Launcher.exe'" in build_script
    assert "Copy-Item -LiteralPath $launcher -Destination $repositoryLauncher -Force" in build_script


def test_model_scanner_reports_ready_incomplete_misplaced_and_unknown(tmp_path):
    from manager.models import scan_models

    catalog = {
        "schema": 1,
        "models": [
            {
                "id": "ready",
                "name": "Ready model",
                "category": "training",
                "path": "./models/Ready",
                "kind": "directory",
                "required_all": ["config.json"],
                "required_any": ["*.safetensors"],
                "download_url": "https://huggingface.co/example/Ready",
            },
            {
                "id": "missing",
                "name": "Missing file",
                "category": "component",
                "path": "./models/Expected/model.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/Expected/blob/main/model.safetensors",
            },
            {
                "id": "incomplete",
                "name": "Incomplete model",
                "category": "training",
                "path": "./models/Incomplete",
                "kind": "directory",
                "required_all": ["config.json", "model.safetensors"],
                "download_url": "https://huggingface.co/example/Incomplete",
            },
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    (tmp_path / "models" / "Ready").mkdir(parents=True)
    (tmp_path / "models" / "Ready" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "Ready" / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / "models" / "Incomplete").mkdir()
    (tmp_path / "models" / "Incomplete" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "Elsewhere").mkdir()
    (tmp_path / "models" / "Elsewhere" / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "models" / "CustomModel").mkdir()

    report = scan_models(repo_root=tmp_path, catalog_path=catalog_path)
    by_id = {item["id"]: item for item in report["models"]}

    assert by_id["ready"]["status"] == "ready"
    assert by_id["missing"]["status"] == "misplaced"
    assert by_id["missing"]["download_url"].endswith("model.safetensors")
    assert "Expected" in by_id["missing"]["detail"]
    assert by_id["incomplete"]["status"] == "incomplete"
    assert "model.safetensors" in by_id["incomplete"]["detail"]
    assert any(item["status"] == "unrecognized" and item["name"] == "CustomModel" for item in report["models"])
    assert report["summary"] == {
        "ready": 1,
        "issues": 2,
        "missing": 0,
        "unrecognized": 1,
        "total": 4,
    }


def test_model_scanner_keeps_related_family_entries_together(tmp_path):
    from manager.models import scan_models

    catalog = {
        "schema": 1,
        "models": [
            {
                "id": "family-model",
                "name": "Family model",
                "family": "Family A",
                "category": "训练模型",
                "path": "./models/family-model.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/family-model",
            },
            {
                "id": "other-model",
                "name": "Other model",
                "category": "训练模型",
                "path": "./models/other-model.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/other-model",
            },
            {
                "id": "family-component",
                "name": "Family component",
                "family": "Family A",
                "category": "模型组件",
                "path": "./models/family-component.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/family-component",
            },
            {
                "id": "family-adapter",
                "name": "Family adapter",
                "family": "Family A",
                "category": "训练适配器",
                "path": "./models/family-adapter.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/family-adapter",
            },
            {
                "id": "family-helper",
                "name": "Family helper",
                "family": "Family A",
                "category": "辅助模型",
                "path": "./models/family-helper.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/family-helper",
            },
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    models_root = tmp_path / "models"
    models_root.mkdir()
    for item in catalog["models"]:
        (tmp_path / item["path"][2:]).write_bytes(b"weights")

    report = scan_models(repo_root=tmp_path, catalog_path=catalog_path)

    assert [item["id"] for item in report["models"]] == [
        "family-model",
        "family-component",
        "family-adapter",
        "family-helper",
        "other-model",
    ]


def test_model_scanner_uses_configured_models_path_for_minimax(tmp_path):
    import sqlite3

    from manager.models import scan_models

    configured_root = tmp_path / "ComfyUI" / "models"
    configured_root.mkdir(parents=True)
    database = sqlite3.connect(tmp_path / "aitk_db.db")
    database.execute(
        'CREATE TABLE "Settings" ("id" INTEGER PRIMARY KEY, "key" TEXT UNIQUE, "value" TEXT)'
    )
    database.execute(
        'INSERT INTO "Settings" ("key", "value") VALUES (?, ?)',
        ("MODELS_PATH", str(configured_root)),
    )
    database.commit()
    database.close()

    catalog = {
        "schema": 1,
        "models_root": "./models",
        "models": [
            {
                "id": "minimax-h3-fl2va",
                "name": "MiniMax-H3 FL2VA Transformer",
                "family": "MiniMax-H3",
                "category": "训练模型",
                "root": "configured_models",
                "path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/Comfy-Org/MiniMax-H3",
            },
            {
                "id": "minimax-h3-video-vae",
                "name": "MiniMax-H3 Video VAE",
                "family": "MiniMax-H3",
                "category": "模型组件",
                "root": "configured_models",
                "path": "vae/minimax_h3_video_vae_fp16.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/Comfy-Org/MiniMax-H3",
            },
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    transformer = (
        configured_root
        / "diffusion_models"
        / "shared"
        / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
    transformer.parent.mkdir(parents=True)
    transformer.write_bytes(b"weights")

    report = scan_models(repo_root=tmp_path, catalog_path=catalog_path)
    by_id = {item["id"]: item for item in report["models"]}

    assert report["configured_models_root"] == str(configured_root.resolve())
    assert by_id["minimax-h3-fl2va"]["status"] == "ready"
    assert by_id["minimax-h3-fl2va"]["absolute_path"] == str(transformer.resolve())
    assert by_id["minimax-h3-fl2va"]["path"].startswith("<MODELS_PATH>/")
    assert by_id["minimax-h3-video-vae"]["status"] == "missing"
    assert "ComfyUI" in by_id["minimax-h3-video-vae"]["detail"]
    assert "MODELS_PATH" in by_id["minimax-h3-video-vae"]["detail"]


def test_model_scanner_detects_lfs_pointer_as_incomplete(tmp_path):
    from manager.models import scan_models

    catalog = {
        "schema": 1,
        "models": [
            {
                "id": "lfs",
                "name": "LFS model",
                "category": "training",
                "path": "./models/LFS/model.safetensors",
                "kind": "file",
                "download_url": "https://huggingface.co/example/LFS/blob/main/model.safetensors",
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    target = tmp_path / "models" / "LFS" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 999\n",
        encoding="utf-8",
    )

    report = scan_models(repo_root=tmp_path, catalog_path=catalog_path)

    assert report["models"][0]["status"] == "incomplete"
    assert "Git LFS" in report["models"][0]["detail"]
