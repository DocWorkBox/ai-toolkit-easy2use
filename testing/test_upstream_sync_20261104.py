"""Contracts for integrating upstream without losing AIgate customizations."""

import ast
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Optional
from urllib import error, request

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def method(path, name, namespace):
    tree = ast.parse(source(path))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def prompt_resolver():
    tree = ast.parse(source("extensions_built_in/captioner/BaseCaptioner.py"))
    regex = next(node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "_CAPTION_TOKEN_RE" for target in node.targets))
    namespace = {"os": os, "re": re, "Optional": Optional}
    exec(compile(ast.Module(body=[regex], type_ignores=[]), "caption_tokens", "exec"), namespace)
    return method("extensions_built_in/captioner/BaseCaptioner.py", "resolve_prompt", namespace)


def test_conflicts_are_resolved_and_fork_version_is_preserved():
    for path in (
        "extensions_built_in/diffusion_models/moss_image/moss_image.py",
        "toolkit/optimizer.py", "ui/src/app/jobs/new/SimpleJob.tsx",
        "ui/src/components/CaptionSimpleJob.tsx", "ui/src/helpers/captionOptions.ts", "version.py",
    ):
        text = source(path)
        assert "<<<<<<<" not in text, path
        assert ">>>>>>>" not in text, path
        if path.endswith(".py"):
            ast.parse(text)
    assert source("version.py").strip() == 'VERSION = "1.19.1"'


def test_new_optimizer_does_not_replace_custom_optimizers():
    backend = source("toolkit/optimizer.py")
    form = source("ui/src/app/jobs/new/SimpleJob.tsx")
    for name in ("automagic3", "automagic4", "automagicexperiment", "singularity", "singularity_group", "adamconvrot"):
        assert f"'{name}'" in backend
        assert f"value: '{name}'" in form


def test_caption_controls_keep_api_options_and_localized_new_features():
    options = source("ui/src/helpers/captionOptions.ts")
    form = source("ui/src/components/CaptionSimpleJob.tsx")
    types = source("ui/src/types.ts")
    for name in ("api_concurrency", "api_base_url", "api_key", "api_protocol", "prompt_template", "target_lang",
                 "mtp", "cuda_graph", "vision_bf16", "video_fps", "system_prompt"):
        assert f"caption.{name}" in options
        assert f"caption.{name}" in form
        assert f"{name}?:" in types
    for label in ("视频采样帧率", "MTP 推测解码", "MTP 草稿深度", "CUDA 图解码", "视觉编码器使用 bf16", "系统提示词"):
        assert label in form
    assert "!selectedCaptionOption?.noCompile" in form
    assert options.count("export const defaultQtype") == 1
    assert "buildCaptionPrompt(defaultCaptionPromptTemplate, defaultCaptionTargetLanguage)" in options
    assert "图片 / 视频" in options


def test_file_info_dialog_is_localized():
    modal = source("ui/src/components/SafetensorsInfoModal.tsx")
    for label in ("正在读取文件信息", "文件大小", "张量", "参数量", "文件头", "元数据", "筛选张量", "形状", "数据类型", "没有匹配的张量", "没有元数据"):
        assert label in modal
    assert 'title="Safetensors 文件信息"' in source("ui/src/components/FilesWidget.tsx")


def test_moss_batch_encoding_preserves_project_assets_and_staging():
    text = source("extensions_built_in/diffusion_models/moss_image/moss_image.py")
    tree = ast.parse(text)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "encode_text_batch" in functions
    assert "_hub_file_folder" not in functions
    assert "_hub_tokenizer_folder" not in functions
    assert "supports_staged_sampling = True" in text
    assert "call_sample_pipeline(self, pipeline" in text
    assert not functions["decode_latents"].decorator_list
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "sample_stage" and node.args
               and isinstance(node.args[0], ast.Constant) and node.args[0].value == "vae"
               for node in ast.walk(functions["decode_latents"]))


def test_caption_token_help_is_localized_and_custom_captioners_expand_tokens():
    docs = source("ui/src/docs.tsx")
    assert "同名" in docs and "&lt;caption.ext&gt;" in docs
    assert "插入" in docs
    for path in ("extensions_built_in/captioner/RemoteAPICaptioner.py",
                 "extensions_built_in/captioner/Qwen2_5OmniH3Captioner.py"):
        assert "self.resolve_prompt(" in source(path), path


def test_caption_tokens_are_per_file_and_do_not_traverse(tmp_path):
    media = tmp_path / "clip.mp4"
    (tmp_path / "clip.txt").write_text(" original words ", encoding="utf-8")
    resolve = prompt_resolver()
    assert resolve(None, "Use <caption.txt> / <caption.json>", str(media)) == "Use original words / "
    assert resolve(None, "<caption.../secret.txt>", str(media)) == "<caption.../secret.txt>"
    assert resolve(None, "<caption.txt>", str(tmp_path / "other.mp4")) == ""


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_custom_api_resolves_prompt_before_building_request(protocol, tmp_path, monkeypatch):
    from extensions_built_in.captioner import remote_api_utils

    captured = []
    response = {"content": [{"type": "text", "text": "caption"}]} if protocol == "anthropic" else {
        "choices": [{"message": {"content": "caption"}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(response).encode()

    def urlopen(req, timeout):
        captured.append(json.loads(req.data))
        assert timeout == 120
        return Response()

    monkeypatch.setattr(request, "urlopen", urlopen)
    namespace = {name: getattr(remote_api_utils, name) for name in (
        "build_anthropic_payload", "build_openai_payload", "parse_anthropic_caption", "parse_openai_caption")}
    namespace.update(json=json, request=request, error=error, encode_image_to_jpeg_base64=lambda _: "abc")
    caption = method("extensions_built_in/captioner/RemoteAPICaptioner.py", "get_caption_for_file", namespace)
    resolve = prompt_resolver()
    fake = SimpleNamespace(
        caption_config=SimpleNamespace(max_res=512, api_protocol=protocol, model_name_or_path="custom-model", max_new_tokens=512),
        endpoint="https://example.invalid/v1", load_pil_image=lambda *_, **__: None,
        build_prompt_for_file=lambda _: "Describe <caption.txt>",
        resolve_prompt=lambda prompt, file_path: resolve(None, prompt, file_path),
        _build_headers=lambda: {"Authorization": "Bearer test"},
    )
    (tmp_path / "clip.txt").write_text("original words", encoding="utf-8")
    assert caption(fake, str(tmp_path / "clip.png")) == "caption"
    content = captured[0]["messages"][0]["content"]
    assert next(item["text"] for item in content if item["type"] == "text") == "Describe original words"
    assert captured[0]["model"] == "custom-model"


def test_h3_prompt_tokens_do_not_replace_confirmed_dialogue(tmp_path):
    resolve = prompt_resolver()
    build = method("extensions_built_in/captioner/Qwen2_5OmniH3Captioner.py", "_build_messages", {"USER_INSTRUCTION": "Inspect audio"})
    fake = SimpleNamespace(caption_config=SimpleNamespace(caption_prompt="Use <caption.txt>"),
                           resolve_prompt=lambda prompt, path: resolve(None, prompt, path))
    (tmp_path / "clip.txt").write_text("source description", encoding="utf-8")
    messages = build(fake, str(tmp_path / "clip.mp4"), "confirmed lyric")
    assert messages[0]["content"][0]["text"] == "Use source description"
    assert "confirmed lyric" in messages[1]["content"][1]["text"]
