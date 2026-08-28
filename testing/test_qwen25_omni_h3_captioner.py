from pathlib import Path


MODEL_PATH = (
    "/datasets/studio/huggingface/models/"
    "Qwen2.5-Omni-7B-H3-Prompt-Rewriter"
)


def test_qwen25_omni_h3_builds_system_prompt_video_messages():
    source = Path(
        "extensions_built_in/captioner/Qwen2_5OmniH3Captioner.py"
    ).read_text(encoding="utf-8")

    assert '"role": "system"' in source
    assert '{"type": "text", "text": self.caption_config.caption_prompt}' in source
    assert '"role": "user"' in source
    assert '{"type": "video", "video": file_path}' in source
    assert "use_audio_in_video=use_audio" in source
    assert "stop_strings=FOLLOWUP_STOP_STRINGS" in source
    assert "eos_token_id=self.processor.tokenizer.eos_token_id" in source


def test_qwen25_omni_h3_is_registered_and_uses_aigate_defaults():
    extension_source = Path("extensions_built_in/captioner/__init__.py").read_text(
        encoding="utf-8"
    )
    options_source = Path("ui/src/helpers/captionOptions.ts").read_text(
        encoding="utf-8"
    )

    assert 'uid = "Qwen2_5OmniH3Captioner"' in extension_source
    assert "Qwen2_5OmniH3CaptionerExtension," in extension_source

    section = options_source.split("name: 'Qwen2_5OmniH3Captioner'", 1)[1].split(
        "name: 'Qwen3OmniCaptioner'", 1
    )[0]
    assert MODEL_PATH in section
    assert "extensionsVideo" in section
    assert "extensionsImage" not in section
    assert "'config.process[0].caption.max_new_tokens': [4096" in section
    assert "qwen25OmniH3CaptionPrompt" in section
    assert "integrated_multimodal_description:" in options_source
    assert "overall_soundscape:" in options_source
    assert "non_diegetic_music:" in options_source
    assert "You are a professional multimodal video analyst" in options_source


def test_qwen25_omni_h3_discards_generated_followup_conversation():
    from extensions_built_in.captioner.caption_output import (
        extract_first_h3_caption,
    )

    raw = """integrated_multimodal_description:
[Shot 1] A man sits at a desk.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A
Assistant
integrated_multimodal_description:
[Shot 1] A duplicate answer.
overall_soundscape:
Duplicate ambience.
non_diegetic_music:
N/A
Human: What is the audio track?
请详细描述视频中人物的面部特征和表情变化。"""

    assert extract_first_h3_caption(raw) == """integrated_multimodal_description:
[Shot 1] A man sits at a desk.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A"""


def test_qwen25_omni_h3_keeps_unstructured_output_for_diagnostics():
    from extensions_built_in.captioner.caption_output import (
        extract_first_h3_caption,
    )

    raw = "The model did not follow the requested output contract."

    assert extract_first_h3_caption(raw) == raw
