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
    assert "transcribe every confirmed utterance verbatim" in source
    assert "Do not answer in multiple turns" in source
    assert "def _transcribe_audio" in source
    assert "Confirmed audio transcription" in source
    assert "inject_h3_dialogue" in source


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
    assert "group: 'image/video/sound'" in section
    assert "extensionsVideo" in section
    assert "extensionsImage" not in section
    assert "'config.process[0].caption.max_new_tokens': [4096" in section
    assert "'config.process[0].caption.batch_size': [1" in section
    assert "qwen25OmniH3CaptionPrompt" in section
    assert "'MiniMax H3 音视频': qwen25OmniH3CaptionPrompt" in section
    assert "'通用': defaultVideoCaptionPrompt" in section
    assert "'caption.batch_size'" in section
    assert "'caption.layer_offloading'" in section
    assert "'caption.thinking'" not in section
    assert "integrated_multimodal_description:" in options_source
    assert "overall_soundscape:" in options_source
    assert "non_diegetic_music:" in options_source
    assert "You are a professional multimodal video analyst" in options_source


def test_qwen25_omni_h3_implements_batching_and_layer_offloading():
    source = Path(
        "extensions_built_in/captioner/Qwen2_5OmniH3Captioner.py"
    ).read_text(encoding="utf-8")

    assert "MemoryManager.attach(" in source
    assert "offload_percent=self.caption_config.layer_offloading_percent" in source
    assert "if self.caption_config.layer_offloading:" in source
    assert (
        "if self.caption_config.low_vram and not self.caption_config.layer_offloading:"
        in source
    )
    assert "batch_size = max(1, int(self.caption_config.batch_size))" in source
    assert "def _caption_batch" in source
    assert "def run_caption_loop" in source


def test_minimax_caption_presets_are_named_for_h3():
    options_source = Path("ui/src/helpers/captionOptions.ts").read_text(
        encoding="utf-8"
    )

    assert "MiniMax H4" not in options_source
    assert "MiniMax H3 视频" in options_source
    assert "MiniMax H3 图片" in options_source


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


def test_qwen25_omni_h3_extracts_tagged_dialogue_from_transcription():
    from extensions_built_in.captioner.caption_output import extract_transcribed_dialogue

    raw = """Assistant
(S1) <d>[Chinese] 目前是有一个硬伤啊。</d>
(S2): <d>[English] I understand.</d>
Human: Continue."""

    assert extract_transcribed_dialogue(raw) == """(S1) <d>[Chinese] 目前是有一个硬伤啊。</d>
(S2): <d>[English] I understand.</d>"""


def test_qwen25_omni_h3_preserves_plain_sung_lyrics_from_transcription():
    from extensions_built_in.captioner.caption_output import extract_transcribed_dialogue

    raw = "Lyrics: I keep on running through the night"

    assert extract_transcribed_dialogue(raw) == (
        "(S1) <d>[Unknown] I keep on running through the night</d>"
    )


def test_qwen25_omni_h3_preserves_language_tagged_untagged_lyrics():
    from extensions_built_in.captioner.caption_output import extract_transcribed_dialogue

    raw = "[Chinese] 我会一直唱到天亮"

    assert extract_transcribed_dialogue(raw) == (
        "(S1) <d>[Chinese] 我会一直唱到天亮</d>"
    )


def test_qwen25_omni_h3_keeps_no_speech_result_empty():
    from extensions_built_in.captioner.caption_output import extract_transcribed_dialogue

    assert extract_transcribed_dialogue("N/A") == ""
    assert extract_transcribed_dialogue("No intelligible speech or singing.") == ""


def test_qwen25_omni_h3_injects_missing_dialogue_into_integrated_section():
    from extensions_built_in.captioner.caption_output import inject_h3_dialogue

    caption = """integrated_multimodal_description:
[Shot 1] A man speaks to the camera.
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A"""
    dialogue = "(S1) <d>[Chinese] 目前是有一个硬伤啊。</d>"

    result = inject_h3_dialogue(caption, dialogue)

    assert dialogue in result.split("overall_soundscape:", 1)[0]
    assert result.count(dialogue) == 1


def test_qwen25_omni_h3_does_not_duplicate_existing_dialogue():
    from extensions_built_in.captioner.caption_output import inject_h3_dialogue

    caption = """integrated_multimodal_description:
(S1) says: <d>[Chinese] 你好。</d>
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A"""

    assert inject_h3_dialogue(caption, "(S1) <d>[Chinese] 你好。</d>") == caption


def test_qwen25_omni_h3_adds_only_missing_dialogue_lines():
    from extensions_built_in.captioner.caption_output import inject_h3_dialogue

    first = "(S1) <d>[Chinese] 你好。</d>"
    second = "(S2) <d>[English] Hello.</d>"
    caption = f"""integrated_multimodal_description:
The first speaker says: {first}
overall_soundscape:
Quiet room tone.
non_diegetic_music:
N/A"""

    result = inject_h3_dialogue(caption, f"{first}\n{second}")

    assert result.count(first) == 1
    assert result.count(second) == 1
