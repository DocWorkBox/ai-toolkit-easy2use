from pathlib import Path


CAPTION_PROMPTS_SOURCE = Path("ui/src/helpers/captionPrompts.ts")


def test_anima_caption_prompt_template_matches_builtin_default_system_prompt():
    source = CAPTION_PROMPTS_SOURCE.read_text(encoding="utf-8")

    assert "Anima格式 (Anima Image LoRA)" in source
    assert "You are an expert image captioning model for Anima Image LoRA training." in source
    assert "Do NOT describe layout or structure." in source
    assert "Now analyze the image and output only the final single-line comma-separated tag caption." in source
    assert "captionPromptTemplatesWithoutLanguageSuffix" in source
    assert "animaCaptionPromptTemplate" in source.split("captionPromptTemplatesWithoutLanguageSuffix", 1)[1]
