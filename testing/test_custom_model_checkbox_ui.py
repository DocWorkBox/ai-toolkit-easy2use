from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_custom_model_checkbox_options_render_as_checkboxes():
    source = (
        ROOT / "ui/src/app/jobs/new/SimpleJob.tsx"
    ).read_text(encoding="utf-8")
    custom_options = source.split(
        "modelArch?.customModelSelectOptions?.map", 1
    )[1].split("modelArch?.modelNotes", 1)[0]

    assert "customOption.type === 'checkbox'" in custom_options
    assert "<Checkbox" in custom_options
    assert "checked={customOption.getValue(jobConfig)}" in custom_options
