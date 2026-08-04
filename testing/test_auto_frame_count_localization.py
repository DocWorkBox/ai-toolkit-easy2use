from pathlib import Path


SIMPLE_JOB_SOURCE = Path("ui/src/app/jobs/new/SimpleJob.tsx")
DOCS_SOURCE = Path("ui/src/docs.tsx")


def test_auto_frame_count_option_and_help_are_localized():
    simple_job = SIMPLE_JOB_SOURCE.read_text(encoding="utf-8")
    docs = DOCS_SOURCE.read_text(encoding="utf-8")

    assert 'label="自动帧数"' in simple_job
    assert "title: '自动帧数'" in docs
    assert "目前仅支持批量大小为 1" in docs
    assert 'label="Auto Frame Count"' not in simple_job
    assert "title: 'Auto Frame Count'" not in docs
