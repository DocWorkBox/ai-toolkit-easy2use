from pathlib import Path


def test_dataset_page_refreshes_all_captions_when_auto_captioning_finishes():
    source = Path("ui/src/app/datasets/[datasetName]/page.tsx").read_text(encoding="utf-8")

    assert "useRef" in source
    assert "wasAutoCaptioningRef" in source
    assert "handleAutoCaptioningChange" in source
    assert "wasAutoCaptioningRef.current && !isActive" in source
    assert "setCaptionRefreshKeys(prev => {" in source
    assert "for (const img of imgList)" in source
    assert "next[img.img_path] = (next[img.img_path] || 0) + 1" in source
    assert "setIsAutoCaptioning={handleAutoCaptioningChange}" in source


def test_auto_caption_button_treats_queued_jobs_as_active_for_dataset_refresh():
    source = Path("ui/src/components/AutoCaptionButton.tsx").read_text(encoding="utf-8")

    assert "const isActive = !!(job && (job.status === 'running' || job.status === 'queued'))" in source
    assert "setIsAutoCaptioning(isActive)" in source
