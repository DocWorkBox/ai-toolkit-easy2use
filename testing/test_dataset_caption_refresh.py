from pathlib import Path


def test_dataset_page_refreshes_all_captions_when_auto_captioning_finishes():
    source = Path("ui/src/app/datasets/[datasetName]/page.tsx").read_text(encoding="utf-8")

    assert "refreshAllCaptions" in source
    assert "setCaptionRefreshKeys(prev => {" in source
    assert "for (const img of imgList)" in source
    assert "next[img.img_path] = (next[img.img_path] || 0) + 1" in source
    assert "onCaptioningFinished={refreshAllCaptions}" in source


def test_auto_caption_button_treats_queued_jobs_as_active_for_dataset_refresh():
    source = Path("ui/src/components/AutoCaptionButton.tsx").read_text(encoding="utf-8")

    assert "const isActive = !!(job && (job.status === 'running' || job.status === 'queued'))" in source
    assert "setIsAutoCaptioning(isActive)" in source


def test_caption_job_start_marks_page_active_before_status_polling():
    button_source = Path("ui/src/components/AutoCaptionButton.tsx").read_text(encoding="utf-8")
    modal_source = Path("ui/src/components/CaptionDatasetModal.tsx").read_text(encoding="utf-8")

    assert "onJobStarted?: () => void" in modal_source
    assert "modalInfo.onJobStarted?.()" in modal_source
    assert "onJobStarted: () => setIsAutoCaptioning?.(true)" in button_source


def test_caption_completion_is_detected_by_job_identity_and_terminal_status():
    source = Path("ui/src/components/AutoCaptionButton.tsx").read_text(encoding="utf-8")

    assert "lastFinishedJobIdRef" in source
    assert "['completed', 'stopped', 'error'].includes(job.status)" in source
    assert "lastFinishedJobIdRef.current !== job.id" in source
    assert "onCaptioningFinished?.()" in source


def test_stale_caption_responses_cannot_replace_a_newer_refresh():
    source = Path("ui/src/hooks/useCaptionBatch.tsx").read_text(encoding="utf-8")

    assert "cacheGenerations" in source
    assert "generationFor(cacheKey) === generation" in source
    assert "cacheGenerations.set(cacheKey, generationFor(cacheKey) + 1)" in source
    assert "requestIdRef.current !== requestId" in source
