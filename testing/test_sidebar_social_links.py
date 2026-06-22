from pathlib import Path


def test_sidebar_social_links_use_doc_workbox_channels():
    sidebar_source = Path("ui/src/components/Sidebar.tsx").read_text(encoding="utf-8")

    assert "grid grid-cols-3 gap-3" in sidebar_source
    assert "https://www.youtube.com/@Doc_workBox" in sidebar_source
    assert "https://space.bilibili.com/12710942" in sidebar_source
    assert "FaYoutube" in sidebar_source
    assert "SiBilibili" in sidebar_source
    assert "FaDiscord" not in sidebar_source
    assert "https://discord.gg/" not in sidebar_source


def test_upsample_prompt_entry_points_do_not_regress_to_english():
    simple_job_source = Path("ui/src/app/jobs/new/SimpleJob.tsx").read_text(encoding="utf-8")

    assert "Upsample Prompts" not in simple_job_source
    assert "title: `Prompt #${i + 1}`" not in simple_job_source
    assert "Edit caption &amp; boxes" not in simple_job_source
