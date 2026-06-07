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
