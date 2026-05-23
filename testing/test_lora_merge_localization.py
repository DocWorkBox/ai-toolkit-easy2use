from pathlib import Path


FILES_WIDGET_SOURCE = Path("ui/src/components/FilesWidget.tsx")
MERGE_MODAL_SOURCE = Path("ui/src/components/MergeLoRAsModal.tsx")


def test_lora_merge_entrypoint_text_is_localized():
    source = FILES_WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "融合" in source
    assert ">merge<" not in source
    assert "Error loading checkpoints" not in source
    assert "No checkpoints available" not in source
    assert "Checkpoint 加载失败" in source
    assert "暂无 Checkpoint 文件" in source


def test_lora_merge_modal_text_is_localized():
    source = MERGE_MODAL_SOURCE.read_text(encoding="utf-8")

    for english_text in [
        'title="Merge LoRAs"',
        ">Merging LoRAs... please do not close this window.<",
        ">Merge failed. See log below.<",
        ">Merge complete.<",
        "'Starting...\\n'",
        'label="Output Filename"',
        'placeholder="Enter output filename"',
        'label="Add LoRA"',
        ">Selected LoRAs<",
        ">Cancel<",
        ">Merge<",
        ">Close<",
        'aria-label="Remove"',
        "Script timed out.",
        "Script exited with code",
        "Unknown error",
    ]:
        assert english_text not in source

    for chinese_text in [
        "融合 LoRA",
        "正在融合 LoRA，请不要关闭此窗口。",
        "融合失败，请查看下方日志。",
        "融合完成。",
        "正在启动...",
        "输出文件名",
        "请输入输出文件名",
        "添加 LoRA",
        "已选择的 LoRA",
        "取消",
        "开始融合",
        "关闭",
        "移除",
        "脚本执行超时。",
        "脚本退出，退出码",
        "未知错误",
    ]:
        assert chinese_text in source
