from pathlib import Path


def test_automagic3_optimizer_is_registered_and_exposed_in_ui():
    optimizer_source = Path("toolkit/optimizer.py").read_text(encoding="utf-8")
    simple_job_source = Path("ui/src/app/jobs/new/SimpleJob.tsx").read_text(
        encoding="utf-8"
    )

    assert "elif lower_type == 'automagic3':" in optimizer_source
    assert "from toolkit.optimizers.automagic3 import Automagic3" in optimizer_source
    assert "elif lower_type == 'automagicexperiment':" in optimizer_source
    assert "from toolkit.optimizers.automagicEXPERIMENT import AutomagicEXPERIMENT" in optimizer_source
    assert "{ value: 'automagic3', label: 'Automagic v3' }" in simple_job_source
    assert "{ value: 'automagicexperiment', label: 'Automagic Experiment' }" in simple_job_source
    assert "{ value: 'singularity_group', label: 'Singularity (group LR)' }" in simple_job_source
