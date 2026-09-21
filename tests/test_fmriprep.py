from for2107_repro.config import load_config
from for2107_repro.fmriprep import build_command


def test_docker_command_mounts_inputs_read_only() -> None:
    command = build_command(load_config("configs/smoke.yaml"), ["sub-0011"])
    rendered = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert ":/data:ro" in rendered
    assert "--participant-label 0011" in rendered
    assert "nipreps/fmriprep:25.1.4" in command
    assert ":/license.txt:ro" in rendered
    assert "--fs-license-file /license.txt" in rendered
