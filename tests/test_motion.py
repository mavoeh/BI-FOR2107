from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from for2107_repro.motion import (
    framewise_displacement,
    motion_exclusion_reasons,
    task_motion_table,
)


def test_framewise_displacement_converts_rotations_on_50mm_sphere() -> None:
    parameters = np.zeros((3, 6))
    parameters[1, 3] = 0.2  # x translation, mm
    parameters[2, 0] = 0.01  # rotation, rad -> 0.5 mm
    fd = framewise_displacement(parameters)
    assert fd == pytest.approx([0.2, 0.2 + 0.5])


def test_task_motion_table_and_exclusions(tmp_path: Path) -> None:
    calm = np.zeros((11, 6))
    jumpy = np.zeros((11, 6))
    jumpy[1::2, 3] = 0.6  # every frame moves 0.6 mm
    for subject, par in (("sub-0001", calm), ("sub-0002", jumpy)):
        for task in ("AP", "FM"):
            mc = tmp_path / f"{task}_feat" / f"{subject}_ses-01.feat" / "mc"
            mc.mkdir(parents=True)
            np.savetxt(mc / "prefiltered_func_data_mcf.par", par if task == "FM" else calm)
    table = task_motion_table(
        tmp_path, ["sub-0001", "sub-0002", "sub-0003"], "ses-01", ["AP", "FM"], 0.5
    )
    assert table.loc[1, "high_motion_pct_FM"] == 100
    assert table.loc[1, "mean_fd"] == pytest.approx(0.3)
    assert np.isnan(table.loc[2, "mean_fd"])
    reasons = motion_exclusion_reasons(table, ["AP", "FM"], 0.5, 20)
    assert reasons.tolist() == [
        "",
        "mean-fd-above-0.5mm-FM",
        "motion-parameters-missing-AP",
    ]


def test_exclusion_reasons_check_each_run() -> None:
    table = pd.DataFrame(
        {
            "mean_fd_AP": [0.2, 0.2],
            "high_motion_pct_AP": [5.0, 25.0],
            "mean_fd_FM": [0.2, 0.2],
            "high_motion_pct_FM": [5.0, 0.0],
        }
    )
    reasons = motion_exclusion_reasons(table, ["AP", "FM"], 0.5, 20)
    assert reasons.tolist() == ["", "high-motion-volumes-above-20pct-AP"]
