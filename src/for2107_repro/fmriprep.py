from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .config import Config, ConfigError


def build_command(config: Config, participant_labels: list[str] | None = None) -> list[str]:
    prep = config.section("preprocessing")
    backend = prep.get("backend", "docker")
    bids_root = config.require_path("outputs.bids_staging")
    derivatives = config.require_path("inputs.fmriprep_derivatives")
    work = config.require_path("outputs.work") / "fmriprep"
    license_path = prep.get("freesurfer_license")
    common = [
        "/data",
        "/out",
        "participant",
        "--work-dir",
        "/work",
        "--nprocs",
        str(prep.get("n_cpus", 4)),
        "--mem-mb",
        str(prep.get("memory_mb", 12000)),
        "--output-spaces",
        *prep.get("output_spaces", ["MNI152NLin2009cAsym:res-2"]),
    ]
    if prep.get("fs_no_reconall", True):
        common.append("--fs-no-reconall")
    if license_path:
        common.extend(["--fs-license-file", "/license.txt"])
    if participant_labels:
        common.extend(["--participant-label", *[label.removeprefix("sub-") for label in participant_labels]])
    image = prep.get("image")
    if backend == "docker":
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{bids_root}:/data:ro",
            "-v",
            f"{derivatives}:/out",
            "-v",
            f"{work}:/work",
        ]
        if license_path:
            command.extend(["-v", f"{license_path}:/license.txt:ro"])
        return [*command, image, *common]
    if backend == "apptainer":
        image_path = prep.get("image_path")
        if not image_path:
            raise ConfigError("preprocessing.image_path is required for Apptainer")
        binds = f"{bids_root}:/data:ro,{derivatives}:/out,{work}:/work"
        if license_path:
            binds += f",{license_path}:/license.txt:ro"
        return ["apptainer", "run", "--cleanenv", "--bind", binds, image_path, *common]
    if backend == "executable":
        executable = prep.get("executable", "fmriprep")
        native = [str(bids_root), str(derivatives), "participant"]
        translated = [str(work) if item == "/work" else item for item in common[3:]]
        return [executable, *native, *translated]
    if backend == "existing":
        return []
    raise ConfigError(f"Unknown preprocessing backend: {backend}")


def printable_command(config: Config, participant_labels: list[str] | None = None) -> str:
    return shlex.join(build_command(config, participant_labels))


def run_fmriprep(config: Config, participant_labels: list[str] | None = None) -> int:
    if not config.get("preprocessing.execute", False):
        raise ConfigError("Refusing to execute: set preprocessing.execute: true after reviewing the command")
    license_path = config.get("preprocessing.freesurfer_license")
    if not license_path or not Path(license_path).is_file():
        raise ConfigError("fMRIPrep requires a valid FreeSurfer license file")
    command = build_command(config, participant_labels)
    if not command:
        return 0
    config.require_path("inputs.fmriprep_derivatives").mkdir(parents=True, exist_ok=True)
    (config.require_path("outputs.work") / "fmriprep").mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, check=False).returncode
