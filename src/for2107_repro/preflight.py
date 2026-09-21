from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Config

PYTHON_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "nibabel",
    "nilearn",
    "yaml",
    "jinja2",
    "pyreadstat",
]


def check(config: Config) -> dict[str, Any]:
    backend = config.get("preprocessing.backend", "docker")
    modules = {name: importlib.util.find_spec(name) is not None for name in PYTHON_MODULES}
    commands = {
        "docker": shutil.which("docker"),
        "apptainer": shutil.which("apptainer"),
        "bids-validator": shutil.which("bids-validator"),
        "fmriprep": shutil.which("fmriprep"),
    }
    paths = {
        "bids_root": config.require_path("inputs.bids_root").exists(),
        "preprocessed_rest_root": bool(
            config.get("inputs.preprocessed_rest_root")
            and Path(config.get("inputs.preprocessed_rest_root")).exists()
        ),
        "clinical_files": [Path(path).exists() for path in config.get("inputs.clinical_files", [])],
        "freesurfer_license": bool(
            config.get("preprocessing.freesurfer_license")
            and Path(config.get("preprocessing.freesurfer_license")).exists()
        ),
    }
    docker_daemon_ready = False
    if commands["docker"]:
        docker_check = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        docker_daemon_ready = docker_check.returncode == 0
    backend_ready = backend == "existing" or bool(commands.get(backend))
    if backend == "docker":
        backend_ready = docker_daemon_ready
    if backend == "executable":
        backend_ready = bool(shutil.which(config.get("preprocessing.executable", "fmriprep")))
    issues = []
    if not all(modules.values()):
        issues.append("Install missing Python packages from environment.yml.")
    if not paths["bids_root"] and not paths["preprocessed_rest_root"]:
        issues.append("Configured BIDS root is missing.")
    if not backend_ready:
        issues.append(f"Preprocessing backend {backend!r} is not installed; reuse derivatives or install it.")
    if backend != "existing" and not paths["freesurfer_license"]:
        issues.append("fMRIPrep requires a FreeSurfer license file, including with --fs-no-reconall.")
    return {
        "python_modules": modules,
        "commands": commands,
        "paths": paths,
        "preprocessing_backend": backend,
        "docker_daemon_ready": docker_daemon_ready,
        "backend_ready": backend_ready,
        "issues": issues,
        "ready_for_inventory": paths["bids_root"] or paths["preprocessed_rest_root"],
        "ready_for_full_pipeline": not issues,
    }


def write_preflight(config: Config) -> dict[str, Any]:
    result = check(config)
    output = config.require_path("outputs.root") / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
