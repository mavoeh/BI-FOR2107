from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}"
    return (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else "unavailable"


def write_provenance(output: Path, config_path: Path, inputs: Iterable[Path] = ()) -> dict:
    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "software": software_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "nibabel", "nilearn", "pyreadstat"]
        ),
        "docker": command_version(["docker", "--version"]),
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs if path.exists() and path.is_file()
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

