from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Config, ConfigError
from .provenance import sha256


def fetch_tozzi_rois(config: Config) -> dict:
    destination = Path(config.get("features.roi_directory"))
    repository = config.get("features.tozzi_repository")
    revision = config.get("features.tozzi_revision")
    if not repository or not revision:
        raise ConfigError("Tozzi repository and immutable revision are required")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tozzi-reference-") as temporary:
        clone = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--quiet", repository, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", revision], check=True)
        actual = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != revision:
            raise ConfigError(f"Resolved Tozzi revision {actual}, expected {revision}")
        for source in sorted((clone / "rois").glob("*.nii")):
            shutil.copy2(source, destination / source.name)
    files = [
        {"name": path.name, "sha256": sha256(path)} for path in sorted(destination.glob("*.nii"))
    ]
    manifest = {
        "source": repository,
        "revision": revision,
        "license_note": "Upstream repository does not declare a license; retain source attribution.",
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

