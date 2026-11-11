from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is incomplete or inconsistent."""


PATH_KEYS = {
    "root",
    "bids_root",
    "clinical_files",
    "primary_clinical_file",
    "subject_map",
    "events_root",
    "fmriprep_derivatives",
    "preprocessed_rest_root",
    "task_data_root",
    "feature_table",
    "feature_dir",
    "catalog",
    "feature_means",
    "correlations",
    "stability_config",
    "for2107_scores",
    "curves",
    "correlation_file",
    "bids_staging",
    "reports",
    "work",
    "freesurfer_license",
    "roi_directory",
    "variable_map",
    "mapping",
}


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing configuration section: {name}")
        return value

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def require_path(self, dotted: str) -> Path:
        value = self.get(dotted)
        if value in (None, ""):
            raise ConfigError(f"Missing required path: {dotted}")
        return Path(value)


def _resolve_path(value: Any, base: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_resolve_path(item, base) for item in value]
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _resolve_paths(node: Any, base: Path, key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_paths(v, base, k) for k, v in node.items()}
    if key in PATH_KEYS and isinstance(node, (str, list)):
        return _resolve_path(node, base)
    return node


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    resolved = _resolve_paths(data, config_path.parent)
    for section in ("project", "inputs", "outputs", "selection", "preprocessing", "analysis"):
        if section not in resolved:
            raise ConfigError(f"Missing configuration section: {section}")
    return Config(config_path, resolved)
