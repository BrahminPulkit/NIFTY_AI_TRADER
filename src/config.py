"""Minimal path configuration for the fresh five-year development cycle."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "config" / "project_data.json"

if not PROJECT_CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Project configuration not found: {PROJECT_CONFIG_PATH}")

PROJECT_CONFIG = json.loads(PROJECT_CONFIG_PATH.read_text(encoding="utf-8"))


def project_path(key: str) -> Path:
    """Resolve and validate a configured repository-relative path."""
    configured = Path(PROJECT_CONFIG[key])
    if configured.is_absolute():
        raise ValueError(f"Configured path must be repository-relative: {configured}")
    resolved = (PROJECT_ROOT / configured).resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Configured path escapes the repository: {configured}")
    return resolved


FIVE_YEAR_INDEX_RAW_DATASET = project_path("five_year_index_raw_dataset")
DHAN_SECURITY_MASTER = project_path("dhan_security_master")
RESET_BACKUP_DIRECTORY = project_path("reset_backup_directory")
