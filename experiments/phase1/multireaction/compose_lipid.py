"""Application boundary for importing and authenticating a COMPOSE-Lipid source release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.corpus.compose_lipid import import_compose_lipid, verify_compose_lipid


def run_import(repo: Path, config: Path, output: Path) -> dict[str, Any]:
    return import_compose_lipid(repo, config, output)


def verify_import(repo: Path, result: Path) -> dict[str, Any]:
    return verify_compose_lipid(repo, result)
