"""Explicit allow-list of FORGE experiment stage implementations.

Each active application owns its specification entries. This module preserves the
installed CLI's registry and import path while checking for duplicate identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from experiments.phase1.hela_potency.specifications import POTENCY_SPECIFICATIONS
from experiments.phase1.multireaction.specifications import (
    FOUNDATION_SPECIFICATIONS,
    SHARED_PROGRAM_SPECIFICATIONS,
)
from experiments.phase1.product_l1.specifications import PRODUCT_L1_SPECIFICATIONS

_LOADED = False


def _merge_specifications(*groups: Mapping[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for group in groups:
        overlap = merged.keys() & group.keys()
        if overlap:
            raise ValueError(f"duplicate experiment identifiers: {sorted(overlap)}")
        merged.update(group)
    return merged


SPECIFICATIONS = _merge_specifications(
    {
        "installation-smoke": "experiments/installation_smoke/experiment.json",
        "phase1-corpus": "experiments/phase1/corpus/experiment.json",
    },
    FOUNDATION_SPECIFICATIONS,
    POTENCY_SPECIFICATIONS,
    SHARED_PROGRAM_SPECIFICATIONS,
    PRODUCT_L1_SPECIFICATIONS,
)


def load_catalog() -> None:
    """Register the stage adapters shipped by this repository exactly once."""

    global _LOADED
    if _LOADED:
        return

    # Importing this allow-listed module applies its @stage decorators. Experiment JSON is never
    # allowed to name an import path or execute arbitrary Python.
    import experiments.phase1.hela_potency.stages  # noqa: F401
    import experiments.phase1.multireaction.compose_lipid_training  # noqa: F401
    import experiments.phase1.multireaction.stages  # noqa: F401
    import experiments.phase1.product_l1.stages  # noqa: F401

    _LOADED = True


def specification_paths(repo: Path) -> tuple[Path, ...]:
    """Return every allow-listed experiment specification in stable identifier order."""

    return tuple((repo / SPECIFICATIONS[name]).resolve() for name in sorted(SPECIFICATIONS))


def resolve_specification(repo: Path, value: str) -> Path:
    """Resolve an allow-listed identifier or an explicit repository-local JSON path."""

    if value in SPECIFICATIONS:
        path = repo / SPECIFICATIONS[value]
    else:
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else repo / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise ValueError(f"experiment specification escapes the repository: {value!r}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"experiment specification not found: {resolved}")
    return resolved


__all__ = ["SPECIFICATIONS", "load_catalog", "resolve_specification", "specification_paths"]
