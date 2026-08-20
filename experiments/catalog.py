"""Explicit allow-list of FORGE experiment stage implementations."""

from __future__ import annotations

from pathlib import Path

_LOADED = False

SPECIFICATIONS = {
    "installation-smoke": "experiments/installation_smoke/experiment.json",
    "phase1-corpus": "experiments/phase1/corpus/experiment.json",
    "phase1-sampling": "experiments/phase1/product_l1/sampling.json",
    "phase1-training-production": "experiments/phase1/product_l1/training_production.json",
    "phase1-training-smoke": "experiments/phase1/product_l1/training_smoke.json",
}


def load_catalog() -> None:
    """Register the stage adapters shipped by this repository exactly once."""

    global _LOADED
    if _LOADED:
        return

    # Importing this allow-listed module applies its @stage decorators. Experiment JSON is never
    # allowed to name an import path or execute arbitrary Python.
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
