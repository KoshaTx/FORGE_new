"""Global constitutional identity partitions for the combinatorial libraries.

Names, block IDs and reaction roles cannot create separate folds for the same molecule. Frozen
fold constraints are read as identity metadata only; no R0 decomposition or fitted preprocessing
is performed here. Mixed-component combinations remain in the ledger with a quarantine label.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule

FOLDS = ("train", "calibration", "heldout")
_ALIASES = {
    "train": "train",
    "R0_train": "train",
    "calibration": "calibration",
    "val": "calibration",
    "R0_cal": "calibration",
    "heldout": "heldout",
    "test": "heldout",
    "R0_heldout": "heldout",
}


def constitution_id(smiles: str) -> str:
    canonical, _ = constitutional_molecule(smiles)
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalize_fold(value: str) -> str:
    if value not in _ALIASES:
        raise LibraryAssemblyError(f"unsupported frozen fold {value!r}")
    return _ALIASES[value]


@dataclass
class FrozenIdentityFolds:
    """Historical identities can have different schemes; the new cohort uses the strictest fold."""

    folds: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, identity: str, fold: str, source: str) -> None:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise LibraryAssemblyError("frozen identity must be a full constitutional SHA-256")
        self.folds[identity].add(normalize_fold(fold))
        self.sources[identity].add(source)

    def effective(self, identity: str) -> str | None:
        observed = self.folds.get(identity)
        return max(observed, key=FOLDS.index) if observed else None


def component_partitions(
    component_smiles: Iterable[str],
    frozen: FrozenIdentityFolds,
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    """Assign identities globally, independent of role/library/row order; preserve frozen guards."""

    if type(seed) is not int or seed < 0:
        raise LibraryAssemblyError("split seed must be a nonnegative integer")
    if (
        set(fractions) != set(FOLDS)
        or any(not 0 < v < 1 for v in fractions.values())
        or abs(sum(fractions.values()) - 1) > 1e-12
    ):
        raise LibraryAssemblyError("split fractions must be positive and sum to one")
    result = {}
    for identity in sorted({constitution_id(smiles) for smiles in component_smiles}):
        existing = frozen.effective(identity)
        draw = (
            int.from_bytes(
                hashlib.sha256(f"{seed}|component|{identity}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        result[identity] = existing or (
            "train"
            if draw < fractions["train"]
            else (
                "calibration" if draw < fractions["train"] + fractions["calibration"] else "heldout"
            )
        )
    return result


def program_partition(
    component_ids: Sequence[str],
    product_id: str,
    intermediate_ids: Sequence[str],
    partitions: Mapping[str, str],
    frozen: FrozenIdentityFolds,
) -> tuple[str, tuple[str, ...]]:
    """A supervised sequence may only expose molecules compatible with a single partition."""

    if not component_ids or any(identity not in partitions for identity in component_ids):
        raise LibraryAssemblyError("program has unassigned precursor identities")
    folds = {partitions[identity] for identity in component_ids}
    reasons = set()
    if len(folds) != 1:
        reasons.add("mixed_component_partitions")
    for identity in (product_id, *intermediate_ids):
        for fixed in (partitions.get(identity), frozen.effective(identity)):
            if fixed is not None:
                if fixed not in folds:
                    reasons.add("protected_product_or_intermediate")
                folds.add(fixed)
    if len(folds) != 1:
        return "quarantine", tuple(sorted(reasons))
    return next(iter(folds)), ()


def conflicting_exposures(rows: Iterable[tuple[str, str, Sequence[str]]]) -> set[str]:
    """Return every product group exposing the same identity in two different active folds."""

    uses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for product_id, fold, identities in rows:
        if fold == "quarantine":
            continue
        if fold not in FOLDS:
            raise LibraryAssemblyError("invalid product fold")
        for identity in identities:
            uses[identity][fold].add(product_id)
    return {
        product_id
        for folds in uses.values()
        if len(folds) > 1
        for products in folds.values()
        for product_id in products
    }
