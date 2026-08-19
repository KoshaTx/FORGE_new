"""The single RDKit boundary for parsing and canonical identity.

Before this module the chemistry primitives lived in `bio/ugi_distributional_applicability.py` and
were reached by importing its private `_canonical` and `_molecule` -- one consumer even aliases that
module to the name `chemistry` to make the call sites read sensibly. Seventeen modules depend on it.

Two canonical forms exist here because the codebase genuinely needs two, and conflating them would
be a correctness change rather than a cleanup:

  `canonical_constitution`            accepts a disconnected input (a salt, `CCN.Cl`) and returns it
  `canonical_connected_constitution`  rejects one, because a lipid product must be a single molecule

An audit of every canonicalizer in the package found exactly these two behaviours and no others.
They agree on molecular identity -- both are stereo-free, as AGENTS.md requires of the model-facing
representation -- and differ only in what input they admit. That distinction is deliberate, so it is
preserved rather than unified.

**Stereo-free is not incidental.** `isomericSmiles=False` discards E/Z, chirality and isotope
labels, which is what makes two source strings collapse to one constitutional graph. Turning it on
would silently change what counts as the same molecule everywhere.

**Caching covers strings, never molecules.** `Chem.Mol` is mutable, so handing the same cached
object to two callers would let one mutate what the other is reading -- a bug that would surface as
irreproducible chemistry rather than an exception. Only the string-to-string functions are memoized.

**The memoization is not a measured speedup, and should not be described as one.** The restructuring
plan assumed that 274 parse sites with no caching meant repeated canonicalization was a bottleneck.
Measured on 12,000 canonicalizations of 1,500 real lipid SMILES from `results/m0_03`, with each
structure seen eight times, the cache took 10,500 hits and the run was **1.0x** -- indistinguishable
from uncached. RDKit canonicalizes a 93-character lipid in microseconds. The cache is kept because
it is bounded and free, not because it earns anything; anyone hunting a slow pipeline should profile
rather than assume this is the cost.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger

from forge.core.types import CanonicalSmiles, Smiles

# RDKit logs parse failures to stderr by default. Failures here are raised, not printed, and the
# chatter otherwise buries real output in any run that probes candidate structures.
RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]  # rdkit ships no stubs

CACHE_SIZE = 100_000


class ChemError(ValueError):
    """A SMILES string cannot be parsed, or violates a declared structural requirement."""


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse SMILES into a molecule, raising rather than returning None.

    Deliberately not cached: `Chem.Mol` is mutable, and a shared instance would let one caller's
    edit leak into another's read. Callers that only need identity should use the canonical
    functions below, which are cached.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ChemError(f"invalid molecular graph: {smiles!r}")
    return molecule


@lru_cache(maxsize=CACHE_SIZE)
def canonical_constitution(smiles: str) -> CanonicalSmiles:
    """Canonical, stereo-free SMILES. Disconnected input is preserved as-is.

    Matches the behaviour of the five existing canonicalizers that accept salts, including
    `bio.ugi_distributional_applicability._canonical`, which most of the package routes through.
    """
    return CanonicalSmiles(
        Chem.MolToSmiles(parse_smiles(smiles), canonical=True, isomericSmiles=False)
    )


@lru_cache(maxsize=CACHE_SIZE)
def canonical_connected_constitution(smiles: str) -> CanonicalSmiles:
    """Canonical, stereo-free SMILES for a single connected molecule.

    Raises on a disconnected input rather than silently accepting a salt or a mixture. A Ugi
    product that arrives in two pieces is not a product, so admitting one would let a malformed
    structure travel downstream looking well formed.
    """
    molecule = parse_smiles(smiles)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ChemError(f"expected a single connected molecule: {smiles!r}")
    return CanonicalSmiles(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))


def is_valid(smiles: str) -> bool:
    """Whether RDKit can parse this SMILES at all."""
    return Chem.MolFromSmiles(smiles) is not None


def same_constitution(left: str, right: str) -> bool:
    """Whether two SMILES denote the same constitutional graph.

    The comparison every module was open-coding. Stereo-free, so a cis and a trans form of the
    same skeleton compare equal -- which is the model-facing identity this project declares.
    """
    return canonical_constitution(left) == canonical_constitution(right)


def cache_stats() -> dict[str, dict[str, int]]:
    """Hit and miss counts, so a slow pipeline can be shown to be re-parsing rather than guessed at."""
    return {
        name: {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
        }
        for name, info in (
            ("canonical_constitution", canonical_constitution.cache_info()),
            ("canonical_connected_constitution", canonical_connected_constitution.cache_info()),
        )
    }


def clear_caches() -> None:
    """Drop memoized results. For benchmarks and for tests that measure parse counts."""
    canonical_constitution.cache_clear()
    canonical_connected_constitution.cache_clear()


__all__ = [
    "CACHE_SIZE",
    "ChemError",
    "CanonicalSmiles",
    "Smiles",
    "cache_stats",
    "canonical_connected_constitution",
    "canonical_constitution",
    "clear_caches",
    "is_valid",
    "parse_smiles",
    "same_constitution",
]
