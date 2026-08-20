"""Provenance verification and recovery for this repository.

Outside the installed package because it cannot work outside a clone: the verifier derives the
repository root from its own location and walks `results/`, `configs/`, `docs/provenance/` and
`provenance/frozen-code/`, none of which ship in the wheel. Shipping the code without the trees it
reads would give an installer a command that can only fail.

`forge.core.provenance_archive` stays in the library on purpose -- that is the resolver, a
primitive that maps a (path, digest) to archived bytes. This is the verifier and archiver built on
top of it: tooling that operates on the repository, like `forge_maintenance` beside it.
"""

from forge_provenance.archive import archive
from forge_provenance.pins import Pin, Report, collect_pins, verify

__all__ = ["Pin", "Report", "archive", "collect_pins", "verify"]
