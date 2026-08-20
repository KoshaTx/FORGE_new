"""Provenance verification and recovery for this repository.

Outside the installed package because it cannot work outside a clone: the verifier derives the
repository root from its own location and walks `results/`, `configs/`, `docs/provenance/` and
`provenance/frozen-code/`, none of which ship in the wheel. Shipping the code without the trees it
reads would give an installer a command that can only fail.

`forge_provenance.resolver` is a compatibility re-export of the generic experiment-runtime
resolver: active applications may need to map a historical `(path, digest)` to archived bytes.
The verifier and archiver here remain repository tooling, like `forge_maintenance` beside them.
"""

from forge_provenance.archive import archive
from forge_provenance.pins import Pin, Report, collect_pins, verify

__all__ = ["Pin", "Report", "archive", "collect_pins", "verify"]
