"""Biological endpoints: the tissue and application targets a lipid is designed for.

This package is now what its name says -- the endpoint interface, the three concrete endpoints
(liver, muscle, vaccine) and the frozen endpoint decision. It held 54 modules until the oracle,
applicability and morphology work moved to `forge.potency`; those were model evaluation wearing a
biology label, and that mislabelling is why `bio` could not be imported without importing
`product`, `route` and `value` back.

AGENTS.md requires the endpoint interface stay generic and that no endpoint be hard-coded, which is
easier to hold to now that model evaluation is not living here as well.

`Endpoint` resolves lazily so that importing the package does not pull its submodules into the
import graph -- the blinded-execution runner checks that graph by exact equality.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.bio.endpoint import Endpoint  # noqa: F401

__all__ = ["Endpoint"]


def __getattr__(name: str) -> object:
    if name == "Endpoint":
        from forge.bio.endpoint import Endpoint

        return Endpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
