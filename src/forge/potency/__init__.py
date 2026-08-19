"""Potency prediction: oracle fitting, applicability, and terminal ranking.

Everything here answers one question -- *how well is this lipid likely to work, and are we entitled
to say so?* That covers fitting the HeLa transfection oracles, deciding whether a candidate falls
inside the applicability domain the oracle was calibrated on, ranking terminals once both are known,
and the morphology and challenger analyses built on top.

Split out of `bio/`, which held 54 modules of which five were biology. The rest were model
evaluation wearing a biology label, and that mislabelling is why `bio` could not be imported without
importing `product`, `route` and `value` back.

**Exports resolve lazily, deliberately.** An eager re-export changes the import graph, and the
blinded-execution runner declares the exact set of `forge.*` modules it may load and checks it by
equality. A package API should describe a surface without altering what gets imported.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # re-exported lazily below; imported here only so type checkers see them
    from forge.potency.oracle_graph import (  # noqa: F401
        GraphFeatureVocabulary,
        OracleGraphRecord,
    )
    from forge.potency.ugi_hela_potency_diagnostic import (  # noqa: F401
        FrozenHeLaOracleWorker,
        HeLaBatchPredictor,
        HeLaPotencyDiagnosticPolicy,
    )
    from forge.potency.ugi_interpolative_conformal import conformal_radius  # noqa: F401

_EXPORTS = {
    "FrozenHeLaOracleWorker": "forge.potency.ugi_hela_potency_diagnostic",
    "GraphFeatureVocabulary": "forge.potency.oracle_graph",
    "HeLaBatchPredictor": "forge.potency.ugi_hela_potency_diagnostic",
    "HeLaPotencyDiagnosticPolicy": "forge.potency.ugi_hela_potency_diagnostic",
    "OracleGraphRecord": "forge.potency.oracle_graph",
    "conformal_radius": "forge.potency.ugi_interpolative_conformal",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_EXPORTS[name]), name)
