"""The generative model: sampling primitives, flows, and the machinery that runs them.

Currently the sampling primitives only. The package exists so that the flow model's core operations
have a home of their own rather than living inside whichever experiment first needed them --
`rstar_step` spent its life as a private function inside a completed feasibility probe while ten
modules reached in for it.

The export list is deliberately short. Everything not named here is internal; a package that
exports forty symbols is not offering an interface, it is offering a directory.
"""

from forge.generate.sampling import SamplingError, rstar_step, sample_categorical

__all__ = ["SamplingError", "rstar_step", "sample_categorical"]
