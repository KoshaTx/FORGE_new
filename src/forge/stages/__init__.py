"""FORGE stage implementations, registered into the generic runner.

This is the seam between `forge_experiment`, which knows how to execute a DAG, and this
repository's science, which knows what the stages do. Importing this module registers them;
the runner never imports it, so the dependency runs one way -- forge -> forge_experiment.
"""

import forge.stages.model_stages as _model_stages  # noqa: F401
