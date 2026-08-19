"""Typed, content-addressed experiment execution for FORGE.

The public surface is intentionally orchestration-only.  Scientific domain packages do not import
this package; registered stage adapters depend on the domains and remain at the outer edge.
"""

# Registration is explicit but central: importing forge.experiment makes infrastructure stages
# available without allowing a JSON spec to import arbitrary Python modules.
import forge.experiment.builtin_stages as _builtin_stages  # noqa: F401
import forge.experiment.model_stages as _model_stages  # noqa: F401
from forge.experiment.backends import ExecutionBackend, LocalBackend, ModalRuntimeBackend
from forge.experiment.doctor import diagnose_experiment
from forge.experiment.errors import (
    BackendError,
    ExperimentError,
    RegistryError,
    RunExistsError,
    SpecError,
    StageError,
    VerificationError,
)
from forge.experiment.registry import StageRegistry, registry, stage
from forge.experiment.runner import (
    ExperimentRunner,
    ExperimentRunResult,
    RunPlan,
    verify_run_directory,
)
from forge.experiment.seed import SeedPlan
from forge.experiment.spec import (
    DeterminismSpec,
    ExperimentSpec,
    OutputSpec,
    ResourceSpec,
    StageSpec,
)
from forge.experiment.stage import ProducedArtifact, RunContext, StageResult

__all__ = [
    "BackendError",
    "DeterminismSpec",
    "ExecutionBackend",
    "ExperimentError",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "LocalBackend",
    "ModalRuntimeBackend",
    "OutputSpec",
    "ProducedArtifact",
    "RegistryError",
    "ResourceSpec",
    "RunContext",
    "RunExistsError",
    "RunPlan",
    "SeedPlan",
    "SpecError",
    "StageError",
    "StageRegistry",
    "StageResult",
    "StageSpec",
    "VerificationError",
    "diagnose_experiment",
    "registry",
    "stage",
    "verify_run_directory",
]
