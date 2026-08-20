"""Typed, content-addressed experiment execution.

Deliberately outside `forge`: this is a generic DAG runner -- typed contracts, keyed seeds,
atomic stage commits, verified resume, a Modal backend -- with nothing lipid-specific in it. The
science enters through the stage registry, which `forge.stages` populates.

"""

# Registration is explicit but central: importing forge_experiment makes infrastructure stages
# available without allowing a JSON spec to import arbitrary Python modules.
import forge_experiment.builtin_stages as _builtin_stages  # noqa: F401
from forge_experiment.backends import ExecutionBackend, LocalBackend, ModalRuntimeBackend
from forge_experiment.doctor import diagnose_experiment
from forge_experiment.errors import (
    BackendError,
    ExperimentError,
    RegistryError,
    RunExistsError,
    SpecError,
    StageError,
    VerificationError,
)
from forge_experiment.registry import StageRegistry, registry, stage
from forge_experiment.runner import (
    ExperimentRunner,
    ExperimentRunResult,
    RunPlan,
    verify_run_directory,
)
from forge_experiment.seed import SeedPlan
from forge_experiment.spec import (
    DeterminismSpec,
    ExperimentSpec,
    OutputSpec,
    ResourceSpec,
    StageSpec,
)
from forge_experiment.stage import ProducedArtifact, RunContext, StageResult

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
