"""Paper build and evidence-reproduction contracts."""

from forge_paper.completed_evidence_v1 import (
    CompletedEvidenceV1Error,
    render_completed_evidence_v1,
)
from forge_paper.contract import PaperContract, PaperContractError
from forge_paper.experiment_matrix import (
    ExperimentMatrix,
    ExperimentMatrixError,
    diagnose_experiment_matrix,
)
from forge_paper.results_v1 import PaperResultsV1Error, render_v1_results
from forge_paper.sample_visualization import (
    ForgeSampleFigureError,
    render_forge_generated_sample_atlas,
    render_forge_generated_sample_figure,
)
from forge_paper.verification import diagnose_paper, verify_paper

__all__ = [
    "ExperimentMatrix",
    "ExperimentMatrixError",
    "CompletedEvidenceV1Error",
    "PaperContract",
    "PaperContractError",
    "PaperResultsV1Error",
    "ForgeSampleFigureError",
    "diagnose_experiment_matrix",
    "diagnose_paper",
    "render_v1_results",
    "render_completed_evidence_v1",
    "render_forge_generated_sample_atlas",
    "render_forge_generated_sample_figure",
    "verify_paper",
]
