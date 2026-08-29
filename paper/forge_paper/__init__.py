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
from forge_paper.gem_prose_results import GemProseResultsError, render_gem_prose_results
from forge_paper.gem_table1 import GemTable1Error, render_gem_table1_final_evidence
from forge_paper.gem_table2 import GemTable2Error, render_gem_table2_forge_row
from forge_paper.gem_table3 import GemTable3Error, render_gem_table3_route_assessment
from forge_paper.gem_table4 import GemTable4Error, render_gem_table4_production_comparison
from forge_paper.gem_table5 import GemTable5Error, render_gem_table5_decoder_source_ablation
from forge_paper.gem_table6 import GemTable6Error, render_gem_table6_exact_l1_counts
from forge_paper.gem_table7 import GemTable7Error, render_gem_table7_lipid_realism
from forge_paper.gem_table8 import GemTable8Error, render_gem_table8_architecture_ablations
from forge_paper.gem_table9 import GemTable9Error, render_gem_table9_catalogue_comparison
from forge_paper.gem_table10 import GemTable10Error, render_gem_table10_route_evidence
from forge_paper.gem_tables12_13 import (
    GemTables12And13Error,
    render_gem_tables12_and_13,
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
    "GemTable1Error",
    "GemTable10Error",
    "GemTable2Error",
    "GemTable3Error",
    "GemTable4Error",
    "GemTable5Error",
    "GemTable6Error",
    "GemTable7Error",
    "GemTable8Error",
    "GemTable9Error",
    "GemTables12And13Error",
    "GemProseResultsError",
    "diagnose_experiment_matrix",
    "diagnose_paper",
    "render_v1_results",
    "render_completed_evidence_v1",
    "render_forge_generated_sample_atlas",
    "render_forge_generated_sample_figure",
    "render_gem_table1_final_evidence",
    "render_gem_table10_route_evidence",
    "render_gem_table2_forge_row",
    "render_gem_table3_route_assessment",
    "render_gem_table4_production_comparison",
    "render_gem_table5_decoder_source_ablation",
    "render_gem_table6_exact_l1_counts",
    "render_gem_table7_lipid_realism",
    "render_gem_table8_architecture_ablations",
    "render_gem_table9_catalogue_comparison",
    "render_gem_tables12_and_13",
    "render_gem_prose_results",
    "verify_paper",
]
