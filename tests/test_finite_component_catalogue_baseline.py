from __future__ import annotations

from pathlib import Path

from experiments.phase1.multireaction.catalogue_baseline import (
    run_finite_component_catalogue_baseline,
)
from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import sha256_file
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.model.finite_component_catalogue import build_finite_component_catalogues
from forge.model.reaction_program_evaluation import (
    load_reaction_program_training_references,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/finite_component_catalogue_baseline_v1.json"
PROGRAM_CONFIG = REPO / "configs/multireaction/lnpdb_reaction_programs_v1.json"
UGI_ASSIGNMENTS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
ATLAS = REPO / "results/phase1/multireaction_program_corpus_v1/reaction_program_atlas.csv.gz"
SPLITS = REPO / "results/phase1/multireaction_program_corpus_v1/component_disjoint_splits.csv.gz"
UGI_REGISTRY = REPO / "data/vendor/qualified_reactions_v1.json"
UGI_PROGRAM = "ugi_3cr_agile"


def test_catalogue_support_is_exactly_the_train_fold_component_support() -> None:
    specs = {spec.program_id: spec for spec in load_reaction_program_specifications(PROGRAM_CONFIG)}
    ugi = Ugi3AssemblyAdapter.from_registry(UGI_REGISTRY)
    catalogues = build_finite_component_catalogues(
        ugi_assignments=UGI_ASSIGNMENTS,
        multireaction_atlas=ATLAS,
        multireaction_splits=SPLITS,
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
        ugi_roles=ugi.roles,
    )
    _, components = load_reaction_program_training_references(
        ugi_assignments=UGI_ASSIGNMENTS,
        multireaction_atlas=ATLAS,
        multireaction_splits=SPLITS,
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
        ugi_roles=ugi.roles,
    )

    assert set(catalogues) == {UGI_PROGRAM, *specs}
    for program_id, catalogue in catalogues.items():
        for role, support in catalogue.role_supports:
            assert set(support.values) == components[program_id][role]
        assert catalogue.tuple_space_upper_bound >= len(catalogue.depth_support.values)


def test_smoke_baseline_is_deterministic_and_reports_the_validity_breadth_tradeoff(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = run_finite_component_catalogue_baseline(
        CONFIG,
        REPO,
        first,
        profile="smoke",
        replicate=0,
    )
    repeated = run_finite_component_catalogue_baseline(
        CONFIG,
        REPO,
        second,
        profile="smoke",
        replicate=0,
    )

    assert result == repeated
    assert result["status"] == "pass"
    assert all(result["gates"].values())
    assert result["attempts_per_program"] == 8
    for metrics in result["metrics"]["per_program"].values():
        assert metrics["samples"] == 8
        assert metrics["component_novelty_fraction"] in {0.0, None}
        assert metrics["unique_open_ended_exact_l1_products_per_1000_attempts"] == 0.0
    for catalogue in result["catalogue"].values():
        assert catalogue["maximum_attainable_component_novelty_fraction"] == 0.0
        assert catalogue["source_product_coverage_by_fold"]["train"]["coverage_fraction"] == 1.0
        assert catalogue["source_product_coverage_by_fold"]["heldout"]["coverage_fraction"] == 0.0
    assert sha256_file(first / "result.json") == sha256_file(second / "result.json")
    assert sha256_file(first / "samples.jsonl.gz") == sha256_file(second / "samples.jsonl.gz")
