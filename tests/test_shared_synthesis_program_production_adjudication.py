from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from experiments.phase1.multireaction.production_adjudication import (
    _finite_catalogue_comparison,
    adjudicate_production_runs,
    paired_seed_bootstrap,
    paired_seed_difference_interval,
)
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
DESIGN = REPO / "configs/multireaction/shared_production_comparison_design_v1.json"


def test_paired_seed_bootstrap_is_deterministic_and_uses_seed_clusters() -> None:
    first = paired_seed_bootstrap(
        [0.90, 0.91, 0.89],
        [0.895, 0.905, 0.885],
        resamples=10_000,
        seed=20260828,
        confidence_level=0.95,
    )
    second = paired_seed_bootstrap(
        [0.90, 0.91, 0.89],
        [0.895, 0.905, 0.885],
        resamples=10_000,
        seed=20260828,
        confidence_level=0.95,
    )

    assert first == second
    assert np.isclose(first["point_estimate"], 0.005)
    assert first["one_sided_bound"] <= 0.005000000000001
    assert first["resampling_unit"] == "paired_independent_training_seed_metric_cell"
    assert first["molecule_rows_treated_as_independent_replicates"] is False


def test_catalogue_comparison_keeps_validity_and_open_endedness_separate() -> None:
    design = json.loads(DESIGN.read_text())
    programmes = tuple(design["programs"])

    def model_metrics() -> dict[str, float | None]:
        return {
            "raw_valid_fraction": 0.9,
            "exact_l1_yield_per_attempt": 0.8,
            "unique_exact_l1_products_per_1000_attempts": 700.0,
            "unique_whole_product_novel_exact_l1_products_per_1000_attempts": 600.0,
            "unique_open_ended_exact_l1_products_per_1000_attempts": 500.0,
            "internal_diversity": 0.7,
            "effective_component_count": 30.0,
            "component_novelty_fraction": 0.6,
            "whole_lipid_novelty_fraction": 0.8,
        }

    def catalogue_metrics() -> dict[str, float]:
        return {
            "valid_fraction": 1.0,
            "exact_l1_yield_per_attempt": 1.0,
            "unique_exact_l1_products_per_1000_attempts": 800.0,
            "unique_whole_product_novel_exact_l1_products_per_1000_attempts": 700.0,
            "unique_open_ended_exact_l1_products_per_1000_attempts": 0.0,
            "mean_pairwise_ecfp4_distance": 0.6,
            "effective_component_count": 20.0,
            "component_novelty_fraction": 0.0,
            "whole_product_novel_to_train_fraction": 0.7,
        }

    production_runs = [
        {
            "evaluation": {
                "checkpoint_metrics": {
                    "shared_three_program_conditioned": {
                        "1700": {
                            "heldout": {program_id: model_metrics() for program_id in programmes}
                        }
                    }
                }
            }
        }
        for _ in range(3)
    ]
    for production in production_runs:
        zero_survivor_metrics = production["evaluation"]["checkpoint_metrics"][
            "shared_three_program_conditioned"
        ]["1700"]["heldout"]["bl_2023_repeated_aza_michael"]
        zero_survivor_metrics["effective_component_count"] = None
        zero_survivor_metrics["component_novelty_fraction"] = None
    catalogue_runs = [
        {
            "result": {
                "metrics": {
                    "per_program": {program_id: catalogue_metrics() for program_id in programmes}
                },
                "catalogue": {
                    program_id: {"component_tuple_space_upper_bound": 100}
                    for program_id in programmes
                },
                "sampled_component_metrics": {
                    program_id: {
                        "effective_component_count": 20.0,
                        "component_novelty_fraction": 0.0,
                    }
                    for program_id in programmes
                },
            }
        }
        for _ in range(3)
    ]

    comparison = _finite_catalogue_comparison(production_runs, catalogue_runs, design)

    ugi = comparison["programs"]["ugi_3cr_agile"]["metrics"]
    assert np.isclose(ugi["raw_valid_fraction"]["forge_minus_catalogue_mean"], -0.1)
    assert (
        ugi["unique_open_ended_exact_l1_products_per_1000_attempts"]["forge_minus_catalogue_mean"]
        == 500.0
    )
    interval = ugi["unique_open_ended_exact_l1_products_per_1000_attempts"][
        "paired_seed_difference_interval"
    ]
    assert interval["lower_bound"] == interval["upper_bound"] == 500.0
    assert comparison["primary_metric"] == ("unique_open_ended_exact_l1_products_per_1000_attempts")
    bl_effective_count = comparison["programs"]["bl_2023_repeated_aza_michael"]["metrics"][
        "effective_component_count"
    ]
    assert bl_effective_count["forge_by_seed"] == [None, None, None]
    assert bl_effective_count["forge_mean"] is None
    assert bl_effective_count["catalogue_mean"] == 20.0
    assert bl_effective_count["forge_minus_catalogue_mean"] is None
    assert bl_effective_count["paired_seed_difference_interval"] is None
    assert bl_effective_count["undefined_cells_imputed"] is False


def test_paired_seed_difference_interval_is_two_sided_and_seed_clustered() -> None:
    interval = paired_seed_difference_interval(
        [3.0, 5.0, 7.0],
        [1.0, 2.0, 3.0],
        resamples=10_000,
        seed=20260828,
        confidence_level=0.95,
    )

    assert interval["point_estimate"] == 3.0
    assert interval["lower_bound"] <= 3.0 <= interval["upper_bound"]
    assert interval["resampling_unit"] == "paired_independent_seed_metric_cell"
    assert interval["molecule_rows_treated_as_independent_replicates"] is False


def _metric_row(value: float = 0.90) -> dict[str, float | int | bool]:
    return {
        "samples": 3_072,
        "raw_valid_fraction": value,
        "connected_fraction": value,
        "exact_l1_decomposition_coverage": value,
        "exact_forward_replay_precision": 0.98,
        "internal_diversity": 0.70,
        "effective_component_count": 20.0,
        "component_novelty_fraction": 0.50,
        "fixed_state_failures": 0,
        "support_overflow_count": 0,
        "coverage_and_precision_reported": True,
    }


def _write_run(
    root: Path,
    *,
    replicate: int,
    seed: int,
    design_pin: dict[str, str],
    challenger_value: float = 0.90,
) -> Path:
    run_id = f"run-{replicate}"
    run_dir = root / "runs" / run_id
    for stage in ("training", "evaluation"):
        (run_dir / "stages" / stage / "artifacts").mkdir(parents=True, exist_ok=True)
    training = {
        "schema_version": "forge.synthesis_program_production_training_result.v1",
        "profile": "full",
        "replicate": replicate,
        "seed": seed,
        "gates": {
            "candidate_selection_absent": True,
            "route_or_oracle_calls_zero": True,
        },
    }
    baseline = _metric_row()
    challenger = _metric_row(challenger_value)
    programs = (
        "ugi_3cr_agile",
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    )
    evaluation = {
        "schema_version": "forge.synthesis_program_production_evaluation_result.v1",
        "status": "pass",
        "profile": "full",
        "replicate": replicate,
        "seed": seed,
        "checkpoint_metrics": {
            "ugi_only_conditioned": {"1700": {"heldout": {"ugi_3cr_agile": baseline}}},
            "shared_three_program_conditioned": {
                "1700": {"heldout": {program_id: challenger for program_id in programs}}
            },
            "shared_three_program_null": {
                "1700": {"heldout": {program_id: _metric_row(0.70) for program_id in programs}}
            },
            "shared_three_program_program_id_cyclic": {
                "1700": {"heldout": {program_id: _metric_row(0.60) for program_id in programs}}
            },
        },
        "component_disjoint_metrics": {
            "ugi_only_conditioned": {"ugi_3cr_agile": {"exact_l1_decomposition_coverage": 0.90}},
            "shared_three_program_conditioned": {
                "ugi_3cr_agile": {"exact_l1_decomposition_coverage": challenger_value}
            },
        },
        "gates": {
            "coverage_and_precision_reported": True,
            "all_arms_evaluated": True,
            "candidate_selection_absent": True,
            "route_or_oracle_calls_zero": True,
        },
        "calls": {"oracle": 0, "route": 0},
        "selection": {"candidate_selection": False},
    }
    result_paths = {}
    for stage, document in (("training", training), ("evaluation", evaluation)):
        path = run_dir / "stages" / stage / "artifacts" / "result.json"
        path.write_text(json.dumps(document))
        result_paths[stage] = path
    samples_path = run_dir / "stages/evaluation/artifacts/samples.jsonl.gz"
    sample_rows = [
        {
            "arm_id": arm_id,
            "program_id": program_id,
            "evaluation_split": "heldout",
            "checkpoint_step": 1700,
            "valid": True,
            "exact_l1_program": arm_id == "shared_three_program_conditioned",
            "forward_verified_trace_count": int(arm_id == "shared_three_program_conditioned"),
        }
        for arm_id in (
            "shared_three_program_conditioned",
            "shared_three_program_null",
        )
        for program_id in programs
        for _ in range(3_072)
    ]
    with gzip.open(samples_path, "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "schema_version": "forge.synthesis_program_production_samples.v1",
                    "rows": len(sample_rows),
                }
            )
            + "\n"
        )
        for row in sample_rows:
            stream.write(json.dumps(row) + "\n")
    stage_documents = {
        "training": {
            "implementation": "model.shared-synthesis-program-production-training.v1",
            "external_inputs": {"production_design": design_pin},
            "artifacts": {
                "result": {
                    "path": "artifacts/result.json",
                    "sha256": str(sha256_file(result_paths["training"])),
                }
            },
        },
        "evaluation": {
            "implementation": "model.shared-synthesis-program-production-evaluation.v1",
            "external_inputs": {"production_design": design_pin},
            "artifacts": {
                "result": {
                    "path": "artifacts/result.json",
                    "sha256": str(sha256_file(result_paths["evaluation"])),
                },
                "samples": {
                    "path": "artifacts/samples.jsonl.gz",
                    "sha256": str(sha256_file(samples_path)),
                },
            },
        },
    }
    for stage, document in stage_documents.items():
        (run_dir / "stages" / stage / "manifest.json").write_text(json.dumps(document))
    run = {
        "experiment_id": "phase1-transformer-synthesis-program-production",
        "profile": "full",
        "status": "complete",
        "replicate": replicate,
        "run_id": run_id,
        "source_sha256": "s" * 64,
        "spec_sha256": "p" * 64,
        "stages": {"training": {}, "evaluation": {}},
    }
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


def test_adjudicator_pins_three_runs_and_preserves_a_negative_result(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.phase1.multireaction.production_adjudication as adjudication

    design_path = tmp_path / "configs/multireaction/design.json"
    design_path.parent.mkdir(parents=True)
    design_path.write_bytes(DESIGN.read_bytes())
    design_pin = {
        "path": "configs/multireaction/design.json",
        "sha256": str(sha256_file(design_path)),
    }
    seeds = [20260825, 20260826, 20260827]
    run_dirs = [
        _write_run(
            tmp_path,
            replicate=replicate,
            seed=seed,
            design_pin=design_pin,
            challenger_value=0.80,
        )
        for replicate, seed in enumerate(seeds)
    ]
    implementation = tmp_path / "experiments/production_adjudication.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# frozen test implementation\n")
    monkeypatch.setattr(adjudication, "verify_run_directory", lambda _: {"status": "verified"})
    monkeypatch.setattr(adjudication, "source_fingerprint", lambda _: "a" * 64)
    monkeypatch.setattr(adjudication, "__file__", str(implementation))

    output = tmp_path / "results/adjudication/result.json"
    result = adjudicate_production_runs(run_dirs, tmp_path, output)

    assert result["status"] == "adjudicated_noninferiority_fail"
    assert result["scientific_decision"]["negative_result_is_valid"] is True
    assert result["gates"]["held_reaction_family_not_a_hard_gate"] is True
    assert result["gates"]["catalogue_free_posthoc_attempt_budgets_match"] is True
    comparison = result["catalogue_free_conditioning_vs_posthoc_filtering"]
    assert (
        comparison["programs"]["ugi_3cr_agile"]["mean_conditioned_minus_posthoc_exact_l1_yield"]
        == 1.0
    )
    assert len(result["evidence"]) == 3
    assert output.is_file()
    assert all(
        row["evaluation_result"]["path"].startswith("results/adjudication/evidence/")
        for row in result["evidence"]
    )
    assert all(
        row["evaluation_samples"]["path"].startswith("results/adjudication/evidence/")
        for row in result["evidence"]
    )
    assert all(
        row["evaluation_samples"]["path"].endswith(".jsonl.gz") for row in result["evidence"]
    )
