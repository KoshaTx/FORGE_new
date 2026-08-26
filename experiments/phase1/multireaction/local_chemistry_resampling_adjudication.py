"""Adjudicate evaluation-only local-chemistry resampling across frozen seeds.

This module compares the constrained terminal decoder with the already frozen unconstrained
production result.  It never retrains a model, selects a candidate, or treats local structural
support as synthesis success.  Every input run is independently verified before aggregation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from experiments._runtime import verify_run_directory
from experiments._runtime.source import source_fingerprint
from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import iter_jsonl, read_json_object, write_json

from .final_production_adjudication import FINAL_ARM, seed_summary
from .production_adjudication import SynthesisProgramProductionAdjudicationError

ADJUDICATION_SCHEMA = "forge.local_chemistry_resampling_adjudication.v1"
EXPERIMENT_ID = "phase1-local-chemistry-resampling"
IMPLEMENTATION = "model.synthesis-program-local-chemistry-resampling.v1"
EVALUATION_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
SAMPLES_SCHEMA = "forge.synthesis_program_production_samples.v1"
TERMINAL_POLICY = "strict_local_chemistry_argmax"
PROGRAMS = (
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
    "ugi_3cr_agile",
)
PAPER_METRICS = (
    "raw_valid_fraction",
    "exact_l1_yield_per_attempt",
    "unique_exact_l1_products_per_1000_attempts",
    "unique_open_ended_exact_l1_products_per_1000_attempts",
    "whole_lipid_novelty_fraction",
    "component_novelty_fraction",
    "internal_diversity",
    "effective_component_count",
)


def _all_true(value: object) -> bool:
    return (
        isinstance(value, Mapping) and bool(value) and all(item is True for item in value.values())
    )


def _artifact_path(run_dir: Path, manifest: Mapping[str, Any], label: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(label), Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            f"local-chemistry run has no {label} artifact"
        )
    relative = artifacts[label].get("path")
    if not isinstance(relative, str) or not relative:
        raise SynthesisProgramProductionAdjudicationError(
            f"local-chemistry {label} artifact path is malformed"
        )
    path = run_dir / "stages/resampling" / relative
    if pin_record(path, run_dir)["sha256"] != artifacts[label].get("sha256"):
        raise SynthesisProgramProductionAdjudicationError(
            f"local-chemistry {label} artifact digest changed"
        )
    return path


def _load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="local-chemistry run manifest",
    )
    if (
        run.get("experiment_id") != EXPERIMENT_ID
        or run.get("profile") != "full"
        or run.get("status") != "complete"
        or set(run.get("stages", {})) != {"resampling"}
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry adjudication requires a complete full-profile resampling run"
        )
    manifest = read_json_object(
        run_dir / "stages/resampling/manifest.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="local-chemistry stage manifest",
    )
    if (
        manifest.get("implementation") != IMPLEMENTATION
        or manifest.get("profile") != "full"
        or manifest.get("status") != "complete"
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry stage implementation or profile changed"
        )
    result_path = _artifact_path(run_dir, manifest, "result")
    samples_path = _artifact_path(run_dir, manifest, "samples")
    result = read_json_object(
        result_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="local-chemistry evaluation result",
    )
    replicate = int(run.get("replicate", -1))
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "pass"
        or result.get("profile") != "full"
        or int(result.get("replicate", -1)) != replicate
        or result.get("terminal_decode_policy") != TERMINAL_POLICY
        or not _all_true(result.get("gates"))
        or result.get("calls") != {"oracle": 0, "route": 0}
        or result.get("selection", {}).get("candidate_selection") is not False
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"local-chemistry result identity or gates failed for replicate {replicate}"
        )
    if result.get("samples", {}).get("sha256") != manifest["artifacts"]["samples"].get("sha256"):
        raise SynthesisProgramProductionAdjudicationError(
            f"local-chemistry result does not bind its sample ledger for replicate {replicate}"
        )
    return {
        "run_dir": run_dir,
        "run": run,
        "manifest": manifest,
        "result": result,
        "result_path": result_path,
        "samples_path": samples_path,
    }


def _liability_audit(
    path: Path,
    *,
    final_step: int,
    attempts_per_program: int,
) -> dict[str, dict[str, Any]]:
    """Audit declared local liabilities directly from final-heldout molecular graphs."""

    counts = {
        program: Counter(
            attempts=0,
            valid=0,
            exact_l1=0,
            policy_applied=0,
            oxygen_oxygen_bonds=0,
            nitrogen_oxygen_bonds=0,
            three_membered_rings=0,
            molecules_with_declared_liability=0,
        )
        for program in PROGRAMS
    }
    rows = iter_jsonl(path)
    try:
        header = next(rows)
    except StopIteration as error:
        raise SynthesisProgramProductionAdjudicationError("sample ledger is empty") from error
    if not isinstance(header, Mapping) or header.get("schema_version") != SAMPLES_SCHEMA:
        raise SynthesisProgramProductionAdjudicationError("sample ledger schema changed")
    observed_rows = 0
    for row in rows:
        observed_rows += 1
        if not isinstance(row, Mapping):
            raise SynthesisProgramProductionAdjudicationError("sample ledger row is malformed")
        if (
            row.get("evaluation_split") != "heldout"
            or int(row.get("checkpoint_step", -1)) != final_step
        ):
            continue
        program = str(row.get("program_id"))
        if program not in counts or row.get("arm_id") != FINAL_ARM:
            raise SynthesisProgramProductionAdjudicationError(
                "final-heldout sample has an unexpected program or arm"
            )
        summary = counts[program]
        summary["attempts"] += 1
        summary["policy_applied"] += int(row.get("local_chemistry_policy_applied") is True)
        valid = row.get("valid") is True
        summary["valid"] += int(valid)
        summary["exact_l1"] += int(valid and row.get("exact_l1_program") is True)
        if not valid:
            continue
        molecule = None
        with rdBase.BlockLogs():
            smiles = row.get("canonical_smiles")
            if isinstance(smiles, str) and smiles:
                molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise SynthesisProgramProductionAdjudicationError(
                "a row marked valid cannot be parsed as one connected molecule"
            )
        local = Counter()
        for bond in molecule.GetBonds():
            symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
            local["oxygen_oxygen_bonds"] += int(symbols == {"O"})
            local["nitrogen_oxygen_bonds"] += int(symbols == {"N", "O"})
        local["three_membered_rings"] = sum(
            len(ring) == 3 for ring in molecule.GetRingInfo().AtomRings()
        )
        for label, value in local.items():
            summary[label] += value
        summary["molecules_with_declared_liability"] += int(any(local.values()))
    if int(header.get("rows", -1)) != observed_rows:
        raise SynthesisProgramProductionAdjudicationError("sample ledger row count changed")
    for program, summary in counts.items():
        if summary["attempts"] != attempts_per_program:
            raise SynthesisProgramProductionAdjudicationError(
                f"{program} has {summary['attempts']} attempts, expected {attempts_per_program}"
            )
        if summary["policy_applied"] != attempts_per_program:
            raise SynthesisProgramProductionAdjudicationError(
                f"the local-chemistry policy was not applied to every {program} attempt"
            )
    return {
        program: {
            **dict(summary),
            "valid_fraction": summary["valid"] / summary["attempts"],
            "exact_l1_yield_per_attempt": summary["exact_l1"] / summary["attempts"],
            "declared_liability_fraction_among_valid": (
                summary["molecules_with_declared_liability"] / summary["valid"]
                if summary["valid"]
                else None
            ),
        }
        for program, summary in counts.items()
    }


def adjudicate_local_chemistry_resampling(
    run_dirs: Sequence[Path],
    original_adjudication_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify, audit and aggregate three constrained frozen-checkpoint replicates."""

    repo = repo.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError as error:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry adjudication output must stay inside the repository"
        ) from error
    runs = sorted((_load_run(path) for path in run_dirs), key=lambda item: item["run"]["replicate"])
    if len(runs) != 3 or [int(item["run"]["replicate"]) for item in runs] != [0, 1, 2]:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry adjudication requires replicates [0, 1, 2]"
        )
    seeds = [int(item["result"]["seed"]) for item in runs]
    if seeds != [20260825, 20260826, 20260827]:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry resampling seeds differ from the frozen production seeds"
        )
    if len({str(item["run"].get("source_sha256")) for item in runs}) != 1:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry replicates do not share one executable source"
        )
    if len({str(item["run"].get("spec_sha256")) for item in runs}) != 1:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry replicates do not share one experiment specification"
        )
    policy_pins = {
        (
            str(item["result"]["local_chemistry_support"]["path"]),
            str(item["result"]["local_chemistry_support"]["sha256"]),
        )
        for item in runs
    }
    design_pins = {
        (str(item["result"]["design"]["path"]), str(item["result"]["design"]["sha256"]))
        for item in runs
    }
    if len(policy_pins) != 1 or len(design_pins) != 1:
        raise SynthesisProgramProductionAdjudicationError(
            "local-chemistry replicates do not share one policy and design"
        )
    policy_path = resolve_pin(
        {"path": next(iter(policy_pins))[0], "sha256": next(iter(policy_pins))[1]},
        repo,
        label="local-chemistry support policy",
    )
    design_path = resolve_pin(
        {"path": next(iter(design_pins))[0], "sha256": next(iter(design_pins))[1]},
        repo,
        label="frozen production design",
    )
    design = read_json_object(
        design_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="frozen production design",
    )
    final_step = int(design["training"]["checkpoint_steps"][-1])
    attempts = int(
        design["evaluation"]["native_sampling"][
            "heldout_samples_per_supported_program_at_final_checkpoint_per_seed"
        ]
    )
    audits = [
        _liability_audit(item["samples_path"], final_step=final_step, attempts_per_program=attempts)
        for item in runs
    ]

    original_path = original_adjudication_path.resolve()
    try:
        original_path.relative_to(repo)
    except ValueError as error:
        raise SynthesisProgramProductionAdjudicationError(
            "original production adjudication must stay inside the repository"
        ) from error
    if not original_path.is_file():
        raise SynthesisProgramProductionAdjudicationError(
            f"original production adjudication is missing: {original_path}"
        )
    original = read_json_object(
        original_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="original production adjudication",
    )
    original_programs = original.get("arm_summaries", {}).get(FINAL_ARM)
    if not isinstance(original_programs, Mapping) or set(original_programs) != set(PROGRAMS):
        raise SynthesisProgramProductionAdjudicationError(
            "original production adjudication does not contain the final FORGE arm"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for program in PROGRAMS:
        per_seed = [
            item["result"]["checkpoint_metrics"][FINAL_ARM][str(final_step)]["heldout"][program]
            for item in runs
        ]
        for metrics, audit in zip(per_seed, audits, strict=True):
            if int(metrics["samples"]) != attempts:
                raise SynthesisProgramProductionAdjudicationError(
                    f"result attempt count changed for {program}"
                )
            if abs(float(metrics["raw_valid_fraction"]) - audit[program]["valid_fraction"]) > 1e-15:
                raise SynthesisProgramProductionAdjudicationError(
                    f"sample ledger and result validity disagree for {program}"
                )
            if (
                abs(
                    float(metrics["exact_l1_yield_per_attempt"])
                    - audit[program]["exact_l1_yield_per_attempt"]
                )
                > 1e-15
            ):
                raise SynthesisProgramProductionAdjudicationError(
                    f"sample ledger and result exact-L1 yield disagree for {program}"
                )
        metric_summaries = {
            metric: seed_summary([float(item[metric]) for item in per_seed])
            for metric in PAPER_METRICS
        }
        abstention = seed_summary(
            [float(item["strict_constraint_abstentions"]) / attempts for item in per_seed]
        )
        original_yield = original_programs[program]["exact_l1_yield_per_attempt"]
        constrained_mean = metric_summaries["exact_l1_yield_per_attempt"]["mean"]
        original_mean = float(original_yield["mean"])
        summaries[program] = {
            **metric_summaries,
            "strict_constraint_abstention_fraction": abstention,
            "yield_comparison_to_original_decoder": {
                "original_by_seed": original_yield["by_seed"],
                "original_mean": original_mean,
                "constrained_mean": constrained_mean,
                "absolute_difference": constrained_mean - original_mean,
                "retained_fraction": constrained_mean / original_mean,
                "descriptive_not_a_retraining_comparison": True,
            },
        }

    liabilities = {
        program: {
            label: sum(int(audit[program][label]) for audit in audits)
            for label in (
                "attempts",
                "valid",
                "exact_l1",
                "policy_applied",
                "oxygen_oxygen_bonds",
                "nitrogen_oxygen_bonds",
                "three_membered_rings",
                "molecules_with_declared_liability",
            )
        }
        for program in PROGRAMS
    }
    no_declared_liabilities = all(
        summary["molecules_with_declared_liability"] == 0 for summary in liabilities.values()
    )
    gates = {
        "three_verified_full_replicates": True,
        "replicates_and_seeds_exact": True,
        "one_executable_source": True,
        "one_experiment_specification": True,
        "one_frozen_checkpoint_design": True,
        "one_training_fold_local_chemistry_policy": True,
        "attempt_budgets_match": True,
        "sample_ledgers_match_reported_validity_and_exact_l1": True,
        "policy_applied_to_every_attempt": all(
            summary["policy_applied"] == summary["attempts"] for summary in liabilities.values()
        ),
        "declared_oxygen_oxygen_bonds_absent": all(
            summary["oxygen_oxygen_bonds"] == 0 for summary in liabilities.values()
        ),
        "declared_nitrogen_oxygen_bonds_absent": all(
            summary["nitrogen_oxygen_bonds"] == 0 for summary in liabilities.values()
        ),
        "three_membered_rings_absent": all(
            summary["three_membered_rings"] == 0 for summary in liabilities.values()
        ),
        "candidate_selection_absent": True,
        "training_calls_zero": True,
        "route_or_oracle_calls_zero": True,
        "repairs_or_retries_absent": all(
            item["result"]["gates"]["no_repairs_or_retries"] is True for item in runs
        ),
        "reductive_amination_substructure_rate_absent": True,
    }
    result = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "complete" if all(gates.values()) else "failed",
        "scientific_decision": {
            "declared_local_liability_control_qualified": no_declared_liabilities,
            "useful_for_qualitative_sampling_and_downstream_acceptance": no_declared_liabilities,
            "supersedes_original_production_metrics": False,
            "reason_not_superseding": (
                "The weights are unchanged and strict decoding trades yield for bounded local "
                "support; the frozen original decoder remains the primary production result."
            ),
            "data_sparse_family_tradeoff_reported": True,
        },
        "gates": gates,
        "programs": list(PROGRAMS),
        "replicates": [0, 1, 2],
        "seeds": seeds,
        "final_checkpoint_step": final_step,
        "attempts_per_program_per_seed": attempts,
        "terminal_decode_policy": TERMINAL_POLICY,
        "program_summaries": summaries,
        "declared_liability_audit_across_seeds": liabilities,
        "per_seed_liability_audit": audits,
        "policy": pin_record(policy_path, repo),
        "design": pin_record(design_path, repo),
        "original_production_adjudication": pin_record(original_path, repo),
        "input_source_sha256": runs[0]["run"]["source_sha256"],
        "input_spec_sha256": runs[0]["run"]["spec_sha256"],
        "adjudicator_source_sha256": source_fingerprint(repo),
        "implementation": pin_record(Path(__file__), repo),
        "evidence": [
            {
                "replicate": int(item["run"]["replicate"]),
                "seed": int(item["result"]["seed"]),
                "run_id": item["run"]["run_id"],
                "run_manifest": pin_record(item["run_dir"] / "run.json", repo),
                "stage_manifest": pin_record(
                    item["run_dir"] / "stages/resampling/manifest.json", repo
                ),
                "result": pin_record(item["result_path"], repo),
                "samples": pin_record(item["samples_path"], repo),
            }
            for item in runs
        ],
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "Training-supported local chemistry is not synthesis success, stability, safety, activity or route closure.",
            "Absence from local training support is an abstention, not proof of chemical impossibility.",
            "This evaluation-only experiment does not change model weights.",
            "The constrained decoder does not use a finite component vocabulary.",
            "BL and LX exact replay do not inherit Ugi prospective validation.",
        ],
    }
    write_json(output_path, result)
    return result


__all__ = [
    "ADJUDICATION_SCHEMA",
    "adjudicate_local_chemistry_resampling",
]
