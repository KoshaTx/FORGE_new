"""Adjudicate the frozen seed-0 role-local ring-morphology diagnostic.

The diagnostic compares two decoders applied to the same trained checkpoint.  It does not retrain,
select candidates, call a route model, or call an oracle.  Its thresholds are hash-pinned before the
full run, and a negative result remains a complete result.
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
from forge.model.local_chemistry_support import LocalChemistrySupport

CONFIG_SCHEMA = "forge.local_morphology_seed0_adjudication_config.v1"
RESULT_SCHEMA = "forge.local_morphology_seed0_adjudication.v1"
EXPERIMENT_ID = "phase1-local-morphology-resampling-seed0-h100-v2"
IMPLEMENTATION = "model.synthesis-program-local-morphology-resampling.v2"
EVALUATION_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
SAMPLES_SCHEMA = "forge.synthesis_program_production_samples.v1"
TERMINAL_POLICY = "strict_local_chemistry_argmax"


class LocalMorphologyAdjudicationError(ValueError):
    """The diagnostic evidence differs from its frozen, nonselecting contract."""


def _artifact_path(run_dir: Path, manifest: Mapping[str, Any], label: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(label), Mapping):
        raise LocalMorphologyAdjudicationError(f"diagnostic run has no {label} artifact")
    relative = artifacts[label].get("path")
    if not isinstance(relative, str) or not relative:
        raise LocalMorphologyAdjudicationError(f"diagnostic {label} path is malformed")
    path = run_dir / "stages/resampling" / relative
    if pin_record(path, run_dir)["sha256"] != artifacts[label].get("sha256"):
        raise LocalMorphologyAdjudicationError(f"diagnostic {label} digest changed")
    return path


def _parse_molecule(smiles: object, *, label: str) -> Chem.Mol:
    molecule = None
    with rdBase.BlockLogs():
        if isinstance(smiles, str) and smiles:
            molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise LocalMorphologyAdjudicationError(f"{label} is not one valid connected molecule")
    return molecule


def _component_values(value: object, *, label: str) -> tuple[object, ...]:
    """Normalize one role payload without collapsing repeated-program components."""

    values = (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else (value,)
    )
    if not values:
        raise LocalMorphologyAdjudicationError(f"{label} has no components")
    return values


def _audit_samples(
    path: Path,
    *,
    support: LocalChemistrySupport,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    programs = tuple(str(value) for value in evaluation["programs"])
    attempts_per_program = int(evaluation["attempts_per_program"])
    counts = {
        program: Counter(
            attempts=0,
            valid=0,
            exact_l1=0,
            policy_applied=0,
            oxygen_oxygen_bonds=0,
            nitrogen_oxygen_bonds=0,
            oxygen_containing_three_or_four_membered_rings=0,
            ugi_oxygen_rings_outside_amine_head_in_exact_l1_components=0,
            unsupported_role_ring_signatures_in_exact_l1_components=0,
            exact_l1_component_rings_assessed=0,
        )
        for program in programs
    }
    rows = iter_jsonl(path)
    try:
        header = next(rows)
    except StopIteration as error:
        raise LocalMorphologyAdjudicationError("diagnostic sample ledger is empty") from error
    if not isinstance(header, Mapping) or header.get("schema_version") != SAMPLES_SCHEMA:
        raise LocalMorphologyAdjudicationError("diagnostic sample schema changed")
    observed_rows = 0
    for row in rows:
        observed_rows += 1
        if not isinstance(row, Mapping):
            raise LocalMorphologyAdjudicationError("diagnostic sample row is malformed")
        if row.get("evaluation_split") != evaluation["split"] or int(
            row.get("checkpoint_step", -1)
        ) != int(evaluation["checkpoint_step"]):
            continue
        program = str(row.get("program_id"))
        if program not in counts or row.get("arm_id") != evaluation["arm_id"]:
            raise LocalMorphologyAdjudicationError("diagnostic sample identity changed")
        summary = counts[program]
        summary["attempts"] += 1
        summary["policy_applied"] += int(row.get("local_chemistry_policy_applied") is True)
        valid = row.get("valid") is True
        exact_l1 = valid and row.get("exact_l1_program") is True
        summary["valid"] += int(valid)
        summary["exact_l1"] += int(exact_l1)
        if not valid:
            continue
        molecule = _parse_molecule(row.get("canonical_smiles"), label="valid generated product")
        for bond in molecule.GetBonds():
            symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
            summary["oxygen_oxygen_bonds"] += int(symbols == {"O"})
            summary["nitrogen_oxygen_bonds"] += int(symbols == {"N", "O"})
        summary["oxygen_containing_three_or_four_membered_rings"] += sum(
            len(ring) in {3, 4}
            and any(molecule.GetAtomWithIdx(index).GetSymbol() == "O" for index in ring)
            for ring in molecule.GetRingInfo().AtomRings()
        )
        if not exact_l1:
            continue
        traces = row.get("exact_l1_traces")
        if not isinstance(traces, list) or not traces:
            raise LocalMorphologyAdjudicationError("exact-L1 row has no decomposition trace")
        for trace in traces:
            components = trace.get("components_by_role") if isinstance(trace, Mapping) else None
            if not isinstance(components, Mapping) or not components:
                raise LocalMorphologyAdjudicationError("exact-L1 component trace is malformed")
            for role, value in components.items():
                role_label = f"exact-L1 {program}/{role}"
                for component_index, smiles in enumerate(
                    _component_values(value, label=role_label)
                ):
                    component = _parse_molecule(
                        smiles,
                        label=f"{role_label} component[{component_index}]",
                    )
                    for ring in component.GetRingInfo().AtomRings():
                        signature = tuple(
                            (str(role), component.GetAtomWithIdx(index).GetSymbol())
                            for index in ring
                        )
                        summary["exact_l1_component_rings_assessed"] += 1
                        summary["unsupported_role_ring_signatures_in_exact_l1_components"] += int(
                            not support.allows_role_cycle(program, signature)
                        )
                        if program == "ugi_3cr_agile" and str(role) != "amine_head":
                            summary[
                                "ugi_oxygen_rings_outside_amine_head_in_exact_l1_components"
                            ] += int(
                                any(
                                    component.GetAtomWithIdx(index).GetSymbol() == "O"
                                    for index in ring
                                )
                            )
    if int(header.get("rows", -1)) != observed_rows:
        raise LocalMorphologyAdjudicationError("diagnostic sample row count changed")
    for program, summary in counts.items():
        if summary["attempts"] != attempts_per_program:
            raise LocalMorphologyAdjudicationError(
                f"{program} has {summary['attempts']} attempts, expected {attempts_per_program}"
            )
        if summary["policy_applied"] != attempts_per_program:
            raise LocalMorphologyAdjudicationError(
                f"role-local morphology was not applied to every {program} attempt"
            )
    return {program: dict(summary) for program, summary in counts.items()}


def _final_metrics(result: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        values = result["checkpoint_metrics"][evaluation["arm_id"]][
            str(evaluation["checkpoint_step"])
        ][evaluation["split"]]
    except (KeyError, TypeError) as error:
        raise LocalMorphologyAdjudicationError("final heldout metric cell is missing") from error
    programs = tuple(str(value) for value in evaluation["programs"])
    if not isinstance(values, Mapping) or set(values) != set(programs):
        raise LocalMorphologyAdjudicationError("final heldout program set changed")
    return {program: values[program] for program in programs}


def adjudicate_local_morphology_seed0(
    run_dir: Path,
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify one full seed-0 run and apply the prespecified morphology/yield gates."""

    repo = repo.resolve()
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError as error:
        raise LocalMorphologyAdjudicationError(
            "adjudication output must stay in the repository"
        ) from error
    if output_path.exists():
        raise LocalMorphologyAdjudicationError(f"adjudication output already exists: {output_path}")
    config = read_json_object(
        config_path,
        error=LocalMorphologyAdjudicationError,
        label="local morphology adjudication config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "evaluation",
        "acceptance",
        "reporting",
        "nonclaims",
    }:
        raise LocalMorphologyAdjudicationError("local morphology adjudication config changed")
    evaluation = config["evaluation"]
    acceptance = config["acceptance"]
    reporting = config["reporting"]
    if not isinstance(evaluation, Mapping) or not isinstance(acceptance, Mapping):
        raise LocalMorphologyAdjudicationError("adjudication policy is malformed")
    if reporting != {
        "candidate_selection": False,
        "training_calls": 0,
        "route_calls": 0,
        "oracle_calls": 0,
        "negative_results_are_reported": True,
        "thresholds_frozen_before_full_seed0_run": True,
    }:
        raise LocalMorphologyAdjudicationError("adjudication reporting guardrails changed")

    baseline_path = resolve_pin(config["inputs"]["baseline_result"], repo, label="baseline_result")
    policy_path = resolve_pin(
        config["inputs"]["morphology_policy"], repo, label="morphology_policy"
    )
    baseline = read_json_object(
        baseline_path,
        error=LocalMorphologyAdjudicationError,
        label="seed-0 local-chemistry baseline",
    )
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            policy_path,
            error=LocalMorphologyAdjudicationError,
            label="role-local morphology policy",
        )
    )
    if not support.enforces_role_cycles:
        raise LocalMorphologyAdjudicationError("morphology policy has no full-cycle support")

    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json", error=LocalMorphologyAdjudicationError, label="diagnostic run"
    )
    if (
        run.get("experiment_id") != EXPERIMENT_ID
        or run.get("profile") != "full"
        or run.get("status") != "complete"
        or int(run.get("replicate", -1)) != int(evaluation["replicate"])
        or set(run.get("stages", {})) != {"resampling"}
    ):
        raise LocalMorphologyAdjudicationError("diagnostic run identity changed")
    manifest = read_json_object(
        run_dir / "stages/resampling/manifest.json",
        error=LocalMorphologyAdjudicationError,
        label="diagnostic stage manifest",
    )
    if (
        manifest.get("implementation") != IMPLEMENTATION
        or manifest.get("profile") != "full"
        or manifest.get("status") != "complete"
    ):
        raise LocalMorphologyAdjudicationError("diagnostic stage identity changed")
    result_path = _artifact_path(run_dir, manifest, "result")
    samples_path = _artifact_path(run_dir, manifest, "samples")
    result = read_json_object(
        result_path, error=LocalMorphologyAdjudicationError, label="diagnostic result"
    )
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "pass"
        or result.get("profile") != "full"
        or int(result.get("replicate", -1)) != int(evaluation["replicate"])
        or int(result.get("seed", -1)) != int(evaluation["seed"])
        or result.get("terminal_decode_policy") != TERMINAL_POLICY
        or result.get("calls") != {"oracle": 0, "route": 0}
        or result.get("selection", {}).get("candidate_selection") is not False
        or result.get("local_chemistry_support", {}).get("sha256")
        != config["inputs"]["morphology_policy"]["sha256"]
    ):
        raise LocalMorphologyAdjudicationError("diagnostic result identity or guardrails failed")

    baseline_metrics = _final_metrics(baseline, evaluation)
    diagnostic_metrics = _final_metrics(result, evaluation)
    sample_audit = _audit_samples(samples_path, support=support, evaluation=evaluation)
    comparisons: dict[str, Any] = {}
    gates: dict[str, bool] = {
        "oxygen_containing_three_or_four_membered_rings_zero": sum(
            row["oxygen_containing_three_or_four_membered_rings"] for row in sample_audit.values()
        )
        <= int(acceptance["maximum_oxygen_containing_three_or_four_membered_rings"]),
        "ugi_oxygen_rings_outside_amine_head_zero": sample_audit["ugi_3cr_agile"][
            "ugi_oxygen_rings_outside_amine_head_in_exact_l1_components"
        ]
        <= int(acceptance["maximum_ugi_oxygen_rings_outside_amine_head_in_exact_l1_components"]),
        "unsupported_exact_l1_role_ring_signatures_zero": sum(
            row["unsupported_role_ring_signatures_in_exact_l1_components"]
            for row in sample_audit.values()
        )
        <= int(acceptance["maximum_unsupported_role_ring_signatures_in_exact_l1_components"]),
    }
    for program in evaluation["programs"]:
        before = baseline_metrics[program]
        after = diagnostic_metrics[program]
        try:
            yield_retained = float(after["exact_l1_yield_per_attempt"]) / float(
                before["exact_l1_yield_per_attempt"]
            )
            diversity_decrease = float(before["internal_diversity"]) - float(
                after["internal_diversity"]
            )
            component_count_retained = float(after["effective_component_count"]) / float(
                before["effective_component_count"]
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise LocalMorphologyAdjudicationError(
                f"comparison metric is undefined for {program}"
            ) from error
        comparisons[program] = {
            "baseline": {
                key: before[key]
                for key in (
                    "exact_l1_yield_per_attempt",
                    "internal_diversity",
                    "effective_component_count",
                )
            },
            "diagnostic": {
                key: after[key]
                for key in (
                    "exact_l1_yield_per_attempt",
                    "internal_diversity",
                    "effective_component_count",
                )
            },
            "exact_l1_yield_retained_fraction": yield_retained,
            "internal_diversity_absolute_decrease": diversity_decrease,
            "effective_component_count_retained_fraction": component_count_retained,
        }
        gates[f"{program}:exact_l1_yield_retained"] = yield_retained >= float(
            acceptance["minimum_exact_l1_yield_retained_fraction_per_program"]
        )
        gates[f"{program}:internal_diversity_preserved"] = diversity_decrease <= float(
            acceptance["maximum_internal_diversity_absolute_decrease_per_program"]
        )
        gates[f"{program}:effective_component_count_preserved"] = component_count_retained >= float(
            acceptance["minimum_effective_component_count_retained_fraction_per_program"]
        )

    qualified = all(gates.values())
    output = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "qualified_for_three_seed_resampling_and_atlas": qualified,
        "scientific_decision": (
            "promote_role_local_morphology_policy"
            if qualified
            else "do_not_promote_report_negative_seed0_result"
        ),
        "config": pin_record(config_path, repo),
        "run": pin_record(run_dir / "run.json", repo),
        "stage_manifest": pin_record(run_dir / "stages/resampling/manifest.json", repo),
        "result": pin_record(result_path, repo),
        "samples": pin_record(samples_path, repo),
        "baseline_result": pin_record(baseline_path, repo),
        "morphology_policy": pin_record(policy_path, repo),
        "adjudicator_source_sha256": source_fingerprint(repo),
        "evaluation": dict(evaluation),
        "acceptance": dict(acceptance),
        "gates": dict(sorted(gates.items())),
        "comparisons": comparisons,
        "sample_audit": sample_audit,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_path, output)
    return output


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "LocalMorphologyAdjudicationError",
    "adjudicate_local_morphology_seed0",
]
