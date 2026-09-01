"""Paired comparison of count-only and amine-semantic Ugi generation.

Both arms use the same frozen checkpoint, ordered coarse morphology programs, flow seed, exact
reaction-core decoder, assessment and attempt budget.  The treatment additionally conditions its
small amine topology/chemistry readout on train-derived graph diameter and N/O counts.  No
component identity, stored graph, SMILES or fragment token enters generation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from experiments.phase1.product_l1.evaluation.ugi_group_balanced_program_prior_comparison import (
    _passes_promotion,
    _program_support_abstentions,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_transformer_topology import (
    UgiTransformerTopologyPolicy,
    enumerate_amine_semantic_topologies,
)
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA = "forge.ugi_amine_semantic_program_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_amine_semantic_program_comparison_result.v1"
DRAW_SCHEMA = "forge.ugi_amine_semantic_program_draw.v1"
INPUT_LABELS = {
    "amine_semantic_program_draw",
    "base_checkpoint_archive",
    "base_design",
    "base_training_result",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "production_cache",
    "qualified_reactions",
    "role_morphology_policy",
    "topology_closure_config",
    "topology_morphology_config",
    "ugi_assignments",
}


class UgiAmineSemanticProgramComparisonError(ValueError):
    """The paired semantic-program experiment contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"]) != INPUT_LABELS
        or not isinstance(config.get("profiles"), Mapping)
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiAmineSemanticProgramComparisonError("semantic comparison config changed")
    runtime = config["profiles"][profile]
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("flow_seed", -1)) < 0
        or runtime.get("terminal_decode_policy")
        != "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
    ):
        raise UgiAmineSemanticProgramComparisonError("semantic comparison runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "component_identity_conditioning": False,
        "heldout_training_access": False,
        "method_blind_assessment": True,
        "paired_coarse_program_order": True,
        "paired_flow_random_stream": True,
        "repairs_or_retries": False,
        "route_or_oracle_calls": 0,
        "training_calls": 0,
    }:
        raise UgiAmineSemanticProgramComparisonError("semantic comparison policy changed")
    if config.get("promotion_gate") != {
        "effective_component_count_minimum_retained_ratio": 0.8,
        "exact_l1_noninferiority_margin": 0.02,
        "local_support_noninferiority_margin": 0.02,
        "mean_pairwise_ecfp4_distance_noninferiority_margin": 0.05,
        "realism_c2st_required_reduction": 0.02,
        "unique_exact_l1_noninferiority_margin": 0.05,
        "validity_noninferiority_margin": 0.02,
    }:
        raise UgiAmineSemanticProgramComparisonError("semantic promotion gate changed")
    if config.get("preflight_gate") != {
        "minimum_control_valid_fraction": 0.75,
        "minimum_semantic_valid_fraction": 0.75,
        "require_estimable_c2st": True,
        "require_zero_program_support_abstentions": True,
    }:
        raise UgiAmineSemanticProgramComparisonError("semantic preflight gate changed")
    return runtime


def _load_draw(
    path: Path, *, count: int
) -> tuple[tuple[UgiMorphologyProgram, ...], tuple[UgiAmineSemanticTarget, ...]]:
    document = read_json_object(
        path,
        error=UgiAmineSemanticProgramComparisonError,
        label="Ugi amine semantic program draw",
    )
    rows = document.get("samples")
    if (
        document.get("schema_version") != DRAW_SCHEMA
        or document.get("status") != "pass"
        or document.get("reaction_id") != "ugi_3cr_agile"
        or document.get("component_identity_conditioning") is not False
        or document.get("heldout_access") is not False
        or document.get("training_generation_repair_retry_route_or_oracle_calls") != 0
        or document.get("candidate_selection") is not False
        or not isinstance(rows, list)
        or int(document.get("rows", 0)) < count
    ):
        raise UgiAmineSemanticProgramComparisonError("semantic program draw is inadmissible")
    programs: list[UgiMorphologyProgram] = []
    targets: list[UgiAmineSemanticTarget] = []
    for index, row in enumerate(rows[:count]):
        if not isinstance(row, Mapping) or row.get("sample_index") != index:
            raise UgiAmineSemanticProgramComparisonError("semantic draw order changed")
        raw_program = row.get("program")
        raw_target = row.get("amine_semantic_target")
        if not isinstance(raw_program, Mapping) or not isinstance(raw_target, Mapping):
            raise UgiAmineSemanticProgramComparisonError("semantic draw row changed")
        programs.append(
            UgiMorphologyProgram(
                node_counts=tuple(int(value) for value in raw_program["node_counts"]),
                junction_budgets=tuple(int(value) for value in raw_program["junction_budgets"]),
                cycle_ranks=tuple(int(value) for value in raw_program["cycle_ranks"]),
                attachment_counts=tuple(int(value) for value in raw_program["attachment_counts"]),
            )
        )
        targets.append(UgiAmineSemanticTarget.from_mapping(raw_target))
    return tuple(programs), tuple(targets)


def _semantic_support_audit(
    programs: tuple[UgiMorphologyProgram, ...],
    targets: tuple[UgiAmineSemanticTarget, ...],
    *,
    topology_policy: UgiTransformerTopologyPolicy,
    ester_policy: UgiEsterChemotypePolicy,
    local_chemistry_support: LocalChemistrySupport,
) -> dict[str, Any]:
    """Exhaustively prove nonempty joint topology and local-chemistry support."""

    if len(programs) != len(targets) or not programs:
        raise UgiAmineSemanticProgramComparisonError(
            "semantic support audit requires paired nonempty requests"
        )
    role_index = ROLE_NAMES.index(ester_policy.amine_role)
    pairs = {
        (
            program.node_counts,
            program.junction_budgets,
            program.cycle_ranks,
            program.attachment_counts,
            target.key,
        ): (program, target)
        for program, target in zip(programs, targets, strict=True)
    }
    coarse_counts: list[int] = []
    counts: list[int] = []
    unsupported: list[dict[str, Any]] = []
    for _, (program, target) in sorted(pairs.items()):
        node_count = int(program.node_counts[role_index])
        cycle_rank = int(program.cycle_ranks[role_index])
        common = {
            "node_count": node_count,
            "junction_budget": int(program.junction_budgets[role_index]),
            "cycle_rank": cycle_rank,
            "attachment_count": int(program.attachment_counts[role_index]),
            "target": target,
            "maximum_children": 3,
            "policy": topology_policy,
            "allowed_ring_sizes": (
                ester_policy.allowed_amine_cycle_sizes(node_count) if cycle_rank > 0 else None
            ),
        }
        coarse_candidates = enumerate_amine_semantic_topologies(**common)
        candidates = enumerate_amine_semantic_topologies(
            **common,
            local_chemistry_support=local_chemistry_support,
        )
        coarse_counts.append(len(coarse_candidates))
        counts.append(len(candidates))
        if not candidates:
            unsupported.append(
                {
                    "amine_program": {
                        "attachment_count": int(program.attachment_counts[role_index]),
                        "cycle_rank": cycle_rank,
                        "junction_budget": int(program.junction_budgets[role_index]),
                        "node_count": node_count,
                    },
                    "target": target.to_mapping(),
                }
            )
    return {
        "attempts": len(programs),
        "unique_program_target_pairs": len(pairs),
        "unsupported_pairs": unsupported,
        "minimum_feasible_topologies_per_pair": min(counts),
        "maximum_feasible_topologies_per_pair": max(counts),
        "minimum_semantic_topologies_before_local_support_per_pair": min(coarse_counts),
        "maximum_semantic_topologies_before_local_support_per_pair": max(coarse_counts),
        "semantic_topologies_before_local_support": sum(coarse_counts),
        "joint_supported_topologies": sum(counts),
        "topologies_removed_by_local_support": sum(coarse_counts) - sum(counts),
        "pairs_with_local_support_filtering": sum(
            joint < coarse for coarse, joint in zip(coarse_counts, counts, strict=True)
        ),
        "all_pairs_supported": not unsupported,
        "maximum_children": 3,
        "component_identity_conditioning": False,
        "local_chemistry_support_conditioned": True,
    }


def run_ugi_amine_semantic_program_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run the paired frozen-checkpoint semantic-factorization comparison."""

    config = read_json_object(
        config_path,
        error=UgiAmineSemanticProgramComparisonError,
        label="Ugi amine semantic program comparison config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiAmineSemanticProgramComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    programs, targets = _load_draw(
        inputs["amine_semantic_program_draw"], count=int(runtime["program_count"])
    )
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiAmineSemanticProgramComparisonError,
            label="role-local chemistry support",
        )
    )
    topology_policy = _topology_policy(inputs)
    core_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
    )
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(inputs["qualified_reactions"]),
        expected_training_assignments_sha256=sha256_file(inputs["ugi_assignments"]),
    )
    semantic_support_audit = _semantic_support_audit(
        programs,
        targets,
        topology_policy=topology_policy,
        ester_policy=ester_policy,
        local_chemistry_support=local_support,
    )
    current_inputs = {
        "checkpoint_archive": inputs["base_checkpoint_archive"],
        "common_ugi_assessment_config": inputs["common_ugi_assessment_config"],
        "current_training_result": inputs["base_training_result"],
        "lipid_realism_config": inputs["lipid_realism_config"],
        "local_chemistry_config": inputs["local_chemistry_config"],
        "production_cache": inputs["production_cache"],
        "production_design": inputs["base_design"],
        "role_morphology_policy": inputs["role_morphology_policy"],
    }
    sampling_runtime = {
        "batch_size": int(runtime["batch_size"]),
        "current_terminal_decode_policy": str(runtime["terminal_decode_policy"]),
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
        "terminal_decoder_seed": None,
    }
    common = {
        "repo": repo,
        "inputs": current_inputs,
        "runtime": sampling_runtime,
        "config_sha256": sha256_file(config_path),
        "device": device,
        "arm_id": str(config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(config["base_checkpoint"]["step"]),
        "programs": programs,
        "program_draw_sha256": sha256_file(inputs["amine_semantic_program_draw"]),
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
    }
    arms = {
        "count_only": {
            "method_id": "forge_seed0_count_only_ugi_program",
            "targets": None,
        },
        "amine_semantic": {
            "method_id": "forge_seed0_amine_semantic_ugi_program",
            "targets": targets,
        },
    }
    methods: dict[str, Any] = {}
    target_sha = sha256_file(inputs["amine_semantic_program_draw"])
    for label, arm in arms.items():
        arm_targets = arm["targets"]
        rows, sampling, sampling_path = _load_or_sample_current(
            output_dir=output_dir / label / "sampling",
            ugi_amine_semantic_targets=arm_targets,
            ugi_amine_semantic_target_sha256=(None if arm_targets is None else target_sha),
            **common,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=str(arm["method_id"]),
        )
        methods[label] = {
            "metrics": dict(assessment["assessment"]["metrics"]),
            "sampling": artifact_record(sampling_path),
            "sampling_summary": sampling,
            "assessment": artifact_record(output_dir / label / "assessment" / "result_index.json"),
            "program_support_abstentions": _program_support_abstentions(sampling),
        }
    control = methods["count_only"]["metrics"]
    treatment = methods["amine_semantic"]["metrics"]
    deltas = _metric_deltas(treatment, control)
    promotes, promotion_checks = _passes_promotion(
        deltas, control, treatment, config["promotion_gate"]
    )
    preflight = config["preflight_gate"]
    preflight_checks = {
        "control_valid_fraction_sufficient": float(control["valid_fraction_per_attempt"])
        >= float(preflight["minimum_control_valid_fraction"]),
        "semantic_valid_fraction_sufficient": float(treatment["valid_fraction_per_attempt"])
        >= float(preflight["minimum_semantic_valid_fraction"]),
        "c2st_estimable": (
            not bool(preflight["require_estimable_c2st"])
            or (
                control.get("realism_c2st_auc") is not None
                and treatment.get("realism_c2st_auc") is not None
            )
        ),
        "program_support_abstentions_zero": (
            not bool(preflight["require_zero_program_support_abstentions"])
            or all(method["program_support_abstentions"] == 0 for method in methods.values())
        ),
    }
    structural_checks = {
        "attempt_denominator_matched": all(
            int(method["sampling_summary"]["samples"]) == int(runtime["program_count"])
            for method in methods.values()
        ),
        "candidate_selection_absent": True,
        "component_identity_conditioning_absent": True,
        "coarse_program_sequence_identical": True,
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": all(
            not method["sampling_summary"].get("repairs") for method in methods.values()
        ),
        "route_or_oracle_calls_zero": True,
        "semantic_target_support_complete": bool(semantic_support_audit["all_pairs_supported"]),
        "training_calls_zero": True,
    }
    status = (
        "complete"
        if all(structural_checks.values())
        and (profile != "h100_preflight" or all(preflight_checks.values()))
        else "fail"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "profile": profile,
        "programs_per_method": int(runtime["program_count"]),
        "methods": methods,
        "amine_semantic_minus_count_only": deltas,
        "promotion_gate": dict(config["promotion_gate"]),
        "promotion_checks": promotion_checks,
        "promotion_decision": (
            "diagnostic_only"
            if profile == "smoke"
            else (
                "eligible_for_full_comparison"
                if profile == "h100_preflight" and status == "complete"
                else (
                    "promote_amine_semantic_program"
                    if profile == "full" and promotes
                    else "do_not_promote"
                )
            )
        ),
        "preflight_checks": preflight_checks,
        "structural_checks": structural_checks,
        "program_pairing": {
            "identical_coarse_programs": True,
            "same_flow_random_seed": True,
            "intervention": "amine heavy-graph diameter, carbon diameter and N/O counts",
        },
        "semantic_support_audit": semantic_support_audit,
        "inputs": {
            "config": pin_record(config_path, repo),
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        },
        "policy": dict(config["policy"]),
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if status != "complete":
        raise UgiAmineSemanticProgramComparisonError(
            f"semantic comparison gates failed: structural={structural_checks}, "
            f"preflight={preflight_checks}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "h100_preflight", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = run_ugi_amine_semantic_program_comparison(
        args.config,
        repo,
        args.output_dir,
        profile=args.profile,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiAmineSemanticProgramComparisonError",
    "run_ugi_amine_semantic_program_comparison",
]
