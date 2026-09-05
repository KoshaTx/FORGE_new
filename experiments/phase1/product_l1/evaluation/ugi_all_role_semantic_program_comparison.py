"""Paired frozen-checkpoint comparison of amine-only and all-role Ugi semantics.

Both arms use the same checkpoint, ordered coarse morphology programs, flow seed, attempt budget,
reaction-core decoder and method-blind assessors.  The baseline is the frozen amine-semantic
joint-support decoder.  The treatment draws one train-only identity-free joint target spanning the
amine head, aldehyde ester arms and both tail unsaturation classes.  Each attempt is decoded once;
there is no component lookup, repair, retry or candidate selection.
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
    _program_support_abstentions,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.synthesis_program_sampling import (
    UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY,
)
from forge.model.ugi_all_role_semantic_program import (
    UgiAllRoleSemanticTarget,
    UgiMeasuredJointAllRoleSemanticPrior,
    UgiMeasuredRoleFactorizedSemanticPrior,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior
from forge.model.ugi_transformer_topology import (
    UgiTransformerTopologyPolicy,
    _enumerate_constructive_ester_offspring,
    enumerate_amine_semantic_topologies,
)
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA = "forge.ugi_all_role_semantic_program_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_all_role_semantic_program_comparison_result.v1"
DRAW_SCHEMA = "forge.ugi_all_role_semantic_program_draw.v1"
INPUT_LABELS = {
    "all_role_semantic_program_draw",
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
LOCAL_CHEMISTRY_INPUT_LABELS = {"semantic_atoms", "semantic_bonds"}


class UgiAllRoleSemanticProgramComparisonError(ValueError):
    """The paired all-role semantic experiment contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(config.get("inputs"), Mapping)
        or frozenset(config["inputs"])
        not in {frozenset(INPUT_LABELS), frozenset(INPUT_LABELS | LOCAL_CHEMISTRY_INPUT_LABELS)}
        or not isinstance(config.get("profiles"), Mapping)
        or not isinstance(config["profiles"].get(profile), Mapping)
    ):
        raise UgiAllRoleSemanticProgramComparisonError("all-role comparison config changed")
    runtime = config["profiles"][profile]
    semantic_guidance = config.get("semantic_guidance")
    baseline_semantic_guidance = config.get("baseline_semantic_guidance")
    treatment_target_source = config.get("treatment_semantic_target_source")
    ordinary_runtime = semantic_guidance is None
    if (
        runtime.get("device") != device
        or int(runtime.get("program_count", 0)) < 4
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or int(runtime.get("flow_seed", -1)) < 0
        or (
            ordinary_runtime
            and runtime.get("terminal_decode_policy")
            != "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
        )
        or (
            not ordinary_runtime
            and (
                runtime.get("treatment_terminal_decode_policy")
                != UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY
                or int(runtime.get("treatment_terminal_decoder_seed", -1)) < 0
            )
        )
    ):
        raise UgiAllRoleSemanticProgramComparisonError("all-role comparison runtime changed")
    if semantic_guidance is not None:
        if not isinstance(semantic_guidance, Mapping):
            raise UgiAllRoleSemanticProgramComparisonError("semantic guidance policy changed")
        parsed_policy = UgiMogSemanticGuidancePolicy.from_mapping(semantic_guidance)
        if parsed_policy.uses_local_reference and not LOCAL_CHEMISTRY_INPUT_LABELS.issubset(
            config["inputs"]
        ):
            raise UgiAllRoleSemanticProgramComparisonError(
                "local-chemistry guidance inputs are missing"
            )
    if treatment_target_source is not None:
        target_source_fields = (
            set(treatment_target_source) if isinstance(treatment_target_source, Mapping) else set()
        )
        selected_roles = (
            treatment_target_source.get("roles", list(ROLE_NAMES))
            if isinstance(treatment_target_source, Mapping)
            else None
        )
        if (
            not isinstance(treatment_target_source, Mapping)
            or target_source_fields not in ({"strategy", "seed"}, {"strategy", "seed", "roles"})
            or treatment_target_source.get("strategy") != "unique_measured_role_factorized"
            or isinstance(treatment_target_source.get("seed"), bool)
            or not isinstance(treatment_target_source.get("seed"), int)
            or int(treatment_target_source["seed"]) < 0
            or not isinstance(selected_roles, list)
            or not selected_roles
            or any(not isinstance(role, str) for role in selected_roles)
            or len(selected_roles) != len(set(selected_roles))
            or not set(selected_roles).issubset(ROLE_NAMES)
        ):
            raise UgiAllRoleSemanticProgramComparisonError(
                "treatment semantic-target source changed"
            )
    if baseline_semantic_guidance is None:
        if (
            not ordinary_runtime
            and runtime.get("baseline_terminal_decode_policy")
            != "strict_ugi_ester_topology_role_local_chemistry_core_saturation"
        ):
            raise UgiAllRoleSemanticProgramComparisonError(
                "baseline terminal decoder policy changed"
            )
    else:
        if ordinary_runtime or not isinstance(baseline_semantic_guidance, Mapping):
            raise UgiAllRoleSemanticProgramComparisonError(
                "baseline semantic guidance policy changed"
            )
        baseline_policy = UgiMogSemanticGuidancePolicy.from_mapping(baseline_semantic_guidance)
        if baseline_policy.uses_joint_realism:
            raise UgiAllRoleSemanticProgramComparisonError(
                "baseline cannot use joint-realism guidance"
            )
        if (
            runtime.get("baseline_terminal_decode_policy")
            != UGI_ESTER_TOPOLOGY_ROLE_LOCAL_MOG_TERMINAL_DECODE_POLICY
            or int(runtime.get("baseline_terminal_decoder_seed", -1)) < 0
        ):
            raise UgiAllRoleSemanticProgramComparisonError(
                "baseline MOG terminal decoder runtime changed"
            )
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
        raise UgiAllRoleSemanticProgramComparisonError("all-role comparison policy changed")
    if config.get("preflight_gate") != {
        "exact_l1_absolute_minimum": 0.95,
        "exact_l1_noninferiority_margin": 0.01,
        "minimum_baseline_valid_fraction": 0.95,
        "minimum_treatment_valid_fraction": 0.95,
        "require_zero_program_support_abstentions": True,
    }:
        raise UgiAllRoleSemanticProgramComparisonError("all-role preflight gate changed")
    return runtime


def _program(raw: Mapping[str, Any]) -> UgiMorphologyProgram:
    if set(raw) != {
        "attachment_counts",
        "cycle_ranks",
        "junction_budgets",
        "node_counts",
    }:
        raise UgiAllRoleSemanticProgramComparisonError("all-role coarse program changed")
    return UgiMorphologyProgram(
        node_counts=tuple(int(value) for value in raw["node_counts"]),
        junction_budgets=tuple(int(value) for value in raw["junction_budgets"]),
        cycle_ranks=tuple(int(value) for value in raw["cycle_ranks"]),
        attachment_counts=tuple(int(value) for value in raw["attachment_counts"]),
    )


def _load_draw(
    path: Path,
    *,
    count: int,
) -> tuple[
    tuple[UgiMorphologyProgram, ...],
    tuple[UgiAmineSemanticTarget, ...],
    tuple[UgiAllRoleSemanticTarget, ...],
]:
    document = read_json_object(
        path,
        error=UgiAllRoleSemanticProgramComparisonError,
        label="Ugi all-role semantic program draw",
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
        raise UgiAllRoleSemanticProgramComparisonError("all-role program draw is inadmissible")
    programs: list[UgiMorphologyProgram] = []
    baseline: list[UgiAmineSemanticTarget] = []
    treatment: list[UgiAllRoleSemanticTarget] = []
    paired_amine_targets = document.get("paired_amine_targets_identical") is True
    for index, row in enumerate(rows[:count]):
        if not isinstance(row, Mapping) or row.get("sample_index") != index:
            raise UgiAllRoleSemanticProgramComparisonError("all-role draw order changed")
        raw_program = row.get("program")
        raw_baseline = row.get("baseline_amine_semantic_target")
        raw_treatment = row.get("all_role_semantic_target")
        if not all(
            isinstance(value, Mapping) for value in (raw_program, raw_baseline, raw_treatment)
        ):
            raise UgiAllRoleSemanticProgramComparisonError("all-role draw row changed")
        program = _program(raw_program)
        baseline_target = UgiAmineSemanticTarget.from_mapping(raw_baseline)
        treatment_target = UgiAllRoleSemanticTarget.from_mapping(raw_treatment)
        if paired_amine_targets and baseline_target != treatment_target.amine:
            raise UgiAllRoleSemanticProgramComparisonError(
                "paired baseline and treatment amine targets differ"
            )
        programs.append(program)
        baseline.append(baseline_target)
        treatment.append(treatment_target)
    return tuple(programs), tuple(baseline), tuple(treatment)


def _semantic_support_audit(
    programs: tuple[UgiMorphologyProgram, ...],
    targets: tuple[UgiAllRoleSemanticTarget, ...],
    *,
    topology_policy: UgiTransformerTopologyPolicy,
    ester_policy: UgiEsterChemotypePolicy,
    local_chemistry_support: LocalChemistrySupport,
) -> dict[str, Any]:
    """Check that every distinct requested semantic tuple has nonempty declared support."""

    if not programs or len(programs) != len(targets):
        raise UgiAllRoleSemanticProgramComparisonError(
            "all-role support audit requires paired nonempty requests"
        )
    amine_index = ROLE_NAMES.index(ester_policy.amine_role)
    aldehyde_index = ROLE_NAMES.index(ester_policy.aldehyde_role)
    isocyanide_index = ROLE_NAMES.index(ester_policy.isocyanide_role)
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
    unsupported: list[dict[str, Any]] = []
    amine_counts: list[int] = []
    aldehyde_counts: list[int] = []
    for _, (program, target) in sorted(pairs.items()):
        amine_nodes = int(program.node_counts[amine_index])
        amine_cycles = int(program.cycle_ranks[amine_index])
        amine_topologies = enumerate_amine_semantic_topologies(
            node_count=amine_nodes,
            junction_budget=int(program.junction_budgets[amine_index]),
            cycle_rank=amine_cycles,
            attachment_count=int(program.attachment_counts[amine_index]),
            target=target.amine,
            maximum_children=3,
            policy=topology_policy,
            allowed_ring_sizes=(
                ester_policy.allowed_amine_cycle_sizes(amine_nodes) if amine_cycles > 0 else None
            ),
            local_chemistry_support=local_chemistry_support,
        )
        tail = target.tail_pair
        aldehyde_topologies = _enumerate_constructive_ester_offspring(
            node_count=int(program.node_counts[aldehyde_index]),
            minimum_side_carbons=int(ester_policy.minimum_ester_side_carbons),
            minimum_long_side_carbons=int(ester_policy.minimum_ester_long_side_carbons),
            exact_full_side_carbons=(
                int(tail.aldehyde_ester_short_side_carbons),
                int(tail.aldehyde_ester_long_side_carbons),
            ),
            exact_alkoxy_handle_and_acyl_side_carbons=(
                None
                if int(tail.aldehyde_alkoxy_handle_side_carbons) == 0
                else (
                    int(tail.aldehyde_alkoxy_handle_side_carbons),
                    int(tail.aldehyde_acyl_side_carbons),
                )
            ),
        )
        tail_bond_support = all(
            (
                count == 0
                or local_chemistry_support.allows_role_edge(
                    "ugi_3cr_agile", role, "C", bond_state, role, "C"
                )
            )
            for role, counts in (
                (
                    ester_policy.aldehyde_role,
                    (
                        int(tail.aldehyde_carbon_carbon_double_bonds),
                        int(tail.aldehyde_carbon_carbon_triple_bonds),
                    ),
                ),
                (
                    ester_policy.isocyanide_role,
                    (
                        int(tail.isocyanide_carbon_carbon_double_bonds),
                        int(tail.isocyanide_carbon_carbon_triple_bonds),
                    ),
                ),
            )
            for bond_state, count in enumerate(counts, start=1)
        )
        isocyanide_support = int(tail.isocyanide_carbon_skeleton_diameter) == int(
            program.node_counts[isocyanide_index]
        )
        amine_counts.append(len(amine_topologies))
        aldehyde_counts.append(len(aldehyde_topologies))
        if (
            not amine_topologies
            or not aldehyde_topologies
            or not tail_bond_support
            or not isocyanide_support
        ):
            unsupported.append(
                {
                    "program": {
                        "attachment_counts": list(program.attachment_counts),
                        "cycle_ranks": list(program.cycle_ranks),
                        "junction_budgets": list(program.junction_budgets),
                        "node_counts": list(program.node_counts),
                    },
                    "target": target.to_mapping(),
                    "amine_topology_support": len(amine_topologies),
                    "aldehyde_topology_support": len(aldehyde_topologies),
                    "tail_bond_support": tail_bond_support,
                    "isocyanide_support": isocyanide_support,
                }
            )
    return {
        "attempts": len(programs),
        "unique_program_target_pairs": len(pairs),
        "unsupported_pairs": unsupported,
        "all_pairs_supported": not unsupported,
        "minimum_amine_topologies_per_pair": min(amine_counts),
        "maximum_amine_topologies_per_pair": max(amine_counts),
        "minimum_aldehyde_topologies_per_pair": min(aldehyde_counts),
        "maximum_aldehyde_topologies_per_pair": max(aldehyde_counts),
        "component_identity_conditioning": False,
        "local_chemistry_support_conditioned": True,
    }


def _preflight_checks(
    *,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    baseline_abstentions: int,
    treatment_abstentions: int,
    gate: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_exact = float(baseline["exact_l1_yield_per_attempt"])
    treatment_exact = float(treatment["exact_l1_yield_per_attempt"])
    return {
        "baseline_valid_fraction_sufficient": float(baseline["valid_fraction_per_attempt"])
        >= float(gate["minimum_baseline_valid_fraction"]),
        "treatment_valid_fraction_sufficient": float(treatment["valid_fraction_per_attempt"])
        >= float(gate["minimum_treatment_valid_fraction"]),
        "treatment_exact_l1_absolute_minimum": treatment_exact
        >= float(gate["exact_l1_absolute_minimum"]),
        "treatment_exact_l1_noninferior": treatment_exact
        >= baseline_exact - float(gate["exact_l1_noninferiority_margin"]),
        "program_support_abstentions_zero": (
            baseline_abstentions == treatment_abstentions == 0
            if gate["require_zero_program_support_abstentions"]
            else True
        ),
    }


def _local_chemistry_method_id(policy: UgiMogSemanticGuidancePolicy) -> str:
    """Give atom-only, bond-only, and joint ranking distinct ledger identities."""

    def with_bond_depth(method_id: str) -> str:
        depth = policy.local_chemistry_bond_maximum_depth_bucket
        suffix = "" if depth is None else f"_bonddepth{depth:03d}"
        if policy.local_chemistry_unsaturation_position_only:
            suffix += "_unsatpos"
        if policy.local_chemistry_bond_position_basis == "terminal_offset":
            suffix += "_terminaloffset"
        if policy.local_chemistry_unsaturation_minimum_support_tier is not None:
            suffix += f"_mintier{policy.local_chemistry_unsaturation_minimum_support_tier}"
        if policy.whole_head_topology_support:
            suffix += "_toposupport"
            if policy.whole_head_topology_score_mode == "tier":
                suffix += "tier"
            if (
                policy.whole_head_topology_rank_weight is not None
                and abs(
                    policy.whole_head_topology_rank_weight
                    - policy.local_chemistry_rank_weight
                )
                > 1e-12
            ):
                topology_weight_milli = round(policy.whole_head_topology_rank_weight * 1000)
                suffix += f"_topow{topology_weight_milli:04d}"
        if not policy.uses_ranked_terminal_chemistry:
            suffix += "_chemargmax"
        elif not policy.uses_ranked_terminal_bonds:
            suffix += "_bondargmax"
        if policy.tail_unsaturation_count_tolerance > 0:
            suffix += f"_unsatslack{policy.tail_unsaturation_count_tolerance}"
        if policy.amine_hydrogen_bond_donors_tolerance > 0:
            suffix += f"_donorslack{policy.amine_hydrogen_bond_donors_tolerance}"
        if policy.amine_heavy_branch_atoms_tolerance > 0:
            suffix += f"_branchslack{policy.amine_heavy_branch_atoms_tolerance}"
        if policy.tail_unsaturation_count_strategy != "flat_symmetric":
            suffix += f"_{policy.tail_unsaturation_count_strategy.replace('_', '')}"
        if policy.tail_unsaturation_position_strategy != "independent_ranked":
            suffix += f"_{policy.tail_unsaturation_position_strategy.replace('_', '')}"
        if policy.tail_unsaturation_frequency_pseudocount is not None:
            pseudocount_milli = round(policy.tail_unsaturation_frequency_pseudocount * 1000)
            suffix += f"_pc{pseudocount_milli:04d}"
        if abs(policy.uniform_probability_mass - 0.25) > 1e-12:
            uniform_milli = round(policy.uniform_probability_mass * 1000)
            suffix += f"_unif{uniform_milli:04d}"
        if policy.local_chemistry_bond_uniform_probability_mass is not None:
            bond_uniform_milli = round(
                policy.local_chemistry_bond_uniform_probability_mass * 1000
            )
            suffix += f"_bondunif{bond_uniform_milli:04d}"
        # Radius-bearing binary-head policies encoded these guards in their historical prefix.
        # Independent entropy guards are now valid without a TV radius and must still receive a
        # distinct method identity so their ledgers cannot collide with an unguarded decoder.
        if policy.whole_head_total_variation_radius is None:
            if policy.whole_head_candidate_effective_count_retention is not None:
                retention_milli = round(
                    policy.whole_head_candidate_effective_count_retention * 1000
                )
                suffix += f"_eff{retention_milli:04d}"
            if policy.whole_head_group_effective_count_retention is not None:
                retention_milli = round(
                    policy.whole_head_group_effective_count_retention * 1000
                )
                suffix += f"_grp{retention_milli:04d}"
        if (
            policy.uses_local_chemistry
            and abs(policy.local_chemistry_rank_weight - 1.0) > 1e-12
        ):
            chemistry_milli = round(policy.local_chemistry_rank_weight * 1000)
            suffix += f"_chemw{chemistry_milli:04d}"
        if (
            policy.local_chemistry_bond_rank_weight is not None
            and policy.local_chemistry_bond_rank_weight > 0
            and abs(
                policy.local_chemistry_bond_rank_weight - policy.local_chemistry_rank_weight
            )
            > 1e-12
        ):
            bond_weight_milli = round(policy.local_chemistry_bond_rank_weight * 1000)
            suffix += f"_bondw{bond_weight_milli:04d}"
        return f"{method_id}{suffix}"

    if policy.local_chemistry_score_mode == "whole_head_support_tier":
        return with_bond_depth("forge_seed0_mog_whole_head_support_complete_semantic_ugi_program")
    if policy.local_chemistry_score_mode == "whole_head_support_distance":
        if policy.whole_head_total_variation_radius is None:
            return with_bond_depth(
                "forge_seed0_mog_whole_head_distance_complete_semantic_ugi_program"
            )
        radius_milli = round(policy.whole_head_total_variation_radius * 1000)
        return with_bond_depth(
            f"forge_seed0_mog_whole_head_distance_tv{radius_milli:03d}"
            "_complete_semantic_ugi_program"
        )
    if policy.local_chemistry_score_mode == "whole_head_support_binary":
        if policy.whole_head_total_variation_radius is not None:
            radius_milli = round(policy.whole_head_total_variation_radius * 1000)
            bond_contract = "hard_bond" if policy.whole_head_hard_bond_support else "soft_bond"
            effective_count_contract = ""
            if policy.whole_head_candidate_effective_count_retention is not None:
                retention_milli = round(
                    policy.whole_head_candidate_effective_count_retention * 1000
                )
                effective_count_contract = f"_eff{retention_milli:04d}"
            if policy.whole_head_group_effective_count_retention is not None:
                retention_milli = round(policy.whole_head_group_effective_count_retention * 1000)
                effective_count_contract += f"_grp{retention_milli:04d}"
            return with_bond_depth(
                "forge_seed0_mog_binary_whole_head_tv"
                f"{radius_milli:03d}{effective_count_contract}_{bond_contract}"
                "_complete_semantic_ugi_program"
            )
        return with_bond_depth(
            "forge_seed0_mog_binary_whole_head_support_complete_semantic_ugi_program"
        )
    if policy.local_chemistry_score_mode == "support_tier":
        return with_bond_depth("forge_seed0_mog_context_support_complete_semantic_ugi_program")
    if policy.uses_local_chemistry_atoms and policy.uses_local_chemistry_bonds:
        return with_bond_depth("forge_seed0_mog_local_chemistry_complete_semantic_ugi_program")
    if policy.uses_local_chemistry_atoms:
        if policy.local_chemistry_atom_total_variation_radius is not None:
            return with_bond_depth(
                "forge_seed0_mog_atom_trust_region_complete_semantic_ugi_program"
            )
        return with_bond_depth("forge_seed0_mog_atom_local_chemistry_complete_semantic_ugi_program")
    if policy.uses_local_chemistry_bonds:
        return with_bond_depth("forge_seed0_mog_bond_local_chemistry_complete_semantic_ugi_program")
    raise UgiAllRoleSemanticProgramComparisonError(
        "local-chemistry method identity requires an active atom or bond rank"
    )


def run_ugi_all_role_semantic_program_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run the matched frozen-checkpoint all-role semantic comparison."""

    config = read_json_object(
        config_path,
        error=UgiAllRoleSemanticProgramComparisonError,
        label="Ugi all-role semantic comparison config",
    )
    runtime = _runtime(config, profile=profile, device=device)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiAllRoleSemanticProgramComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    programs, baseline_targets, all_role_targets = _load_draw(
        inputs["all_role_semantic_program_draw"], count=int(runtime["program_count"])
    )
    local_support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiAllRoleSemanticProgramComparisonError,
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
    target_source = config.get("treatment_semantic_target_source")
    semantic_target_reference: Mapping[str, Any] | None = None
    if target_source is not None:
        cache = SynthesisProgramProductionCache(inputs["production_cache"])
        try:
            factorized_prior = UgiMeasuredRoleFactorizedSemanticPrior.from_training_data(
                assignments_path=inputs["ugi_assignments"],
                cache=cache,
                ester_policy=ester_policy,
                reaction_id="ugi_3cr_agile",
                expected_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
                include_amine_substitution_semantics=any(
                    target.amine.hydrogen_bond_donors is not None
                    for target in all_role_targets
                ),
            )
        finally:
            cache.close()
        factorized_targets = factorized_prior.sample_for_programs(
            programs,
            seed=int(target_source["seed"]),
        )
        selected_roles = set(target_source.get("roles", ROLE_NAMES))
        combined_targets = []
        for original, factorized in zip(all_role_targets, factorized_targets, strict=True):
            original_tail = original.tail_pair
            factorized_tail = factorized.tail_pair
            combined_targets.append(
                UgiAllRoleSemanticTarget(
                    amine=(
                        factorized.amine if "amine_head" in selected_roles else original.amine
                    ),
                    tail_pair=type(original_tail)(
                        aldehyde_ester_short_side_carbons=(
                            factorized_tail.aldehyde_ester_short_side_carbons
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_ester_short_side_carbons
                        ),
                        aldehyde_ester_long_side_carbons=(
                            factorized_tail.aldehyde_ester_long_side_carbons
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_ester_long_side_carbons
                        ),
                        aldehyde_carbon_carbon_double_bonds=(
                            factorized_tail.aldehyde_carbon_carbon_double_bonds
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_carbon_carbon_double_bonds
                        ),
                        aldehyde_carbon_carbon_triple_bonds=(
                            factorized_tail.aldehyde_carbon_carbon_triple_bonds
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_carbon_carbon_triple_bonds
                        ),
                        isocyanide_carbon_carbon_double_bonds=(
                            factorized_tail.isocyanide_carbon_carbon_double_bonds
                            if "isocyanide_tail" in selected_roles
                            else original_tail.isocyanide_carbon_carbon_double_bonds
                        ),
                        isocyanide_carbon_carbon_triple_bonds=(
                            factorized_tail.isocyanide_carbon_carbon_triple_bonds
                            if "isocyanide_tail" in selected_roles
                            else original_tail.isocyanide_carbon_carbon_triple_bonds
                        ),
                        isocyanide_carbon_skeleton_diameter=(
                            factorized_tail.isocyanide_carbon_skeleton_diameter
                            if "isocyanide_tail" in selected_roles
                            else original_tail.isocyanide_carbon_skeleton_diameter
                        ),
                        aldehyde_alkoxy_handle_side_carbons=(
                            factorized_tail.aldehyde_alkoxy_handle_side_carbons
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_alkoxy_handle_side_carbons
                        ),
                        aldehyde_acyl_side_carbons=(
                            factorized_tail.aldehyde_acyl_side_carbons
                            if "oxoester_aldehyde_body_tail" in selected_roles
                            else original_tail.aldehyde_acyl_side_carbons
                        ),
                    ),
                )
            )
        all_role_targets = tuple(combined_targets)
        semantic_target_reference = {
            **factorized_prior.audit,
            "sample_seed": int(target_source["seed"]),
            "sampled_roles": sorted(selected_roles),
            "sampled_targets_sha256": str(
                sha256_json([target.to_mapping() for target in all_role_targets])
            ),
        }
    support_audit = _semantic_support_audit(
        programs,
        all_role_targets,
        topology_policy=topology_policy,
        ester_policy=ester_policy,
        local_chemistry_support=local_support,
    )
    baseline_semantic_guidance_policy = (
        None
        if config.get("baseline_semantic_guidance") is None
        else UgiMogSemanticGuidancePolicy.from_mapping(config["baseline_semantic_guidance"])
    )
    semantic_guidance_policy = (
        None
        if config.get("semantic_guidance") is None
        else UgiMogSemanticGuidancePolicy.from_mapping(config["semantic_guidance"])
    )
    joint_realism_reference: Mapping[str, Any] | None = None
    local_chemistry_reference: Mapping[str, Any] | None = None
    if semantic_guidance_policy is not None and semantic_guidance_policy.uses_joint_realism:
        cache = SynthesisProgramProductionCache(inputs["production_cache"])
        try:
            joint_prior = UgiMeasuredJointAllRoleSemanticPrior.from_training_data(
                assignments_path=inputs["ugi_assignments"],
                cache=cache,
                ester_policy=ester_policy,
                reaction_id="ugi_3cr_agile",
                expected_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
            )
        finally:
            cache.close()
        semantic_guidance_policy = semantic_guidance_policy.bind_joint_realism(joint_prior)
        joint_realism_reference = semantic_guidance_policy.joint_realism_audit()
    if semantic_guidance_policy is not None and semantic_guidance_policy.uses_local_reference:
        local_prior = UgiRoleChemistryPrior.from_training_data(
            assignments_path=inputs["ugi_assignments"],
            semantic_atoms_path=inputs["semantic_atoms"],
            semantic_bonds_path=inputs["semantic_bonds"],
            reaction_id="ugi_3cr_agile",
            expected_assignments_sha256=str(config["inputs"]["ugi_assignments"]["sha256"]),
            expected_semantic_atoms_sha256=str(config["inputs"]["semantic_atoms"]["sha256"]),
            expected_semantic_bonds_sha256=str(config["inputs"]["semantic_bonds"]["sha256"]),
            maximum_bond_depth_bucket=(
                semantic_guidance_policy.local_chemistry_bond_maximum_depth_bucket
            ),
        )
        semantic_guidance_policy = semantic_guidance_policy.bind_local_chemistry(local_prior)
        local_chemistry_reference = semantic_guidance_policy.local_chemistry_audit()
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
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
    }
    common = {
        "repo": repo,
        "inputs": current_inputs,
        "config_sha256": sha256_file(config_path),
        "device": device,
        "arm_id": str(config["base_checkpoint"]["arm_id"]),
        "checkpoint_step": int(config["base_checkpoint"]["step"]),
        "programs": programs,
        "program_draw_sha256": sha256_file(inputs["all_role_semantic_program_draw"]),
        "ugi_topology_policy": topology_policy,
        "local_chemistry_support": local_support,
        "reaction_core_saturation_policy": core_policy,
        "ugi_ester_chemotype_policy": ester_policy,
    }
    draw_sha = sha256_file(inputs["all_role_semantic_program_draw"])
    treatment_target_sha = str(
        sha256_json([target.to_mapping() for target in all_role_targets])
    )
    baseline_all_role_targets = tuple(
        UgiAllRoleSemanticTarget(
            amine=amine,
            tail_pair=treatment.tail_pair,
        )
        for amine, treatment in zip(baseline_targets, all_role_targets, strict=True)
    )
    substitution_treatment = any(
        target.amine.hydrogen_bond_donors is not None for target in all_role_targets
    )
    arm_specs = {
        "amine_semantic": {
            "method_id": (
                "forge_seed0_mog_semantic_joint_support_ugi_program"
                if baseline_semantic_guidance_policy is not None
                else "forge_seed0_amine_semantic_joint_support_ugi_program"
            ),
            "amine_targets": (
                baseline_targets if baseline_semantic_guidance_policy is None else None
            ),
            "all_role_targets": (
                baseline_all_role_targets if baseline_semantic_guidance_policy is not None else None
            ),
            "semantic_guidance_policy": baseline_semantic_guidance_policy,
        },
        "all_role_semantic": {
            "method_id": (
                (
                    _local_chemistry_method_id(semantic_guidance_policy)
                    if semantic_guidance_policy is not None
                    and semantic_guidance_policy.uses_local_reference
                    else (
                        "forge_seed0_mog_substitution_semantic_joint_support_ugi_program"
                        if substitution_treatment
                        else "forge_seed0_mog_semantic_joint_support_ugi_program"
                    )
                )
                if semantic_guidance_policy is not None
                else "forge_seed0_all_role_semantic_joint_support_ugi_program"
            ),
            "amine_targets": None,
            "all_role_targets": all_role_targets,
            "semantic_guidance_policy": semantic_guidance_policy,
        },
    }
    if target_source is not None:
        selected_role_suffix = (
            ""
            if "roles" not in target_source
            else "_".join(
                {
                    "amine_head": "amine",
                    "oxoester_aldehyde_body_tail": "aldehyde",
                    "isocyanide_tail": "isocyanide",
                }[role]
                for role in target_source["roles"]
            )
        )
        arm_specs["all_role_semantic"]["method_id"] = (
            f"{arm_specs['all_role_semantic']['method_id']}_rolefactorized"
            f"{('_' + selected_role_suffix) if selected_role_suffix else ''}"
        )
    methods: dict[str, Any] = {}
    for label, spec in arm_specs.items():
        arm_runtime = dict(sampling_runtime)
        if semantic_guidance_policy is None:
            arm_runtime.update(
                {
                    "current_terminal_decode_policy": str(runtime["terminal_decode_policy"]),
                    "terminal_decoder_seed": None,
                }
            )
        elif label == "amine_semantic":
            arm_runtime.update(
                {
                    "current_terminal_decode_policy": str(
                        runtime["baseline_terminal_decode_policy"]
                    ),
                    "terminal_decoder_seed": (
                        None
                        if baseline_semantic_guidance_policy is None
                        else int(runtime["baseline_terminal_decoder_seed"])
                    ),
                    "terminal_temperature": float(runtime.get("terminal_temperature", 1.0)),
                }
            )
        else:
            arm_runtime.update(
                {
                    "current_terminal_decode_policy": str(
                        runtime["treatment_terminal_decode_policy"]
                    ),
                    "terminal_decoder_seed": int(runtime["treatment_terminal_decoder_seed"]),
                    "terminal_temperature": float(runtime.get("terminal_temperature", 1.0)),
                }
            )
        rows, sampling, sampling_path = _load_or_sample_current(
            output_dir=output_dir / label / "sampling",
            ugi_amine_semantic_targets=spec["amine_targets"],
            ugi_amine_semantic_target_sha256=(
                draw_sha if spec["amine_targets"] is not None else None
            ),
            ugi_all_role_semantic_targets=spec["all_role_targets"],
            ugi_all_role_semantic_target_sha256=(
                treatment_target_sha if spec["all_role_targets"] is not None else None
            ),
            ugi_mog_semantic_guidance_policy=spec["semantic_guidance_policy"],
            runtime=arm_runtime,
            **common,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / label / "assessment",
            inputs=current_inputs,
            repo=repo,
            method_id=str(spec["method_id"]),
        )
        methods[label] = {
            "metrics": dict(assessment["assessment"]["metrics"]),
            "sampling": artifact_record(sampling_path),
            "sampling_summary": sampling,
            "assessment": artifact_record(output_dir / label / "assessment" / "result_index.json"),
            "program_support_abstentions": _program_support_abstentions(sampling),
        }
    baseline_metrics = methods["amine_semantic"]["metrics"]
    treatment_metrics = methods["all_role_semantic"]["metrics"]
    deltas = _metric_deltas(treatment_metrics, baseline_metrics)
    preflight_checks = _preflight_checks(
        baseline=baseline_metrics,
        treatment=treatment_metrics,
        baseline_abstentions=int(methods["amine_semantic"]["program_support_abstentions"]),
        treatment_abstentions=int(methods["all_role_semantic"]["program_support_abstentions"]),
        gate=config["preflight_gate"],
    )
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
        "semantic_target_support_complete": bool(support_audit["all_pairs_supported"]),
        "mog_guidance_bound_as_configured": (
            bool(
                methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_mog_semantic_guidance_applied", False
                )
            )
            is (baseline_semantic_guidance_policy is not None)
            and bool(
                methods["all_role_semantic"]["sampling_summary"].get(
                    "ugi_mog_semantic_guidance_applied", False
                )
            )
            is (semantic_guidance_policy is not None)
        ),
        "joint_realism_arm_bound_exclusively": (
            semantic_guidance_policy is None
            or not semantic_guidance_policy.uses_joint_realism
            or (
                not methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_joint_semantic_realism_guidance_applied", False
                )
                and methods["all_role_semantic"]["sampling_summary"].get(
                    "ugi_joint_semantic_realism_guidance_applied", False
                )
                and joint_realism_reference is not None
                and joint_realism_reference.get("calibration_or_heldout_access") is False
            )
        ),
        "local_chemistry_arm_bound_exclusively": (
            semantic_guidance_policy is None
            or not semantic_guidance_policy.uses_local_reference
            or (
                not methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_local_chemistry_mog_guidance_applied", False
                )
                and not methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_whole_head_topology_support_applied", False
                )
                and (
                    methods["all_role_semantic"]["sampling_summary"].get(
                        "ugi_local_chemistry_mog_guidance_applied", False
                    )
                    or methods["all_role_semantic"]["sampling_summary"].get(
                        "ugi_whole_head_topology_support_applied", False
                    )
                )
                and local_chemistry_reference is not None
                and local_chemistry_reference.get("component_identity_conditioning") is False
                and local_chemistry_reference.get("component_graph_conditioning") is False
                and local_chemistry_reference.get("fragment_vocabulary_conditioning") is False
            )
        ),
        "local_chemistry_paths_match_configuration": (
            semantic_guidance_policy is None
            or (
                methods["all_role_semantic"]["sampling_summary"].get(
                    "ugi_local_atom_chemistry_mog_guidance_applied", False
                )
                is semantic_guidance_policy.uses_local_chemistry_atoms
                and methods["all_role_semantic"]["sampling_summary"].get(
                    "ugi_local_bond_chemistry_mog_guidance_applied", False
                )
                is semantic_guidance_policy.uses_local_chemistry_bonds
                and not methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_local_atom_chemistry_mog_guidance_applied", False
                )
                and not methods["amine_semantic"]["sampling_summary"].get(
                    "ugi_local_bond_chemistry_mog_guidance_applied", False
                )
            )
        ),
        "training_calls_zero": True,
    }
    status = (
        "complete"
        if all(structural_checks.values())
        and (profile == "smoke" or all(preflight_checks.values()))
        else "fail"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "profile": profile,
        "programs_per_method": int(runtime["program_count"]),
        "methods": methods,
        "all_role_minus_amine_semantic": deltas,
        "preflight_gate": dict(config["preflight_gate"]),
        "preflight_checks": preflight_checks,
        "promotion_decision": (
            "diagnostic_only"
            if profile == "smoke"
            else (
                "eligible_for_full_comparison"
                if profile == "h100_preflight" and status == "complete"
                else "requires_measured_role_panel_adjudication"
            )
        ),
        "structural_checks": structural_checks,
        "program_pairing": {
            "identical_coarse_programs": True,
            "same_flow_random_seed": True,
            "baseline": (
                "MOG-style ranked stochastic bands over supported joint role semantics"
                if baseline_semantic_guidance_policy is not None
                else "frozen amine-head joint-support semantics"
            ),
            "intervention": (
                (
                    "MOG-style stochastic chemistry plus novelty-neutral measured-train "
                    "complete-head topology support"
                    if semantic_guidance_policy is not None
                    and semantic_guidance_policy.whole_head_topology_support
                    and not semantic_guidance_policy.uses_local_chemistry_atoms
                    and not semantic_guidance_policy.uses_local_chemistry_bonds
                    else (
                        "MOG-style ranked stochastic bands plus measured-train complete-head "
                        "arrangement and topology support"
                        if semantic_guidance_policy is not None
                        and semantic_guidance_policy.local_chemistry_score_mode
                        in {
                            "whole_head_support_tier",
                            "whole_head_support_binary",
                            "whole_head_support_distance",
                        }
                        else (
                            "MOG-style ranked stochastic bands plus novelty-neutral "
                            "measured-train local-context support tiers"
                            if semantic_guidance_policy is not None
                            and semantic_guidance_policy.local_chemistry_score_mode == "support_tier"
                            else (
                                "MOG-style ranked stochastic bands plus train-fold atom-context "
                                "ranking; bond placement remains Transformer-driven"
                                if semantic_guidance_policy is not None
                                and semantic_guidance_policy.uses_local_chemistry_atoms
                                and not semantic_guidance_policy.uses_local_chemistry_bonds
                                else (
                                    "MOG-style ranked stochastic bands plus train-fold "
                                    "local-chemistry support"
                                )
                            )
                        )
                    )
                )
                if semantic_guidance_policy is not None
                and semantic_guidance_policy.uses_local_reference
                else (
                    "MOG-style ranked stochastic bands plus train-fold joint-realism density"
                    if semantic_guidance_policy is not None
                    and semantic_guidance_policy.uses_joint_realism
                    else (
                        "MOG-style ranked stochastic bands over supported joint role semantics"
                        if semantic_guidance_policy is not None
                        else "joint amine, ester-arm and tail-unsaturation semantics"
                    )
                )
            ),
        },
        "baseline_semantic_guidance": (
            None
            if baseline_semantic_guidance_policy is None
            else baseline_semantic_guidance_policy.to_mapping()
        ),
        "semantic_guidance": (
            None if semantic_guidance_policy is None else semantic_guidance_policy.to_mapping()
        ),
        "joint_realism_reference": joint_realism_reference,
        "local_chemistry_reference": local_chemistry_reference,
        "semantic_support_audit": support_audit,
        "semantic_target_reference": semantic_target_reference,
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
        raise UgiAllRoleSemanticProgramComparisonError(
            f"all-role comparison gates failed: structural={structural_checks}, "
            f"preflight={preflight_checks}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "h100_preflight", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = run_ugi_all_role_semantic_program_comparison(
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
    "UgiAllRoleSemanticProgramComparisonError",
    "run_ugi_all_role_semantic_program_comparison",
]
