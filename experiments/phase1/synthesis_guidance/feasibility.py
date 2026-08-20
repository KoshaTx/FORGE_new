"""Route-aware feasibility census for the frozen matched Ugi candidate pools.

The census is deliberately downstream of molecular generation and biological
applicability/ranking, but upstream of any prospective panel lock.  It applies
one authenticated L1/L2/L3 route policy to the broad-prior and promoted
applicability-enriched arms and reports two denominators:

* every continuously supported terminal produced at the fixed generation
  budget (the production-yield cohort); and
* the already frozen, equal-size oracle-scored subset (the causal matched
  cohort).

It does not add route evidence, invoke Graph2Edits, relax exact closure, select
an experimental candidate, or interpret missing route knowledge as chemical
impossibility.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiments.phase1.product_l1.sampling.ugi_selected_restartable_generator import (
    _atom_vocabulary_states,
)
from experiments.phase1.product_l1.sampling.ugi_selected_restartable_generator_v2 import (
    CLOSURE_CHECKPOINT_SHA256,
    COMPONENT_RECOVERY_CONTRACT_SHA256,
    DECLARED_GRAPH_SUPPORT_SHA256,
    GENERATOR_CHECKPOINT_SHA256,
    L1_REACTION_SHA256,
    MODEL_CONFIG_SHA256,
    _canonical_sha256,
)
from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    adapt_restartable_completion_row_for_route_support,
    canonical_morphology_program_bytes,
    native_completion_record_from_locked_terminal,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_matched_planner_cache_binding import (
    preflight_lazy_matched_planner_cache_binding,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_production_terminal_route_evaluator import (
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object
from forge.core.io import stable_json as _stable_json
from forge.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    declared_graph_support_context_sha256,
)
from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.synthesis.engine.planner_cache import FilePlannerCache
from forge.synthesis.matched import (
    MatchedArm,
    MatchedAssessmentContext,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.synthesis.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    assess_locked_ugi_terminal_routes,
    required_three_role_route_reservation,
)
from forge.synthesis.value.exact_closure import (
    UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
    exact_closure_potential_from_product_value,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_route_aware_panel_feasibility_config.v1"
CONFIG_SCHEMA_VERSION_V2 = "phase1_ugi_route_aware_panel_feasibility_config.v2"
CONFIG_SCHEMA_VERSION_V3 = "phase1_ugi_route_aware_panel_feasibility_config.v3"
RESULT_SCHEMA_VERSION = "phase1_ugi_route_aware_panel_feasibility.v1"
LEDGER_SCHEMA_VERSION = "forge.ugi_route_aware_panel_feasibility_ledger.v1"
ARM_ORDER = ("support_enriched", "broad_prior")
EXPECTED_SCOPE = {
    "post_generation_route_assessment": True,
    "assessment_before_candidate_lock": True,
    "matched_broad_and_support_arms": True,
    "exact_l1_l2_l3_policy": True,
    "new_route_evidence": False,
    "graph2edits": False,
    "biological_model_calls": 0,
    "potency_tilting": False,
    "candidate_selection": False,
    "prospective_panel_lock": False,
    "sealed_holdout_access": False,
}


class UgiRouteAwarePanelFeasibilityError(RuntimeError):
    """Raised when the frozen route-aware census cannot be reproduced exactly."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (_stable_json(value) + "\n").encode()


def _jsonl_gzip_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    raw = b"".join(_canonical_json_bytes(dict(row)) for row in rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(raw)
    return output.getvalue()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiRouteAwarePanelFeasibilityError, label=label)


def _pin(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise UgiRouteAwarePanelFeasibilityError(f"malformed input pin: {label}")
    path = (repo / str(value["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiRouteAwarePanelFeasibilityError(
            f"input pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != value["sha256"]:
        raise UgiRouteAwarePanelFeasibilityError(f"input pin changed: {label}")
    return path


def _read_generation_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiRouteAwarePanelFeasibilityError("generation ledger contains a non-object row")
    return rows


def _read_ranking_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str, *, label: str) -> bool:
    if value not in {"True", "False"}:
        raise UgiRouteAwarePanelFeasibilityError(f"{label} is not a serialized boolean")
    return value == "True"


def _load_contract(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    config = _load_json(config_path, label="route-aware feasibility config")
    schema_version = config.get("schema_version")
    if schema_version not in {
        CONFIG_SCHEMA_VERSION,
        CONFIG_SCHEMA_VERSION_V2,
        CONFIG_SCHEMA_VERSION_V3,
    }:
        raise UgiRouteAwarePanelFeasibilityError("unsupported route-aware feasibility config")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiRouteAwarePanelFeasibilityError("route-aware feasibility scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UgiRouteAwarePanelFeasibilityError("route-aware feasibility inputs are missing")
    expected_inputs = {
        "generation_result",
        "generation_ledger",
        "ranking_result",
        "ranking_ledger",
        "production_generator_manifest",
        "selected_model_config",
        "atom_vocabulary",
        "qualified_reaction_registry",
        "current_route_source",
        "route_completion_utility",
    }
    if set(inputs) != expected_inputs:
        raise UgiRouteAwarePanelFeasibilityError("route-aware feasibility input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    generation_result = _load_json(paths["generation_result"], label="generation result")
    generation_artifact = generation_result.get("artifacts", {}).get("terminal_ledger.jsonl.gz")
    expected_generation_status = (
        "complete_matched_production_candidate_generation"
        if schema_version in {CONFIG_SCHEMA_VERSION_V2, CONFIG_SCHEMA_VERSION_V3}
        else "complete_matched_morphology_allocation_terminal_generation"
    )
    if (
        generation_result.get("status") != expected_generation_status
        or not isinstance(generation_artifact, Mapping)
        or generation_artifact.get("sha256") != sha256_file(paths["generation_ledger"])
        or generation_result.get("scope", {}).get("route_calls") != 0
    ):
        raise UgiRouteAwarePanelFeasibilityError("frozen matched generation is not authentic")

    ranking_result = _load_json(paths["ranking_result"], label="ranking result")
    ranking_artifacts = ranking_result.get("artifacts", {})
    ranking_artifact_name = (
        "terminal_rescoring.csv.gz"
        if schema_version == CONFIG_SCHEMA_VERSION_V3
        else "terminal_ranking.csv.gz"
    )
    ranking_pin = ranking_artifacts.get(ranking_artifact_name)
    expected_ranking_status = {
        CONFIG_SCHEMA_VERSION: "continuous_novelty_matched_ranking_complete",
        CONFIG_SCHEMA_VERSION_V2: "complete_matched_production_terminal_ranking",
        CONFIG_SCHEMA_VERSION_V3: "complete_full_interpolative_support_terminal_rescoring",
    }[schema_version]
    if (
        ranking_result.get("status") != expected_ranking_status
        or not isinstance(ranking_pin, Mapping)
        or ranking_pin.get("sha256") != sha256_file(paths["ranking_ledger"])
    ):
        raise UgiRouteAwarePanelFeasibilityError("frozen continuous ranking is not authentic")

    source = _load_json(paths["current_route_source"], label="current route source")
    current_source = source.get("current_cumulative_source")
    if (
        source.get("status") != "current_source_l3_and_zero_guidance_route_values_requalified"
        or not isinstance(current_source, Mapping)
        or current_source.get("assessment_at_utc") != config.get("assessment_as_of_utc")
        or current_source.get("inputs_sha256") != config.get("cumulative_source_inputs_sha256")
        or current_source.get("all_l3_windows_cover_decision_horizon") is not True
    ):
        raise UgiRouteAwarePanelFeasibilityError("route source is not current at the frozen time")

    utility = _load_json(paths["route_completion_utility"], label="route utility")
    if (
        utility.get("status") != "binary_exact_dossier_route_completion_utility_qualified"
        or utility.get("implementation_policy", {}).get("policy_sha256")
        != UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256
    ):
        raise UgiRouteAwarePanelFeasibilityError("route-completion policy is not qualified")

    generation_rows = _read_generation_rows(paths["generation_ledger"])
    ranking_rows = _read_ranking_rows(paths["ranking_ledger"])
    generation_by_key = {
        (str(row["arm_id"]), int(row["draw_index"])): row for row in generation_rows
    }
    if len(generation_by_key) != len(generation_rows):
        raise UgiRouteAwarePanelFeasibilityError("generation arm/draw keys are not unique")

    cohort: list[dict[str, Any]] = []
    for rank in ranking_rows:
        arm = rank.get("arm_id", "")
        eligibility_field = (
            "oracle_scored" if schema_version == CONFIG_SCHEMA_VERSION_V3 else "eligible"
        )
        if arm not in ARM_ORDER or not _bool(
            rank.get(eligibility_field, ""), label=eligibility_field
        ):
            continue
        draw_index = int(rank["draw_index"])
        generated = generation_by_key.get((arm, draw_index))
        if generated is None:
            raise UgiRouteAwarePanelFeasibilityError(
                "ranked terminal lacks its native generation row"
            )
        native = generated.get("native_terminal")
        components = native.get("component_smiles_by_role") if isinstance(native, Mapping) else None
        if (
            not isinstance(components, Mapping)
            or native.get("smiles") != rank.get("canonical_product")
            or components.get("amine_head") != rank.get("canonical_amine")
            or components.get("oxoester_aldehyde_body_tail") != rank.get("canonical_aldehyde")
            or components.get("isocyanide_tail") != rank.get("canonical_isocyanide")
            or native.get("l1_forward_verification", {}).get("exact_product_reconstructed")
            is not True
        ):
            raise UgiRouteAwarePanelFeasibilityError(
                "ranking identity differs from its native exact-L1 terminal"
            )
        selected_field = (
            "oracle_scored" if schema_version == CONFIG_SCHEMA_VERSION_V3 else "oracle_selected"
        )
        oracle_selected = _bool(rank[selected_field], label=selected_field)
        cohort.append(
            {
                "arm_id": arm,
                "draw_index": draw_index,
                "pattern_id": rank.get("pattern_id") or rank.get("lane_id"),
                "oracle_selected": oracle_selected,
                "conservative_high_potency": _bool(
                    rank["conservative_high_potency"], label="conservative_high_potency"
                ),
                "authority_tier": rank.get("authority_tier") or "legacy_unspecified",
                "potency_utility": (
                    None
                    if not oracle_selected or not rank["potency_utility"]
                    else float(rank["potency_utility"])
                ),
                "rank": rank,
                "generation": generated,
            }
        )

    expected = config.get("cohort", {}).get("eligible_rows_per_arm")
    observed = Counter(row["arm_id"] for row in cohort)
    if expected != dict(observed):
        raise UgiRouteAwarePanelFeasibilityError(
            f"eligible cohort changed: expected {expected}, observed {dict(observed)}"
        )
    selected = Counter(row["arm_id"] for row in cohort if row["oracle_selected"])
    expected_selected_key = (
        "oracle_scored_rows_per_arm"
        if schema_version == CONFIG_SCHEMA_VERSION_V3
        else "matched_oracle_rows_per_arm"
    )
    if config.get("cohort", {}).get(expected_selected_key) != dict(selected):
        raise UgiRouteAwarePanelFeasibilityError("matched oracle-scored cohort changed")
    cohort.sort(key=lambda row: (ARM_ORDER.index(row["arm_id"]), row["draw_index"]))
    return config, paths, cohort


def _build_frozen_graph_support(
    paths: Mapping[str, Path],
) -> tuple[DeclaredGraphSupportContext, Any, QualifiedUgiL1Reverifier]:
    model_config = _load_json(paths["selected_model_config"], label="selected model config").get(
        "model"
    )
    if not isinstance(model_config, dict) or _canonical_sha256(model_config) != MODEL_CONFIG_SHA256:
        raise UgiRouteAwarePanelFeasibilityError("selected graph-support model config changed")
    graph_support = DeclaredGraphSupportContext(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        model_config=MappingProxyType(model_config.copy()),
        atom_vocabulary=_atom_vocabulary_states(paths["atom_vocabulary"]),
    )
    if declared_graph_support_context_sha256(graph_support) != DECLARED_GRAPH_SUPPORT_SHA256:
        raise UgiRouteAwarePanelFeasibilityError("frozen declared graph support changed")
    reaction = load_ugi_reaction_contract(paths["qualified_reaction_registry"])
    reverifier = QualifiedUgiL1Reverifier(
        reaction_contract=reaction,
        l1_reaction_sha256=L1_REACTION_SHA256,
    )
    return graph_support, reaction, reverifier


def _post_hoc_lock_sha256(cohort: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        [row["arm_id"], row["draw_index"], row["rank"]["canonical_product"]]
        for row in cohort
        if row["arm_id"] == "broad_prior"
    ]
    return _sha256_payload(identities)


def _route_one(
    row: Mapping[str, Any],
    *,
    factory: Any,
    binding: Any,
    reaction: Any,
    graph_support: DeclaredGraphSupportContext,
    reverifier: QualifiedUgiL1Reverifier,
    post_hoc_lock_sha256: str,
) -> dict[str, Any]:
    arm = str(row["arm_id"])
    generated = row["generation"]
    native = generated["native_terminal"]
    treatment = MatchedArm.GUIDED if arm == "support_enriched" else MatchedArm.POST_HOC
    entry = MatchedScheduleEntry(
        unit_id=f"route-feasibility:{arm}:{int(row['draw_index']):04d}",
        morphology_program=canonical_morphology_program_bytes(native["program"]),
        program_index=int(generated["support_index"]),
        particle_index=int(generated["particle_index"]),
        checkpoint_index=int(generated["checkpoint_index"]),
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        rollout_index=int(generated["rollout_index"]),
        productive_generation_calls=1,
        route_reservation=RouteComputeUsage(),
    )
    request = MatchedGenerationRequest(
        arm=treatment,
        entry=entry,
        productive_seed=int(generated["terminal_seed"]),
    )
    terminal = adapt_restartable_completion_row_for_route_support(
        native,
        generation_request=request,
        l1_reaction=reaction,
        l1_reaction_sha256=L1_REACTION_SHA256,
        component_recovery_contract_sha256=COMPONENT_RECOVERY_CONTRACT_SHA256,
        graph_support=graph_support,
        l1_reverifier=reverifier,
    ).locked_terminal
    required = required_three_role_route_reservation(factory.planner_context.budget_limits)
    context = MatchedAssessmentContext(
        arm=treatment,
        route_seed=int(generated["terminal_seed"]),
        remaining_budget=required,
        unit_reservation=required,
        cache_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        cache_clone_id=f"panel-feasibility-{treatment.value}-v1",
        post_hoc_lock_manifest_sha256=(
            post_hoc_lock_sha256 if treatment is MatchedArm.POST_HOC else None
        ),
    )
    planner = factory.build_planner(
        binding.bind(context), factory.planner_context, context, terminal
    )
    receipt = assess_locked_ugi_terminal_routes(
        terminal,
        l1_reverifier=factory.l1_reverifier,
        planner=planner,
        planner_context=factory.planner_context,
        assessment_context=context,
        assessment_at_utc=factory.assessment_as_of_utc,
    )
    potential = exact_closure_potential_from_product_value(receipt.product_value)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "arm_id": arm,
        "draw_index": int(row["draw_index"]),
        "pattern_id": row["pattern_id"],
        "authority_tier": row["authority_tier"],
        "oracle_selected": bool(row["oracle_selected"]),
        "conservative_high_potency": bool(row["conservative_high_potency"]),
        "potency_utility": row["potency_utility"],
        "canonical_product": row["rank"]["canonical_product"],
        "canonical_components": {
            "amine_head": row["rank"]["canonical_amine"],
            "oxoester_aldehyde_body_tail": row["rank"]["canonical_aldehyde"],
            "isocyanide_tail": row["rank"]["canonical_isocyanide"],
        },
        "terminal_sha256": terminal.terminal_sha256,
        "generation_trace_sha256": terminal.generation_trace_sha256,
        "route_complete": potential.strict_route_complete,
        "strict_complete_role_count": potential.strict_complete_role_count,
        "route_reasons": [reason.value for reason in potential.reasons],
        "potential": potential.to_dict(),
        "product_value": receipt.product_value.to_dict(),
        "assessment_receipt": receipt.to_dict(),
        "support_audit": planner.support_audit.to_dict(),
    }


def summarize_route_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize row-, product- and role-level feasibility without selection."""

    by_arm: dict[str, list[Mapping[str, Any]]] = {
        arm: [row for row in records if row.get("arm_id") == arm]
        for arm in ("broad_prior", "support_enriched")
    }
    summary: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        unique: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            product = str(row["canonical_product"])
            if product in unique and bool(unique[product]["route_complete"]) != bool(
                row["route_complete"]
            ):
                raise UgiRouteAwarePanelFeasibilityError(
                    "one canonical product received inconsistent route outcomes"
                )
            unique.setdefault(product, row)
        selected = [row for row in rows if row["oracle_selected"]]
        selected_unique = {str(row["canonical_product"]): row for row in selected}
        high = [row for row in selected if row["conservative_high_potency"]]
        patterns: dict[str, dict[str, int]] = {}
        for pattern in sorted({str(row["pattern_id"]) for row in rows}):
            subset = [row for row in rows if row["pattern_id"] == pattern]
            patterns[pattern] = {
                "rows": len(subset),
                "unique_products": len({row["canonical_product"] for row in subset}),
                "route_complete_rows": sum(bool(row["route_complete"]) for row in subset),
                "oracle_selected_rows": sum(bool(row["oracle_selected"]) for row in subset),
                "conservative_high_rows": sum(
                    bool(row["oracle_selected"] and row["conservative_high_potency"])
                    for row in subset
                ),
                "route_complete_conservative_high_rows": sum(
                    bool(
                        row["oracle_selected"]
                        and row["conservative_high_potency"]
                        and row["route_complete"]
                    )
                    for row in subset
                ),
            }
        authority_tiers = dict(
            sorted(Counter(str(row.get("authority_tier")) for row in rows).items())
        )
        role_outcomes: dict[str, Any] = {}
        component_state: dict[str, dict[str, bool]] = defaultdict(dict)
        for row in rows:
            for role in row["potential"]["roles"]:
                component = row["canonical_components"][role["role"]]
                observed = component_state[role["role"]].get(component)
                strict = bool(role["strict_complete"])
                if observed is not None and observed != strict:
                    raise UgiRouteAwarePanelFeasibilityError(
                        "one canonical component received inconsistent route outcomes"
                    )
                component_state[role["role"]][component] = strict
        for role, states in sorted(component_state.items()):
            decisions = [
                role_record
                for row in rows
                for role_record in row["potential"]["roles"]
                if role_record["role"] == role
            ]
            role_outcomes[role] = {
                "unique_components": len(states),
                "strict_route_complete_unique_components": sum(states.values()),
                "strict_route_complete_rows": sum(
                    bool(value["strict_complete"]) for value in decisions
                ),
                "assessment_outcomes": dict(
                    sorted(Counter(value["assessment_outcome"] for value in decisions).items())
                ),
                "reasons": dict(sorted(Counter(value["reason"] for value in decisions).items())),
            }
        summary[arm] = {
            "eligible_rows": len(rows),
            "eligible_unique_products": len(unique),
            "route_complete_rows": sum(bool(row["route_complete"]) for row in rows),
            "route_complete_unique_products": sum(
                bool(row["route_complete"]) for row in unique.values()
            ),
            "oracle_selected_rows": len(selected),
            "oracle_selected_unique_products": len(selected_unique),
            "route_complete_oracle_selected_rows": sum(
                bool(row["route_complete"]) for row in selected
            ),
            "route_complete_oracle_selected_unique_products": sum(
                bool(row["route_complete"]) for row in selected_unique.values()
            ),
            "conservative_high_rows": len(high),
            "route_complete_conservative_high_rows": sum(
                bool(row["route_complete"]) for row in high
            ),
            "patterns": patterns,
            "authority_tiers": authority_tiers,
            "roles": role_outcomes,
        }
    broad = summary["broad_prior"]
    support = summary["support_enriched"]
    return {
        "arms": summary,
        "balanced_fillable_route_complete_unique_products": {
            "production_yield_cohort": min(
                broad["route_complete_unique_products"],
                support["route_complete_unique_products"],
            ),
            "causal_matched_oracle_cohort": min(
                broad["route_complete_oracle_selected_unique_products"],
                support["route_complete_oracle_selected_unique_products"],
            ),
            "conservative_high_cohort_rows": min(
                broad["route_complete_conservative_high_rows"],
                support["route_complete_conservative_high_rows"],
            ),
        },
    }


def run_route_aware_panel_feasibility(
    repo: Path,
    config_path: Path,
    cache_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen nonselecting route census once and persist its receipts."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    cache_root = cache_root.resolve()
    config, paths, cohort = _load_contract(repo, config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiRouteAwarePanelFeasibilityError("output directory must be absent or empty")
    if cache_root.exists() and any(cache_root.iterdir()):
        raise UgiRouteAwarePanelFeasibilityError("cache root must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    graph_support, reaction, reverifier = _build_frozen_graph_support(paths)
    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc=config["assessment_as_of_utc"],
        expected_cumulative_source_inputs_sha256=config["cumulative_source_inputs_sha256"],
        selected_generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        graph_support=graph_support,
        l1_reverifier=reverifier,
        candidate_record_resolver=native_completion_record_from_locked_terminal,
    )
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(cache_root / "base"),
        FilePlannerCache(cache_root / "guided"),
        FilePlannerCache(cache_root / "post_hoc"),
        factory.planner_context,
        assessment_at_utc=factory.assessment_as_of_utc,
    )
    lock_sha256 = _post_hoc_lock_sha256(cohort)
    records = [
        _route_one(
            row,
            factory=factory,
            binding=binding,
            reaction=reaction,
            graph_support=graph_support,
            reverifier=reverifier,
            post_hoc_lock_sha256=lock_sha256,
        )
        for row in cohort
    ]
    cache_audit = binding.finalize()
    summary = summarize_route_records(records)
    minimum_target = int(config["panel_targets"]["minimum_evaluable_per_causal_arm"])
    preferred_target = int(config["panel_targets"]["preferred_evaluable_per_causal_arm"])
    matched_fill = summary["balanced_fillable_route_complete_unique_products"][
        "causal_matched_oracle_cohort"
    ]
    production_fill = summary["balanced_fillable_route_complete_unique_products"][
        "production_yield_cohort"
    ]
    status = (
        "route_evidence_supports_preferred_causal_panel"
        if matched_fill >= preferred_target
        else (
            "route_evidence_supports_minimum_causal_panel"
            if matched_fill >= minimum_target
            else "route_evidence_gap_blocks_causal_panel_lock"
        )
    )

    ledger_path = output_dir / "route_assessment_ledger.jsonl.gz"
    cache_path = output_dir / "cache_audit.json"
    _atomic_write(ledger_path, _jsonl_gzip_bytes(records))
    _atomic_write(cache_path, _canonical_json_bytes(cache_audit.to_dict()))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "assessment_as_of_utc": config["assessment_as_of_utc"],
        "cumulative_source_inputs_sha256": config["cumulative_source_inputs_sha256"],
        "route_policy_sha256": UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
        "declared_graph_support_sha256": DECLARED_GRAPH_SUPPORT_SHA256,
        "generator_checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
        "cohort_interpretation": {
            "production_yield_cohort": (
                "all continuously supported terminals available after equal generation budgets"
            ),
            "causal_matched_oracle_cohort": (
                "the frozen equal-size oracle-scored subset; no additional oracle calls"
            ),
            "route_assessment_timing": "post_generation_pre_panel_lock",
        },
        "summary": summary,
        "panel_targets": {
            **config["panel_targets"],
            "balanced_matched_route_complete_unique_products": matched_fill,
            "balanced_production_route_complete_unique_products": production_fill,
            "minimum_target_fillable": matched_fill >= minimum_target,
            "preferred_target_fillable": matched_fill >= preferred_target,
        },
        "cache": {
            "preflight_sha256": binding.preflight.preflight_sha256,
            "audit_sha256": cache_audit.audit_sha256,
            "base_unchanged": cache_audit.base_before == cache_audit.base_after,
            "guided_entries": cache_audit.guided_after.entry_count,
            "post_hoc_entries": cache_audit.post_hoc_after.entry_count,
        },
        "artifacts": {
            "route_assessment_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(records),
                "rows": len(records),
            },
            "cache_audit.json": {
                "path": cache_path.name,
                "sha256": sha256_file(cache_path),
                "audit_sha256": cache_audit.audit_sha256,
            },
        },
        "scope": dict(EXPECTED_SCOPE),
        "decision": (
            "Do not lock a causal prospective panel unless the frozen route-evidence census "
            "fills the declared arm target. Missing knowledge identifies an evidence gap, not "
            "intrinsic unsynthesizability."
        ),
        "nonclaims": [
            "A missing or incomplete dossier does not prove that a molecule is unsynthesizable.",
            "No route proposal, evidence lookup, manual rescue or candidate selection was run.",
            "This census does not authorize potency tilting or a prospective panel lock.",
        ],
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    _atomic_write(output_dir / "result.json", _canonical_json_bytes(result))
    return result


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION_V2",
    "CONFIG_SCHEMA_VERSION_V3",
    "EXPECTED_SCOPE",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiRouteAwarePanelFeasibilityError",
    "run_route_aware_panel_feasibility",
    "summarize_route_records",
]
