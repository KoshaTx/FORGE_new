from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.data.r1_prime_audit import sha256_file
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES
from forge.design.corpus.ugi_generated_terminal_support import DeclaredGraphSupportContext
from forge.design.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.design.schedule.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.design.schedule.ugi_matched_planner_cache_binding import (
    preflight_lazy_matched_planner_cache_binding,
)
from forge.design.schedule.ugi_production_terminal_route_evaluator import (
    UgiProductionTerminalRouteEvaluatorError,
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.route.engine.planner_cache import FilePlannerCache
from forge.route.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    ValidatedUgiTerminalPayload,
    assess_locked_ugi_terminal_routes,
    required_three_role_route_reservation,
)
from forge.route.assessment.ugi3_support_boundary import UGI_COMPONENT_ROLES

REPO = Path(__file__).resolve().parents[1]
ASSESSMENT_AT = "2026-08-03T04:00:00Z"
SOURCE_INPUTS_SHA256 = "0fca91fae36da762eda53695405ec74cf5f9eacd19aee6ca3e4495a6a4d427c6"
L1_VARIANT = REPO / "configs/assembly/ugi_variant.yaml"
L1_VARIANT_SHA256 = "5b97e062b115fcc137b4a05d8f72b74e9cc67a584c05c054ef5f983969bf1427"
REGISTRY_SHA256 = "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf"
COMPONENTS = {
    "amine_head": "CN",
    "oxoester_aldehyde_body_tail": "CC=O",
    "isocyanide_tail": "[C-]#[N+]C",
}
PRODUCT = "CNC(=O)C(C)NC"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reaction():
    registry = REPO / "data/vendor/qualified_reactions_v1.json"
    assert sha256_file(registry) == REGISTRY_SHA256
    return load_ugi_reaction_contract(registry)


def _payload(*, l1_sha256: str = L1_VARIANT_SHA256) -> ValidatedUgiTerminalPayload:
    return ValidatedUgiTerminalPayload.from_recovered_components(
        product_smiles=PRODUCT,
        components_by_role=COMPONENTS,
        l1_reaction=_reaction(),
        l1_reaction_sha256=l1_sha256,
        component_recovery_contract_sha256=_hash("component-recovery"),
    )


def _reverifier(*, l1_sha256: str = L1_VARIANT_SHA256) -> QualifiedUgiL1Reverifier:
    return QualifiedUgiL1Reverifier(_reaction(), l1_sha256)


def _terminal(payload: ValidatedUgiTerminalPayload, **overrides: object) -> LockedMatchedTerminal:
    values: dict[str, object] = {
        "unit_id": "unit-1",
        "morphology_program_sha256": _hash("program"),
        "checkpoint_index": 1,
        "generator_checkpoint_sha256": _hash("selected-generator"),
        "closure_checkpoint_sha256": _hash("closure"),
        "terminal_id": "terminal-1",
        "terminal_locked": True,
        "terminal_valid": True,
        "exact_l1": True,
        "terminal_bytes": payload.canonical_bytes,
        "generation_trace_bytes": b"native-generation-trace\n",
        "payload": payload,
    }
    values.update(overrides)
    return LockedMatchedTerminal(**values)  # type: ignore[arg-type]


def _candidate(payload: ValidatedUgiTerminalPayload) -> dict[str, object]:
    return {
        "valid": True,
        "component_reconstruction_valid": True,
        "smiles": payload.product_smiles,
        "component_smiles_by_role": {
            role: payload.by_role()[role].canonical_smiles for role in ROLE_NAMES
        },
        "program": {
            "node_counts": [1, 1, 1],
            "junction_budgets": [0, 0, 0],
            "cycle_ranks": [0, 0, 0],
            "attachment_counts": [1, 1, 1],
        },
        "offspring_by_role": {role: [0] for role in ROLE_NAMES},
    }


def _atom_vocabulary(product: str) -> frozenset[tuple[str, int, bool, int]]:
    molecule = Chem.MolFromSmiles(product)
    assert molecule is not None
    return frozenset(
        (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumExplicitHs(),
        )
        for atom in molecule.GetAtoms()
    )


def _graph_support(payload: ValidatedUgiTerminalPayload) -> DeclaredGraphSupportContext:
    return DeclaredGraphSupportContext(
        generator_checkpoint_sha256=_hash("selected-generator"),
        model_config={
            "maximum_total_atoms": 96,
            "maximum_component_atoms": 48,
            "maximum_junction_budget": 4,
            "maximum_cycle_rank": 4,
            "maximum_attachment_count": 4,
            "maximum_children": 4,
        },
        atom_vocabulary=_atom_vocabulary(payload.product_smiles),
    )


def _factory():
    assert sha256_file(L1_VARIANT) == L1_VARIANT_SHA256
    payload = _payload()
    terminal = _terminal(payload)
    candidate = _candidate(payload)

    def resolve(observed: LockedMatchedTerminal):
        if observed.terminal_sha256 != terminal.terminal_sha256:
            raise KeyError(observed.terminal_sha256)
        return candidate

    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=REPO,
        assessment_as_of_utc=ASSESSMENT_AT,
        expected_cumulative_source_inputs_sha256=SOURCE_INPUTS_SHA256,
        selected_generator_checkpoint_sha256=_hash("selected-generator"),
        graph_support=_graph_support(payload),
        l1_reverifier=_reverifier(),
        candidate_record_resolver=resolve,
    )
    return factory, terminal, candidate


def _binding(tmp_path: Path, factory):
    return preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        factory.planner_context,
        assessment_at_utc=ASSESSMENT_AT,
    )


def _assessment_context(binding, factory, *, arm: MatchedArm, **overrides: object):
    required = required_three_role_route_reservation(factory.planner_context.budget_limits)
    values: dict[str, object] = {
        "arm": arm,
        "route_seed": 20260803,
        "remaining_budget": required,
        "unit_reservation": required,
        "cache_snapshot_sha256": binding.preflight.base.snapshot_sha256,
        "cache_clone_id": f"{arm.value}-clone",
        "post_hoc_lock_manifest_sha256": (
            _hash("post-hoc-lock") if arm is MatchedArm.POST_HOC else None
        ),
    }
    values.update(overrides)
    return MatchedAssessmentContext(**values)  # type: ignore[arg-type]


def test_factory_derives_production_context_and_exact_internal_roles() -> None:
    factory, _, _ = _factory()

    assert factory.planner_context.l1_reaction_sha256 == L1_VARIANT_SHA256
    assert factory.planner_context.l3_snapshot_sha256 == SOURCE_INPUTS_SHA256
    assert factory.planner_context.l3_region == "US"
    assert factory.planner_context.l3_accessed_at_utc == "2026-08-03T03:04:10Z"
    assert factory.planner_context.l3_expires_at_utc == "2026-08-09T05:16:00Z"
    assert factory.planner_context.budget_limits.to_dict() == {
        "maximum_depth": 4,
        "maximum_logical_planner_calls": 1,
        "maximum_expansions": 4,
        "maximum_product_candidates": 4,
        "maximum_verifier_calls": 4,
        "maximum_elapsed_milliseconds": 0,
    }
    roles = factory.authenticated_internal_roles.roles
    assert len(roles) == 35
    assert not roles.intersection(UGI_COMPONENT_ROLES)
    assert all("*" not in role and "?" not in role for role in roles)
    assert "ugi3_retrieved_upstream_material:ugi-component-50d403b09b1f4ce09cd5" in roles
    assert (
        "ugi3_targeted_upstream:ugi-component-6895b1b938ea2c6e1f6f:mo_2018_eicosanal_pcc" in roles
    )
    assert "exact_c18_terminal_acid" in roles
    assert "terminal_protected_propargyl_alcohol" in roles
    assert len([role for role in roles if role.startswith("ugi3_upstream_material:")]) == 24
    qualification = factory.planner_context_qualification
    assert qualification.source_inputs_sha256 == SOURCE_INPUTS_SHA256
    assert qualification.qualified_l1_registry_artifact.sha256 == REGISTRY_SHA256
    assert qualification.qualified_l1_registry_artifact.path == (
        "data/vendor/qualified_reactions_v1.json"
    )
    assert qualification.internal_role_manifest_sha256 == (
        factory.internal_role_manifest.manifest_sha256
    )
    assert len(qualification.budget_sources) == 4
    assert qualification.to_dict()["scope"]["scalar_value"] is None
    assert qualification.to_dict()["scope"]["success_probability"] is None


def test_terminal_aware_factory_uses_composer_bound_cache_and_external_assessment(
    tmp_path: Path,
) -> None:
    factory, terminal, _ = _factory()
    binding = _binding(tmp_path, factory)
    guided_context = _assessment_context(binding, factory, arm=MatchedArm.GUIDED)
    guided_cache = binding.bind(guided_context)
    planner = factory.build_planner(
        guided_cache,
        factory.planner_context,
        guided_context,
        terminal,
    )

    audit = planner.support_audit.to_dict()
    assert audit["support"]["terminal_sha256"] == terminal.terminal_sha256
    assert audit["support"]["l1_reverification"]["exact_product_reconstructed"] is True
    assert [item["role"] for item in audit["support"]["handle_rechecks"]] == list(ROLE_NAMES)
    assert [item["role"] for item in audit["support"]["root_targets"]] == list(ROLE_NAMES)
    assert len(audit["support"]["root_qualifications"]) == 3
    assert audit["scalar_value"] is None
    assert audit["success_probability"] is None

    receipt = assess_locked_ugi_terminal_routes(
        terminal,
        l1_reverifier=factory.l1_reverifier,
        planner=planner,
        planner_context=factory.planner_context,
        assessment_context=guided_context,
        assessment_at_utc=ASSESSMENT_AT,
    )
    assert tuple(item.role for item in receipt.role_assessments) == ROLE_NAMES
    assert receipt.product_value.to_dict()["scalar_value"] is None
    assert receipt.product_value.to_dict()["success_probability"] is None
    assert receipt.realized_route_usage.fits_within(guided_context.unit_reservation)

    post_hoc_context = _assessment_context(binding, factory, arm=MatchedArm.POST_HOC)
    post_hoc_cache = binding.bind(post_hoc_context)
    post_hoc_planner = factory(
        post_hoc_cache,
        factory.planner_context,
        post_hoc_context,
        terminal,
    )
    post_hoc_receipt = assess_locked_ugi_terminal_routes(
        terminal,
        l1_reverifier=factory.l1_reverifier,
        planner=post_hoc_planner,
        planner_context=factory.planner_context,
        assessment_context=post_hoc_context,
        assessment_at_utc=ASSESSMENT_AT,
    )
    cache_audit = binding.finalize()
    assert receipt.product_value_sha256 == post_hoc_receipt.product_value_sha256
    assert cache_audit.base_before == cache_audit.base_after
    assert cache_audit.guided_after == cache_audit.post_hoc_after


def test_runtime_budget_clone_context_candidate_and_checkpoint_fail_closed(
    tmp_path: Path,
) -> None:
    factory, terminal, candidate = _factory()
    binding = _binding(tmp_path, factory)
    context = _assessment_context(binding, factory, arm=MatchedArm.GUIDED)
    cache = binding.bind(context)
    required = context.unit_reservation
    oversized = RouteComputeUsage(
        logical_planner_calls=required.logical_planner_calls + 1,
        physical_planner_calls=required.physical_planner_calls,
        logical_verifier_calls=required.logical_verifier_calls,
        physical_verifier_calls=required.physical_verifier_calls,
    )
    with pytest.raises(UgiProductionTerminalRouteEvaluatorError, match="unit reservation"):
        factory.build_planner(
            cache,
            factory.planner_context,
            replace(context, unit_reservation=oversized),
            terminal,
        )
    with pytest.raises(UgiProductionTerminalRouteEvaluatorError, match="clone IDs"):
        factory.build_planner(
            cache,
            factory.planner_context,
            replace(context, cache_clone_id="different-clone"),
            terminal,
        )
    wrong_context = replace(factory.planner_context, planner_id="wrong-planner")
    with pytest.raises(UgiProductionTerminalRouteEvaluatorError, match="runtime planner context"):
        factory.build_planner(cache, wrong_context, context, terminal)

    candidate["smiles"] = "CCO"
    with pytest.raises(
        UgiProductionTerminalRouteEvaluatorError,
        match="per-terminal support qualification",
    ):
        factory.build_planner(cache, factory.planner_context, context, terminal)
    with pytest.raises(UgiProductionTerminalRouteEvaluatorError, match="selected checkpoint"):
        factory.build_planner(
            cache,
            factory.planner_context,
            context,
            replace(terminal, generator_checkpoint_sha256=_hash("other-generator")),
        )


def test_factory_rejects_wrong_l1_identity_exact_expiry_and_mutated_graph_support(
    tmp_path: Path,
) -> None:
    payload = _payload()
    candidate = _candidate(payload)
    with pytest.raises(
        UgiProductionTerminalRouteEvaluatorError,
        match="context or internal-role qualification failed",
    ):
        build_production_ugi_terminal_aware_planner_factory(
            repo_root=REPO,
            assessment_as_of_utc=ASSESSMENT_AT,
            expected_cumulative_source_inputs_sha256=SOURCE_INPUTS_SHA256,
            selected_generator_checkpoint_sha256=_hash("selected-generator"),
            graph_support=_graph_support(payload),
            l1_reverifier=_reverifier(l1_sha256=_hash("wrong-l1")),
            candidate_record_resolver=lambda _: candidate,
        )

    with pytest.raises(
        UgiProductionTerminalRouteEvaluatorError,
        match="cumulative production Ugi source failed authentication",
    ):
        build_production_ugi_terminal_aware_planner_factory(
            repo_root=REPO,
            assessment_as_of_utc="2026-08-09T05:16:00Z",
            expected_cumulative_source_inputs_sha256=SOURCE_INPUTS_SHA256,
            selected_generator_checkpoint_sha256=_hash("selected-generator"),
            graph_support=_graph_support(payload),
            l1_reverifier=_reverifier(),
            candidate_record_resolver=lambda _: candidate,
        )

    factory, terminal, _ = _factory()
    binding = _binding(tmp_path, factory)
    context = _assessment_context(binding, factory, arm=MatchedArm.GUIDED)
    cache = binding.bind(context)
    assert isinstance(factory.graph_support.model_config, dict)
    factory.graph_support.model_config["maximum_total_atoms"] = 95
    with pytest.raises(
        UgiProductionTerminalRouteEvaluatorError,
        match="graph-support context changed",
    ):
        factory.build_planner(cache, factory.planner_context, context, terminal)
