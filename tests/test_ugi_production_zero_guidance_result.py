from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_generated_terminal_support import (
    QualifiedGeneratedUgiTerminalSupport,
    RoleHandleRecheck,
)
from forge.product.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.product.ugi_production_terminal_route_evaluator import (
    ProductionQualifiedRoutePlanner,
    ProductionTerminalSupportAudit,
    _support_sha256,
)
from forge.product.ugi_production_zero_guidance_result import (
    ArmTerminalSupportAuditSnapshot,
    AuthenticatedZeroGuidanceComposerResult,
    ProductionTerminalSupportAuditCollector,
    ProductionZeroGuidanceExecutionResult,
    UgiProductionZeroGuidanceResultError,
)
from forge.product.ugi_zero_guidance_rehearsal import ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION
from forge.route.planner import RouteTarget
from forge.route.terminal_assessment import ExactL1ForwardVerification
from forge.route.ugi3_support_boundary import (
    MolecularSupportState,
    TargetQualification,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _composer_payload(
    *,
    guided_clone: str = "guided-clone",
    post_hoc_clone: str = "post-hoc-clone",
    terminal_sha256: str | None = None,
) -> dict[str, object]:
    terminal_sha256 = terminal_sha256 or _hash("terminal")
    common_record = {
        "unit_id": "unit-0",
        "terminal_sha256": terminal_sha256,
        "route_assessed": True,
    }
    content: dict[str, object] = {
        "schema_version": ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION,
        "status": "zero_guidance_rehearsal_complete",
        "preflight": {"preflight_sha256": _hash("preflight")},
        "matched_run": {
            "zero_guidance": True,
            "zero_guidance_bitwise_equivalent": True,
            "shared_censored_unit_ids": [],
            "guided_cache_clone_id": guided_clone,
            "post_hoc_cache_clone_id": post_hoc_clone,
            "guided_records": [{**common_record, "cache_clone_id": guided_clone}],
            "post_hoc_records": [{**common_record, "cache_clone_id": post_hoc_clone}],
        },
        "chemistry_projections": [{"unit_id": "unit-0"}],
        "chemistry_projection_identity_proven": True,
        "guided_route_receipts": [{"unit_id": "unit-0"}],
        "post_hoc_route_receipts": [{"unit_id": "unit-0"}],
        "cache_audit": {"base_snapshot_sha256": _hash("cache")},
        "scope": {
            "guidance_strength": 0,
            "synthesis_scalar_defined": False,
            "nonzero_synthesis_guidance": False,
            "biological_guidance": False,
            "candidate_selection": False,
            "private_holdout_accessed": False,
        },
    }
    return {**content, "result_sha256": _hash_payload(content)}


def _hash_payload(value: object) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _authenticate(value: dict[str, object] | None = None):
    payload = value or _composer_payload()
    return AuthenticatedZeroGuidanceComposerResult.from_canonical_bytes(
        (_stable_json(payload) + "\n").encode()
    )


def _support(
    *,
    terminal_sha256: str | None = None,
    generator_sha256: str | None = None,
    product_smiles: str = "CCN",
) -> QualifiedGeneratedUgiTerminalSupport:
    terminal_sha256 = terminal_sha256 or _hash("terminal")
    generator_sha256 = generator_sha256 or _hash("generator")
    return QualifiedGeneratedUgiTerminalSupport(
        terminal_sha256=terminal_sha256,
        generator_checkpoint_sha256=generator_sha256,
        product_smiles=product_smiles,
        l1_reverification=ExactL1ForwardVerification(
            exact_product_reconstructed=True,
            maximum_outcomes=64,
            maximum_outcomes_saturated=False,
            outcome_count=1,
        ),
        handle_rechecks=tuple(
            RoleHandleRecheck(
                role=role,
                raw_handle_matches=1,
                symmetry_distinct_handle_sites=1,
                forbidden_substructure_match=False,
                passes_registry_handle_policy=True,
            )
            for role in ROLE_NAMES
        ),
        root_targets=tuple(
            RouteTarget(role=role, canonical_smiles=smiles)
            for role, smiles in zip(
                ROLE_NAMES,
                ("CN", "CC=O", "[C-]#[N+]C"),
                strict=True,
            )
        ),
        root_qualifications=tuple(
            TargetQualification(
                exact_l1_eligible=True,
                supported_ugi_role=True,
                role_handle_qualified=True,
                molecular_support_state=MolecularSupportState.WITHIN_DECLARED_SUPPORT,
            )
            for _ in ROLE_NAMES
        ),
    )


def _audit(
    clone_id: str,
    *,
    support: QualifiedGeneratedUgiTerminalSupport | None = None,
    source_inputs_sha256: str | None = None,
) -> ProductionTerminalSupportAudit:
    support = support or _support()
    return ProductionTerminalSupportAudit(
        terminal_sha256=support.terminal_sha256,
        selected_generator_checkpoint_sha256=support.generator_checkpoint_sha256,
        source_inputs_sha256=source_inputs_sha256 or _hash("source-inputs"),
        graph_support_sha256=_hash("graph-support"),
        internal_role_registry_sha256=_hash("role-registry"),
        internal_role_manifest_sha256=_hash("role-manifest"),
        support_sha256=_support_sha256(support),
        planner_context_sha256=_hash("planner-context"),
        planner_context_qualification_sha256=_hash("planner-context-qualification"),
        assessment_as_of_utc="2026-08-03T04:00:00Z",
        cache_clone_id=clone_id,
        support=support,
    )


def _snapshots(
    *,
    guided: ProductionTerminalSupportAudit | None = None,
    post_hoc: ProductionTerminalSupportAudit | None = None,
) -> tuple[ArmTerminalSupportAuditSnapshot, ...]:
    guided = guided or _audit("guided-clone")
    post_hoc = post_hoc or replace(guided, cache_clone_id="post-hoc-clone")
    return (
        ArmTerminalSupportAuditSnapshot(MatchedArm.GUIDED, "unit-0", guided),
        ArmTerminalSupportAuditSnapshot(MatchedArm.POST_HOC, "unit-0", post_hoc),
    )


def _result(
    snapshots: tuple[ArmTerminalSupportAuditSnapshot, ...] | None = None,
    *,
    composer: AuthenticatedZeroGuidanceComposerResult | None = None,
) -> ProductionZeroGuidanceExecutionResult:
    return ProductionZeroGuidanceExecutionResult(
        composer_result=composer or _authenticate(),
        support_audits=snapshots or _snapshots(),
    )


def test_complete_composer_and_support_audits_are_retained_byte_stably() -> None:
    composer = _authenticate()
    result = _result(composer=composer)

    serialized = result.to_dict()
    assert serialized["composer_result"] == composer.value
    assert serialized["composer_result_sha256"] == composer.result_sha256
    assert serialized["paired_support_count"] == 1
    assert serialized["paired_support_identity_except_cache_clone_proven"] is True
    assert len(serialized["support_audits"]) == 2
    assert (
        serialized["support_audits"][0]["audit"]["support"]
        == (_snapshots()[0].audit.to_dict()["support"])
    )
    assert result.canonical_bytes == result.canonical_bytes
    assert json.loads(result.canonical_bytes)["result_sha256"] == result.result_sha256


@pytest.mark.parametrize(
    "field",
    [
        "source_inputs_sha256",
        "graph_support_sha256",
        "internal_role_registry_sha256",
        "internal_role_manifest_sha256",
        "planner_context_sha256",
        "planner_context_qualification_sha256",
        "assessment_as_of_utc",
    ],
)
def test_any_support_pair_metadata_mismatch_fails_closed(field: str) -> None:
    guided = _audit("guided-clone")
    value = "2026-08-03T04:00:01Z" if field == "assessment_as_of_utc" else _hash("drift")
    post_hoc = replace(guided, cache_clone_id="post-hoc-clone", **{field: value})

    with pytest.raises(UgiProductionZeroGuidanceResultError, match="differ beyond"):
        _result(_snapshots(guided=guided, post_hoc=post_hoc))


def test_nested_support_mismatch_fails_closed() -> None:
    guided = _audit("guided-clone")
    drifted_support = _support(product_smiles="CCCN")
    post_hoc = _audit("post-hoc-clone", support=drifted_support)

    with pytest.raises(UgiProductionZeroGuidanceResultError, match="differ beyond"):
        _result(_snapshots(guided=guided, post_hoc=post_hoc))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_missing_extra_or_duplicate_support_audit_fails_closed(mutation: str) -> None:
    snapshots = list(_snapshots())
    if mutation == "missing":
        snapshots.pop()
    elif mutation == "extra":
        snapshots.append(
            ArmTerminalSupportAuditSnapshot(
                MatchedArm.GUIDED,
                "unit-extra",
                _audit("guided-clone"),
            )
        )
    else:
        snapshots.append(snapshots[0])

    with pytest.raises(UgiProductionZeroGuidanceResultError, match="exactly cover"):
        _result(tuple(snapshots))


@pytest.mark.parametrize(
    ("guided_clone", "post_hoc_clone", "error"),
    [
        ("wrong-guided", "post-hoc-clone", "bind"),
        ("guided-clone", "wrong-post-hoc", "bind"),
        ("guided-clone", "guided-clone", "nonisolated"),
    ],
)
def test_cache_clone_mismatch_or_nonisolation_fails_closed(
    guided_clone: str,
    post_hoc_clone: str,
    error: str,
) -> None:
    snapshots = _snapshots(
        guided=_audit(guided_clone),
        post_hoc=_audit(post_hoc_clone),
    )
    composer = _authenticate(
        _composer_payload(
            guided_clone=("guided-clone" if guided_clone == "wrong-guided" else guided_clone),
            post_hoc_clone=(
                "post-hoc-clone" if post_hoc_clone == "wrong-post-hoc" else post_hoc_clone
            ),
        )
    )

    with pytest.raises(UgiProductionZeroGuidanceResultError, match=error):
        _result(snapshots, composer=composer)


def test_tampered_composer_checksum_fails_closed() -> None:
    payload = _composer_payload()
    payload["chemistry_projections"] = [{"unit_id": "tampered"}]

    with pytest.raises(UgiProductionZeroGuidanceResultError, match="checksum mismatch"):
        _authenticate(payload)


def test_collector_transparently_retains_each_runtime_support_audit_once() -> None:
    audit = _audit("guided-clone")
    planner = object.__new__(ProductionQualifiedRoutePlanner)
    object.__setattr__(planner, "support_audit", audit)
    delegate_calls: list[tuple[object, object]] = []

    def delegate(cache, planner_context, assessment_context, terminal):
        delegate_calls.append((cache, planner_context))
        assert assessment_context.arm is MatchedArm.GUIDED
        assert terminal.unit_id == "unit-0"
        return planner

    collector = ProductionTerminalSupportAuditCollector(delegate)
    assessment_context = MatchedAssessmentContext(
        arm=MatchedArm.GUIDED,
        route_seed=7,
        remaining_budget=RouteComputeUsage(),
        unit_reservation=RouteComputeUsage(),
        cache_snapshot_sha256=_hash("cache"),
        cache_clone_id="guided-clone",
        post_hoc_lock_manifest_sha256=None,
    )
    terminal = LockedMatchedTerminal(
        unit_id="unit-0",
        morphology_program_sha256=_hash("program"),
        checkpoint_index=8,
        generator_checkpoint_sha256=_hash("generator"),
        closure_checkpoint_sha256=_hash("closure"),
        terminal_id="terminal-0",
        terminal_locked=True,
        terminal_valid=True,
        exact_l1=True,
        terminal_bytes=b"terminal",
        generation_trace_bytes=b"trace",
    )
    cache = object()
    planner_context = object()

    assert collector.build_planner(cache, planner_context, assessment_context, terminal) is planner
    assert delegate_calls == [(cache, planner_context)]
    assert collector.snapshots == (
        ArmTerminalSupportAuditSnapshot(MatchedArm.GUIDED, "unit-0", audit),
    )
    with pytest.raises(UgiProductionZeroGuidanceResultError, match="duplicate"):
        collector.build_planner(cache, planner_context, assessment_context, terminal)


def test_noncanonical_or_duplicate_key_composer_bytes_fail_closed() -> None:
    payload = _composer_payload()
    compact_without_newline = _stable_json(payload).encode()
    with pytest.raises(UgiProductionZeroGuidanceResultError, match="not canonical"):
        AuthenticatedZeroGuidanceComposerResult.from_canonical_bytes(compact_without_newline)

    duplicate = b'{"schema_version":"a","schema_version":"b"}\n'
    with pytest.raises(UgiProductionZeroGuidanceResultError, match="duplicate JSON key"):
        AuthenticatedZeroGuidanceComposerResult.from_canonical_bytes(duplicate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("guidance_strength", 1),
        ("nonzero_synthesis_guidance", True),
        ("biological_guidance", True),
        ("candidate_selection", True),
        ("private_holdout_accessed", True),
    ],
)
def test_expansive_composer_scope_fails_even_with_recomputed_checksum(
    field: str,
    value: object,
) -> None:
    payload = _composer_payload()
    scope = dict(payload["scope"])
    scope[field] = value
    payload["scope"] = scope
    content = {key: value for key, value in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = _hash_payload(content)

    with pytest.raises(UgiProductionZeroGuidanceResultError, match="not a completed"):
        _authenticate(payload)
