from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from forge.route.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.planner_cache import (
    CachedRoutePlanner,
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
    PlannerCacheKey,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _limits(**overrides: int) -> PlannerBudgetLimits:
    values = {
        "maximum_depth": 3,
        "maximum_logical_planner_calls": 2,
        "maximum_expansions": 4,
        "maximum_product_candidates": 4,
        "maximum_verifier_calls": 4,
        "maximum_elapsed_milliseconds": 1_000,
    }
    values.update(overrides)
    return PlannerBudgetLimits(**values)


def _context(**overrides: object) -> PlannerCacheContext:
    values: dict[str, object] = {
        "planner_id": "exact-evidence-lookup-v1",
        "planner_sha256": _hash("planner"),
        "search_policy_sha256": _hash("search"),
        "value_policy_sha256": _hash("value"),
        "l1_reaction_sha256": _hash("l1"),
        "upstream_reaction_registry_sha256": _hash("upstream"),
        "variant_registry_sha256": _hash("variants"),
        "verifier_sha256": _hash("verifier"),
        "l3_snapshot_sha256": _hash("l3"),
        "l3_region": "US",
        "l3_accessed_at_utc": "2026-08-01T00:00:00Z",
        "l3_expires_at_utc": "2026-08-31T00:00:00Z",
        "software_versions": (("rdkit", "2025.9.6"), ("route_schema", "v1")),
        "identity_policy": "constitutional_canonical_isomeric_false",
        "stereochemistry_policy": "not_represented",
        "budget_limits": _limits(),
    }
    values.update(overrides)
    return PlannerCacheContext(**values)  # type: ignore[arg-type]


def _terminal_result() -> KnowledgeResult:
    evidence = EvidenceRecord(
        evidence_id="terminal-a",
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        source_sha256=_hash("terminal-source"),
        source_locator="procurement:item-a",
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_APPLICABLE,
        availability=AvailabilityState.CURRENT_CLOSED,
    )
    return KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(evidence,),
        detail="accepted terminal",
    )


@dataclass
class CountingSource:
    calls: int = 0

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls += 1
        return _terminal_result()


def _assessment():
    return RecursiveRouteAssessor(CountingSource()).assess(
        RouteTarget("amine_head", "CCO"),
        PlannerBudgetLedger(_limits()),
    )


def test_cache_key_canonicalizes_constitutional_smiles() -> None:
    context = _context()
    first = PlannerCacheKey.build(
        RouteTarget("amine_head", "C(C)O", ("C(C)N",)),
        context,
        PlannerBudgetLedger(context.budget_limits),
    )
    second = PlannerCacheKey.build(
        RouteTarget("amine_head", "CCO", ("CCN",)),
        context,
        PlannerBudgetLedger(context.budget_limits),
    )

    assert first == second
    assert first.payload["target"] == {
        "role": "amine_head",
        "canonical_smiles": "CCO",
        "product_context_smiles": ["CCN"],
    }


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("planner_sha256", _hash("planner-new")),
        ("search_policy_sha256", _hash("search-new")),
        ("value_policy_sha256", _hash("value-new")),
        ("l1_reaction_sha256", _hash("l1-new")),
        ("upstream_reaction_registry_sha256", _hash("upstream-new")),
        ("variant_registry_sha256", _hash("variants-new")),
        ("verifier_sha256", _hash("verifier-new")),
        ("l3_snapshot_sha256", _hash("l3-new")),
        ("l3_region", "EU"),
        ("l3_expires_at_utc", "2026-09-30T00:00:00Z"),
        ("software_versions", (("rdkit", "2026.1"), ("route_schema", "v1"))),
        ("identity_policy", "different_identity"),
        ("stereochemistry_policy", "represented"),
        ("budget_limits", _limits(maximum_expansions=3)),
    ],
)
def test_cache_key_invalidates_on_every_planner_contract_axis(
    field_name: str,
    replacement: object,
) -> None:
    target = RouteTarget("amine_head", "CCO")
    baseline_context = _context()
    changed_context = replace(baseline_context, **{field_name: replacement})
    baseline = PlannerCacheKey.build(
        target,
        baseline_context,
        PlannerBudgetLedger(baseline_context.budget_limits),
    )
    changed = PlannerCacheKey.build(
        target,
        changed_context,
        PlannerBudgetLedger(changed_context.budget_limits),
    )

    assert baseline.digest != changed.digest


def test_file_cache_roundtrip_is_byte_deterministic(tmp_path: Path) -> None:
    assessment = _assessment()
    context = _context()
    key = PlannerCacheKey.build(
        assessment.target,
        context,
        PlannerBudgetLedger(context.budget_limits),
    )
    first = FilePlannerCache(tmp_path / "first")
    second = FilePlannerCache(tmp_path / "second")
    first.put(key, assessment)
    second.put(key, assessment)

    assert first.get(key) == assessment
    assert second.get(key) == assessment
    assert first.path_for(key).read_bytes() == second.path_for(key).read_bytes()
    entry = json.loads(first.path_for(key).read_text())
    assert isinstance(entry["assessment"], dict)
    assert entry["assessment"]["outcome"] == "complete"


def test_cache_detects_assessment_corruption(tmp_path: Path) -> None:
    assessment = _assessment()
    context = _context()
    key = PlannerCacheKey.build(
        assessment.target,
        context,
        PlannerBudgetLedger(context.budget_limits),
    )
    cache = FilePlannerCache(tmp_path)
    cache.put(key, assessment)
    value = json.loads(cache.path_for(key).read_text())
    value["assessment"]["outcome"] = "missing_knowledge"
    cache.path_for(key).write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(PlannerCacheError, match="checksum mismatch"):
        cache.get(key)


def test_cached_planner_separates_logical_calls_hits_and_misses(tmp_path: Path) -> None:
    source = CountingSource()
    context = _context()
    planner = CachedRoutePlanner(
        RecursiveRouteAssessor(source),
        FilePlannerCache(tmp_path),
        context,
    )
    target = RouteTarget("amine_head", "CCO")
    first_budget = PlannerBudgetLedger(context.budget_limits)
    second_budget = PlannerBudgetLedger(context.budget_limits)

    first = planner.assess(target, first_budget)
    second = planner.assess(target, second_budget)

    assert first == second
    assert source.calls == 1
    assert first_budget.logical_planner_calls == 1
    assert first_budget.physical_cache_misses == 1
    assert first_budget.physical_cache_hits == 0
    assert second_budget.logical_planner_calls == 1
    assert second_budget.physical_cache_misses == 0
    assert second_budget.physical_cache_hits == 1


def test_cached_planner_rejects_budget_context_drift(tmp_path: Path) -> None:
    context = _context()
    planner = CachedRoutePlanner(
        RecursiveRouteAssessor(CountingSource()),
        FilePlannerCache(tmp_path),
        context,
    )

    with pytest.raises(PlannerCacheError, match="budget limits differ"):
        planner.assess(
            RouteTarget("amine_head", "CCO"),
            PlannerBudgetLedger(_limits(maximum_expansions=3)),
        )
