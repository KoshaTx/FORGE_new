from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from forge.route.engine.planner import (
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
from forge.route.engine.planner_cache import (
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
    PlannerCacheKey,
)
from forge.route.engine.planner_cache_snapshot import (
    PlannerCacheSnapshotManifest,
    ReadOnlyFilePlannerCache,
    build_file_planner_cache_snapshot_manifest,
    open_matched_file_planner_cache_overlays,
    require_current_l3_context,
)

ASSESSMENT_AT_UTC = "2026-08-15T00:00:00Z"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _limits() -> PlannerBudgetLimits:
    return PlannerBudgetLimits(
        maximum_depth=3,
        maximum_logical_planner_calls=2,
        maximum_expansions=4,
        maximum_product_candidates=4,
        maximum_verifier_calls=4,
        maximum_elapsed_milliseconds=1_000,
    )


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


def _missing_result() -> KnowledgeResult:
    return KnowledgeResult(
        disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
        evidence=(),
        detail="missing route knowledge",
    )


@dataclass
class FixedSource:
    result: KnowledgeResult

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return self.result


def _assessment_for(smiles: str, *, result: KnowledgeResult | None = None):
    return RecursiveRouteAssessor(FixedSource(result or _terminal_result())).assess(
        RouteTarget("amine_head", smiles),
        PlannerBudgetLedger(_limits()),
    )


def _key_for(assessment, context: PlannerCacheContext) -> PlannerCacheKey:
    return PlannerCacheKey.build(
        assessment.target,
        context,
        PlannerBudgetLedger(context.budget_limits),
    )


def _manifest(cache: FilePlannerCache, context: PlannerCacheContext):
    return build_file_planner_cache_snapshot_manifest(
        cache,
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )


def test_snapshot_manifest_is_deterministic_root_independent_and_roundtrips(
    tmp_path: Path,
) -> None:
    context = _context()
    assessment = _assessment_for("CCO")
    key = _key_for(assessment, context)
    first = FilePlannerCache(tmp_path / "first")
    second = FilePlannerCache(tmp_path / "second")
    first.put(key, assessment)
    second.put(key, assessment)

    first_manifest = _manifest(first, context)
    second_manifest = _manifest(second, context)

    assert first_manifest == second_manifest
    assert first_manifest.entry_count == 1
    assert first_manifest.entries[0].key_sha256 == key.digest
    assert PlannerCacheSnapshotManifest.from_dict(first_manifest.to_dict()) == first_manifest
    assert first_manifest.snapshot_sha256 == second_manifest.snapshot_sha256


def test_snapshot_manifest_fails_on_noncanonical_or_corrupt_files(tmp_path: Path) -> None:
    context = _context()
    cache = FilePlannerCache(tmp_path / "cache")
    cache.root.mkdir(parents=True)
    unexpected = cache.root / "unexpected.json"
    unexpected.write_text("{}\n")

    with pytest.raises(PlannerCacheError, match="noncanonical file"):
        _manifest(cache, context)

    unexpected.unlink()
    assessment = _assessment_for("CCO")
    key = _key_for(assessment, context)
    cache.put(key, assessment)
    value = json.loads(cache.path_for(key).read_text())
    value["assessment"]["outcome"] = "missing_knowledge"
    cache.path_for(key).write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(PlannerCacheError, match="checksum mismatch"):
        _manifest(cache, context)


def test_read_only_snapshot_rejects_writes_and_proves_base_unchanged(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    first = _assessment_for("CCO")
    first_key = _key_for(first, context)
    base.put(first_key, first)
    snapshot = ReadOnlyFilePlannerCache(
        base,
        _manifest(base, context),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )

    assert snapshot.get(first_key) == first
    with pytest.raises(PlannerCacheError, match="read-only"):
        snapshot.put(first_key, first)

    second = _assessment_for("CCN")
    second_key = _key_for(second, context)
    base.put(second_key, second)
    assert snapshot.get(second_key) is None
    with pytest.raises(PlannerCacheError, match="base cache changed"):
        snapshot.assert_unchanged()


def test_matched_overlays_are_isolated_and_produce_deterministic_audit(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    base_assessment = _assessment_for("CCO")
    base_key = _key_for(base_assessment, context)
    base.put(base_key, base_assessment)
    caches = open_matched_file_planner_cache_overlays(
        base,
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
        guided_clone_id="guided-clone",
        post_hoc_clone_id="post-hoc-clone",
    )
    new_assessment = _assessment_for("CCN")
    new_key = _key_for(new_assessment, context)

    assert caches.guided.get(base_key) == base_assessment
    assert caches.post_hoc.get(base_key) == base_assessment
    caches.guided.put(new_key, new_assessment)
    assert caches.guided.get(new_key) == new_assessment
    assert caches.post_hoc.get(new_key) is None
    caches.post_hoc.put(new_key, new_assessment)

    audit = caches.finalize()
    serialized = audit.to_dict()
    assert audit.base_before == audit.base_after
    assert audit.guided_before.entry_count == audit.post_hoc_before.entry_count == 0
    assert audit.guided_after.entry_count == audit.post_hoc_after.entry_count == 1
    assert serialized["base_unchanged"] is True
    assert serialized["audit_sha256"] == audit.audit_sha256
    assert caches.finalize().to_dict() == serialized


def test_overlay_conflicts_fail_closed(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    complete = _assessment_for("CCO")
    missing = _assessment_for("CCO", result=_missing_result())
    key = _key_for(complete, context)
    base.put(key, complete)
    caches = open_matched_file_planner_cache_overlays(
        base,
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
        guided_clone_id="guided-clone",
        post_hoc_clone_id="post-hoc-clone",
    )

    with pytest.raises(PlannerCacheError, match="immutable base assessment"):
        caches.guided.put(key, missing)

    new_complete = _assessment_for("CCN")
    new_missing = _assessment_for("CCN", result=_missing_result())
    new_key = _key_for(new_complete, context)
    caches.guided.put(new_key, new_complete)
    with pytest.raises(PlannerCacheError, match="different assessment"):
        caches.guided.put(new_key, new_missing)


def test_overlay_reads_before_base_and_base_corruption_fails_finalize(tmp_path: Path) -> None:
    context = _context()
    assessment = _assessment_for("CCO")
    key = _key_for(assessment, context)
    base = FilePlannerCache(tmp_path / "base")
    guided = FilePlannerCache(tmp_path / "guided")
    post_hoc = FilePlannerCache(tmp_path / "post_hoc")
    for cache in (base, guided, post_hoc):
        cache.put(key, assessment)
    caches = open_matched_file_planner_cache_overlays(
        base,
        guided,
        post_hoc,
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
        guided_clone_id="guided-clone",
        post_hoc_clone_id="post-hoc-clone",
    )
    base.path_for(key).write_text("corrupt\n")

    assert caches.guided.get(key) == assessment
    assert caches.post_hoc.get(key) == assessment
    with pytest.raises(PlannerCacheError, match="invalid JSON"):
        caches.finalize()


def test_matched_factory_rejects_overlap_and_unequal_initial_states(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    with pytest.raises(PlannerCacheError, match="must not overlap"):
        open_matched_file_planner_cache_overlays(
            base,
            FilePlannerCache(base.root / "guided"),
            FilePlannerCache(tmp_path / "post_hoc"),
            context,
            assessment_at_utc=ASSESSMENT_AT_UTC,
            guided_clone_id="guided-clone",
            post_hoc_clone_id="post-hoc-clone",
        )

    guided = FilePlannerCache(tmp_path / "guided")
    post_hoc = FilePlannerCache(tmp_path / "post_hoc_unequal")
    assessment = _assessment_for("CCO")
    guided.put(_key_for(assessment, context), assessment)
    with pytest.raises(PlannerCacheError, match="begin byte-identically"):
        open_matched_file_planner_cache_overlays(
            base,
            guided,
            post_hoc,
            context,
            assessment_at_utc=ASSESSMENT_AT_UTC,
            guided_clone_id="guided-clone",
            post_hoc_clone_id="post-hoc-clone",
        )


@pytest.mark.parametrize(
    ("assessment_at_utc", "message"),
    [
        ("2026-07-31T23:59:59Z", "predates"),
        ("2026-08-31T00:00:00Z", "expired"),
        ("2026-09-01T00:00:00Z", "expired"),
    ],
)
def test_snapshot_fails_closed_outside_l3_validity_window(
    assessment_at_utc: str,
    message: str,
) -> None:
    with pytest.raises(PlannerCacheError, match=message):
        require_current_l3_context(_context(), assessment_at_utc=assessment_at_utc)


def test_l3_context_rejects_invalid_window() -> None:
    with pytest.raises(PlannerCacheError, match="must precede"):
        require_current_l3_context(
            _context(
                l3_accessed_at_utc="2026-08-31T00:00:00Z",
                l3_expires_at_utc="2026-08-31T00:00:00Z",
            ),
            assessment_at_utc=ASSESSMENT_AT_UTC,
        )
