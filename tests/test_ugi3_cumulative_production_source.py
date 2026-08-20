from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from forge.route.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    RouteTarget,
)
from forge.route.sources.ugi3_cumulative_production_source import (
    LAYER_ORDER,
    AuthenticatedExactTerminalDeltaOverlay,
    CumulativeUgi3ProductionPaths,
    Ugi3CumulativeProductionSourceError,
    load_authenticated_fresh_exact_terminal_delta,
    load_cumulative_production_ugi3_source,
)

REPO = Path(__file__).resolve().parents[1]
ASSESSMENT = "2026-08-03T04:00:00Z"


@pytest.fixture(scope="module")
def cumulative_source():
    return load_cumulative_production_ugi3_source(
        repo_root=REPO,
        assessment_as_of_utc=ASSESSMENT,
    )


def test_production_chain_order_and_manifest_are_frozen(cumulative_source) -> None:
    source, metadata = cumulative_source

    assert source.layer_order == LAYER_ORDER
    assert metadata["layer_order"] == list(LAYER_ORDER)
    assert metadata["unified_l3_window"] == {
        "accessed_utc": "2026-08-03T03:04:10Z",
        "expires_utc": "2026-08-09T05:16:00Z",
        "interval_semantics": "accessed_inclusive_expires_exclusive",
    }
    assert list(metadata["inputs"]) == sorted(metadata["inputs"])
    assert metadata["inputs_sha256"] == (
        "adfc6767d44d9aa56cfb46bba2253e485ec35f5dace0af0f5a11f9cd72bcd3c6"
    )
    assert metadata["scope"] == {
        "zero_guidance_annotation_rehearsal": True,
        "support_boundary_wrapped": False,
        "synthesis_scalar_computed": False,
        "generator_guidance_authorized": False,
        "biology_used": False,
        "selection_used": False,
        "holdout_accessed": False,
    }


def test_all_late_exact_layers_are_reachable(cumulative_source) -> None:
    source, _ = cumulative_source
    terminals = (
        RouteTarget("amine_head", "CCCN"),
        RouteTarget("amine_head", "NC1CCCCC1"),
        RouteTarget("amine_head", "NC1CCNCC1"),
        RouteTarget("amine_head", "CCCCCCCCCCCCCCCCCCN"),
        RouteTarget("oxoester_aldehyde_body_tail", "CCCCCCCCCCCCCCCCC=O"),
    )
    for target in terminals:
        result = source.lookup(target)
        assert result.disposition is KnowledgeDisposition.TERMINAL
        assert result.evidence[0].tier is EvidenceTier.ACCEPTED_TERMINAL
        assert result.evidence[0].exact_substrate is True

    for smiles in ("C#CCCCCCCCCCCCCCCCC=O", "C#CCCCCCCCCCCCCCC=O"):
        result = source.lookup(RouteTarget("oxoester_aldehyde_body_tail", smiles))
        assert result.disposition is KnowledgeDisposition.EXPAND
        assert result.evidence[0].tier is EvidenceTier.EXACT_SOURCE


class _FamilyOnlySource:
    def __init__(self) -> None:
        self.calls: list[RouteTarget] = []

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(
                EvidenceRecord(
                    evidence_id="family-only",
                    tier=EvidenceTier.FAMILY_PROJECTED,
                    source_sha256="a" * 64,
                    source_locator="frozen://family-only",
                    exact_substrate=False,
                    forward_verification=ForwardVerificationState.NOT_RUN,
                ),
            ),
            detail="family evidence remains nonclosing",
        )


def test_exact_terminal_overlay_is_role_and_identity_exact_only() -> None:
    base = _FamilyOnlySource()
    exact = KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(
            EvidenceRecord(
                evidence_id="exact-terminal",
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256="b" * 64,
                source_locator="frozen://exact-terminal",
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=AvailabilityState.CURRENT_CLOSED,
            ),
        ),
        detail="exact terminal",
    )
    overlay = AuthenticatedExactTerminalDeltaOverlay(
        base_source=base,
        terminal_results={("amine_head", "NC1CCNCC1"): exact},
    )

    assert overlay.lookup(RouteTarget("amine_head", "C1CNCCC1N")) is exact
    near = overlay.lookup(RouteTarget("amine_head", "NC1CCNCCC1"))
    wrong_role = overlay.lookup(RouteTarget("oxoester_aldehyde_body_tail", "NC1CCNCC1"))
    assert near.disposition is KnowledgeDisposition.MISSING_KNOWLEDGE
    assert wrong_role.disposition is KnowledgeDisposition.MISSING_KNOWLEDGE
    assert near.evidence[0].tier is EvidenceTier.FAMILY_PROJECTED
    assert wrong_role.evidence[0].tier is EvidenceTier.FAMILY_PROJECTED
    assert base.calls == [
        RouteTarget("amine_head", "NC1CCNCCC1"),
        RouteTarget("oxoester_aldehyde_body_tail", "NC1CCNCC1"),
    ]


def test_exact_terminal_overlay_rejects_tier_promotion() -> None:
    family = _FamilyOnlySource().lookup(RouteTarget("amine_head", "CCN"))
    with pytest.raises(
        Ugi3CumulativeProductionSourceError,
        match="cannot promote non-exact or non-current evidence",
    ):
        AuthenticatedExactTerminalDeltaOverlay(
            base_source=_FamilyOnlySource(),
            terminal_results={("amine_head", "CCN"): family},
        )


def test_family_projection_in_cumulative_source_is_not_promoted(cumulative_source) -> None:
    source, _ = cumulative_source
    ledger_path = REPO / "results/phase1/ugi3_hybrid_search/assessment_ledger.json.gz"
    with gzip.open(ledger_path, "rt") as handle:
        rows = json.load(handle)["records"]

    found = None
    for row in rows:
        if row.get("search_channel") != "family_projection":
            continue
        target = RouteTarget(
            role=row["role"],
            canonical_smiles=row["canonical_smiles"],
        )
        result = source.lookup(target)
        if (
            result.disposition is KnowledgeDisposition.MISSING_KNOWLEDGE
            and result.evidence
            and result.evidence[0].tier is EvidenceTier.FAMILY_PROJECTED
        ):
            found = result
            break
    assert found is not None
    assert all(record.exact_substrate is False for record in found.evidence)


def test_fresh_terminal_delta_rejects_exact_evidence_expiry() -> None:
    with pytest.raises(
        Ugi3CumulativeProductionSourceError,
        match="fresh_v2_piperazine_head evidence is not current",
    ):
        load_authenticated_fresh_exact_terminal_delta(
            base_source=_FamilyOnlySource(),
            repo_root=REPO,
            assessment_as_of_utc="2026-08-28T18:23:53Z",
            paths=CumulativeUgi3ProductionPaths.from_repo(REPO),
        )


@pytest.mark.parametrize(
    "assessment_as_of_utc",
    ("2026-08-09T05:16:00Z", "2026-08-09T05:16:01Z"),
)
def test_expired_unified_window_fails_closed(assessment_as_of_utc: str) -> None:
    with pytest.raises(
        Ugi3CumulativeProductionSourceError,
        match="outside the unified authenticated L3 window",
    ):
        load_cumulative_production_ugi3_source(
            repo_root=REPO,
            assessment_as_of_utc=assessment_as_of_utc,
        )


def test_nested_input_pin_mismatch_fails_before_source_exposure(tmp_path: Path) -> None:
    frozen = CumulativeUgi3ProductionPaths.from_repo(REPO)
    config = json.loads(frozen.exact_config.read_text())
    config["inputs"]["component_program"]["expected_sha256"] = "0" * 64
    changed = tmp_path / "exact-config.json"
    changed.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        Ugi3CumulativeProductionSourceError,
        match="exact_evidence_base input pin mismatch: component_program",
    ):
        load_cumulative_production_ugi3_source(
            repo_root=REPO,
            assessment_as_of_utc=ASSESSMENT,
            paths=replace(frozen, exact_config=changed),
        )
