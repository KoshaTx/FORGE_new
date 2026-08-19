from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.data.source_evidence_adjudication import (
    SourceEvidenceAdjudicationError,
    _canonical,
    _load_config,
    _member_identity_resolved,
    _new_record,
    _render_ledger,
    _verify_hash,
    _virtual_candidate_is_exact,
)

REPO = Path(__file__).resolve().parents[1]


def test_exact_and_transform_only_records_remain_distinct() -> None:
    exact = _new_record(
        level="L1",
        target_id="measured",
        reaction_family_id="ugi_3cr_agile",
        target_smiles="CCNC(=O)CN",
        components=("CN", "CC=O", "[C-]#[N+]C"),
        source_review_id="source",
        source_asset={"filename": "source.pdf", "sha256": "a" * 64},
        source_locator="page 1",
        evidence_basis="exact_reported_library_execution",
        disposition="admit_exact",
        experimental_outcome_label="reported_library_execution",
        route_closure="L1_only",
        reason_codes=("source",),
    )
    computed = _new_record(
        level="L1",
        target_id="virtual",
        reaction_family_id="ugi_3cr_agile",
        target_smiles="CCNC(=O)CN",
        components=("CN", "CC=O", "[C-]#[N+]C"),
        source_review_id="",
        source_asset=None,
        source_locator="",
        evidence_basis="computed_transform_consistency",
        disposition="admit_transform_consistency",
        experimental_outcome_label="not_observed",
        route_closure="L1_only",
        reason_codes=("not_source_reported_execution",),
    )

    assert exact["disposition"] == "admit_exact"
    assert computed["disposition"] == "admit_transform_consistency"
    assert exact["evidence_record_id"] != computed["evidence_record_id"]
    assert computed["source_asset"] == ""


def test_ledger_rendering_is_byte_deterministic() -> None:
    record = {
        "evidence_record_id": "",
        "level": "",
        "target_id": "",
        "reaction_family_id": "",
        "target_canonical_smiles": "",
        "component_smiles_json": "",
        "source_review_id": "",
        "source_asset": "",
        "source_asset_sha256": "",
        "source_locator": "",
        "evidence_basis": "",
        "disposition": "",
        "experimental_outcome_label": "",
        "route_closure": "",
        "reason_codes_json": "",
    }

    assert _render_ledger([record]) == _render_ledger([record])


def test_canonicalization_is_atom_order_invariant() -> None:
    assert _canonical("CCO", "first") == _canonical("OCC", "second")


def test_virtual_exactness_collapses_equivalent_raw_target_outcomes() -> None:
    row = {
        "decomposition_status": "one_exact_qualified_ugi_decomposition",
        "candidate_count": "1",
    }

    assert _virtual_candidate_is_exact(
        row,
        [
            {
                "target_matching_forward_outcomes": 3,
                "distinct_target_reacting_sites": 1,
            }
        ],
    )
    assert not _virtual_candidate_is_exact(
        row,
        [
            {
                "target_matching_forward_outcomes": 2,
                "distinct_target_reacting_sites": 2,
            }
        ],
    )


def test_member_identity_allows_disclosed_reagent_name_conflict() -> None:
    assert _member_identity_resolved(
        {
            "product_smiles": "CCO",
            "source_structure_status": (
                "exact_lnpdb_match_with_product_specific_proton_nmr_" "and_reagent_name_conflict"
            ),
        }
    )
    assert not _member_identity_resolved(
        {
            "source_structure_status": "unresolved_source_to_lnpdb_conflict",
        }
    )


def test_abstention_can_preserve_unknown_target_identity() -> None:
    record = _new_record(
        level="L2",
        target_id="unresolved",
        reaction_family_id="aldehyde_series",
        target_smiles="",
        components=(),
        source_review_id="source",
        source_asset={"filename": "source.pdf", "sha256": "a" * 64},
        source_locator="page 1",
        evidence_basis="source_conflict",
        disposition="abstain",
        experimental_outcome_label="unresolved",
        route_closure="unresolved",
        reason_codes=("unresolved_source_identity",),
    )

    assert record["target_canonical_smiles"] == ""


def test_mixture_abstention_cannot_invent_a_single_target_graph() -> None:
    record = _new_record(
        level="L1",
        target_id="agile-measured-A1B4C1",
        reaction_family_id="ugi_3cr_agile",
        target_smiles="",
        components=(),
        source_review_id="source",
        source_asset={"filename": "source.pdf", "sha256": "a" * 64},
        source_locator="page 1",
        evidence_basis="reported_mixture_library_execution",
        disposition="abstain_mixture_single_graph",
        experimental_outcome_label="reported_mixture_measurement",
        route_closure="mixture_execution_not_single_graph",
        reason_codes=("B4_cis_trans_mixture",),
    )

    assert record["target_canonical_smiles"] == ""
    assert record["component_smiles_json"] == "[]"


def test_hash_mismatch_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")

    with pytest.raises(SourceEvidenceAdjudicationError, match="hash mismatch"):
        _verify_hash(source, "0" * 64, "supplement")


def test_config_rejects_hidden_human_blocker(tmp_path: Path) -> None:
    source = REPO / "configs/corpus/m0_05_source_evidence_adjudication.json"
    config = json.loads(source.read_text())
    config["policy"]["human_review_required_for_exact_admission"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(SourceEvidenceAdjudicationError, match="human blocker"):
        _load_config(path)


@pytest.mark.needs_vendor
def test_frozen_source_adjudication_contract_when_present() -> None:
    result_path = REPO / "results/m0_05_source_adjudication/result.json"
    if not result_path.exists():
        pytest.skip("source-adjudication result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_05_source_evidence_adjudication.v1"
    assert result["status"] == "completed_automated_source_evidence_gate"
    assert result["summary"]["measured_l1_admit_exact"] == 1100
    assert result["summary"]["measured_l1_abstain_mixture"] == 100
    assert result["summary"]["virtual_l1_admit_transform_consistency"] == 12276
    assert result["summary"]["l2_admit_exact"] == 44
    assert result["summary"]["l2_abstain"] == 1
    assert result["decision"]["human_chemist_review_blocks_training"] is False
    assert result["decision"]["l1_mixture_records_excluded_from_single_graph_training"] == 100
    assert all(result["adversarial_checks"].values())
    ledger = REPO / "results/m0_05_source_adjudication/evidence_ledger.csv.gz"
    payload = ledger.read_bytes()
    assert len(payload) == result["artifacts"]["evidence_ledger.csv.gz"]["bytes"]
    assert (
        hashlib.sha256(payload).hexdigest()
        == result["artifacts"]["evidence_ledger.csv.gz"]["sha256"]
    )
