from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_route_registry_pair_contract import (
    PUBLIC_AGGREGATE_FIELDS,
    RECORD_LEDGER_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    TARGET_KEY_SHA256,
    TARGET_ROLE,
    TARGET_SMILES,
    Ugi3RouteRegistryPairContractError,
    component_key_sha256,
    exact_l1_eligible,
    validate_binding,
    validate_protocol,
    validate_public_aggregate,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json"
EXACT_C18_CONFIG = REPO / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
EXACT_C18_RESULT = REPO / "results/phase1/ugi3_exact_c18_route_v1/result.json"
EXACT_C18_ASSESSMENT = REPO / "results/phase1/ugi3_exact_c18_route_v1/assessment.json.gz"


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_gzip_json(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(_stable_json_bytes(value))


def _component_record(
    role: str,
    smiles: str,
    *,
    route_complete: bool,
    exact_l2_steps: int = 0,
    terminal_availability: str = "open",
    family_template_admitted: bool = False,
    source_class: str = "synthetic_test_record",
) -> dict[str, Any]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "component_key_sha256": component_key_sha256(role, smiles),
        "assessment_outcome": "complete" if route_complete else "missing_knowledge",
        "route_complete": route_complete,
        "exact_l2_steps": exact_l2_steps,
        "terminal_availability": terminal_availability,
        "family_template_admitted": family_template_admitted,
        "source_class": source_class,
    }


def _anchor() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text())
    sealed = protocol["sealed_holdout_anchor"]
    return {
        "contract_sha256": sealed["contract"]["sha256"],
        "seal_sha256": sealed["seal"]["sha256"],
        "program_draw_sha256": sealed["program_draw"]["sha256"],
        "program_rows": sealed["program_draw"]["rows"],
        "seeds": sealed["seeds"],
    }


def _snapshot(
    tmp_path: Path,
    *,
    name: str,
    role: str,
    semantic_label: str,
    records: list[dict[str, Any]],
    parent: dict[str, str] | None,
    source_final_component_ledger: dict[str, str],
    anchor: dict[str, Any] | None = None,
) -> Path:
    ledger_path = tmp_path / f"{name}.records.json.gz"
    _write_gzip_json(
        ledger_path,
        {"schema_version": RECORD_LEDGER_SCHEMA_VERSION, "records": records},
    )
    manifest_path = tmp_path / f"{name}.manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "status": "frozen_before_molecular_holdout_reveal",
            "snapshot_id": name,
            "snapshot_role": role,
            "semantic_label": semantic_label,
            "sealed_holdout_anchor": anchor or _anchor(),
            "identity_policy": {
                "match_on_exact_role_and_canonical_smiles": True,
                "permit_similarity_or_family_matching": False,
                "permit_neighbor_homologue_admission": False,
                "permit_reaction_family_template_promotion": False,
            },
            "source_final_component_ledger": source_final_component_ledger,
            "parent_snapshot": parent,
            "records": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
                "schema_version": RECORD_LEDGER_SCHEMA_VERSION,
                "rows": len(records),
            },
        },
    )
    return manifest_path


def _closed_c18_artifacts(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path, Path]:
    if complete:
        return EXACT_C18_CONFIG, EXACT_C18_RESULT, EXACT_C18_ASSESSMENT

    with gzip.open(EXACT_C18_ASSESSMENT, "rt") as handle:
        assessment_payload = json.load(handle)
    value = assessment_payload["synthesis_value"]
    value["assessment_outcome"] = "missing_knowledge"
    value["current_terminal_leaf_count"] = 0
    value["unassessed_terminal_leaf_count"] = value["leaf_count"]
    assessment = tmp_path / "assessment.json.gz"
    _write_gzip_json(assessment, assessment_payload)

    result_payload = json.loads(EXACT_C18_RESULT.read_text())
    result_payload["summary"]["route_complete"] = False
    result_payload["summary"]["terminal_availability"] = "unassessed"
    result_payload["adjudication"]["current_l3_procurement_closed"] = False
    result_payload["adjudication"]["route_complete"] = False
    result_payload["artifacts"][assessment.name]["sha256"] = sha256_file(assessment)
    result = tmp_path / "exact_c18_result.json"
    _write_json(result, result_payload)
    return EXACT_C18_CONFIG, result, assessment


def _binding(
    tmp_path: Path,
    *,
    r0_records: list[dict[str, Any]] | None = None,
    r1_records: list[dict[str, Any]] | None = None,
    c18_complete: bool = True,
    r1_anchor: dict[str, Any] | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shared_c16 = _component_record(
        "oxoester_aldehyde_body_tail",
        "CCCCCCCCCCCCCCCC=O",
        route_complete=True,
        exact_l2_steps=2,
        terminal_availability="current_closed",
        source_class="independently_qualified_c16",
    )
    r0_target = _component_record(TARGET_ROLE, TARGET_SMILES, route_complete=False)
    r1_target = _component_record(
        TARGET_ROLE,
        TARGET_SMILES,
        route_complete=True,
        exact_l2_steps=3,
        terminal_availability="current_closed",
        family_template_admitted=False,
        source_class="exact_c18_three_step_route_current_terminal",
    )
    if r0_records is None:
        r0_records = [shared_c16, r0_target]
    if r1_records is None:
        r1_records = [dict(shared_c16), r1_target]
    source_path = tmp_path / "source_final_component_ledger.json.gz"
    source_schema = "synthetic_frozen_final_component_ledger.v1"
    _write_gzip_json(
        source_path,
        {
            "schema_version": source_schema,
            "status": "frozen_final_component_ledger",
            "records": r0_records,
        },
    )
    source_specification = {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "schema_version": source_schema,
    }
    r0_path = _snapshot(
        tmp_path,
        name="r0_without_c18",
        role="immutable_parent_registry",
        semantic_label="without_c18",
        records=r0_records,
        parent=None,
        source_final_component_ledger=source_specification,
    )
    r0_spec = {"path": str(r0_path), "sha256": sha256_file(r0_path)}
    r1_path = _snapshot(
        tmp_path,
        name="r1_with_c18",
        role="immutable_child_registry",
        semantic_label="with_c18",
        records=r1_records,
        parent=r0_spec,
        source_final_component_ledger=source_specification,
        anchor=r1_anchor,
    )
    config_path, result_path, assessment_path = _closed_c18_artifacts(
        tmp_path, complete=c18_complete
    )
    r0_manifest = json.loads(r0_path.read_text())
    r1_manifest = json.loads(r1_path.read_text())
    diff_path = tmp_path / "registry_diff.json"
    _write_json(
        diff_path,
        {
            "schema_version": "phase1_ugi3_route_registry_diff.v1",
            "status": "exactly_one_authorized_record_change",
            "visibility": "private_hash_pinned",
            "r0_semantic_label": "without_c18",
            "r1_semantic_label": "with_c18",
            "record_count_r0": len(r0_records),
            "record_count_r1": len(r1_records),
            "added_count": 0,
            "deleted_count": 0,
            "changed_count": 1,
            "non_target_mutation_count": 0,
            "regression_count": 0,
            "changed_component": {
                "component_key_sha256": TARGET_KEY_SHA256,
                "role": TARGET_ROLE,
                "r0_route_complete": False,
                "r1_route_complete": True,
                "exact_l2_steps": 3,
                "terminal_availability": "current_closed",
                "family_template_admitted": False,
            },
            "all_non_c18_records_canonical_json_equal": True,
            "source_final_component_ledger_sha256": sha256_file(source_path),
            "exact_c18_config_sha256": sha256_file(config_path),
            "exact_c18_result_sha256": sha256_file(result_path),
            "exact_c18_assessment_sha256": sha256_file(assessment_path),
            "r0_records_sha256": r0_manifest["records"]["sha256"],
            "r1_records_sha256": r1_manifest["records"]["sha256"],
        },
    )
    binding_path = tmp_path / "binding.json"
    _write_json(
        binding_path,
        {
            "schema_version": "phase1_ugi3_route_registry_pair_binding.v1",
            "status": "frozen_before_molecular_holdout_reveal",
            "protocol": {
                "path": str(PROTOCOL.relative_to(REPO)),
                "sha256": sha256_file(PROTOCOL),
            },
            "r0_manifest": r0_spec,
            "r1_manifest": {"path": str(r1_path), "sha256": sha256_file(r1_path)},
            "registry_diff": {"path": str(diff_path), "sha256": sha256_file(diff_path)},
            "exact_c18_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "exact_c18_result": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
            },
            "exact_c18_assessment": {
                "path": str(assessment_path),
                "sha256": sha256_file(assessment_path),
            },
        },
    )
    return binding_path


def test_live_protocol_fails_closed_after_one_shot_reveal() -> None:
    with pytest.raises(
        Ugi3RouteRegistryPairContractError,
        match="sealed molecular sample exists before immutable registry binding",
    ):
        validate_protocol(REPO, PROTOCOL)


def test_frozen_protocol_validates_in_isolated_prereveal_view(
    prereveal_registry_view: Any,
) -> None:
    result = validate_protocol(
        prereveal_registry_view.repo,
        prereveal_registry_view.protocol,
    )

    assert result["protocol_valid"] is True
    assert result["registry_pair_bound"] is False
    assert result["holdout_reveal_authorized"] is False
    assert result["sealed_holdout"]["molecular_holdout_generated"] is False
    assert result["sealed_holdout"]["molecular_holdout_inspected"] is False


@pytest.mark.parametrize(
    ("valid", "reconstructed", "forward", "eligible"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, None, False),
    ],
)
def test_exact_l1_eligibility_requires_all_three_exact_gates(
    valid: bool,
    reconstructed: bool,
    forward: bool | None,
    eligible: bool,
) -> None:
    row = {
        "valid": valid,
        "component_reconstruction_valid": reconstructed,
        "l1_forward_verification": {"exact_product_reconstructed": forward},
    }
    assert exact_l1_eligible(row) is eligible


def test_component_key_is_exact_role_plus_canonical_identity() -> None:
    assert component_key_sha256(TARGET_ROLE, TARGET_SMILES) == TARGET_KEY_SHA256
    assert component_key_sha256("another_role", TARGET_SMILES) != TARGET_KEY_SHA256


def test_complete_immutable_pair_authorizes_one_time_reveal(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    result = validate_binding(
        prereveal_registry_view.repo,
        prereveal_registry_view.protocol,
        _binding(tmp_path),
    )

    assert result["registry_pair_bound"] is True
    assert result["holdout_reveal_authorized"] is True
    assert result["authorized_delta_count"] == 1
    assert result["registry_record_count"] == 2


def test_non_c18_evidence_must_be_identical_in_both_registries(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    shared = _component_record(
        "oxoester_aldehyde_body_tail",
        "CCCCCCCCCCCCCCCC=O",
        route_complete=True,
        exact_l2_steps=2,
        terminal_availability="current_closed",
        source_class="independently_qualified_c16",
    )
    changed = dict(shared)
    changed["source_class"] = "mutated_after_parent_freeze"
    r0 = [shared, _component_record(TARGET_ROLE, TARGET_SMILES, route_complete=False)]
    r1 = [
        changed,
        _component_record(
            TARGET_ROLE,
            TARGET_SMILES,
            route_complete=True,
            exact_l2_steps=3,
            terminal_availability="current_closed",
        ),
    ]
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="exactly one authorized"):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            _binding(tmp_path, r0_records=r0, r1_records=r1),
        )


def test_added_or_deleted_registry_identity_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    r0 = [
        _component_record(TARGET_ROLE, TARGET_SMILES, route_complete=False),
        _component_record("amine_head", "CN", route_complete=True),
    ]
    r1 = [
        _component_record(
            TARGET_ROLE,
            TARGET_SMILES,
            route_complete=True,
            exact_l2_steps=3,
            terminal_availability="current_closed",
        )
    ]
    with pytest.raises(
        Ugi3RouteRegistryPairContractError,
        match="(record count|key set) changed",
    ):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            _binding(tmp_path, r0_records=r0, r1_records=r1),
        )


def test_missing_source_component_ledger_lineage_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    binding_path = _binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    r0_path = Path(binding["r0_manifest"]["path"])
    r1_path = Path(binding["r1_manifest"]["path"])

    r0 = json.loads(r0_path.read_text())
    r0.pop("source_final_component_ledger")
    _write_json(r0_path, r0)
    binding["r0_manifest"]["sha256"] = sha256_file(r0_path)

    r1 = json.loads(r1_path.read_text())
    r1["parent_snapshot"] = dict(binding["r0_manifest"])
    _write_json(r1_path, r1)
    binding["r1_manifest"]["sha256"] = sha256_file(r1_path)
    _write_json(binding_path, binding)

    with pytest.raises(Ugi3RouteRegistryPairContractError, match="lacks an exact frozen"):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            binding_path,
        )


def test_conflicting_source_component_ledgers_fail_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    binding_path = _binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    r1_path = Path(binding["r1_manifest"]["path"])
    r1 = json.loads(r1_path.read_text())

    original_source = Path(r1["source_final_component_ledger"]["path"])
    with gzip.open(original_source, "rt") as handle:
        alternate = json.load(handle)
    alternate["records"] = list(reversed(alternate["records"]))
    alternate_source = tmp_path / "alternate_source_component_ledger.json.gz"
    _write_gzip_json(alternate_source, alternate)
    r1["source_final_component_ledger"] = {
        "path": str(alternate_source),
        "sha256": sha256_file(alternate_source),
        "schema_version": alternate["schema_version"],
    }
    _write_json(r1_path, r1)
    binding["r1_manifest"]["sha256"] = sha256_file(r1_path)
    _write_json(binding_path, binding)

    with pytest.raises(Ugi3RouteRegistryPairContractError, match="do not own the same"):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            binding_path,
        )


def test_incomplete_c18_l3_evidence_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="not fully closed"):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            _binding(tmp_path, c18_complete=False),
        )


def test_r1_seed_or_parent_anchor_change_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    wrong_anchor = _anchor()
    wrong_anchor["seeds"] = {**wrong_anchor["seeds"], "terminal": 99}
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="holdout anchor changed"):
        validate_binding(
            prereveal_registry_view.repo,
            prereveal_registry_view.protocol,
            _binding(tmp_path, r1_anchor=wrong_anchor),
        )


def test_retry_replanning_or_post_reveal_evidence_policy_cannot_be_relaxed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    protocol["execution_policy"]["permit_replanning"] = True
    protocol["execution_policy"]["permit_post_reveal_evidence"] = True
    path = tmp_path / "relaxed_protocol.json"
    _write_json(path, protocol)
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="execution policy"):
        validate_protocol(prereveal_registry_view.repo, path)


def test_future_dated_protocol_freeze_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    protocol["frozen_at_utc"] = "2099-01-01T00:00:00Z"
    path = tmp_path / "future_protocol.json"
    _write_json(path, protocol)
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="future-dated"):
        validate_protocol(prereveal_registry_view.repo, path)


def test_public_result_is_aggregate_only() -> None:
    aggregate = {field: 0 for field in PUBLIC_AGGREGATE_FIELDS}
    aggregate["confidence_interval_method"] = "two_sided_clopper_pearson_95pct"
    aggregate["confidence_interval_95pct"] = [0.0, 1.0]
    aggregate["aggregate_counts_by_evidence_family"] = {}
    validate_public_aggregate(aggregate)

    aggregate["canonical_product_smiles"] = "CC"
    with pytest.raises(Ugi3RouteRegistryPairContractError, match="fields changed"):
        validate_public_aggregate(aggregate)


def test_snapshot_and_binding_hashes_are_deterministic(tmp_path: Path) -> None:
    first = _binding(tmp_path / "first")
    second = _binding(tmp_path / "second")
    first_payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())

    # Paths differ, but record-ledger bytes and semantic contents are identical.
    assert sha256_file(
        Path(first_payload["r0_manifest"]["path"]).with_name("r0_without_c18.records.json.gz")
    ) == sha256_file(
        Path(second_payload["r0_manifest"]["path"]).with_name("r0_without_c18.records.json.gz")
    )
