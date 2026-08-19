from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_route_registry_pair_builder import (
    OUTPUT_LABELS,
    REQUIRED_BINDING_POLICY,
    Ugi3RouteRegistryPairBuilderError,
    build_registry_pair,
)
from forge.route.ugi3_route_registry_pair_contract import (
    TARGET_ROLE,
    TARGET_SMILES,
    Ugi3RouteRegistryPairContractError,
    validate_binding,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json"
UNBOUND_CONFIG = REPO / "configs/route/phase1_ugi3_route_registry_pair_builder_v1.json"
EXACT_C18_CONFIG = REPO / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
EXACT_C18_RESULT = REPO / "results/phase1/ugi3_exact_c18_route_v1/result.json"
EXACT_C18_ASSESSMENT = REPO / "results/phase1/ugi3_exact_c18_route_v1/assessment.json.gz"
SOURCE_SCHEMA = "synthetic_frozen_final_component_ledger.v1"


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(_stable_json_bytes(value))


def _read_gzip_payload(payload: bytes) -> dict[str, Any]:
    return json.loads(gzip.decompress(payload))


def _value(
    role: str,
    smiles: str,
    *,
    complete: bool,
    steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": "forge.synthesis_value.v1",
        "target": {
            "role": role,
            "canonical_smiles": smiles,
            "product_context_smiles": [],
        },
        "assessment_outcome": "complete" if complete else "missing_knowledge",
        "forward_consistency": "exact_unique" if complete else "not_applicable",
        "evidence_support": "exact_identity" if complete else "missing",
        "route_step_count": steps,
        "maximum_route_depth": steps,
        "leaf_count": 1,
        "current_terminal_leaf_count": 1 if complete else 0,
        "unavailable_terminal_leaf_count": 0,
        "unassessed_terminal_leaf_count": 0,
        "missing_knowledge_leaf_count": 0 if complete else 1,
        "outside_support_leaf_count": 0,
        "incompatible_leaf_count": 0,
        "budget_exhausted_leaf_count": 0,
        "invalid_input_leaf_count": 0,
        "execution_error_leaf_count": 0,
        "protection_burden": {"knowledge": "unknown", "count": None},
        "purification_burden": {"knowledge": "unknown", "count": None},
    }


def _source_record(
    role: str,
    smiles: str,
    *,
    complete: bool,
    steps: int,
    source_class: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "source_class": source_class,
        "family_template_admitted": False,
        "exact_l2_steps": steps if complete else 0,
        "value": _value(role, smiles, complete=complete, steps=steps),
    }


def _closed_c18(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path, Path]:
    if complete:
        return EXACT_C18_CONFIG, EXACT_C18_RESULT, EXACT_C18_ASSESSMENT

    with gzip.open(EXACT_C18_ASSESSMENT, "rt") as handle:
        assessment_payload = json.load(handle)
    assessment_payload["synthesis_value"] = _value(
        TARGET_ROLE, TARGET_SMILES, complete=False, steps=3
    )
    assessment = tmp_path / "assessment.json.gz"
    _write_gzip_json(assessment, assessment_payload)

    result_payload = json.loads(EXACT_C18_RESULT.read_text())
    result_payload["summary"]["route_complete"] = False
    result_payload["summary"]["terminal_availability"] = "unassessed"
    result_payload["adjudication"]["current_l3_procurement_closed"] = False
    result_payload["adjudication"]["route_complete"] = False
    result_payload["artifacts"][assessment.name]["sha256"] = sha256_file(assessment)
    result = tmp_path / "c18_result.json"
    _write_json(result, result_payload)
    return EXACT_C18_CONFIG, result, assessment


def _bound_config(
    tmp_path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    c18_complete: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = [
            _source_record(
                "oxoester_aldehyde_body_tail",
                "CCCCCCCCCCCCCCCC=O",
                complete=True,
                steps=2,
                source_class="independently_qualified_c16",
            ),
            _source_record(
                TARGET_ROLE,
                TARGET_SMILES,
                complete=False,
                steps=0,
                source_class="missing_exact_c18_route",
            ),
            _source_record(
                "amine_head",
                "CN",
                complete=True,
                steps=0,
                source_class="current_terminal_material",
            ),
        ]
    source = tmp_path / "final_components.json.gz"
    _write_gzip_json(
        source,
        {
            "schema_version": SOURCE_SCHEMA,
            "status": "frozen_final_component_ledger",
            "records": records,
        },
    )
    exact_config, result, assessment = _closed_c18(tmp_path, complete=c18_complete)
    outputs = {label: str(tmp_path / "outputs" / f"{label}.json") for label in OUTPUT_LABELS}
    outputs["r0_records"] += ".gz"
    outputs["r1_records"] += ".gz"
    config = tmp_path / "builder.json"
    _write_json(
        config,
        {
            "schema_version": "phase1_ugi3_route_registry_pair_builder_config.v1",
            "status": "frozen_inputs_bound_before_molecular_holdout_reveal",
            "protocol": {
                "path": str(PROTOCOL.relative_to(REPO)),
                "sha256": sha256_file(PROTOCOL),
            },
            "inputs": {
                "final_component_ledger": {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "schema_version": SOURCE_SCHEMA,
                },
                "exact_c18_config": {
                    "path": str(exact_config),
                    "sha256": sha256_file(exact_config),
                },
                "exact_c18_result": {
                    "path": str(result),
                    "sha256": sha256_file(result),
                },
                "exact_c18_assessment": {
                    "path": str(assessment),
                    "sha256": sha256_file(assessment),
                },
            },
            "outputs": outputs,
            "binding_policy": REQUIRED_BINDING_POLICY,
            "decision": {
                "inputs_bound": True,
                "registry_pair_materialized": False,
                "holdout_reveal_authorized": False,
            },
        },
    )
    return config


def _write_artifacts(config_path: Path, payloads: dict[str, bytes]) -> dict[str, Path]:
    config = json.loads(config_path.read_text())
    paths = {label: Path(path) for label, path in config["outputs"].items()}
    for label, payload in payloads.items():
        paths[label].parent.mkdir(parents=True, exist_ok=True)
        paths[label].write_bytes(payload)
    return paths


def test_checked_in_builder_is_bound_and_immutable_after_materialization() -> None:
    config = json.loads(UNBOUND_CONFIG.read_text())
    assert config["status"] == "frozen_inputs_bound_before_molecular_holdout_reveal"
    assert config["decision"] == {
        "inputs_bound": True,
        "registry_pair_materialized": False,
        "holdout_reveal_authorized": False,
    }
    for label, specification in config["inputs"].items():
        path = REPO / specification["path"]
        assert sha256_file(path) == specification["sha256"], label
    for path in config["outputs"].values():
        assert (REPO / path).is_file()
    with pytest.raises(
        Ugi3RouteRegistryPairContractError,
        match="sealed molecular sample exists before immutable registry binding",
    ):
        build_registry_pair(REPO, UNBOUND_CONFIG)


def test_builder_is_deterministic_and_preserves_every_non_c18_record(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    first = build_registry_pair(prereveal_registry_view.repo, config)
    second = build_registry_pair(prereveal_registry_view.repo, config)

    assert first == second
    r0 = _read_gzip_payload(first["r0_records"])["records"]
    r1 = _read_gzip_payload(first["r1_records"])["records"]
    changed = [index for index, pair in enumerate(zip(r0, r1, strict=True)) if pair[0] != pair[1]]
    target_index = next(
        index for index, record in enumerate(r0) if record["canonical_smiles"] == TARGET_SMILES
    )
    c16_index = next(
        index
        for index, record in enumerate(r0)
        if record["source_class"] == "independently_qualified_c16"
    )
    assert len(r0) == len(r1) == 3
    assert changed == [target_index]
    assert r0[c16_index] == r1[c16_index]
    assert r0[target_index]["route_complete"] is False
    assert r1[target_index]["route_complete"] is True

    diff = json.loads(first["registry_diff"])
    assert diff["changed_count"] == 1
    assert diff["added_count"] == 0
    assert diff["deleted_count"] == 0
    assert diff["non_target_mutation_count"] == 0
    assert diff["all_non_c18_records_canonical_json_equal"] is True


def test_builder_outputs_pass_the_independent_pair_contract(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    payloads = build_registry_pair(prereveal_registry_view.repo, config)
    paths = _write_artifacts(config, payloads)

    result = validate_binding(
        prereveal_registry_view.repo,
        prereveal_registry_view.protocol,
        paths["binding"],
    )
    assert result["holdout_reveal_authorized"] is True
    assert result["authorized_delta_count"] == 1
    assert result["registry_record_count"] == 3


def test_parent_hash_mismatch_fails_closed(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    payload = json.loads(config.read_text())
    payload["inputs"]["final_component_ledger"]["sha256"] = "0" * 64
    _write_json(config, payload)
    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="ledger hash changed"):
        build_registry_pair(prereveal_registry_view.repo, config)


def test_repinned_broader_c18_config_fails_reproduction(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    builder = json.loads(config.read_text())

    exact_config = json.loads(EXACT_C18_CONFIG.read_text())
    exact_config["limitations"]["homologue_promotion_authorized"] = True
    altered_config = tmp_path / "altered_exact_c18_config.json"
    _write_json(altered_config, exact_config)

    exact_result = json.loads(EXACT_C18_RESULT.read_text())
    exact_result["config_sha256"] = sha256_file(altered_config)
    altered_result = tmp_path / "altered_exact_c18_result.json"
    _write_json(altered_result, exact_result)

    builder["inputs"]["exact_c18_config"] = {
        "path": str(altered_config),
        "sha256": sha256_file(altered_config),
    }
    builder["inputs"]["exact_c18_result"] = {
        "path": str(altered_result),
        "sha256": sha256_file(altered_result),
    }
    _write_json(config, builder)

    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="failed reproduction"):
        build_registry_pair(prereveal_registry_view.repo, config)


def test_repinned_c18_assessment_bytes_fail_reproduction(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    builder = json.loads(config.read_text())

    with gzip.open(EXACT_C18_ASSESSMENT, "rt") as handle:
        assessment_payload = json.load(handle)
    assessment_payload["synthesis_value"]["protection_burden"] = {
        "knowledge": "known",
        "count": 0,
    }
    altered_assessment = tmp_path / "assessment.json.gz"
    _write_gzip_json(altered_assessment, assessment_payload)

    result_payload = json.loads(EXACT_C18_RESULT.read_text())
    result_payload["artifacts"][altered_assessment.name]["sha256"] = sha256_file(altered_assessment)
    altered_result = tmp_path / "altered_exact_c18_result.json"
    _write_json(altered_result, result_payload)

    builder["inputs"]["exact_c18_result"] = {
        "path": str(altered_result),
        "sha256": sha256_file(altered_result),
    }
    builder["inputs"]["exact_c18_assessment"] = {
        "path": str(altered_assessment),
        "sha256": sha256_file(altered_assessment),
    }
    _write_json(config, builder)

    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="does not reproduce"):
        build_registry_pair(prereveal_registry_view.repo, config)


def test_parent_with_c18_already_complete_is_rejected(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    records = [
        _source_record(
            TARGET_ROLE,
            TARGET_SMILES,
            complete=True,
            steps=3,
            source_class="premature_c18_closure",
        )
    ]
    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="already marks"):
        build_registry_pair(
            prereveal_registry_view.repo,
            _bound_config(tmp_path, records=records),
        )


def test_incomplete_owned_c18_assessment_is_rejected(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="not fully closed"):
        build_registry_pair(
            prereveal_registry_view.repo,
            _bound_config(tmp_path, c18_complete=False),
        )


def test_duplicate_exact_component_identity_is_rejected(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    target = _source_record(
        TARGET_ROLE,
        TARGET_SMILES,
        complete=False,
        steps=0,
        source_class="missing_exact_c18_route",
    )
    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="duplicate exact"):
        build_registry_pair(
            prereveal_registry_view.repo,
            _bound_config(tmp_path, records=[target, dict(target)]),
        )


def test_existing_output_blocks_immutable_rebuild(
    tmp_path: Path,
    prereveal_registry_view: Any,
) -> None:
    config = _bound_config(tmp_path)
    output = Path(json.loads(config.read_text())["outputs"]["registry_diff"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("already frozen")
    with pytest.raises(Ugi3RouteRegistryPairBuilderError, match="already exist"):
        build_registry_pair(prereveal_registry_view.repo, config)
