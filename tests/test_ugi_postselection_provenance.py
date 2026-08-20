from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.design.guidance.ugi_postselection_provenance import (
    ABSENT_SUBSTRATUM,
    CATALOG_ABSENT,
    EXPANDED_SUBSTRATUM,
    KNOWN_EXPANSION,
    ORIGINAL,
    ORIGINAL_SUBSTRATUM,
    OUTSIDE_ROUTE_TIER,
    PRODUCT_CATALOG_ABSENT,
    PRODUCT_KNOWN_EXPANSION,
    PRODUCT_ORIGINAL,
    TRANSFER_SUBSTRATUM,
    UgiPostselectionProvenanceError,
    _route_diagnostic,
    build_postselection_provenance_audit,
    classify_component_provenance,
    classify_product_provenance,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_postselection_provenance_audit_v1.json"


def _paths() -> tuple[Path, ...]:
    config = json.loads(CONFIG.read_text())
    inputs = config["inputs"]
    return (
        CONFIG,
        REPO / inputs["production_manifest"]["path"],
        REPO / inputs["selected_sample"]["path"],
        REPO / inputs["component_registry"]["path"],
        REPO / inputs["component_expansion_result"]["path"],
        REPO / inputs["route_readiness_result"]["path"],
        REPO / inputs["route_readiness_ledger"]["path"],
    )


def _rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as handle:
            return list(csv.DictReader(handle))


def _registry_row(*, current: str, source_classes: list[str]) -> dict[str, str]:
    return {
        "component_id": "component-test",
        "is_current_catalog": current,
        "source_classes_json": json.dumps(source_classes, separators=(",", ":")),
    }


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            _registry_row(current="true", source_classes=["current_phase1_catalog"]),
            (ORIGINAL, ORIGINAL_SUBSTRATUM),
        ),
        (
            _registry_row(current="false", source_classes=["cross_platform_hydrophobic_transfer"]),
            (KNOWN_EXPANSION, TRANSFER_SUBSTRATUM),
        ),
        (
            _registry_row(current="false", source_classes=["bounded_head_capability"]),
            (KNOWN_EXPANSION, EXPANDED_SUBSTRATUM),
        ),
        (None, (CATALOG_ABSENT, ABSENT_SUBSTRATUM)),
    ],
)
def test_component_provenance_uses_only_explicit_registry_state(
    row: dict[str, str] | None, expected: tuple[str, str]
) -> None:
    assert (
        classify_component_provenance(
            row, transfer_source_class="cross_platform_hydrophobic_transfer"
        )
        == expected
    )


def test_product_provenance_is_mutually_exclusive_and_prioritizes_catalog_absence() -> None:
    assert classify_product_provenance([ORIGINAL, ORIGINAL, ORIGINAL]) == PRODUCT_ORIGINAL
    assert (
        classify_product_provenance([ORIGINAL, KNOWN_EXPANSION, ORIGINAL])
        == PRODUCT_KNOWN_EXPANSION
    )
    assert (
        classify_product_provenance([KNOWN_EXPANSION, CATALOG_ABSENT, ORIGINAL])
        == PRODUCT_CATALOG_ABSENT
    )


def test_catalog_absence_is_missing_knowledge_not_infeasibility() -> None:
    diagnostic = _route_diagnostic(None)

    assert diagnostic == {
        "tier": OUTSIDE_ROUTE_TIER,
        "category": "missing_route_knowledge",
        "complete": "false",
        "gap": "route_evidence_not_assessed",
        "basis": "component_absent_from_frozen_424_catalog_no_route_assessment",
    }


def test_full_frozen_audit_matches_expected_counts_and_is_byte_deterministic() -> None:
    first = build_postselection_provenance_audit(*_paths())
    second = build_postselection_provenance_audit(*_paths())

    assert first == second
    result, component_payload, product_payload = first
    assert result["summary"]["component_reconstructed_products"] == 1007
    assert result["summary"]["component_occurrences"] == 3021
    assert result["summary"]["products_by_structural_provenance"] == {
        PRODUCT_ORIGINAL: 30,
        PRODUCT_KNOWN_EXPANSION: 130,
        PRODUCT_CATALOG_ABSENT: 847,
    }
    assert result["scope"]["outside_catalog_components_are_infeasible"] is False
    components = _rows(component_payload)
    products = _rows(product_payload)
    assert len(components) == 3021
    assert len(products) == 1007
    outside = [row for row in components if row["structural_provenance_stratum"] == CATALOG_ABSENT]
    assert len(outside) == 1378
    assert {row["static_route_readiness_tier"] for row in outside} == {OUTSIDE_ROUTE_TIER}
    assert {row["static_route_evidence_category"] for row in outside} == {"missing_route_knowledge"}
    assert {row["static_route_complete"] for row in outside} == {"false"}


def test_frozen_config_hashes_match_every_input() -> None:
    config = json.loads(CONFIG.read_text())
    for specification in config["inputs"].values():
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]


def test_malformed_or_ambiguous_registry_provenance_fails_closed() -> None:
    with pytest.raises(UgiPostselectionProvenanceError, match="current-catalog flag"):
        classify_component_provenance(
            _registry_row(current="unknown", source_classes=[]),
            transfer_source_class="cross_platform_hydrophobic_transfer",
        )
    row = _registry_row(current="false", source_classes=[])
    row["source_classes_json"] = "not-json"
    with pytest.raises(UgiPostselectionProvenanceError, match="invalid JSON"):
        classify_component_provenance(
            row, transfer_source_class="cross_platform_hydrophobic_transfer"
        )
