from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pinned_artifact(metadata: dict[str, object]) -> None:
    path = REPO / str(metadata["path"])
    assert path.stat().st_size == metadata["bytes"]
    assert _sha256(path) == metadata["sha256"]


def test_m0_01_authoritative_index_is_current() -> None:
    result = json.loads((REPO / "results/m0_01/result.json").read_text())

    assert result["schema_version"] == "m0_01_ugi_variant_decision.v1"
    assert result["decision"]["variant"] == "ugi_3cr_agile"
    assert result["decision"]["reactant_count"] == 3
    assert result["decision"]["carboxylic_acid_reactant_component"] is False
    for metadata in result["inputs"].values():
        _assert_pinned_artifact(metadata)


def test_m0_02_authoritative_index_is_current() -> None:
    result = json.loads((REPO / "results/m0_02/result.json").read_text())

    assert result["schema_version"] == "m0_02_vendor_verification.v1"
    assert result["verification"]["assets_verified"] == 30
    _assert_pinned_artifact(result["inputs"]["vendor_manifest"])


def test_m0_07_authoritative_index_is_current() -> None:
    result = json.loads((REPO / "results/m0_07/result.json").read_text())

    assert result["schema_version"] == "m0_07_authoritative_index.v1"
    assert result["decision"]["guidance_action"] == "abstain"
    assert result["decision"]["any_biological_guidance_domain_authorized"] is False
    for metadata in result["authoritative_artifacts"].values():
        _assert_pinned_artifact(metadata)
