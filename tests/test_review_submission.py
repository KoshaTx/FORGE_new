"""Small offline fixtures for reviewer integrity and honest missing-artifact reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from review_submission import (
    GENERATED,
    MANIFEST,
    ROOT,
    audit,
    check_conditioned_counts,
    inspect_pin,
)


def _pin(repo: Path, name: str, content: str) -> dict[str, str]:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {"path": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}


@pytest.fixture
def review_repo(tmp_path: Path) -> Path:
    # Copy only the small published tables; no model/data/runtime dependency.
    table_pins = []
    for name in ("production_seed_exact_counts_rows.tex", "shared_program_figure_rows.tex"):
        table_pins.append(
            _pin(tmp_path, str(GENERATED / name), (ROOT / GENERATED / name).read_text())
        )
    manifest = {
        "schema_version": "forge.submission_review.v1",
        "review_assets": [_pin(tmp_path, "paper.pdf", "synthetic paper bytes"), *table_pins],
        "historical_inputs": [{"path": "missing/checkpoint.tar", "sha256": "a" * 64}],
        "historical_inventory_scope": "synthetic direct input",
        "limitations": ["not a full reproduction"],
    }
    path = tmp_path / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return tmp_path


def test_default_check_discloses_absent_weights_without_claiming_reproduction(review_repo: Path):
    report = audit(review_repo)
    assert report["status"] == "pass"
    assert report["historical_inputs"][0]["status"] == "missing"
    assert report["historical_inventory_complete"] is False
    assert report["full_reproduction_verified"] is False
    ugi = report["conditioned_arithmetic"][0]
    assert ugi["mean_percent"] == pytest.approx(96.38671875)
    assert round(ugi["sample_sd_percent"], 1) == 2.1
    assert audit(review_repo, require_run_artifacts=True)["status"] == "fail"


def test_corrupt_paper_fails(review_repo: Path):
    (review_repo / "paper.pdf").write_text("different paper")
    report = audit(review_repo)
    assert report["status"] == "fail"
    assert report["review_assets"][0]["status"] == "mismatch"


def test_missing_required_asset_fails(review_repo: Path):
    (review_repo / "paper.pdf").unlink()
    assert audit(review_repo)["status"] == "fail"


def test_changed_optional_historical_artifact_is_not_treated_as_merely_missing(review_repo: Path):
    _pin(review_repo, "missing/checkpoint.tar", "incorrect model")
    report = audit(review_repo)
    assert report["status"] == "fail"
    assert report["historical_inputs"][0]["status"] == "mismatch"


def test_complete_inventory_does_not_claim_a_model_rerun(review_repo: Path):
    path = review_repo / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["historical_inputs"] = [_pin(review_repo, "checkpoint.tar", "fixture model")]
    path.write_text(json.dumps(manifest))
    report = audit(review_repo, require_run_artifacts=True)
    assert report["status"] == "pass"
    assert report["historical_inventory_complete"] is True
    assert report["full_reproduction_verified"] is False


@pytest.mark.parametrize(
    "old,new,reason",
    [
        ("3{,}072", "3{,}073", "denominator"),
        ("Ugi & 1", "Ugi & 0", "duplicate"),
        ("95.12", "95.11", "percentage"),
    ],
)
def test_seed_table_contract_is_enforced(review_repo: Path, old: str, new: str, reason: str):
    path = review_repo / GENERATED / "production_seed_exact_counts_rows.tex"
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=reason):
        check_conditioned_counts(review_repo)


def test_population_sd_cannot_replace_sample_sd(review_repo: Path):
    path = review_repo / GENERATED / "shared_program_figure_rows.tex"
    path.write_text(path.read_text().replace(r"96.4\pm2.1", r"96.4\pm1.7"))
    with pytest.raises(ValueError, match="sample SD"):
        check_conditioned_counts(review_repo)


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_pin_paths_must_remain_in_repo(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="unsafe"):
        inspect_pin(tmp_path, {"path": name, "sha256": "a" * 64})


def test_symlink_cannot_escape_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside bytes")
    (repo / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        inspect_pin(repo, {"path": "link", "sha256": "a" * 64})


def test_checked_in_review_assets_and_arithmetic():
    report = audit(ROOT)
    assert report["status"] == "pass", report["errors"]
    assert len(report["conditioned_arithmetic"]) == 3
