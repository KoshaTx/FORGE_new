from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.corpus.decomposition_precision_audit import (
    ANSWER_FIELDS,
    BLIND_FIELDS,
    ReviewCase,
    _assign_review_ids,
    _gzip_bytes,
    _stratified_round_robin,
    run_audit,
)
from forge.corpus.decomposition_precision_review import score_reviews, wilson_interval
from forge.corpus.r1_prime_audit import AuditError, sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/corpus/m0_05_decomposition_precision_audit.json"
RESULTS = REPO / "results/m0_05"


def _case(case_key: str, *, rare: bool = False, branching: bool = False) -> ReviewCase:
    return ReviewCase(
        case_key=case_key,
        sampling_frame="observed_corpus_decomposition",
        case_origin="test",
        control_type="",
        proposed_reaction_id="test_reaction",
        product_smiles="CC",
        components=("C", "C"),
        proposed_role_to_component=(("left", 0), ("right", 1)),
        proposed_reactive_atoms=(("left", (0,)), ("right", (0,))),
        expected_reaction_id="test_reaction",
        expected_role_to_component=(("left", 0), ("right", 1)),
        expected_components=("C", "C"),
        expected_reactive_atoms=(("left", (0,)), ("right", (0,))),
        source_structure_id=case_key,
        source_schemes=("test",),
        source_decomposition_ids=(),
        source_studies=("test",),
        known_synthesis_alignment="test",
        chemical_expectation="human_review_required",
        sampling_strata=(),
        anchor_distance=0.7 if rare else 0.2,
        component_frequencies=(("left", 1 if rare else 10), ("right", 1 if rare else 10)),
        handle_match_counts=(("left", 1), ("right", 1)),
        structural_flags=("branching",) if branching else (),
        mechanical_forward_reconstruction=True,
        proposed_role_assignment_mechanically_correct=True,
        proposed_reactive_atoms_mechanically_correct=True,
        target_matching_forward_outcomes=1,
        target_site_signatures=((("left", (0,)), ("right", (0,))),),
    )


def test_seeded_stratified_sampling_and_review_ids_are_deterministic() -> None:
    cases = (
        _case("case-a", rare=True),
        _case("case-b", branching=True),
        _case("case-c"),
        _case("case-d", rare=True, branching=True),
    )

    first = _assign_review_ids(_stratified_round_robin(cases, 3, 7), 11)
    second = _assign_review_ids(_stratified_round_robin(cases, 3, 7), 11)

    assert first == second
    assert len({case.review_id for case in first}) == 3
    assert any("rare_component" in case.sampling_strata for case in first)
    assert any("branching" in case.sampling_strata for case in first)
    assert _gzip_bytes(b"same") == _gzip_bytes(b"same")


@pytest.mark.needs_vendor
def test_frozen_review_packet_has_complete_blinding_and_provenance_contract() -> None:
    if not (RESULTS / "result.json").exists():
        pytest.skip("M0-05 review packet has not been generated")
    result = json.loads((RESULTS / "result.json").read_text())
    manifest = json.loads((RESULTS / "sampling_manifest.json").read_text())
    with (RESULTS / "review_packet.csv").open(newline="") as handle:
        blind_rows = list(csv.DictReader(handle))
    with gzip.open(RESULTS / "answer_key.csv.gz", "rt", newline="") as handle:
        answer_rows = list(csv.DictReader(handle))

    assert result["schema_version"] == "m0_05_decomposition_precision_audit.v1"
    assert result["status"] == "awaiting_human_chemist_review"
    assert result["human_review"]["precision_per_family"] is None
    assert result["human_review"]["specificity_per_control_type"] is None
    assert result["packet_summary"]["rows"] == 319
    assert manifest["packet_rows"] == 319
    assert len(blind_rows) == len(answer_rows) == 319
    assert set(blind_rows[0]) == set(BLIND_FIELDS)
    assert set(answer_rows[0]) == set(ANSWER_FIELDS)
    assert {row["review_id"] for row in blind_rows} == {row["review_id"] for row in answer_rows}
    assert all(
        not row[field]
        for row in blind_rows
        for field in (
            "reviewer_id",
            "plausible_final_assembly",
            "component_identities_and_roles_correct",
            "reactive_atoms_correct",
            "exact_forward_reconstruction",
            "confidence",
        )
    )
    forbidden_blind_fields = {
        "sampling_frame",
        "case_origin",
        "control_type",
        "source_structure_id",
        "chemical_expectation",
    }
    assert forbidden_blind_fields.isdisjoint(blind_rows[0])

    exact_blind_keys = {
        (
            row["proposed_reaction_family"],
            row["product_smiles"],
            row["proposed_roles_json"],
            row["proposed_components_json"],
            row["proposed_reactive_atoms_json"],
        )
        for row in blind_rows
    }
    assert len(exact_blind_keys) == len(blind_rows)

    controls = result["packet_summary"]["adversarial_controls"]
    assert set(controls) == {
        "role_swap",
        "cross_record_component_swap",
        "wrong_reaction_family",
        "non_ugi_as_ugi",
        "wrong_attachment_site",
        "degenerate_reconstructing",
    }
    assert set(controls.values()) == {8}
    families = result["packet_summary"]["observed_frame"]["families"]
    assert len(families) == 12
    assert families["iphos_amine_dioxaphospholane"]["sampled"] == 0
    for details in families.values():
        assert details["sampled"] == min(
            30,
            details["deduplicated_exact_decompositions"],
        )

    for name, record in result["artifacts"].items():
        artifact_path = REPO / record["path"]
        assert artifact_path.name == name
        assert sha256_file(artifact_path) == record["sha256"]
        assert artifact_path.stat().st_size == record["bytes"]
    for record in result["inputs"].values():
        path = Path(record["path"])
        if not path.is_absolute():
            path = REPO / path
        assert sha256_file(path) == record["sha256"]


@pytest.mark.needs_vendor
def test_rendered_packet_contains_every_review_id_and_inline_svg() -> None:
    if not (RESULTS / "review_packet.html.gz").exists():
        pytest.skip("M0-05 rendered review packet has not been generated")
    with (RESULTS / "review_packet.csv").open(newline="") as handle:
        review_ids = [row["review_id"] for row in csv.DictReader(handle)]
    with gzip.open(RESULTS / "review_packet.html.gz", "rt") as handle:
        rendered = handle.read()

    assert rendered.count("class='review-card'") == len(review_ids)
    assert rendered.count("<svg") >= 3 * len(review_ids)
    assert all(review_id in rendered for review_id in review_ids)
    assert "case_origin" not in rendered
    assert "control_type" not in rendered


def test_input_hash_mismatch_fails_before_packet_generation(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["expected_inputs"]["m0_04_result"]["sha256"] = "0" * 64
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(AuditError, match="input hash mismatch for m0_04_result"):
        run_audit(config_path, tmp_path / "output", REPO)

    assert not (tmp_path / "output").exists()


def _write_answer_key(path: Path) -> list[dict[str, str]]:
    rows = []
    cases = (
        ("positive-1", "", "ugi_3cr_agile"),
        ("positive-2", "", "ugi_3cr_agile"),
        ("role", "role_swap", "ugi_3cr_agile"),
        ("cross", "cross_record_component_swap", "ugi_3cr_agile"),
        ("family", "wrong_reaction_family", "ugi_3cr_agile"),
        ("non-ugi", "non_ugi_as_ugi", "amide_coupling_acid_amine"),
        ("site", "wrong_attachment_site", "ugi_3cr_agile"),
        ("degenerate", "degenerate_reconstructing", "reductive_amination_amine_aldehyde"),
    )
    for review_id, control_type, family in cases:
        row = {field: "" for field in ANSWER_FIELDS}
        row.update(
            {
                "review_id": review_id,
                "sampling_frame": (
                    "adversarial_control" if control_type else "observed_corpus_decomposition"
                ),
                "control_type": control_type,
                "expected_reaction_family": family,
            }
        )
        rows.append(row)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANSWER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_annotations(
    path: Path,
    answer_rows: list[dict[str, str]],
    reviewer_id: str,
    *,
    disagree_on_second_positive: bool = False,
) -> None:
    rows = []
    for answer in answer_rows:
        row = {field: "" for field in BLIND_FIELDS}
        row.update(
            {
                "review_id": answer["review_id"],
                "reviewer_id": reviewer_id,
                "plausible_final_assembly": "yes",
                "component_identities_and_roles_correct": "yes",
                "reactive_atoms_correct": "yes",
                "exact_forward_reconstruction": "yes",
                "plausible_alternative_decomposition": "no",
                "confidence": "4",
            }
        )
        control_type = answer["control_type"]
        if control_type == "role_swap":
            row["component_identities_and_roles_correct"] = "no"
        elif control_type == "cross_record_component_swap":
            row["exact_forward_reconstruction"] = "no"
        elif control_type in {"wrong_reaction_family", "non_ugi_as_ugi"}:
            row["plausible_final_assembly"] = "no"
        elif control_type == "wrong_attachment_site":
            row["reactive_atoms_correct"] = "no"
        elif control_type == "degenerate_reconstructing":
            row["plausible_final_assembly"] = "no"
        if disagree_on_second_positive and answer["review_id"] == "positive-2":
            row["plausible_final_assembly"] = "uncertain"
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_human_review_scoring_reports_precision_specificity_and_agreement(
    tmp_path: Path,
) -> None:
    answer_key = tmp_path / "answer.csv.gz"
    answer_rows = _write_answer_key(answer_key)
    first = tmp_path / "reviewer-1.csv"
    second = tmp_path / "reviewer-2.csv"
    _write_annotations(first, answer_rows, "chemist-1")
    _write_annotations(
        second,
        answer_rows,
        "chemist-2",
        disagree_on_second_positive=True,
    )

    result = score_reviews(
        answer_key,
        (first, second),
        tmp_path / "scored.json",
        "2026-07-29T20:00:00+00:00",
    )

    assert result["review_rows"] == 8
    assert set(result["per_reviewer"]) == {"chemist-1", "chemist-2"}
    assert (
        result["per_reviewer"]["chemist-1"]["precision_per_family"]["ugi_3cr_agile"]["rate"] == 1.0
    )
    assert result["consensus"]["precision_per_family"]["ugi_3cr_agile"]["successes"] == 1
    assert all(
        details["rate"] == 1.0
        for details in result["consensus"]["specificity_per_negative_control_type"].values()
    )
    assert result["reviewer_agreement"]["mean_exact_agreement"] < 1.0
    assert result["family_disposition"]["ugi_3cr_agile"] == "human_signoff_required"
    assert (
        sha256_file(tmp_path / "scored.json")
        == hashlib.sha256((tmp_path / "scored.json").read_bytes()).hexdigest()
    )


def test_review_scoring_rejects_duplicate_reviewer_identity(tmp_path: Path) -> None:
    answer_key = tmp_path / "answer.csv.gz"
    answer_rows = _write_answer_key(answer_key)
    first = tmp_path / "reviewer-1.csv"
    second = tmp_path / "reviewer-2.csv"
    _write_annotations(first, answer_rows, "chemist")
    _write_annotations(second, answer_rows, "chemist")

    with pytest.raises(AuditError, match="distinct reviewer IDs"):
        score_reviews(
            answer_key,
            (first, second),
            tmp_path / "scored.json",
            "2026-07-29T20:00:00+00:00",
        )


def test_wilson_interval_handles_empty_and_invalid_counts() -> None:
    assert wilson_interval(0, 0)["rate"] is None
    interval = wilson_interval(8, 10)
    assert interval["lower"] < interval["rate"] < interval["upper"]
    with pytest.raises(AuditError, match="0 <= successes <= total"):
        wilson_interval(2, 1)
