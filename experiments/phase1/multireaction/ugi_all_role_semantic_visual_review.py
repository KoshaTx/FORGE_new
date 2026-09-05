"""Prepare a blinded, nonselecting morphology review for the all-role Ugi experiment.

The packet uses attempt indices frozen before the production molecules exist.  Every selected
attempt is displayed, including failures, and the two method identities are independently swapped
within each pair.  The blinding key is emitted as a separate artifact and is not included in the
review packet.
"""

from __future__ import annotations

import hashlib
import html
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from forge.core.hashing import artifact_record, pin_record
from forge.core.io import atomic_write, read_json_object, stable_json, write_json
from forge.model.common_ugi_benchmark import CommonUgiAttempt, load_attempt_ledger

CONFIG_SCHEMA = "forge.ugi_all_role_semantic_visual_review_config.v1"
RESULT_SCHEMA = "forge.ugi_all_role_semantic_visual_review.v1"
REVIEW_SHEET_SCHEMA = "forge.ugi_all_role_semantic_blinded_review_sheet.v1"
BLINDING_KEY_SCHEMA = "forge.ugi_all_role_semantic_blinding_key.v1"
FROZEN_ATTEMPT_INDICES = (
    23,
    329,
    333,
    467,
    543,
    580,
    743,
    762,
    813,
    858,
    982,
    1004,
    1086,
    1321,
    1369,
    1655,
    1688,
    1691,
    1754,
    1762,
    1880,
    2012,
    2469,
    2840,
)

ORIGINAL_ARMS = {
    "amine_semantic": {
        "archive_member": "amine_semantic/assessment/attempts.jsonl.gz",
        "method_id": "forge_seed0_amine_semantic_joint_support_ugi_program",
        "seed": 0,
    },
    "all_role_semantic": {
        "archive_member": "all_role_semantic/assessment/attempts.jsonl.gz",
        "method_id": "forge_seed0_all_role_semantic_joint_support_ugi_program",
        "seed": 0,
    },
}

COMPLETE_SEMANTIC_ARMS = {
    "amine_semantic": ORIGINAL_ARMS["amine_semantic"],
    "all_role_semantic": {
        "archive_member": "all_role_semantic/assessment/attempts.jsonl.gz",
        "method_id": "forge_seed0_mog_substitution_semantic_joint_support_ugi_program",
        "seed": 0,
    },
}


class UgiAllRoleSemanticVisualReviewError(ValueError):
    """The blinded visual-review contract or one of its dependencies changed."""


def _validate_config(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "scientific_question",
        "inputs",
        "arms",
        "expected_attempts",
        "attempt_indices",
        "blinding_seed",
        "criteria",
        "decision_rule",
        "policy",
        "nonclaims",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiAllRoleSemanticVisualReviewError("visual-review config fields changed")
    if value.get("schema_version") != CONFIG_SCHEMA or value.get("inputs") != {}:
        raise UgiAllRoleSemanticVisualReviewError("visual-review config schema changed")
    if value.get("expected_attempts") != 3072:
        raise UgiAllRoleSemanticVisualReviewError("visual-review attempt denominator changed")
    if value.get("arms") not in (ORIGINAL_ARMS, COMPLETE_SEMANTIC_ARMS):
        raise UgiAllRoleSemanticVisualReviewError("visual-review arms changed")
    indices = value.get("attempt_indices")
    if not isinstance(indices, list) or tuple(indices) != FROZEN_ATTEMPT_INDICES:
        raise UgiAllRoleSemanticVisualReviewError("visual-review fixed indices changed")
    if value.get("blinding_seed") != 2026090106:
        raise UgiAllRoleSemanticVisualReviewError("visual-review blinding seed changed")
    if value.get("criteria") != [
        "tail_morphology_preference",
        "head_tail_balance_preference",
        "unsupported_ring_or_heteroatom_pathology",
    ]:
        raise UgiAllRoleSemanticVisualReviewError("visual-review criteria changed")
    if value.get("decision_rule") != {
        "maximum_treatment_pathology_pairs": 2,
        "minimum_assessable_pairs_per_criterion": 20,
        "minimum_net_treatment_preferences": 4,
        "require_treatment_pathologies_no_more_than_baseline": True,
        "require_treatment_preference_majority": True,
    }:
        raise UgiAllRoleSemanticVisualReviewError("visual-review decision rule changed")
    if value.get("policy") != {
        "candidate_selection": False,
        "fixed_indices_selected_before_generation": True,
        "include_invalid_or_failed_attempts": True,
        "independent_left_right_blinding_per_pair": True,
        "repair_or_retry": False,
        "render_complete_molecule_only": True,
    }:
        raise UgiAllRoleSemanticVisualReviewError("visual-review policy changed")
    return dict(value)


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise UgiAllRoleSemanticVisualReviewError(f"unsafe comparison member: {name!r}")
    return path


def _load_arms(
    config: Mapping[str, Any], comparison_archive: Path
) -> dict[str, tuple[CommonUgiAttempt, ...]]:
    loaded: dict[str, tuple[CommonUgiAttempt, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="forge-all-role-visual-") as temporary:
        root = Path(temporary)
        with tarfile.open(comparison_archive, mode="r:*") as archive:
            available = {member.name: member for member in archive.getmembers()}
            for arm_id, arm in sorted(config["arms"].items()):
                member_name = str(arm["archive_member"])
                member_path = _safe_member(member_name)
                member = available.get(member_name)
                if member is None or not member.isfile():
                    raise UgiAllRoleSemanticVisualReviewError(
                        f"comparison archive omits {member_name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise UgiAllRoleSemanticVisualReviewError(
                        f"comparison member cannot be read: {member_name}"
                    )
                destination = root.joinpath(*member_path.parts)
                atomic_write(destination, source.read())
                loaded[arm_id] = load_attempt_ledger(
                    destination,
                    expected_method=str(arm["method_id"]),
                    expected_seed=int(arm["seed"]),
                    expected_attempts=int(config["expected_attempts"]),
                )
    return loaded


def _slot_order(seed: int, attempt_index: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{attempt_index}".encode()).digest()
    arms = ("amine_semantic", "all_role_semantic")
    return arms if digest[0] % 2 == 0 else tuple(reversed(arms))


def _molecule_svg(attempt: CommonUgiAttempt) -> str:
    if attempt.status != "generated" or attempt.product_smiles is None:
        return (
            "<div class='missing'>No generated molecule was returned for this frozen attempt."
            "</div>"
        )
    molecule = Chem.MolFromSmiles(attempt.product_smiles)
    if molecule is None:
        return "<div class='missing'>The frozen output could not be rendered.</div>"
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(760, 360)
    options = drawer.drawOptions()
    options.padding = 0.04
    options.bondLineWidth = 2.2
    options.fixedFontSize = 18
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _review_html(
    config: Mapping[str, Any],
    arms: Mapping[str, tuple[CommonUgiAttempt, ...]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[str] = []
    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for rank, attempt_index in enumerate(config["attempt_indices"], start=1):
        pair_id = f"P{rank:02d}"
        order = _slot_order(int(config["blinding_seed"]), int(attempt_index))
        slots = {"A": order[0], "B": order[1]}
        panels: list[str] = []
        for slot in ("A", "B"):
            attempt = arms[slots[slot]][int(attempt_index)]
            panels.append(
                "<section class='molecule'><h3>"
                + slot
                + "</h3>"
                + _molecule_svg(attempt)
                + "</section>"
            )
        sections.append(
            f"<article><h2>{html.escape(pair_id)}</h2><div class='pair'>"
            + "".join(panels)
            + "</div></article>"
        )
        sheet_rows.append(
            {
                "pair_id": pair_id,
                "tail_morphology_preference": None,
                "head_tail_balance_preference": None,
                "unsupported_ring_or_heteroatom_pathology": None,
                "reviewer_note": None,
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "attempt_index": int(attempt_index),
                "slot_A_arm": slots["A"],
                "slot_B_arm": slots["B"],
            }
        )
    document = "\n".join(
        (
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Blinded complete-lipid morphology review</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;color:#202020}"
            "article{break-inside:avoid;border-top:1px solid #ccc;padding:12px 0 24px}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
            ".molecule{border:1px solid #ddd;background:#fff;padding:8px;text-align:center}"
            ".molecule svg{width:100%;height:auto}.missing{height:330px;display:flex;align-items:center;"
            "justify-content:center;color:#666}.rules{background:#f5f5f5;padding:14px}</style></head><body>",
            "<h1>Blinded complete-lipid morphology review</h1>",
            "<div class='rules'><p>Every pair is a frozen generation-attempt index selected before "
            "the molecules existed. A and B are independently randomized within each pair. No "
            "molecule was filtered, repaired, retried or replaced.</p><p>For each pair, record which "
            "side has more plausible tail morphology, which has better head-tail balance, and "
            "whether either side contains a recurrent unsupported ring or heteroatom pathology. "
            "Use tie or unassessable when appropriate.</p></div>",
            *sections,
            "</body></html>",
        )
    )
    return document, sheet_rows, key_rows


def run_ugi_all_role_semantic_visual_review(
    config_path: Path,
    repo: Path,
    comparison_result_path: Path,
    comparison_archive_path: Path,
    adjudication_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the blinded packet only after the quantitative gate passes."""

    config = _validate_config(
        read_json_object(
            config_path,
            error=UgiAllRoleSemanticVisualReviewError,
            label="all-role visual-review config",
        )
    )
    comparison = read_json_object(
        comparison_result_path,
        error=UgiAllRoleSemanticVisualReviewError,
        label="all-role comparison result",
    )
    adjudication = read_json_object(
        adjudication_result_path,
        error=UgiAllRoleSemanticVisualReviewError,
        label="all-role adjudication result",
    )
    if (
        comparison.get("schema_version")
        != "forge.ugi_all_role_semantic_program_comparison_result.v1"
        or comparison.get("status") != "complete"
        or comparison.get("profile") != "full"
        or comparison.get("programs_per_method") != int(config["expected_attempts"])
    ):
        raise UgiAllRoleSemanticVisualReviewError("full comparison is inadmissible")
    if (
        adjudication.get("schema_version") != "forge.ugi_all_role_semantic_adjudication.v1"
        or adjudication.get("status") != "complete"
    ):
        raise UgiAllRoleSemanticVisualReviewError("quantitative adjudication is inadmissible")
    output_dir.mkdir(parents=True, exist_ok=False)
    quantitative_pass = (
        adjudication.get("quantitative_decision") == "pass_seed0_quantitative_gate"
        and adjudication.get("promotion_decision") == "eligible_for_blinded_visual_review"
    )
    if quantitative_pass:
        arms = _load_arms(config, comparison_archive_path)
        review_html, review_rows, key_rows = _review_html(config, arms)
        atomic_write(output_dir / "review_packet.html", review_html.encode("utf-8"))
        write_json(
            output_dir / "review_sheet.json",
            {
                "schema_version": REVIEW_SHEET_SCHEMA,
                "status": "awaiting_blinded_review",
                "reviewer_id": None,
                "allowed_preferences": ["A", "B", "tie", "unassessable"],
                "allowed_pathology_labels": ["A", "B", "both", "neither", "unassessable"],
                "decision_rule": dict(config["decision_rule"]),
                "reviews": review_rows,
            },
        )
        write_json(
            output_dir / "blinding_key.json",
            {
                "schema_version": BLINDING_KEY_SCHEMA,
                "status": "sealed_until_review_complete",
                "rows": key_rows,
            },
        )
        atomic_write(
            output_dir / "instructions.txt",
            (
                b"Open review_packet.html without opening blinding_key.json. Complete every field "
                b"in review_sheet.json using only A, B, tie/unassessable, or the allowed pathology "
                b"labels. Do not replace any frozen pair. Unblind only after the sheet is complete.\n"
            ),
        )
        status = "ready_for_blinded_review"
    else:
        atomic_write(
            output_dir / "review_packet.html",
            b"<!doctype html><html><body><h1>Visual review not reached</h1><p>The frozen quantitative gate did not pass.</p></body></html>",
        )
        write_json(
            output_dir / "review_sheet.json",
            {
                "schema_version": REVIEW_SHEET_SCHEMA,
                "status": "not_reached",
                "reviewer_id": None,
                "decision_rule": dict(config["decision_rule"]),
                "reviews": [],
            },
        )
        write_json(
            output_dir / "blinding_key.json",
            {"schema_version": BLINDING_KEY_SCHEMA, "status": "not_reached", "rows": []},
        )
        atomic_write(
            output_dir / "instructions.txt", b"The quantitative promotion gate did not pass.\n"
        )
        status = "not_reached"
    indices_sha = hashlib.sha256(stable_json(config["attempt_indices"]).encode("utf-8")).hexdigest()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "quantitative_gate_passed": quantitative_pass,
        "fixed_attempt_indices": list(config["attempt_indices"]),
        "fixed_attempt_indices_sha256": indices_sha,
        "pairs": len(config["attempt_indices"]) if quantitative_pass else 0,
        "blinded": True,
        "candidate_selection": False,
        "invalid_or_failed_attempts_retained": True,
        "repair_or_retry": False,
        "inputs": {
            "config": pin_record(config_path, repo),
            "comparison_result": artifact_record(
                comparison_result_path, logical_path="compare_full/result.json"
            ),
            "comparison_archive": artifact_record(
                comparison_archive_path, logical_path="compare_full/comparison_details.tar"
            ),
            "adjudication_result": artifact_record(
                adjudication_result_path, logical_path="adjudicate/result.json"
            ),
        },
        "criteria": list(config["criteria"]),
        "decision_rule": dict(config["decision_rule"]),
        "policy": dict(config["policy"]),
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "BLINDING_KEY_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "REVIEW_SHEET_SCHEMA",
    "UgiAllRoleSemanticVisualReviewError",
    "run_ugi_all_role_semantic_visual_review",
]
