"""Prepare a blinded, output-blind visual review for a Ugi development comparison."""

from __future__ import annotations

import argparse
import hashlib
import html
import random
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, stable_json, write_json
from forge.model.common_ugi_benchmark import CommonUgiAttempt, load_attempt_ledger

CONFIG_SCHEMA = "forge.ugi_development_visual_review_config.v1"
RESULT_SCHEMA = "forge.ugi_development_visual_review.v1"
SHEET_SCHEMA = "forge.ugi_development_blinded_review_sheet.v1"
KEY_SCHEMA = "forge.ugi_development_blinding_key.v1"

LEGACY_PREFERENCE_CRITERIA = (
    "tail_morphology_preference",
    "head_tail_balance_preference",
)
WHOLE_HEAD_PREFERENCE_CRITERIA = (
    "head_arrangement_preference",
    "overall_lipid_plausibility_preference",
)
PATHOLOGY_CRITERION = "unsupported_ring_or_heteroatom_pathology"
SUPPORTED_CRITERIA = {
    (*LEGACY_PREFERENCE_CRITERIA, PATHOLOGY_CRITERION),
    (*WHOLE_HEAD_PREFERENCE_CRITERIA, PATHOLOGY_CRITERION),
}


class UgiDevelopmentVisualReviewError(ValueError):
    """The pinned development comparison or visual-review contract is malformed."""


def _validate_config(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "scientific_question",
        "inputs",
        "arms",
        "baseline_arm",
        "treatment_arm",
        "expected_attempts",
        "attempt_indices",
        "selection_seed",
        "blinding_seed",
        "criteria",
        "decision_rule",
        "policy",
        "nonclaims",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiDevelopmentVisualReviewError("development visual-review fields changed")
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise UgiDevelopmentVisualReviewError("development visual-review schema changed")
    arms = value.get("arms")
    if not isinstance(arms, Mapping) or len(arms) != 2:
        raise UgiDevelopmentVisualReviewError("development visual review requires two arms")
    baseline = value.get("baseline_arm")
    treatment = value.get("treatment_arm")
    if baseline == treatment or {baseline, treatment} != set(arms):
        raise UgiDevelopmentVisualReviewError("development visual-review arm labels changed")
    attempts = value.get("expected_attempts")
    indices = value.get("attempt_indices")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or not isinstance(indices, list)
        or not indices
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or indices != sorted(set(indices))
        or indices[0] < 0
        or indices[-1] >= attempts
    ):
        raise UgiDevelopmentVisualReviewError("development visual-review indices are invalid")
    criteria = value.get("criteria")
    if not isinstance(criteria, list) or tuple(criteria) not in SUPPORTED_CRITERIA:
        raise UgiDevelopmentVisualReviewError("development visual-review criteria changed")
    if value.get("policy") != {
        "candidate_selection": False,
        "indices_selected_by_output_blind_rng_before_review": True,
        "include_invalid_or_failed_attempts": True,
        "independent_left_right_blinding_per_pair": True,
        "repair_or_retry": False,
        "render_complete_molecule_only": True,
    }:
        raise UgiDevelopmentVisualReviewError("development visual-review policy changed")
    return dict(value)


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise UgiDevelopmentVisualReviewError(f"unsafe comparison member: {name!r}")
    return path


def _load_arms(
    config: Mapping[str, Any], comparison_archive: Path
) -> dict[str, tuple[CommonUgiAttempt, ...]]:
    loaded: dict[str, tuple[CommonUgiAttempt, ...]] = {}
    with tempfile.TemporaryDirectory(prefix="forge-development-visual-") as temporary:
        root = Path(temporary)
        with tarfile.open(comparison_archive, mode="r:*") as archive:
            available = {member.name: member for member in archive.getmembers()}
            for arm_id, raw_arm in sorted(config["arms"].items()):
                if not isinstance(raw_arm, Mapping) or set(raw_arm) != {
                    "archive_member",
                    "method_id",
                    "seed",
                }:
                    raise UgiDevelopmentVisualReviewError("visual-review arm fields changed")
                member_name = str(raw_arm["archive_member"])
                member_path = _safe_member(member_name)
                member = available.get(member_name)
                if member is None or not member.isfile():
                    raise UgiDevelopmentVisualReviewError(f"comparison archive omits {member_name}")
                source = archive.extractfile(member)
                if source is None:
                    raise UgiDevelopmentVisualReviewError(
                        f"comparison member cannot be read: {member_name}"
                    )
                destination = root.joinpath(*member_path.parts)
                atomic_write(destination, source.read())
                loaded[str(arm_id)] = load_attempt_ledger(
                    destination,
                    expected_method=str(raw_arm["method_id"]),
                    expected_seed=int(raw_arm["seed"]),
                    expected_attempts=int(config["expected_attempts"]),
                )
    return loaded


def _slot_order(
    seed: int, attempt_index: int, baseline_arm: str, treatment_arm: str
) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{attempt_index}".encode()).digest()
    arms = (baseline_arm, treatment_arm)
    return arms if digest[0] % 2 == 0 else tuple(reversed(arms))


def _molecule_svg(attempt: CommonUgiAttempt) -> str:
    if attempt.status != "generated" or attempt.product_smiles is None:
        return "<div class='missing'>No generated molecule was returned.</div>"
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


def run_ugi_development_visual_review(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the packet only when the pinned six-metric panel uniformly improved."""

    config = _validate_config(
        read_json_object(
            config_path,
            error=UgiDevelopmentVisualReviewError,
            label="development visual-review config",
        )
    )
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {
        "comparison_archive",
        "realism_panel_result",
    }:
        raise UgiDevelopmentVisualReviewError("development visual-review input pins changed")
    inputs = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    panel = read_json_object(
        inputs["realism_panel_result"],
        error=UgiDevelopmentVisualReviewError,
        label="realism-panel result",
    )
    treatment = str(config["treatment_arm"])
    comparison = panel.get("panel_comparison", {}).get("comparisons", {}).get(treatment, {})
    if (
        panel.get("schema_version") != "forge.ugi_development_realism_comparison.v2"
        or panel.get("status") != "pass"
        or comparison.get("classification") != "uniformly_improved"
        or comparison.get("primary_metrics_improved") != 6
        or comparison.get("primary_metrics_total") != 6
    ):
        raise UgiDevelopmentVisualReviewError(
            "the pinned six-metric realism panel did not uniformly improve"
        )
    expected_indices = sorted(
        random.Random(int(config["selection_seed"])).sample(
            range(int(config["expected_attempts"])), len(config["attempt_indices"])
        )
    )
    if expected_indices != config["attempt_indices"]:
        raise UgiDevelopmentVisualReviewError("attempt indices do not match the output-blind draw")
    arms = _load_arms(config, inputs["comparison_archive"])
    output_dir.mkdir(parents=True, exist_ok=False)
    sections: list[str] = []
    grid_molecules: list[Chem.Mol | None] = []
    grid_legends: list[str] = []
    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    baseline = str(config["baseline_arm"])
    preference_criteria = tuple(str(value) for value in config["criteria"][:-1])
    pathology_criterion = str(config["criteria"][-1])
    for rank, attempt_index in enumerate(config["attempt_indices"], start=1):
        pair_id = f"P{rank:02d}"
        order = _slot_order(int(config["blinding_seed"]), int(attempt_index), baseline, treatment)
        slots = {"A": order[0], "B": order[1]}
        panels = []
        for slot in ("A", "B"):
            attempt = arms[slots[slot]][int(attempt_index)]
            panels.append(
                "<section class='molecule'><h3>"
                + slot
                + "</h3>"
                + _molecule_svg(attempt)
                + "</section>"
            )
            molecule = (
                Chem.MolFromSmiles(attempt.product_smiles)
                if attempt.status == "generated" and attempt.product_smiles is not None
                else None
            )
            grid_molecules.append(molecule)
            grid_legends.append(f"{pair_id} / {slot}")
        sections.append(
            f"<article><h2>{html.escape(pair_id)}</h2><div class='pair'>"
            + "".join(panels)
            + "</div></article>"
        )
        sheet_rows.append(
            {
                "pair_id": pair_id,
                **{criterion: None for criterion in preference_criteria},
                pathology_criterion: None,
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
            "<title>Blinded Ugi development morphology review</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;color:#202020}"
            "article{break-inside:avoid;border-top:1px solid #ccc;padding:12px 0 24px}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
            ".molecule{border:1px solid #ddd;background:#fff;padding:8px;text-align:center}"
            ".molecule svg{width:100%;height:auto}.missing{height:330px;display:flex;align-items:center;"
            "justify-content:center;color:#666}.rules{background:#f5f5f5;padding:14px}</style></head><body>",
            "<h1>Blinded Ugi development morphology review</h1>",
            "<div class='rules'><p>Indices were drawn by an output-blind RNG before this visual "
            "review. A and B are independently randomized. No molecule was filtered, repaired, "
            "retried or replaced.</p><p>For each pair, score only the two named preference fields "
            f"({html.escape(preference_criteria[0])} and "
            f"{html.escape(preference_criteria[1])}) and record whether either side contains an "
            "unsupported ring or heteroatom pathology. Use tie or unassessable when "
            "appropriate.</p></div>",
            *sections,
            "</body></html>",
        )
    )
    atomic_write(output_dir / "review_packet.html", document.encode("utf-8"))
    grid_svg = Draw.MolsToGridImage(
        grid_molecules, legends=grid_legends, molsPerRow=2, subImgSize=(700, 300), useSVG=True
    )
    atomic_write(output_dir / "review_packet.svg", str(grid_svg).encode("utf-8"))
    page_names = []
    for page_index, start in enumerate(range(0, len(grid_molecules), 12), start=1):
        page_name = f"review_packet_page_{page_index:02d}.svg"
        page_svg = Draw.MolsToGridImage(
            grid_molecules[start : start + 12],
            legends=grid_legends[start : start + 12],
            molsPerRow=2,
            subImgSize=(700, 300),
            useSVG=True,
        )
        atomic_write(output_dir / page_name, str(page_svg).encode("utf-8"))
        page_names.append(page_name)
    write_json(
        output_dir / "review_sheet.json",
        {
            "schema_version": SHEET_SCHEMA,
            "status": "awaiting_blinded_review",
            "reviewer_id": None,
            "allowed_preferences": ["A", "B", "tie", "unassessable"],
            "allowed_pathology_labels": ["A", "B", "both", "neither", "unassessable"],
            "preference_criteria": list(preference_criteria),
            "pathology_criterion": pathology_criterion,
            "decision_rule": dict(config["decision_rule"]),
            "reviews": sheet_rows,
        },
    )
    write_json(
        output_dir / "blinding_key.json",
        {
            "schema_version": KEY_SCHEMA,
            "status": "sealed_until_review_complete",
            "rows": key_rows,
        },
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "ready_for_blinded_review",
        # Arm identities are part of the frozen packet contract even though they are never
        # rendered into the blinded packet.  The adjudicator must not infer treatment status from
        # a mutable substring in an arm label.
        "baseline_arm": baseline,
        "treatment_arm": treatment,
        "preference_criteria": list(preference_criteria),
        "pathology_criterion": pathology_criterion,
        "pairs": len(sheet_rows),
        "review_packet_pages": len(page_names),
        "fixed_attempt_indices": list(config["attempt_indices"]),
        "fixed_attempt_indices_sha256": hashlib.sha256(
            stable_json(config["attempt_indices"]).encode("utf-8")
        ).hexdigest(),
        "blinded": True,
        "candidate_selection": False,
        "repair_or_retry": False,
        "inputs": {
            "config": pin_record(config_path, repo),
            "comparison_archive": artifact_record(
                inputs["comparison_archive"], logical_path="comparison_details.tar"
            ),
            "realism_panel_result": artifact_record(
                inputs["realism_panel_result"], logical_path="realism_panel/result.json"
            ),
        },
        "artifacts": {
            name: artifact_record(output_dir / name, logical_path=name)
            for name in (
                "review_packet.html",
                "review_packet.svg",
                "review_sheet.json",
                "blinding_key.json",
                *page_names,
            )
        },
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_ugi_development_visual_review(
        args.config.resolve(), args.repo.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()
