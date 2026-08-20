from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from experiments.phase1.hela_potency.preview import (
    CARD_WARNING,
    UgiHelaDiagnosticPreviewError,
    _pdf_bytes,
    eligible_candidates,
    select_panel,
)


def _row(
    index: int,
    smiles: str,
    roles: list[str],
    mean: float,
    *,
    bin_name: str = "interpolative",
) -> dict[str, str]:
    return {
        "sample_index": str(index),
        "product_id": f"p{index}",
        "product_smiles": smiles,
        "exact_unseen_roles_json": json.dumps(roles, separators=(",", ":")),
        "overall_distribution_bin": bin_name,
        "ensemble_mean_descriptive_only": str(mean),
        "ensemble_standard_deviation": "0.2",
    }


def test_eligibility_requires_interpolative_supported_role_pattern_and_deduplicates() -> None:
    patterns = {
        ("amine",): ("amine_only", "held_head_5fold"),
        ("aldehyde", "isocyanide"): (
            "aldehyde_isocyanide",
            "held_aldehyde_isocyanide_pair_5fold",
        ),
    }
    q90 = {
        "held_head_5fold": 4.0,
        "held_aldehyde_isocyanide_pair_5fold": 6.0,
    }
    candidates, counts = eligible_candidates(
        [
            _row(0, "CCCCN", ["amine"], 7.0),
            _row(1, "NCCCC", ["amine"], 6.5),
            _row(2, "CCCCCC", ["aldehyde", "isocyanide"], 8.0),
            _row(3, "CCCCCCC", ["aldehyde"], 9.0),
            _row(4, "CCCCCCCC", ["amine"], 9.0, bin_name="boundary"),
        ],
        patterns=patterns,
        q90_by_scheme=q90,
    )
    assert len(candidates) == 2
    assert candidates[0].canonical_smiles == "CCCCN"
    assert candidates[0].lcb90 == pytest.approx(3.0)
    assert counts["all_records"] == 5
    assert counts["supported_interpolative_records"] == 3
    assert counts["supported_interpolative_unique_products"] == 2
    assert counts["supported_interpolative_duplicate_records"] == 1


def test_panel_pairs_diverse_leads_with_lower_same_pattern_controls() -> None:
    patterns = {
        ("amine",): ("amine_only", "held_head_5fold"),
        ("aldehyde", "isocyanide"): (
            "aldehyde_isocyanide",
            "held_aldehyde_isocyanide_pair_5fold",
        ),
    }
    q90 = {
        "held_head_5fold": 4.0,
        "held_aldehyde_isocyanide_pair_5fold": 6.0,
    }
    candidates, _ = eligible_candidates(
        [
            _row(0, "CCCCCCCCN", ["amine"], 8.0),
            _row(1, "CCCCCCCN", ["amine"], 6.0),
            _row(2, "CCCCCCCCCC", ["aldehyde", "isocyanide"], 9.0),
            _row(3, "CCCCCCCCC", ["aldehyde", "isocyanide"], 7.0),
            _row(4, "c1ccccc1CC", ["aldehyde", "isocyanide"], 8.5),
            _row(5, "c1ccccc1C", ["aldehyde", "isocyanide"], 6.5),
        ],
        patterns=patterns,
        q90_by_scheme=q90,
    )
    panel = select_panel(candidates, top_count=2, cluster_distance_threshold=0.2)
    assert len(panel) == 4
    assert [row["selection_class"] for row in panel] == [
        "top_lcb_diverse",
        "matched_lower_lcb_control",
        "top_lcb_diverse",
        "matched_lower_lcb_control",
    ]
    for lead, control in zip(panel[::2], panel[1::2], strict=True):
        assert lead["supported_pattern"] == control["supported_pattern"]
        assert lead["descriptive_lcb90"] > control["descriptive_lcb90"]
        assert lead["card_warning"] == CARD_WARNING
        assert control["biological_guidance_authorized"] is False
        assert control["synthesis_assessed"] is False


def test_panel_fails_when_no_lower_same_pattern_control_exists() -> None:
    patterns = {("amine",): ("amine_only", "held_head_5fold")}
    candidates, _ = eligible_candidates(
        [_row(0, "CCCCN", ["amine"], 7.0)],
        patterns=patterns,
        q90_by_scheme={"held_head_5fold": 4.0},
    )
    with pytest.raises(UgiHelaDiagnosticPreviewError, match="no lower-LCB"):
        select_panel(candidates, top_count=1, cluster_distance_threshold=0.2)


def test_pdf_render_is_deterministic() -> None:
    png = io.BytesIO()
    Image.new("RGB", (8, 5), (240, 241, 242)).save(png, format="PNG")
    first = _pdf_bytes(png.getvalue())
    second = _pdf_bytes(png.getvalue())
    assert first == second
    assert first.startswith(b"%PDF-1.4")
    assert first.endswith(b"%%EOF\n")
