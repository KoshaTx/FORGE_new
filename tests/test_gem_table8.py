from __future__ import annotations

import math
from pathlib import Path

from forge_paper.gem_table8 import render_gem_table8_architecture_ablations


def test_gem_table8_renders_all_matched_three_seed_arms(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = tmp_path / "architecture_ablation_rows.tex"
    result = render_gem_table8_architecture_ablations(
        repo / "configs/reproduction/gem_table8_architecture_ablations_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert len(result["sources"]["training_results"]) == 3
    assert len(result["sources"]["evaluation_results"]) == 3
    assert result["scope_note"].startswith("The full_transformer row is the matched")
    assert math.isclose(
        result["summaries"]["full_transformer"]["ugi_l1_per_1000"]["mean"],
        (792.3177083333334 + 955.4036458333334 + 822.9166666666666) / 3,
    )
    rendered = rows.read_text()
    assert "TBD" not in rendered
    assert rendered.count("\\cellcolor{forgerow}") == 8
    assert rendered.count("\\midrule") == 2
    assert "Input-only program" in rendered
    assert "FACT-matched" in rendered
    assert "FACT-generous" in rendered
    assert "FORGE Transformer" not in rendered
    assert "\\cellcolor{forgerow}{FORGE}" in rendered
    assert "\\pm" in rendered
    assert "$856.9\\pm86.7$" in rendered
    assert "$767.7\\pm35.9$" in rendered
    assert "$318.8\\pm158.3$" in rendered
    assert "$56.9\\pm24.4$" in rendered
    assert "$455.0\\pm191.9$" in rendered
