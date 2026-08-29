from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table5 import render_gem_table5_decoder_source_ablation


def test_gem_table5_renders_three_completed_seed0_rows(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = tmp_path / "decoder_source_ablation_rows.tex"
    result = render_gem_table5_decoder_source_ablation(
        repo / "configs/reproduction/gem_table5_decoder_source_ablation_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    rendered = rows.read_text()
    assert "Program-role" in rendered
    assert "Global" in rendered
    assert "95.21" in rendered
    assert "99.90" in rendered
    assert "95.12" in rendered
    assert "95.31" in rendered
    assert "78.35" in rendered
    assert "74.67" in rendered
    assert "86.95" in rendered
    assert "99.63" in rendered
    assert "86.62" in rendered
