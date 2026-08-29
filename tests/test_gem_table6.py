from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table6 import render_gem_table6_exact_l1_counts


def test_gem_table6_renders_exact_final_model_counts(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = tmp_path / "production_seed_exact_counts_rows.tex"
    result = render_gem_table6_exact_l1_counts(
        repo / "configs/reproduction/gem_table1_core_saturation_final_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert len(result["rows"]) == 9
    assert [row["verified_exact_l1_count"] for row in result["rows"]] == [
        2922,
        3037,
        2924,
        2176,
        2203,
        2274,
        1561,
        1727,
        1535,
    ]
    assert {row["attempts"] for row in result["rows"]} == {3072}
    assert [row["replicate"] for row in result["rows"]] == [0, 1, 2] * 3
    rendered = rows.read_text()
    assert "2{,}922" in rendered
    assert "3{,}037" in rendered
    assert "1{,}535" in rendered
    assert rendered.count("3{,}072") == 9
