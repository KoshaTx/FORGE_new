from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table4 import render_gem_table4_production_comparison


def test_gem_table4_uses_all_nine_matched_production_cells(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows_path = tmp_path / "production_comparison_transposed_rows.tex"
    result = render_gem_table4_production_comparison(
        repo / "configs/reproduction/gem_table1_core_saturation_complete_v1.json",
        repo,
        rows_path,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["arm_order"] == ["conditioned", "shared_null", "cyclic_program"]
    assert len(result["sources"]) == 10
    assert all(result["gates"].values())
    rows = rows_path.read_text()
    assert "TBD" not in rows
    assert "Ugi-only" not in rows
    assert rows.count("\\cellcolor{forgerow}") == 18
    assert (
        r"Exact-L1/attempt & \cellcolor{forgerow} $96.4\pm2.1$ & $34.8\pm7.1$"
        r" & $93.8\pm4.2$ & \cellcolor{forgerow} $72.2\pm1.6$"
    ) in rows
    assert len(rows.splitlines()) == 6
