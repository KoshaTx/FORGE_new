from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table1 import render_gem_table1_final_evidence


def test_gem_table1_uses_three_pinned_final_evaluations(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_gem_table1_final_evidence(
        repo / "configs/reproduction/gem_table1_core_saturation_complete_v1.json",
        repo,
        tmp_path / "generated",
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["pending_columns"] == []
    assert len(result["sources"]) == 10
    rows = (tmp_path / "generated/shared_program_figure_rows.tex").read_text()
    assert r"Ugi & $96.4\pm2.1$ & $34.8\pm7.1$ & $93.8\pm4.2$ & 99.9/100.0" in rows
    assert r"Aza-Michael & $72.2\pm1.6$ & $19.2\pm4.1$ & $71.5\pm2.2$ & 100.0/100.0" in rows
    assert r"Reductive amination & $52.3\pm3.4$ & $37.0\pm2.2$ & $50.7\pm4.9$ & 86.6/100.0" in rows
    assert "TBD" not in rows
    macros = (tmp_path / "generated/gem_table1_macros.tex").read_text()
    assert r"\newcommand{\ForgeTableOneNullUgiDifferencePP}{61.6}" in macros
    assert r"\newcommand{\ForgeTableOneCyclicBLRangeLowPP}{-2.4}" in macros
