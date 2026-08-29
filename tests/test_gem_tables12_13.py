from __future__ import annotations

from pathlib import Path

from forge_paper.gem_tables12_13 import render_gem_tables12_and_13


def test_gem_tables12_and_13_render_all_pinned_common_ugi_rows(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_gem_tables12_and_13(
        repo / "configs/reproduction/natbiotech_v1_completed_evidence_v1.json",
        repo,
        tmp_path / "generated",
        result_path=tmp_path / "result.json",
    )

    seed_rows = (tmp_path / "generated/common_ugi_seed_rows.tex").read_text()
    decomposition_rows = (
        tmp_path / "generated/common_ugi_decomposition_rows.tex"
    ).read_text()
    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert len(result["sources"]) == 27
    assert seed_rows.count(r"\\") == 27
    assert decomposition_rows.count(r"\\") == 9
    assert "TBD" not in seed_rows
    assert "TBD" not in decomposition_rows
    assert "FORGE Transformer &\\cellcolor{forgerow} 20260825" in seed_rows
    assert "951.2" in seed_rows
    assert "$99.9\\pm0.0$" in decomposition_rows
    assert decomposition_rows.count("N/E") == 5
