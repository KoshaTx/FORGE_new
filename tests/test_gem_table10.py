from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table10 import render_gem_table10_route_evidence


def test_gem_table10_uses_final_common_ledgers_and_method_blind_evidence(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_gem_table10_route_evidence(
        repo / "configs/reproduction/gem_table10_route_evidence_core_saturation_v2.json",
        repo,
        tmp_path / "generated",
        result_path=tmp_path / "result.json",
    )

    rows = (tmp_path / "generated/route_evidence_completed_rows.tex").read_text()
    macros = (tmp_path / "generated/gem_table10_macros.tex").read_text()
    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert result["route_union_components"] == 11021
    assert result["route_evidence_dispositions"]["complete"] == 30
    assert len(result["sources"]) == 30
    assert rows.count(r"\\") == 8
    assert "TBD" not in rows
    assert r"$963.9\pm21.4$" in rows
    assert r"$0.7\pm0.3$" in rows
    assert r"$999.3\pm0.3$" in rows
    assert r"\cellcolor{forgerow}FORGE Transformer" in rows
    assert r"\renewcommand{\ForgeCommonRouteUnionComponents}{11{,}021}" in macros
    assert r"\renewcommand{\ForgeCommonRouteEvidenceCompleteComponents}{30}" in macros
    assert r"\renewcommand{\ForgeCommonRouteForgeExactPerThousand}{$963.9\pm21.4$}" in macros
