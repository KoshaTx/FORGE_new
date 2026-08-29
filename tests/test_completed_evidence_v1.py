from __future__ import annotations

from pathlib import Path

from forge_paper.completed_evidence_v1 import render_completed_evidence_v1


def test_completed_evidence_renderer_uses_only_pinned_nonselecting_results(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = repo / "configs/reproduction/natbiotech_v1_completed_evidence_v1.json"

    result_path = tmp_path / "result.json"
    generated = tmp_path / "generated"
    result = render_completed_evidence_v1(config, repo, generated, result_path=result_path)

    assert result["status"] == "complete_for_currently_available_evidence"
    assert result["candidate_selection"] is False
    assert result["held_reaction_family_included"] is False
    assert result["missing_common_methods"] == []
    assert result["method_blind_route_union_included"] is True
    assert result_path.is_file()
    assert "FORGE Transformer" in (generated / "common_ugi_completed_rows.tex").read_text()
    assert "RGFN" not in (generated / "common_ugi_completed_rows.tex").read_text()
    assert "RGFN" in (generated / "common_ugi_reaction_external_rows.tex").read_text()
    assert "DeFoG unconditional" in (generated / "common_ugi_generic_external_rows.tex").read_text()
    assert "GenMol/SAFE" in (generated / "common_ugi_generic_external_rows.tex").read_text()
    assert (
        "Learned inventory selector" in (generated / "common_ugi_formulation_rows.tex").read_text()
    )
    benchmark = (generated / "common_ugi_benchmark_completed_rows.tex").read_text()
    # The group headings are named in the caption, not in the table, so the groups are asserted
    # through a member of each: reaction-space, whole-molecule, matched control, then our own.
    assert "RGFN" in benchmark
    assert "DeFoG unconditional" in benchmark
    assert "Finite catalogue oracle" in benchmark
    assert "FORGE Transformer" in benchmark
    assert "N/E" in (generated / "common_ugi_decomposition_rows.tex").read_text()
    assert "DeFoG unconditional" in (generated / "lipid_realism_rows.tex").read_text()
    assert "50{,}000" not in (generated / "compute_parity_external_rows.tex").read_text()
    assert "50,000" in (generated / "compute_parity_external_rows.tex").read_text()
    route_rows = (generated / "route_evidence_completed_rows.tex").read_text()
    assert "\\TABLEPENDING" not in route_rows
    assert "FORGE Transformer" in route_rows
    assert "DeFoG + AiZynthFinder" not in route_rows
    assert "generated atoms and bonds" not in route_rows
    assert "reaction $\\times$ building block" not in route_rows
    assert "route" not in (generated / "common_ugi_completed_rows.tex").read_text().lower()
    assert "ForgeRouteCompleteProducts" in (generated / "completed_evidence_macros.tex").read_text()
    macros = (generated / "completed_evidence_macros.tex").read_text()
    assert "ForgeUgiRetentionRangeLowPP" in macros
    assert "ForgeNullUgiSeedDifferencesPP" in macros
    assert "ForgeSemanticNullCoordinatesRangeLowPP" not in macros
    assert "ForgeSemanticNullCoordinatesSeedDifferencesPP" not in macros
    production_transposed = (generated / "production_comparison_transposed_rows.tex").read_text()
    assert "Exact-L1/attempt" in production_transposed
    assert "\\cellcolor{forgerow}" in production_transposed
    seed_counts = (generated / "production_seed_exact_counts_rows.tex").read_text()
    assert "2{,}220" in seed_counts
    assert "3{,}072" in seed_counts
    catalogue_transposed = (generated / "catalogue_comparison_transposed_rows.tex").read_text()
    assert "Verifier-recovered component-novel/1k" in catalogue_transposed
