from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table3 import render_gem_table3_route_assessment

from forge.core.io import iter_jsonl


def test_gem_table3_is_reproducible_and_preserves_denominators(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = render_gem_table3_route_assessment(
        repo / "configs/reproduction/gem_table3_core_saturation_v1.json",
        repo,
        tmp_path / "generated",
        result_path=tmp_path / "result" / "result.json",
    )

    summary = result["summary"]
    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert summary["products"] == 256
    assert sum(summary["product_states"].values()) == 256
    assert sum(summary["component_states"].values()) == summary["components"]
    assert all(result["gates"].values())

    products = list(iter_jsonl(tmp_path / "result/product_route_ledger.jsonl.gz"))
    assert products[0] == {
        "schema_version": "forge.gem_table3_route_assessed_products.v1",
        "rows": 256,
    }
    assert [row["assessment_rank"] for row in products[1:]] == list(range(1, 257))
    assert all(
        row["final_product_state"] in {"complete", "unresolved", "search_censored"}
        for row in products[1:]
    )

    macros = (tmp_path / "generated/gem_table3_route_macros.tex").read_text()
    assert r"\newcommand{\ForgeRouteEligiblePool}{7696}" in macros
    assert r"\renewcommand{\ForgeRouteCandidates}{256}" in macros
    assert r"\renewcommand{\ForgeRouteExactLone}{256}" in macros
