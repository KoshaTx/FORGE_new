from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table9 import render_gem_table9_catalogue_comparison


def test_gem_table9_renders_final_forge_and_fixed_catalogue(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = tmp_path / "catalogue_comparison_transposed_rows.tex"
    result = render_gem_table9_catalogue_comparison(
        repo / "configs/reproduction/gem_table9_catalogue_comparison_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert len(result["sources"]) == 4
    rendered = rows.read_text()
    assert "TBD" not in rendered
    assert "963.9" in rendered
    assert "721.9" in rendered
    assert "523.3" in rendered
    assert "1000.0" in rendered
    assert "723.2" in rendered
    assert "839.2" in rendered
    assert "159.3" in rendered
    assert "58.7" in rendered
    assert "0.643/419.9" in rendered
    assert "0.593/118.0" in rendered
    assert "0.776/173.0" in rendered
    assert rendered.count("\\cellcolor{forgerow}") == 21
    assert "superseded forge arm summaries" in result["ignored_catalogue_artifact_fields"]
