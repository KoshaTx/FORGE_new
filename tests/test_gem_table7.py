from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table7 import render_gem_table7_lipid_realism


def test_gem_table7_renders_all_pinned_structural_realism_rows(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = tmp_path / "lipid_realism_rows.tex"
    result = render_gem_table7_lipid_realism(
        repo / "configs/reproduction/gem_table7_lipid_realism_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert result["attempts_per_method_per_seed"] == 3072
    rendered = rows.read_text()
    assert "\\TABLEPENDING" not in rendered
    assert "TBD" not in rendered
    assert "RGFN" in rendered
    assert "DeFoG unconditional" in rendered
    assert "GenMol/SAFE" in rendered
    assert "Finite catalogue oracle" in rendered
    assert "Learned inventory selector" in rendered
    assert "FORGE Transformer" in rendered
    assert "$965.0\\pm21.9$" in rendered
    assert "$29.5\\pm2.6$/$2.21\\pm0.03$" in rendered
    assert "$2489.2\\pm66.8$" in rendered
    assert rendered.count("N/E") == 3
    assert "\\cellcolor{forgerow}{FORGE Transformer}" in rendered
