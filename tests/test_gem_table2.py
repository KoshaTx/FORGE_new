from __future__ import annotations

from pathlib import Path

from forge_paper.gem_table2 import render_gem_table2_forge_row

from forge.core.io import atomic_write


def test_gem_table2_replaces_only_final_forge_row(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "paper/v1/generated/common_ugi_benchmark_completed_rows.tex"
    rows = tmp_path / source.name
    before = source.read_text()
    atomic_write(rows, before.encode("utf-8"))

    result = render_gem_table2_forge_row(
        repo / "configs/reproduction/natbiotech_v1_completed_evidence_v1.json",
        repo,
        rows,
        result_path=tmp_path / "result.json",
    )

    after = rows.read_text()
    assert result["status"] == "complete"
    assert len(result["sources"]) == 3
    assert after.count("FORGE Transformer") == 1
    assert r"$965.0\pm21.9$" in after
    assert r"$963.9\pm21.4$" in after
    assert r"$862.6\pm19.2$" in after
    assert r"$1.4\pm1.7$" in after
    assert r"$0.643\pm0.022$" in after
    assert [line for line in before.splitlines() if "FORGE Transformer" not in line] == [
        line for line in after.splitlines() if "FORGE Transformer" not in line
    ]
