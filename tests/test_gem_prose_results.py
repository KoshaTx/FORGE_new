from __future__ import annotations

import re
from pathlib import Path

from forge_paper.gem_prose_results import render_gem_prose_results


def test_gem_result_prose_is_generated_from_final_evidence(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    macros = tmp_path / "gem_prose_results_macros.tex"
    result = render_gem_prose_results(
        repo / "configs/reproduction/gem_prose_results_core_saturation_v1.json",
        repo,
        macros,
        result_path=tmp_path / "result.json",
    )

    assert result["status"] == "complete"
    assert result["candidate_selection"] is False
    assert all(result["gates"].values())
    assert len(result["sources"]) == 37
    assert result["macros"]["ForgeProseCommonForgeExactPerThousand"] == "$963.9\\pm21.4$"
    assert result["macros"]["ForgeProseCommonForgeDistinctPerThousand"] == "$862.6\\pm19.2$"
    assert result["macros"]["ForgeProseCommonForgeHeldPerThousand"] == "$1.4\\pm1.7$"
    assert result["macros"]["ForgeProseCommonHeldTotal"] == "13"
    assert result["macros"]["ForgeProseCatalogueForgeUgiExactPerThousand"] == "963.9"
    assert result["macros"]["ForgeProseMechanismFullUgi"] == "856.9"

    paper = (repo / "paper/v1_neuripsgem/FORGE_GEM2026_paper.tex").read_text()
    rendered = macros.read_text()
    used = set(re.findall(r"\\(ForgeProse[A-Za-z]+)", paper))
    defined = set(re.findall(r"\\newcommand\{\\(ForgeProse[A-Za-z]+)\}", rendered))
    assert used == defined
    for stale in (
        r"786.8\pm100.8",
        r"751.2\pm80.2",
        "320.8, 156.9 and 102.9",
        r"$0.1\pm0.2$ for FORGE",
    ):
        assert stale not in paper
