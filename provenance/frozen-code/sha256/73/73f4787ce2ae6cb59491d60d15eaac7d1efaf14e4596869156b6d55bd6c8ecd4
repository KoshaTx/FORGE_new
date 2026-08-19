import shutil
from pathlib import Path

from forge.route.ugi3_source_qualified_cumulative_inputs import (
    build_source_qualified_cumulative_inputs,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_source_qualified_cumulative_inputs_v1.json"


def test_source_qualified_inputs_reproduce_all_historical_science(tmp_path: Path) -> None:
    output = REPO / "results/phase1/_test_source_qualified_cumulative_inputs"
    shutil.rmtree(output, ignore_errors=True)
    try:
        result = build_source_qualified_cumulative_inputs(REPO, CONFIG, output)
    finally:
        shutil.rmtree(output, ignore_errors=True)

    assert result["status"] == "source_qualified_cumulative_inputs_reproduced"
    assert result["summary"] == {
        "qualified_layers": 4,
        "scientific_summaries_changed": 0,
        "ledger_byte_changes": 0,
    }
    assert result["adjudication"]["cumulative_source_requalification_may_advance"] is True
    assert result["adjudication"]["nonzero_guidance_authorized"] is False
    assert result["adjudication"]["sealed_holdout_accessed"] is False
