from pathlib import Path

from experiments.archive.phase1.synthesis_value_coverage.ugi3_fresh_pool_route_coverage_v6 import (
    build_fresh_pool_route_coverage_v6,
)
from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v6.json"
BASE = REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v5"


def test_v6_requalifies_v5_without_changing_values() -> None:
    result, components, products = build_fresh_pool_route_coverage_v6(REPO, CONFIG)

    assert result["status"] == "immutable_v5_values_requalified_under_current_source"
    assert result["summary"]["value_records_changed"] == 0
    assert result["summary"]["source_lineage_requalified"] is True
    assert components == (BASE / "component_synthesis_values.json.gz").read_bytes()
    assert products == (BASE / "product_synthesis_values.json.gz").read_bytes()
    assert result["artifacts"]["component_synthesis_values.json.gz"]["sha256"] == sha256_file(
        BASE / "component_synthesis_values.json.gz"
    )
    assert result["adjudication"]["nonzero_guidance_authorized"] is False
    assert result["adjudication"]["sealed_holdout_accessed"] is False
