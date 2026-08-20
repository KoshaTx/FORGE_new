from forge.value.coverage.ugi3_fresh_pool_route_priority import _coverage_thresholds


def test_coverage_thresholds_count_ranked_components() -> None:
    assert _coverage_thresholds([4, 3, 2, 1], 10) == {
        "25pct": 1,
        "50pct": 2,
        "75pct": 3,
        "90pct": 3,
    }
