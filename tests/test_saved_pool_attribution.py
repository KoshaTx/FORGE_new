import pytest

from experiments.phase1.multireaction import saved_pool_selector as selector
from experiments.phase1.multireaction.saved_pool_attribution import tie_costs


def option(ordinal, *, exact=True, observed=False):
    parts = (selector.ComponentIdentity("tail", "CCC", True, True),) if exact else ()
    return selector.SelectionCandidate(
        0, ordinal, "C" * (ordinal + 2), True, exact, observed, True, parts
    )


def test_ordering_changes_tie_only_and_preserves_candidate_identity():
    first, second = option(0), option(1)
    clean = frozenset({(0, 0), (0, 1)})
    a, _ = selector.select_design_supported([[first, second]], baseline=[first], design_pass=clean)
    b, _ = selector.select_design_supported(
        [[first, second]], baseline=[first], design_pass=clean, tie_costs={(0, 0): 1, (0, 1): 0}
    )
    assert a == [first] and b == [second]
    selector.check_strict_noninferiority(a, b)


def test_lower_priority_cost_cannot_override_design_or_exactness():
    first, second, invalid = option(0), option(1), option(2, exact=False)
    chosen, _ = selector.select_design_supported(
        [[first, second, invalid]],
        baseline=[first],
        design_pass=frozenset({(0, 0)}),
        tie_costs={(0, 0): 1000, (0, 1): 1, (0, 2): 0},
    )
    assert chosen == [first]


@pytest.mark.parametrize("costs", [{(0, 0): 0}, {(0, 0): 0, (0, 1): -1}, {(0, 0): 0, (0, 1): 0.2}])
def test_malformed_costs_rejected(costs):
    first, second = option(0), option(1)
    with pytest.raises(ValueError, match="Tie costs"):
        selector.select_design_supported(
            [[first, second]], baseline=[first], design_pass=frozenset(), tie_costs=costs
        )


def test_seeded_order_is_reproducible_per_request_and_complete():
    pool = [option(i) for i in range(20)]
    assert tie_costs([pool], "permuted_2026092671") == tie_costs(
        [list(reversed(pool))], "permuted_2026092671"
    )
    assert sorted(tie_costs([pool], "permuted_2026092671").values()) == list(range(20))
    assert tie_costs([pool], "permuted_2026092671") != tie_costs([pool], "permuted_2026092672")
