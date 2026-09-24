from dataclasses import replace

import pytest

from forge.model.compose_lipid_quality_selection import (
    ComponentIdentity,
    SelectionCandidate,
    check_noninferiority,
    first_exact,
    select_reference_supported,
)


def candidate(request, ordinal, head, *, supported=False, novel=True, smiles=None):
    return SelectionCandidate(
        request=request,
        ordinal=ordinal,
        smiles=smiles or f"product-{request}-{ordinal}",
        connected=True,
        exact=True,
        local_features_observed=supported,
        product_train_novel=novel,
        components=(ComponentIdentity("head", head, novel, novel),),
    )


def test_joint_swap_can_improve_support_without_concentrating_components():
    pools = [
        [candidate(0, 0, "A"), candidate(0, 1, "B", supported=True)],
        [candidate(1, 0, "B"), candidate(1, 1, "A", supported=True)],
    ]
    selected, report = select_reference_supported(pools)
    assert report["status"] == "optimal_independently_verified"
    assert [c.ordinal for c in selected] == [1, 1]
    assert report["supported_distinct_after"] == 2
    assert select_reference_supported(pools)[0] == selected


def test_supported_option_cannot_collapse_heads_or_reduce_novelty():
    pools = [
        [candidate(0, 0, "A")],
        [
            candidate(1, 0, "B"),
            candidate(1, 1, "A", supported=True),
            candidate(1, 2, "C", supported=True, novel=False),
        ],
    ]
    selected, report = select_reference_supported(pools)
    assert [c.ordinal for c in selected] == [0, 0]
    assert report["supported_distinct_after"] == 0


def test_unknown_exact_baseline_can_be_replaced_and_failure_requests_remain():
    failed = SelectionCandidate(2, 0, None, False, False, False, False)
    pools = [[candidate(0, 0, "A"), candidate(0, 1, "C", supported=True)], [failed]]
    selected, _ = select_reference_supported(pools)
    assert selected[0].ordinal == 1
    assert selected[1] == failed
    assert len(selected) == 2


def test_exact_only_uses_frozen_order_and_never_drops_requests():
    failed = SelectionCandidate(0, 0, "raw", True, False, False, True)
    exact = candidate(0, 1, "A")
    assert first_exact([exact, failed]) == exact
    assert first_exact([failed]) == failed
    with pytest.raises(ValueError):
        first_exact([exact, exact])


def test_shannon_guard_is_independent_of_simpson_and_unique_count():
    before_heads = list("AAAAABBCCD")
    after_heads = list("AAAABBBBCD")
    before = [candidate(i, 0, head) for i, head in enumerate(before_heads)]
    after = [candidate(i, 1, head) for i, head in enumerate(after_heads)]
    with pytest.raises(ValueError, match="Shannon"):
        check_noninferiority(before, after)


def test_changed_exact_coverage_or_population_is_rejected():
    before = [candidate(0, 0, "A")]
    with pytest.raises(ValueError, match="coverage"):
        check_noninferiority(before, [SelectionCandidate(0, 0, None, False, False, False, False)])
    with pytest.raises(ValueError, match="population"):
        check_noninferiority(before, [replace(before[0], request=1)])


def test_unproved_solver_retains_reviewable_baseline(monkeypatch):
    from types import SimpleNamespace

    import forge.model.compose_lipid_quality_selection as module

    pools = [[candidate(0, 0, "A"), candidate(0, 1, "B", supported=True)]]
    monkeypatch.setattr(
        module, "milp", lambda *args, **kwargs: SimpleNamespace(status=1, message="limit", x=None)
    )
    selected, report = select_reference_supported(pools)
    assert selected == [pools[0][0]]
    assert report["status"] == "censored_retained_exact_only"


def test_invalid_incumbent_cannot_waive_unique_count(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    import forge.model.compose_lipid_quality_selection as module

    pools = [[candidate(0, 0, "A")], [candidate(1, 0, "B"), candidate(1, 1, "A", supported=True)]]

    def invalid(objective, **kwargs):
        x = np.zeros(len(objective))
        x[0] = x[2] = 1
        return SimpleNamespace(status=0, message="fake", x=x)

    monkeypatch.setattr(module, "milp", invalid)
    selected, report = select_reference_supported(pools)
    assert selected == [pool[0] for pool in pools]
    assert report["status"] == "failed_independent_check_retained_exact_only"
