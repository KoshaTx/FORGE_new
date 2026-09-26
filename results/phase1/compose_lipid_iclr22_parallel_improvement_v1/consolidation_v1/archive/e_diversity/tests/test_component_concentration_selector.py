"""Selection contracts on small synthetic identity ledgers; no chemistry is executed."""

from types import SimpleNamespace

import pytest

from experiments.phase1.multireaction import component_concentration_helpers as helper
from experiments.phase1.multireaction import component_concentration_selector as selector


def row(request, ordinal, component, *, novel=True, observed=True):
    return selector.SelectionCandidate(
        request,
        ordinal,
        f"product-{request}-{ordinal}",
        True,
        True,
        observed,
        novel,
        (selector.ComponentIdentity("isocyanide", component, novel, novel),),
    )


def fixture():
    base = [row(i, 0, c) for i, c in enumerate(("A", "A", "B", "C"))]
    pools = [[c] for c in base]
    pools[0].append(row(0, 1, "D"))
    clean = frozenset((c.request, c.ordinal) for pool in pools for c in pool)
    return pools, base, clean


def solve(pools, baseline, clean, **kwargs):
    return selector.select_design_supported(pools, baseline=baseline, design_pass=clean, **kwargs)


def test_concentration_reduces_repetition_after_equal_primary_objectives():
    pools, base, clean = fixture()
    control, cr = solve(pools, base, clean, minimize_concentration=False)
    changed, er = solve(pools, base, clean)
    assert control == base
    assert changed[0].components[0].smiles == "D"
    assert [s["objective_value"] for s in cr["stages"][:3]] == [
        s["objective_value"] for s in er["stages"][:3]
    ]
    assert er["stages"][-2]["objective_value"] == 4
    selector.check_strict_noninferiority(base, changed)


@pytest.mark.parametrize("field", ["design", "observed", "novel"])
def test_concentration_cannot_purchase_priority_or_floor_loss(field):
    pools, base, clean = fixture()
    if field == "design":
        clean = clean - {(0, 1)}
    else:
        pools[0][1] = row(0, 1, "D", **{field: False})
    changed, report = solve(pools, base, clean)
    assert changed == base
    assert report["status"] == "optimal_independently_verified"


def test_no_alternative_is_exact_noop():
    _, base, clean = fixture()
    chosen, report = solve([[c] for c in base], base, clean - {(0, 1)})
    assert chosen == base
    assert report["status"] == "no_alternative"


def test_censored_solver_does_not_publish_incumbent(monkeypatch):
    pools, base, clean = fixture()
    monkeypatch.setattr(
        selector,
        "milp",
        lambda *a, **k: SimpleNamespace(status=1, x=None, message="bounded test censoring"),
    )
    chosen, report = solve(pools, base, clean)
    assert chosen == base
    assert report["status"] == "censored_retained_current_baseline"


def test_exact_integer_check_rejects_new_concentration():
    _, base, _ = fixture()
    collapsed = [row(i, 1, "A") for i in range(4)]
    with pytest.raises(ValueError, match="unique-identity"):
        selector.check_strict_noninferiority(base, collapsed)


def test_context_eligibility_rejects_new_flag():
    from dataclasses import asdict

    pools, base, _ = fixture()
    rows = [{"candidate": asdict(c)} for pool in pools for c in pool]
    gates = {
        (c.request, c.ordinal): {"chemical": {"flags": []}, "qualified_design_pass": True}
        for pool in pools
        for c in pool
    }
    gates[0, 1]["chemical"]["flags"] = [{"tier": "context_required", "code": "new-context"}]
    filtered, _, _, excluded = helper.build(selector, rows, [asdict(c) for c in base], gates)
    assert filtered[0] == [base[0]]
    assert excluded == [(0, 1)]
