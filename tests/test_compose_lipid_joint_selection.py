from itertools import product

import pytest

from forge.model.compose_lipid_joint_selection import admit_joint_component_proposals


def exact(head, tail):
    return dict(
        status="evaluated",
        exact=True,
        checks=[dict(accepted_components=[dict(head=head, tail=tail)])],
    )


def row(index, head=None, tail=None, options=(), old=None):
    return dict(
        index=index,
        family="test",
        baseline_smiles=old or f"old{index}",
        baseline_check=exact(head, tail) if head else dict(status="evaluated", exact=False),
        proposals=[
            dict(smiles=f"new{index}_{j}", check=exact(h, t)) for j, (h, t) in enumerate(options)
        ],
    )


def test_balanced_pair_unlocks_without_relaxing_diversity():
    rows = [
        row(0, "A", "X"),
        row(1, "B", "Y"),
        row(2, options=[("A", "X")]),
        row(3, options=[("B", "Y")]),
    ]
    selected, report = admit_joint_component_proposals(rows, is_train_product=lambda _: False)
    assert all(r["accepted"] for r in selected)
    assert report["test"]["admitted"] == 2
    assert [r["selected_smiles"] for r in selected[:2]] == ["old0", "old1"]


def test_unbalanced_pool_cannot_concentrate_components():
    rows = [
        row(0, "A", "X"),
        row(1, "B", "Y"),
        row(2, options=[("A", "X")]),
        row(3, options=[("A", "X")]),
    ]
    selected, _ = admit_joint_component_proposals(rows, is_train_product=lambda _: False)
    assert sum(r["accepted"] for r in selected) == 2


def test_full_source_novelty_duplicates_and_exact_baselines_are_guarded():
    rows = [
        row(0, "A", "X", options=[("C", "Z")]),
        row(1, options=[("B", "Y")]),
        row(2, options=[("B", "Y")]),
        row(3, options=[("D", "W")]),
    ]
    rows[2]["proposals"][0]["smiles"] = "old0"
    rows[3]["proposals"][0]["check"]["exact"] = False
    selected, _ = admit_joint_component_proposals(rows, is_train_product=lambda s: s == "new1_0")
    assert [r["selected_smiles"] for r in selected] == [f"old{i}" for i in range(4)]
    assert [r["proposal_decisions"][0]["disposition"] for r in selected] == [
        "baseline_exact_immutable",
        "would_reduce_train_novelty",
        "already_present_baseline_product",
        "not_full_source_exact",
    ]


def test_solver_matches_exhaustive_cardinality_on_small_pools():
    from collections import Counter

    choices = [("A", "X"), ("B", "Y"), ("A", "Z")]
    for combination in product(choices, repeat=3):
        rows = [row(0, "A", "X"), row(1, "B", "Y")] + [
            row(i + 2, options=[option]) for i, option in enumerate(combination)
        ]
        best = 0
        for bits in product((0, 1), repeat=3):
            parts = [("A", "X"), ("B", "Y")] + [
                p for p, bit in zip(combination, bits, strict=True) if bit
            ]
            counts = [Counter(p[j] for p in parts) for j in range(2)]
            if all(len(parts) ** 2 >= 2 * sum(v * v for v in c.values()) for c in counts):
                best = max(best, sum(bits))
        selected, report = admit_joint_component_proposals(rows, is_train_product=lambda _: False)
        assert report["test"]["optimal"]
        assert sum(r["accepted"] for r in selected) == 2 + best


def test_fractional_or_invalid_incumbent_fails_closed(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    import forge.model.compose_lipid_joint_selection as module

    rows = [row(0, "A", "X"), row(1, "B", "Y"), row(2, options=[("A", "X")])]
    monkeypatch.setattr(
        module,
        "milp",
        lambda *a, **kw: SimpleNamespace(status=1, message="bounded", x=np.array([0.5])),
    )
    with pytest.raises(ValueError, match="nonintegral"):
        admit_joint_component_proposals(rows, is_train_product=lambda _: False)
    monkeypatch.setattr(
        module,
        "milp",
        lambda *a, **kw: SimpleNamespace(status=1, message="bounded", x=np.array([1.0])),
    )
    with pytest.raises(ValueError, match="decreases"):
        admit_joint_component_proposals(rows, is_train_product=lambda _: False)


def test_no_incumbent_retains_every_request(monkeypatch):
    from types import SimpleNamespace

    import forge.model.compose_lipid_joint_selection as module

    rows = [row(0, "A", "X"), row(1, options=[("B", "Y")])]
    monkeypatch.setattr(
        module, "milp", lambda *a, **kw: SimpleNamespace(status=1, message="bounded", x=None)
    )
    selected, report = admit_joint_component_proposals(rows, is_train_product=lambda _: False)
    assert len(selected) == 2 and [r["selected_smiles"] for r in selected] == ["old0", "old1"]
    assert report["test"]["disposition"] == "no_incumbent_retained_baseline"
