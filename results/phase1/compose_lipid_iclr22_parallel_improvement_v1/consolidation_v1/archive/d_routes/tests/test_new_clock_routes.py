"""Focused exact source/terminal/clock admission controls; no search or HTTP."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments/phase1/route_improvement")
)
import new_clock_routes as module  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    protocol = module.read(module.previous.BASE / "protocol.json")
    policy = module.prior.MakeabilityPolicy.from_mapping(module.prior.read_pin(protocol["policy"]))
    rows = module.read(module.DEST / "candidate_paths.json")["paths"]
    return protocol, policy, rows


@pytest.mark.parametrize("index", [0, 1])
def test_intact_and_graft_pass(inputs, index):
    _, policy, rows = inputs
    assert (
        module.validate_path(rows[index], policy=policy, as_of=module.old.time(module.AS_OF))[
            "terminal_occurrences"
        ]
        > 0
    )


@pytest.mark.parametrize(
    "case",
    [
        "drop_child",
        "alter_identity",
        "inject_listing",
        "change_basis",
        "drop_terminal",
        "experimental_claim",
    ],
)
def test_tree_tampering_fails(inputs, case):
    _, policy, rows = inputs
    row = copy.deepcopy(rows[1])
    if case == "drop_child":
        row["root"]["step"]["reactants"].pop()
    elif case == "alter_identity":
        row["root"]["identity"] = "CCO"
    elif case == "inject_listing":
        row["root"]["listings"] = [{"identity": row["root"]["identity"]}]
    elif case == "change_basis":
        row["basis"] = "planner_solved"
    elif case == "drop_terminal":
        row["terminal_identities"].pop()
    else:
        row["experimental_execution"] = True
    with pytest.raises(ValueError):
        module.validate_path(row, policy=policy, as_of=module.old.time(module.AS_OF))


@pytest.mark.parametrize("case", ["future", "expired", "identity", "no_listing"])
def test_new_listing_admission_controls(inputs, monkeypatch, case):
    _, policy, rows = inputs
    listings = copy.deepcopy(module.listings())
    for row in listings:
        if row["identity"] != "C=CCOC":
            continue
        if case == "future":
            row["observed_at"] = "2026-09-27T06:22:31+00:00"
        elif case == "expired":
            row["observed_at"] = "2026-08-20T06:22:31+00:00"
        elif case == "identity":
            row["identity"] = "CCCOC"
        else:
            row["vendor_count"] = 0
    monkeypatch.setattr(module, "listings", lambda: listings)
    with pytest.raises(ValueError, match="unknown, future or expired"):
        module.validate_path(rows[0], policy=policy, as_of=module.old.time(module.AS_OF))


def test_old_clock_rejected(inputs):
    protocol, policy, _ = inputs
    with pytest.raises(ValueError, match="unreviewed new evaluation clock"):
        module.clock_check(policy, module.old.time(module.wrapper.COMMON_TIME), protocol["policy"])


def test_policy_pin_tamper_rejected(inputs):
    protocol, policy, _ = inputs
    ref = {**protocol["policy"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="policy pin changed"):
        module.clock_check(policy, module.old.time(module.AS_OF), ref)
