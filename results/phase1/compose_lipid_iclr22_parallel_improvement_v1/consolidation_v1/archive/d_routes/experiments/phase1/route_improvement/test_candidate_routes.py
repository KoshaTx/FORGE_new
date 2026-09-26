"""Saved-structure boundary controls; no search, training, or network."""

import copy

import candidate_routes as candidate
import pytest


@pytest.fixture(scope="module")
def qualified():
    return candidate.context(), candidate.read(candidate.CAND / "candidate_paths.json")["paths"]


def test_all_candidates_pass_exact_original_contract(qualified):
    context, rows = qualified
    assert len(rows) == 6
    assert all(context.composition(row)["segments"] > 1 for row in rows)


@pytest.mark.parametrize(
    "change",
    [
        "drop_child",
        "change_identity",
        "drop_graft",
        "inject_listing",
        "change_policy",
        "change_clock",
        "change_source_tree",
        "change_basis",
        "change_target",
        "solved_flag",
    ],
)
def test_candidate_tampering_fails(qualified, monkeypatch, change):
    context, rows = qualified
    row = copy.deepcopy(rows[0])
    pointer = candidate.old.pointer
    proof = copy.deepcopy(pointer(row["receipt"]))
    if change == "drop_child":
        row["root"]["step"]["reactants"].pop()
    elif change == "change_identity":
        row["root"]["step"]["reactants"][0]["identity"] = "CC"
    elif change == "drop_graft":
        proof["segments"].pop()
    elif change == "inject_listing":
        row["root"]["listings"] = [{"identity": row["target_identity"]}]
    elif change == "change_policy":
        proof["policy"]["sha256"] = "0" * 64
    elif change == "change_clock":
        proof["as_of_utc"] = "2026-01-01T00:00:00+00:00"
    elif change == "change_source_tree":
        proof["segments"][1]["source"]["original_tree"]["sha256"] = "0" * 64
    elif change == "change_basis":
        row["basis"] = "planner_solved"
    elif change == "change_target":
        row["target_identity"] = "CC"
    elif change == "solved_flag":
        row["planner_solved"] = True
    proof["root"] = copy.deepcopy(row["root"])
    monkeypatch.setattr(
        candidate.old, "pointer", lambda ref: proof if ref == row["receipt"] else pointer(ref)
    )
    with pytest.raises((ValueError, KeyError)):
        context.composition(row)


def test_all_four_searches_required(monkeypatch):
    reader = candidate.read
    completion = reader(candidate.SEARCH / "completion.json")
    completion["declared_indices"] = [0, 1, 2]
    monkeypatch.setattr(
        candidate,
        "read",
        lambda path: completion if path == candidate.SEARCH / "completion.json" else reader(path),
    )
    with pytest.raises(ValueError, match="denominator"):
        candidate.source_census()


def test_existing_listing_view_cannot_fabricate_observation(monkeypatch):
    reader = candidate.read
    view = reader(candidate.CAND / "listing_view.json")
    view["listings"][0]["vendor_count"] += 1
    monkeypatch.setattr(
        candidate,
        "read",
        lambda path: view if path == candidate.CAND / "listing_view.json" else reader(path),
    )
    with pytest.raises(ValueError, match="Listing union"):
        candidate.context()


def test_current_clock_cannot_change(qualified):
    context, _ = qualified
    with pytest.raises(ValueError, match="clock"):
        candidate.wrapper.check_policy_clock(
            context.policy, candidate.old.time("2026-10-26T01:56:45+00:00"), context.policy_pin
        )
