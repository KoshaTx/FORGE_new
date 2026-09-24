"""Merge fixed/repeated evidence without losing protection or scientific checks."""

import copy
import hashlib

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_grouped_readiness import validate_replay_row


def make_records(kind):
    prepared = {
        "target_id": "t",
        "family": "family",
        "constitution_id": hashlib.sha256(b"CCN").hexdigest(),
        "component_instances": [["head", "global1", 1], ["tail", "global2", 1]],
        "construction_basis": "supplied",
        "eligible_for_program_preparation": True,
        "old_split": "train",
        "corrected_split": "train",
        "exclusion_reasons": [],
        "pending_reasons": [],
    }
    row = {
        k: copy.deepcopy(prepared[k])
        for k in (
            "target_id",
            "family",
            "constitution_id",
            "component_instances",
            "construction_basis",
        )
    }
    replay = {"computed_consistency_pass": True, "disposition": "exact_computed_reconstruction"}
    if kind == "fixed":
        replay.update(
            forward_products=["CCN"],
            verified_target_constitution_id=prepared["constitution_id"],
            checks={
                k: True
                for k in (
                    "unique_unfiltered_forward_exact",
                    "unique_unfiltered_inverse_exact",
                    "source_reactive_site_witness",
                    "full_element_hydrogen_charge_balance",
                    "distinct_source_role_handles_retained",
                )
            },
        )
    else:
        replay.update(
            forward_layers=[["CN"], ["CCN"]],
            bound_reasons=[],
            checks={
                k: True
                for k in (
                    "complete_search",
                    "declared_event_count_replayed",
                    "unique_forward_exact",
                    "unique_declared_side_inverse",
                    "full_element_hydrogen_charge_balance",
                )
            },
        )
    if kind in ("staged", "grouped"):
        replay.update(
            stage_balance_checks=[True],
            inverse={
                "complete_search": True,
                "bound_reasons": [],
                "candidate_components": [{"head": "CN"}],
            },
            checks={
                k: True
                for k in (
                    "declared_component_equalities",
                    "complete_search",
                    "every_stage_replayed",
                    "unique_each_forward_stage",
                    "unique_forward_exact",
                    "unique_complete_inverse",
                    "all_stage_inventories_balance",
                    "full_element_hydrogen_charge_balance",
                    "terminal_constraints",
                    "product_constraints",
                )
            },
        )
    if kind == "grouped":
        replay.pop("stage_balance_checks")
        replay.update(
            event_searches=[
                {"stage": 1, "declared_events": 2, "layers": [["CN"], ["CNC", "NCC"], ["CCN"]]}
            ],
            event_balance_checks=[True, True, True, True],
            declared_quantities={"head": 1, "tail": 2},
            checks={
                k: True
                for k in (
                    "complete_search",
                    "every_declared_event_replayed",
                    "unique_each_completed_source_stage",
                    "unique_forward_exact",
                    "unique_complete_inverse",
                    "all_event_inventories_balance",
                    "full_element_hydrogen_charge_balance",
                    "terminal_constraints",
                    "product_constraints",
                )
            },
        )
    row.update(replay=replay, training_admitted=False, experimental_execution_admitted=False)
    return row, prepared


@pytest.fixture(params=["fixed", "repeated", "staged", "grouped"])
def records(request):
    return make_records(request.param)


@pytest.fixture
def staged_records():
    return make_records("staged")


def test_exact_fixed_and_repeated_evidence_is_accepted(records):
    assert validate_replay_row(*records)


@pytest.mark.parametrize(
    "corruption",
    [
        "heldout",
        "unassigned",
        "excluded",
        "pending",
        "training",
        "experiment",
        "target",
        "roles",
        "omitted_check",
        "failed_check",
        "identity",
        "ambiguous",
    ],
)
def test_corruption_cannot_be_promoted(records, corruption):
    row, prepared = records
    r = row["replay"]
    if corruption == "heldout":
        prepared["old_split"] = "heldout"
    elif corruption == "unassigned":
        prepared["corrected_split"] = "unassigned"
    elif corruption == "excluded":
        prepared["exclusion_reasons"] = ["held_component"]
    elif corruption == "pending":
        prepared["pending_reasons"] = ["partition"]
    elif corruption == "training":
        row["training_admitted"] = True
    elif corruption == "experiment":
        row["experimental_execution_admitted"] = True
    elif corruption == "target":
        row["target_id"] = "another"
    elif corruption == "roles":
        row["component_instances"][0][0] = "another_role"
    elif corruption == "omitted_check":
        r["checks"].pop("full_element_hydrogen_charge_balance")
    elif corruption == "failed_check":
        r["checks"]["full_element_hydrogen_charge_balance"] = False
    elif corruption == "identity":
        r["verified_target_constitution_id"] = "0" * 64
    else:
        if "forward_products" in r:
            r["forward_products"].append("CCC")
        else:
            r["forward_layers"][-1].append("CCC")
    with pytest.raises(ComposeLipidError):
        validate_replay_row(row, prepared)


def test_failed_reconstruction_stays_unresolved(records):
    row, prepared = records
    row["replay"] = {
        "computed_consistency_pass": False,
        "disposition": "unsupported_source_role_tuple",
    }
    assert not validate_replay_row(row, prepared)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("corruption", [None, "quantities", "events"])
def test_merge_preserves_entire_universe_prior_evidence_and_protection(
    tmp_path, monkeypatch, grouped, corruption
):
    import gzip
    import json

    from forge.corpus import compose_lipid_grouped_readiness as merge
    from forge.corpus.compose_lipid_source_view import dump, pin
    from forge.corpus.compose_lipid_supplement import rows

    def write_rows(name, values):
        path = tmp_path / name
        with gzip.open(path, "wt") as stream:
            for value in values:
                stream.write(json.dumps(value) + "\n")
        return pin(tmp_path, path)

    prepared = []
    old = []
    for target, eligible in [("a", True), ("b", False), ("c", True), ("d", False)]:
        p = {
            "target_id": target,
            "family": "f",
            "constitution_id": hashlib.sha256(b"CCN").hexdigest(),
            "component_instances": [["head", "id", 1]],
            "construction_basis": "supplied",
            "eligible_for_program_preparation": eligible,
            "old_split": "train" if eligible else "heldout",
            "corrected_split": "train" if eligible else "unassigned",
            "exclusion_reasons": [] if eligible else ["protected"],
            "pending_reasons": [],
            "disposition": "eligible" if eligible else "protected",
        }
        prepared.append(p)
        old.append(
            {
                "target_id": target,
                "family": "f",
                "preparation_disposition": p["disposition"],
                "training_admitted": False,
                "passes_completed_recipe_checks": False,
                "additional_replays": [{"receipt": "prior", "exact": False}],
                "reconstruction_disposition": "pending",
            }
        )
    prep = {"artifact": write_rows("prepared.jsonl.gz", prepared)}
    dump(tmp_path / "prepared.json", prep)
    prep_pin = pin(tmp_path, tmp_path / "prepared.json")
    previous = {
        "schema_version": "forge.compose_lipid_readiness_extension.v1",
        "inputs": {"preparation": prep_pin},
        "summary": {
            "exact_reconstructions": 0,
            "source_records": 4,
            "preparation_dispositions": {"eligible": 2, "protected": 2},
        },
        "remaining_gates": ["partition", "training_dataset"],
        "artifact": write_rows("prior.jsonl.gz", old),
    }
    dump(tmp_path / "prior.json", previous)
    identifier = "program_id" if grouped else "reaction_id"
    cfg = {"inputs": {"preparation": prep_pin}, "families": {"f": {identifier: "reaction"}}}
    dump(tmp_path / "replay-config.json", cfg)
    updated = []
    for p in (prepared[0], prepared[2]):
        row = {
            k: copy.deepcopy(p[k])
            for k in (
                "target_id",
                "family",
                "constitution_id",
                "component_instances",
                "construction_basis",
            )
        }
        exact = p["target_id"] == "a"
        replay = {
            "computed_consistency_pass": exact,
            "disposition": (
                "exact_computed_reconstruction" if exact else "unsupported_source_role_tuple"
            ),
        }
        if exact:
            replay.update(
                forward_products=["CCN"],
                verified_target_constitution_id=p["constitution_id"],
                checks={
                    k: True
                    for k in (
                        "unique_unfiltered_forward_exact",
                        "unique_unfiltered_inverse_exact",
                        "source_reactive_site_witness",
                        "full_element_hydrogen_charge_balance",
                        "distinct_source_role_handles_retained",
                    )
                },
            )
        if grouped and exact:
            replay = make_records("grouped")[0]["replay"]
            if corruption == "quantities":
                replay["declared_quantities"]["tail"] = 3
            elif corruption == "events":
                replay["event_searches"][0]["declared_events"] = 3
                replay["event_searches"][0]["layers"].insert(1, ["CN"])
        row[identifier] = "reaction"
        row.update(
            replay=replay,
            training_admitted=False,
            experimental_execution_admitted=False,
        )
        updated.append(row)
    counts = {"rows": 2, "exact_computed_reconstruction": 1, "unsupported_source_role_tuple": 1}
    engine = merge.grouped if grouped else merge.fixed
    result = {
        "schema_version": engine.RESULT_SCHEMA,
        "policy": engine.POLICY,
        "inputs": cfg["inputs"],
        "implementation": {},
        "controls": {},
        "config": pin(tmp_path, tmp_path / "replay-config.json"),
        "artifact": write_rows("replay.jsonl.gz", updated),
        "summary": {"by_family": {"f": counts}, "totals": counts},
    }
    dump(tmp_path / "replay.json", result)
    from pathlib import Path

    local_implementation = tmp_path / "merge.py"
    local_implementation.write_bytes(Path(merge.__file__).read_bytes())
    monkeypatch.setattr(merge, "__file__", str(local_implementation))
    from types import SimpleNamespace

    program = SimpleNamespace(
        quantities={"head": 1, "tail": 2}, specification={"stages": [{"events": 2}]}
    )
    monkeypatch.setattr(engine, "IMPLEMENTATION", ())
    monkeypatch.setattr(
        engine, "load_contract", lambda *_: (cfg, {}, {"f": {"program": program}}, {})
    )
    config = {
        "schema_version": "forge.compose_lipid_grouped_readiness_config.v1",
        "inputs": {
            "preparation": prep_pin,
            "previous_report": pin(tmp_path, tmp_path / "prior.json"),
            "new_replay": pin(tmp_path, tmp_path / "replay.json"),
        },
        "replay_inputs": ["new_replay"],
    }
    dump(tmp_path / "config.json", config)
    if grouped and corruption:
        with pytest.raises(ComposeLipidError, match="quantities or event counts"):
            merge.merge_readiness(tmp_path, tmp_path / "config.json", tmp_path / "output")
        assert not (tmp_path / "output").exists()
        return
    report = merge.merge_readiness(tmp_path, tmp_path / "config.json", tmp_path / "output")
    actual = list(rows(tmp_path / report["artifact"]["path"]))
    assert [r["target_id"] for r in actual] == ["a", "b", "c", "d"]
    assert [r["passes_completed_recipe_checks"] for r in actual] == [True, False, False, False]
    assert all(r["additional_replays"][0] == {"receipt": "prior", "exact": False} for r in actual)
    assert all(not r["training_admitted"] for r in actual)
    assert report["summary"]["source_records"] == 4
    assert report["summary"]["new_exact_reconstructions"] == 1
    assert report["summary"]["preparation_dispositions"] == {"eligible": 2, "protected": 2}
    assert report["remaining_gates"] == previous["remaining_gates"]


@pytest.mark.parametrize(
    "missing",
    [
        "declared_component_equalities",
        "every_stage_replayed",
        "unique_each_forward_stage",
        "unique_complete_inverse",
        "all_stage_inventories_balance",
        "terminal_constraints",
        "product_constraints",
    ],
)
def test_staged_checks_cannot_be_omitted(staged_records, missing):
    row, prepared = staged_records
    row["replay"]["checks"].pop(missing)
    with pytest.raises(ComposeLipidError, match="required"):
        validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "change",
    ["balance", "empty_balance", "inverse_bound", "inverse_ambiguity", "intermediate_ambiguity"],
)
def test_staged_raw_evidence_cannot_be_hidden_behind_passing_flags(staged_records, change):
    row, prepared = staged_records
    r = row["replay"]
    if change == "balance":
        r["stage_balance_checks"] = [False]
    elif change == "empty_balance":
        r["stage_balance_checks"] = []
    elif change == "inverse_bound":
        r["inverse"]["bound_reasons"] = ["saturated"]
    elif change == "inverse_ambiguity":
        r["inverse"]["candidate_components"] *= 2
    else:
        r["forward_layers"][0].append("CC")
    with pytest.raises(ComposeLipidError):
        validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "change",
    [
        "empty_stages",
        "stage_number",
        "event_boolean",
        "event_zero",
        "missing_layer",
        "empty_layer",
        "wrong_start",
        "wrong_end",
        "balance",
        "empty_balance",
        "inverse_bound",
        "inverse_ambiguity",
        "stage_ambiguity",
    ],
)
def test_grouped_stage_evidence_cannot_be_hidden_by_pass_flags(change):
    row, prepared = make_records("grouped")
    r = row["replay"]
    stage = r["event_searches"][0]
    if change == "empty_stages":
        r["event_searches"] = []
    elif change == "stage_number":
        stage["stage"] = 2
    elif change == "event_boolean":
        stage["declared_events"] = True
    elif change == "event_zero":
        stage["declared_events"] = 0
    elif change == "missing_layer":
        stage["layers"].pop(1)
    elif change == "empty_layer":
        stage["layers"][1] = []
    elif change == "wrong_start":
        stage["layers"][0] = ["CC"]
    elif change == "wrong_end":
        stage["layers"][-1] = ["CC"]
    elif change == "balance":
        r["event_balance_checks"][0] = False
    elif change == "empty_balance":
        r["event_balance_checks"] = []
    elif change == "inverse_bound":
        r["inverse"]["bound_reasons"] = ["saturated"]
    elif change == "inverse_ambiguity":
        r["inverse"]["candidate_components"] *= 2
    else:
        r["forward_layers"][0].append("CC")
    with pytest.raises(ComposeLipidError):
        validate_replay_row(row, prepared)


@pytest.mark.parametrize(
    "missing",
    [
        "every_declared_event_replayed",
        "unique_each_completed_source_stage",
        "unique_complete_inverse",
        "all_event_inventories_balance",
        "terminal_constraints",
        "product_constraints",
    ],
)
def test_grouped_check_omission_never_admits(missing):
    row, prepared = make_records("grouped")
    row["replay"]["checks"].pop(missing)
    with pytest.raises(ComposeLipidError, match="required"):
        validate_replay_row(row, prepared)
