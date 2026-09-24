"""B5 source profiles bind original task inputs without looking at the target graph."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_b5_profiles import B5SourceProfiles
from forge.corpus.compose_lipid_b5_readiness import validate_profiled_row
from forge.corpus.compose_lipid_b5_replay import load_contract, qualify_controls, replay_record

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
SOURCE = ROOT / "results/phase1/compose_lipid_b5_source_v1/control-transcriptions.json"


@pytest.fixture(scope="module")
def profiles():
    return B5SourceProfiles.from_registry(REGISTRY, expected_sha256=sha256_file(REGISTRY))


def example(profiles, label="I81"):
    control = next(c for c in json.loads(SOURCE.read_text())["controls"] if c["label"] == label)
    spec = next(
        p for p in profiles.specification["profiles"].values() if p["profile"] == control["profile"]
    )
    structures = {f"fixture:{r}": s for r, s in control["components"].items()}
    metadata = {
        **spec["selector"],
        "head_axis": "reported_head_" + spec["head_kind"],
        "tail_axis": "__".join(
            next(k for k, rule in spec["tail_rules"][r].items() if rule["kind"] == "reported")
            for r in spec["tail_roles"]
        ),
        "head_id": "source-scoped-fixture-head",
    }
    task = {
        **metadata,
        "schema": profiles.specification["schema"],
        "task_id": "fixture-task",
        "family": profiles.specification["family"],
        "training_admissible": False,
        "events": copy.deepcopy(spec["events"]),
        "reagents": [
            {"role": generic, "smiles": control["components"][r]}
            for r, generic in spec["reagent_order"]
        ],
    }
    prepared = {
        "eligible_for_program_preparation": True,
        "family": task["family"],
        "component_instances": [
            [alias, f"fixture:{r}", 1] for r, alias in spec["registry_to_source_roles"].items()
        ],
    }
    construction = {
        "task_id": task["task_id"],
        "reaction_steps_sites": {
            "task": {**spec["selector"], "events": copy.deepcopy(task["events"])}
        },
        "declared_precursor_identifiers": [
            {
                "field": "head_id",
                "roles": ["series_head"],
                "value": metadata["head_id"],
                "scoped_alias": task["family"] + "|head_id|" + metadata["head_id"],
            }
        ],
    }
    return prepared, construction, metadata, task, structures


@pytest.mark.parametrize("label", ["I71", "I81", "I91", "I93", "I95", "I97"])
def test_independent_source_controls_qualify_every_input(profiles, label):
    args = example(profiles, label)
    before = copy.deepcopy(args)
    result = profiles.assess(*args)
    assert result["profile_qualified"], result["checks"]
    assert all(result["checks"].values())
    assert not result["training_admitted"] and not result["experimental_execution_admitted"]
    assert args == before


def test_profile_binding_does_not_consult_target_graph(profiles):
    args = example(profiles)
    first = profiles.assess(*args)
    args[1]["constitution"] = "invalid-do-not-parse"
    args[0]["constitution_id"] = "different-product"
    assert profiles.assess(*args) == first


@pytest.mark.parametrize("label", ["I71", "I81", "I91", "I93", "I95", "I97"])
def test_protected_record_rejected_before_any_structure_decoding(profiles, label):
    args = example(profiles, label)
    args[0]["eligible_for_program_preparation"] = False
    args[4].clear()
    with pytest.raises(ComposeLipidError, match="Protected"):
        profiles.assess(*args)


@pytest.mark.parametrize("quantity", [2, 3])
def test_original_unit_quantity_cannot_change(profiles, quantity):
    args = example(profiles)
    args[0]["component_instances"][0][2] = quantity
    assert not profiles.assess(*args)["profile_qualified"]


@pytest.mark.parametrize("quantity", [True, 0, -1, 1.0])
def test_invalid_quantity_is_loud(profiles, quantity):
    args = example(profiles)
    args[0]["component_instances"][0][2] = quantity
    with pytest.raises(ComposeLipidError):
        profiles.assess(*args)


@pytest.mark.parametrize("key", ["series", "design_lane", "head_axis", "tail_axis", "head_id"])
def test_exported_metadata_must_match_original(profiles, key):
    args = example(profiles)
    args[2][key] = "changed"
    result = profiles.assess(*args)
    assert not result["profile_qualified"]
    assert not result["checks"]["source_metadata_agrees"]


def test_head_source_alias_never_compared_as_global_component_id(profiles):
    args = example(profiles)
    result = profiles.assess(*args)
    assert result["profile_qualified"]
    assert args[3]["head_id"] not in args[4]
    args[1]["declared_precursor_identifiers"][0]["value"] = "wrong-source-alias"
    assert not profiles.assess(*args)["profile_qualified"]


def test_same_generic_roles_cannot_swap_original_tail_positions(profiles):
    args = example(profiles, "I91")
    reagents = args[3]["reagents"]
    reagents[1], reagents[2] = reagents[2], reagents[1]
    result = profiles.assess(*args)
    assert not result["checks"]["complete_original_component_structure_agreement"]
    assert not result["profile_qualified"]


def test_source_stereochemistry_not_erased_by_constitutional_model_identity(profiles):
    args = example(profiles)
    args[3]["reagents"][0]["smiles"] = args[3]["reagents"][0]["smiles"].replace("@@", "@")
    result = profiles.assess(*args)
    assert result["checks"]["complete_original_component_structure_agreement"]
    assert not result["checks"]["raw_source_core_stereochemistry"]
    assert not result["profile_qualified"]


@pytest.mark.parametrize("change", ["swap", "wrong_site", "wrong_operation", "missing"])
def test_original_source_event_order_and_linkage_are_required(profiles, change):
    args = example(profiles, "I91")
    events = args[3]["events"]
    if change == "swap":
        events[0], events[1] = events[1], events[0]
    elif change == "wrong_site":
        events[0]["site"] = "secondary"
    elif change == "wrong_operation":
        events[-1]["operation"] = "ester"
    else:
        events.pop()
    result = profiles.assess(*args)
    assert not result["checks"]["source_events_agree"]
    assert not result["profile_qualified"]


def test_incorrect_reported_tail_label_abstains_despite_generic_compatibility(profiles):
    args = example(profiles)
    replacement = next(
        c for c in json.loads(SOURCE.read_text())["controls"] if c["label"] == "I91"
    )["components"]["tail_acid_1"]
    args[4]["fixture:tail_acid"] = replacement
    args[3]["reagents"][-1]["smiles"] = replacement
    result = profiles.assess(*args)
    assert result["checks"]["complete_original_component_structure_agreement"]
    assert result["checks"]["complete_component_scope"]
    assert not result["checks"]["tail_axis_identity"]
    assert not result["profile_qualified"]


def test_two_declared_homo_tails_keep_two_roles(profiles):
    args = example(profiles, "I95")
    instances = args[0]["component_instances"]
    # Identical full precursors may legitimately share one global ID in two roles.
    for row in instances:
        if row[0] == "tail_acid_2":
            row[1] = "fixture:tail_acid_1"
    assert profiles.assess(*args)["profile_qualified"]
    instances[:] = [r for r in instances if r[0] != "tail_acid_2"]
    for row in instances:
        if row[0] == "tail_acid_1":
            row[2] = 2
    assert not profiles.assess(*args)["profile_qualified"]


def test_unknown_profile_abstains_without_parsing_structures(profiles):
    args = example(profiles)
    args[3]["design_lane"] = "arbitrary_cartesian_pool"
    args[4].clear()
    assert not profiles.assess(*args)["profile_qualified"]


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, ROOT / "configs/multireaction/compose_lipid_b5_replay_v1.json")


def replay_example(profiles, label):
    p, construction, metadata, task, structures = example(profiles, label)
    control = next(c for c in json.loads(SOURCE.read_text())["controls"] if c["label"] == label)
    target = control["expected_product"]
    p["constitution_id"] = hashlib.sha256(constitutional_molecule(target)[0].encode()).hexdigest()
    item = {"preparation": p, "source": {"constitution": target, "primary_metadata": metadata}}
    original = {
        "preparation": p,
        "construction": construction,
        "primary_metadata": metadata,
        "original_task": task,
    }
    return item, original, structures


@pytest.mark.parametrize("label", ["I71", "I81", "I91", "I93", "I95", "I97"])
def test_source_profile_replay_matches_all_independent_source_endpoints(contract, label):
    profiles, programs = contract[2:4]
    args = replay_example(profiles, label)
    before = copy.deepcopy(args)
    result = replay_record(*args, profiles, programs)
    assert result["computed_consistency_pass"]
    assert result["binding"]["profile_qualified"]
    assert all(result["replay"]["checks"].values())
    assert result["verified_target_constitution_id"] == args[0]["preparation"]["constitution_id"]
    assert args == before


def test_failed_input_scope_never_runs_program_or_parses_product(contract):
    profiles, _ = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    item["source"]["constitution"] = "invalid-do-not-parse"
    original["original_task"]["events"].reverse()
    result = replay_record(item, original, structures, profiles, {})
    assert not result["computed_consistency_pass"]
    assert "replay" not in result


def test_protected_record_cannot_reach_source_profile_replay(contract):
    profiles, programs = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    item["preparation"]["eligible_for_program_preparation"] = False
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record(item, original, {}, profiles, programs)


def test_bad_product_never_changes_the_source_input_profile(contract):
    profiles, programs = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    item["source"]["constitution"] = "CCO"
    result = replay_record(item, original, structures, profiles, programs)
    assert result["binding"]["profile_qualified"]
    assert not result["computed_consistency_pass"]


def test_exact_replay_must_agree_with_protected_preparation_identity(contract):
    profiles, programs = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    item["preparation"]["constitution_id"] = "0" * 64
    with pytest.raises(ComposeLipidError, match="identity"):
        replay_record(item, original, structures, profiles, programs)


def test_source_metadata_substitution_fails_loudly(contract):
    profiles, programs = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    item["source"]["primary_metadata"] = {"series": "I7"}
    with pytest.raises(ComposeLipidError, match="substitution"):
        replay_record(item, original, structures, profiles, programs)


@pytest.mark.parametrize("change", ["stage", "formula", "product", "missing_program"])
def test_primary_source_control_gates_cannot_be_skipped(contract, change):
    document = json.loads(SOURCE.read_text())
    registry = json.loads(REGISTRY.read_text())
    if change == "stage":
        document["controls"][0]["expected_stage_products"][0] = "CCO"
    elif change == "formula":
        document["controls"][0]["expected_formula"] = "C2H6O"
    elif change == "product":
        document["controls"][0]["expected_product"] = "CCO"
    else:
        document["controls"].pop()
    with pytest.raises(ComposeLipidError):
        qualify_controls(document, registry, contract[2], contract[3])


@pytest.fixture
def evidence_row(contract):
    profiles, programs = contract[2:4]
    item, original, structures = replay_example(profiles, "I81")
    prepared = item["preparation"]
    prepared.update(
        target_id="fixture-target",
        old_split="train",
        corrected_split="train",
        exclusion_reasons=[],
        pending_reasons=[],
        construction_basis="v8_exact_family_task",
    )
    replay = replay_record(item, original, structures, profiles, programs)
    row = {
        k: prepared[k]
        for k in (
            "target_id",
            "family",
            "constitution_id",
            "component_instances",
            "construction_basis",
        )
    }
    row.update(
        result=replay, training_admitted=False, evidence_basis="computed_transform_consistency"
    )
    return copy.deepcopy(row), copy.deepcopy(prepared), copy.deepcopy(replay["binding"])


def test_profiled_evidence_merger_requires_all_original_and_staged_checks(evidence_row):
    assert validate_profiled_row(*evidence_row)


@pytest.mark.parametrize(
    "key,value",
    [
        ("old_split", "test"),
        ("corrected_split", "calibration"),
        ("exclusion_reasons", ["held_component"]),
        ("pending_reasons", ["unassigned_partition"]),
        ("eligible_for_program_preparation", False),
    ],
)
def test_profiled_evidence_cannot_release_protected_or_unassigned_rows(evidence_row, key, value):
    row, prepared, binding = evidence_row
    prepared[key] = value
    with pytest.raises(ComposeLipidError):
        validate_profiled_row(row, prepared, binding)


@pytest.mark.parametrize(
    "key", ["target_id", "family", "constitution_id", "component_instances", "construction_basis"]
)
def test_profiled_evidence_cannot_change_original_identity_or_quantities(evidence_row, key):
    row, prepared, binding = evidence_row
    row[key] = [] if key == "component_instances" else "changed"
    with pytest.raises(ComposeLipidError):
        validate_profiled_row(row, prepared, binding)


@pytest.mark.parametrize(
    "key",
    [
        "source_events_agree",
        "complete_component_scope",
        "tail_axis_identity",
        "complete_original_component_structure_agreement",
        "matched_body_identity",
    ],
)
def test_forged_exact_flag_cannot_override_failed_source_profile(evidence_row, key):
    row, prepared, binding = evidence_row
    binding["checks"][key] = False
    row["result"]["binding"] = copy.deepcopy(binding)
    with pytest.raises(ComposeLipidError):
        validate_profiled_row(row, prepared, binding)


@pytest.mark.parametrize(
    "key",
    [
        "complete_search",
        "every_stage_replayed",
        "unique_each_forward_stage",
        "unique_forward_exact",
        "unique_complete_inverse",
        "full_element_hydrogen_charge_balance",
        "terminal_constraints",
    ],
)
def test_complete_input_scope_cannot_override_a_failed_staged_gate(evidence_row, key):
    row, prepared, binding = evidence_row
    row["result"]["replay"]["checks"][key] = False
    with pytest.raises(ComposeLipidError):
        validate_profiled_row(row, prepared, binding)


@pytest.mark.parametrize(
    "change",
    [
        "training",
        "execution",
        "basis",
        "profile_substitution",
        "bad_target_identity",
        "nonboolean_exact",
        "missing_inverse",
        "unqualified_replayed",
        "unqualified_promoted",
    ],
)
def test_profiled_evidence_tier_and_payload_cannot_be_promoted(evidence_row, change):
    row, prepared, binding = evidence_row
    result = row["result"]
    if change == "training":
        row["training_admitted"] = True
    elif change == "execution":
        result["experimental_execution_admitted"] = True
    elif change == "basis":
        row["evidence_basis"] = "exact_executed_characterized"
    elif change == "profile_substitution":
        result["binding"]["program_id"] = "source_b5_I7"
    elif change == "bad_target_identity":
        result["verified_target_constitution_id"] = "0" * 64
    elif change == "nonboolean_exact":
        result["computed_consistency_pass"] = 1
    elif change == "missing_inverse":
        result["replay"]["inverse"]["candidate_components"] = []
    else:
        binding["profile_qualified"] = False
        result["binding"] = copy.deepcopy(binding)
        if change == "unqualified_replayed":
            result["computed_consistency_pass"] = False
            result["replay"]["computed_consistency_pass"] = False
    with pytest.raises(ComposeLipidError):
        validate_profiled_row(row, prepared, binding)
