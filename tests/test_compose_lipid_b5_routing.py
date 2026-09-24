"""Generation lanes cannot override exact source precursor pairs or original provenance."""

import copy
import hashlib
import json

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_b5_readiness import validate_profiled_row
from forge.corpus.compose_lipid_b5_replay import replay_record
from forge.corpus.compose_lipid_b5_replay_v2 import load_contract
from forge.corpus.compose_lipid_b5_routing import B5SourcePairProfiles
from tests.test_compose_lipid_b5_profiles import ROOT, SOURCE, example

REGISTRY = ROOT / "data/vendor/qualified_b5_source_pair_routing_v2.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, ROOT / "configs/multireaction/compose_lipid_b5_replay_v2.json")


@pytest.fixture(scope="module")
def profiles(contract):
    return contract[2]


def original_lane(profiles, label):
    args = example(profiles.base, label)
    kind = "amine" if label == "I95" else "alcohol"
    lane = "one_tail_acid_knob_" + kind
    args[2]["design_lane"] = args[3]["design_lane"] = lane
    args[1]["reaction_steps_sites"]["task"]["design_lane"] = lane
    return args


@pytest.mark.parametrize("label", ["I95", "I97"])
def test_exact_published_pair_selects_homo_without_rewriting_original(profiles, label):
    args = original_lane(profiles, label)
    before = copy.deepcopy(args)
    baseline = profiles.base.assess(*args)
    assert not baseline["profile_qualified"]
    assert {k for k, v in baseline["checks"].items() if not v} == {"tail_axis_identity"}
    result = profiles.assess(*args)
    assert result["profile_qualified"]
    assert "homo" in result["program_id"]
    assert result["source_pair_routing"]["baseline_binding"] == baseline
    assert not result["source_pair_routing"]["target_used_for_selection"]
    assert args == before


@pytest.mark.parametrize("label", ["I71", "I81", "I91", "I93", "I95", "I97"])
def test_previously_qualified_source_profiles_unchanged(profiles, label):
    args = example(profiles.base, label)
    assert profiles.assess(*args) == profiles.base.assess(*args)


def test_selection_never_reads_target(profiles):
    args = original_lane(profiles, "I95")
    first = profiles.assess(*args)
    args[1]["constitution"] = "invalid-target-must-not-be-parsed"
    args[0]["constitution_id"] = "not-a-target"
    assert profiles.assess(*args) == first


@pytest.mark.parametrize("changed_role", ["tail_acid_1", "tail_acid_2", "both"])
def test_other_reported_tail_pairs_are_not_routed(profiles, changed_role):
    args = original_lane(profiles, "I95")
    wrong = example(profiles.base, "I91")[4]["fixture:tail_acid_1"]
    roles = ["tail_acid_1", "tail_acid_2"] if changed_role == "both" else [changed_role]
    spec = profiles.base.specification["profiles"]["one_tail_acid_knob_amine"]
    for role in roles:
        args[4]["fixture:" + role] = wrong
        index = next(i for i, (r, _) in enumerate(spec["reagent_order"]) if r == role)
        args[3]["reagents"][index]["smiles"] = wrong
    result = profiles.assess(*args)
    assert "source_pair_routing" not in result
    assert result == profiles.base.assess(*args)


@pytest.mark.parametrize(
    "mutation",
    [
        "quantity",
        "metadata",
        "events",
        "task_id",
        "training_flag",
        "structure",
        "source_stereo",
        "head_alias",
        "tail_axis",
        "head_axis",
    ],
)
def test_pair_does_not_override_other_source_failures(profiles, mutation):
    args = original_lane(profiles, "I95")
    prepared, construction, metadata, task, structures = args
    if mutation == "quantity":
        prepared["component_instances"][0][2] = 2
    elif mutation == "metadata":
        metadata["head_id"] = "changed"
    elif mutation == "events":
        task["events"].reverse()
    elif mutation == "task_id":
        construction["task_id"] = "changed"
    elif mutation == "training_flag":
        task["training_admissible"] = True
    elif mutation == "structure":
        structures["fixture:tail_acid_1"] = "CCCC(=O)O"
    elif mutation == "source_stereo":
        task["reagents"][0]["smiles"] = task["reagents"][0]["smiles"].replace("@@", "@")
    elif mutation == "head_alias":
        construction["declared_precursor_identifiers"][0]["value"] = "changed"
    elif mutation == "tail_axis":
        metadata["tail_axis"] = task["tail_axis"] = "invented__reported_tail_acid"
    elif mutation == "head_axis":
        metadata["head_axis"] = task["head_axis"] = "invented"
    result = profiles.assess(*args)
    assert not result["profile_qualified"]
    assert "source_pair_routing" not in result


def test_protected_record_rejected_before_decoding(profiles):
    args = original_lane(profiles, "I95")
    args[0]["eligible_for_program_preparation"] = False
    args[4].clear()
    with pytest.raises(ComposeLipidError, match="Protected"):
        profiles.assess(*args)


@pytest.mark.parametrize("label", ["I95", "I97"])
def test_routed_control_passes_all_staged_gates_and_strict_merge(contract, label):
    profiles, programs = contract[2:4]
    args = original_lane(profiles, label)
    p, construction, metadata, task, structures = args
    control = next(c for c in json.loads(SOURCE.read_text())["controls"] if c["label"] == label)
    target = control["expected_product"]
    p.update(
        constitution_id=hashlib.sha256(constitutional_molecule(target)[0].encode()).hexdigest(),
        target_id="fixture-target",
        construction_basis="v8_exact_family_task",
        old_split="train",
        corrected_split="train",
        exclusion_reasons=[],
        pending_reasons=[],
    )
    original = dict(
        preparation=p, construction=construction, primary_metadata=metadata, original_task=task
    )
    item = {"preparation": p, "source": {"primary_metadata": metadata, "constitution": target}}
    result = replay_record(item, original, structures, profiles, programs)
    assert result["computed_consistency_pass"]
    row = {
        k: p[k]
        for k in (
            "target_id",
            "family",
            "constitution_id",
            "construction_basis",
            "component_instances",
        )
    }
    row.update(
        result=result, training_admitted=False, evidence_basis="computed_transform_consistency"
    )
    assert validate_profiled_row(row, p, profiles.assess(*args))
    row["result"]["binding"]["source_pair_routing"]["target_used_for_selection"] = True
    with pytest.raises(ComposeLipidError, match="source profile"):
        validate_profiled_row(row, p, profiles.assess(*args))


@pytest.mark.parametrize("mutation", ["pair", "profile", "duplicate", "quantity", "axis"])
def test_overlay_cannot_expand_source_scope(tmp_path, mutation):
    overlay = json.loads(REGISTRY.read_text())
    rule = overlay["routes"][0]
    if mutation == "pair":
        rule["required_components"]["tail_acid_1"] = "CCCC(=O)O"
    elif mutation == "profile":
        rule["source_profile"] = "matched_identical_tail_acids_alcohol"
    elif mutation == "duplicate":
        overlay["routes"].append(copy.deepcopy(rule))
    elif mutation == "quantity":
        rule["minimum_reported_tails"] = 1
    elif mutation == "axis":
        rule["task_fields"] = {}
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(overlay))
    with pytest.raises(ComposeLipidError):
        B5SourcePairProfiles.from_registry(ROOT, path, expected_sha256=sha256_file(path))


def test_overlay_checksum_required():
    with pytest.raises(ComposeLipidError, match="checksum"):
        B5SourcePairProfiles.from_registry(ROOT, REGISTRY, expected_sha256="0" * 64)
