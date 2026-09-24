"""Reported-source role synonyms require an explicit whole-tuple, source-lane binding."""

import copy
import json
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_fixed_profiles import load_contract, select_executor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def executor():
    c = load_contract(
        ROOT, ROOT / "configs/multireaction/compose_lipid_supplied_ketone_ugi4_profiles_v1.json"
    )
    return c[2]["ketone_ugi4"]


def item(executor):
    p = executor["source_role_profiles"][0]
    return {
        "source": {
            "primary_metadata": copy.deepcopy(p["required_source_metadata"]),
            "constitution": "not parsed for binding",
        },
        "preparation": {
            "component_instances": [
                [r, "same-id-not-inferred-from-target", 1]
                for r in p["registry_to_source_roles"].values()
            ]
        },
    }


def test_exact_profile_binds_semantic_roles_without_rewriting_original_instances(executor):
    source = item(executor)
    before = copy.deepcopy(source)
    selected = select_executor(source, executor)
    assert selected["mapping"]["amine_head"] == "amine"
    assert selected["mapping"]["coupled_ketone"] == "ketone"
    assert source == before
    assert selected["run"] is executor["run"]


@pytest.mark.parametrize(
    "change", ["lane", "release", "extra_role", "duplicate_role", "partial_alias"]
)
def test_role_labels_alone_are_insufficient_for_alias_binding(executor, change):
    source = item(executor)
    if change == "lane":
        source["source"]["primary_metadata"]["evidence_lane"] = "another_lane"
    elif change == "release":
        source["source"]["primary_metadata"].pop("source_release")
    elif change == "extra_role":
        source["preparation"]["component_instances"].append(["extra", "x", 1])
    elif change == "duplicate_role":
        source["preparation"]["component_instances"].append(
            copy.deepcopy(source["preparation"]["component_instances"][0])
        )
    else:
        source["preparation"]["component_instances"][0][0] = "amine_head"
    assert select_executor(source, executor) is executor


def test_overlapping_profiles_fail_instead_of_selecting_by_target(executor):
    bad = {**executor, "source_role_profiles": executor["source_role_profiles"] * 2}
    with pytest.raises(ComposeLipidError, match="Ambiguous"):
        select_executor(item(executor), bad)


def test_canonical_rows_keep_canonical_binding(executor):
    source = item(executor)
    source["preparation"]["component_instances"] = [
        [r, "id", 1] for r in executor["mapping"].values()
    ]
    assert select_executor(source, executor) is executor


def test_alias_report_preserves_all_seven_original_complete_tuples():
    report = json.loads(
        (
            ROOT / "results/phase1/compose_lipid_ketone_ugi4_source_v1/source-role-alias-audit.json"
        ).read_text()
    )
    assert len(report["rows"]) == 7
    for row in report["rows"]:
        assert {r for r, _, _ in row["component_instances"]} == {
            "amine",
            "ketone",
            "carboxylic_acid",
            "isocyanide",
        }
        assert all(q == 1 for _, _, q in row["component_instances"])
    assert not report["training_admitted"]
