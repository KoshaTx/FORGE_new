"""Independent quality diagnostics preserve unknown chemistry and bounded support."""

import math

import pytest
from rdkit import Chem

from forge.model.compose_lipid_quality import (
    StructuralSupport,
    canonical_molecule,
    fingerprint_distribution,
    identity_diversity,
    representation_diagnostics,
    ring_systems,
    source_role_mapping,
)


def test_unknown_environment_is_not_reported_as_invalid_or_qualified():
    support = StructuralSupport()
    support.add("CSC")
    familiar = support.assess(Chem.MolFromSmiles("CSC"))
    assert familiar["all_local_features_observed"] is True
    sulfurane = canonical_molecule("CS(C)(C)(C)(C)C")
    assert sulfurane is not None
    unknown = support.assess(sulfurane[1])
    assert unknown["unknown_atom_environments"]
    assert unknown["all_local_features_observed"] is False
    assert unknown["status"] == "assessed"
    assert "chemically_invalid" not in unknown
    assert "quality_pass" not in unknown


def test_complete_neighborhood_detects_unseen_combinations_of_seen_bonds():
    support = StructuralSupport()
    support.add("COC")
    support.add("CO")
    assert support.assess(Chem.MolFromSmiles("OC(O)(O)O"))["unknown_atom_environments"]
    assert support.assess(Chem.MolFromSmiles("CO"))["all_local_features_observed"]


def test_ring_systems_are_whole_and_atom_order_invariant():
    spiro = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
    reordered = Chem.RenumberAtoms(spiro, list(reversed(range(spiro.GetNumAtoms()))))
    assert ring_systems(spiro) == ring_systems(reordered)
    assert len(ring_systems(spiro)) == 1
    assert len(ring_systems(Chem.MolFromSmiles("C1CCCCC1CC2CCCCC2"))) == 2
    support = StructuralSupport()
    support.add("C1CCCCC1")
    assert support.assess(spiro)["unknown_ring_systems"]


def test_reference_multiplicity_does_not_change_support():
    support = StructuralSupport()
    for _ in range(25):
        support.add("OCC")
    assert len(support.identities) == 1
    assert support.assess(Chem.MolFromSmiles("CCO"))["all_local_features_observed"]
    with pytest.raises(ValueError):
        support.add("CC.O")
    assert (
        StructuralSupport().assess(Chem.MolFromSmiles("CC"))["all_local_features_observed"] is None
    )


def test_diversity_retains_attempt_denominator_and_distinct_effective_counts():
    result = identity_diversity(["A", "A", "B"], requests=6)
    assert result["unique_per_request"] == pytest.approx(1 / 3)
    assert result["simpson_effective_count"] == pytest.approx(1.8)
    assert result["shannon_effective_count"] == pytest.approx(
        math.exp(-2 / 3 * math.log(2 / 3) - 1 / 3 * math.log(1 / 3))
    )
    assert identity_diversity([], requests=6)["simpson_effective_count"] == 0
    with pytest.raises(ValueError):
        identity_diversity(["A", "B"], requests=1)


def test_full254_atom_and_bromine_support_is_preserved():
    mol = Chem.MolFromSmiles("C" * 253 + "Br")
    result = representation_diagnostics(
        mol, elements={"C", "Br"}, maximum_heavy_atoms=254, maximum_cycles=12
    )
    assert result["heavy_atoms"] == 254
    assert result["within_declared_size_element_cycle_bounds"]
    larger = Chem.MolFromSmiles("C" * 254 + "Br")
    assert not representation_diagnostics(
        larger, elements={"C", "Br"}, maximum_heavy_atoms=254, maximum_cycles=12
    )["within_declared_size_element_cycle_bounds"]


def test_fingerprint_precision_has_all_request_denominator():
    reference = ["CC", "CCC", "CCCC", "CCO", "CCN", "CCOC", "CCCO"]
    result = fingerprint_distribution(["CC", "CC", "CCC"], reference, requests=10, neighbors=1)
    assert result["unique_generated"] == 2
    assert result["fingerprint_precision_among_unique"] == 1
    assert result["distinct_in_reference_manifold_per_request"] == pytest.approx(0.2)
    assert result["fingerprint_coverage"] == pytest.approx(5 / 7)
    assert result["fingerprint_recall"] == pytest.approx(1.0)
    assert fingerprint_distribution([], reference, requests=10)["status"].startswith("unassessed")


def test_source_role_mapping_uses_only_the_accepted_ordered_contract():
    components = {"thiol_first": "CS", "thiol_second": "CS"}
    mapping = {"thiol_first": "thiol_tail", "thiol_second": "thiol_tail"}
    executors = [{"mapping": {"unrelated": "other"}}, {"mapping": mapping}]
    assessment = {"checks": [{"accepted_components": []}, {"accepted_components": [components]}]}
    assert source_role_mapping(components, assessment, executors) == mapping
    with pytest.raises(ValueError, match="order"):
        source_role_mapping(components, assessment, executors[:1])
    assessment["checks"][0]["accepted_components"] = [components]
    with pytest.raises(ValueError, match="ambiguous"):
        source_role_mapping(components, assessment, executors)
