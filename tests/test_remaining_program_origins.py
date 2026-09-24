"""Source controls, repeated participation, introduction provenance and search refusal."""

import inspect
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.grouped_atom_origins import trace_grouped_program
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.introduced_atom_origins import trace_single_introduction_program
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import resolve_pin
from forge.potency.annotations import _outcome_annotation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contracts():
    answer = {}
    for kind, filename in (
        ("han", "results/phase1/compose_lipid_han_db_source_v1/replay-config.json"),
        ("ugi3", "configs/multireaction/compose_lipid_v8_ugi3_program_v1.json"),
    ):
        cfg = json.loads((ROOT / filename).read_text())
        registry = resolve_pin(cfg["inputs"]["registry"], ROOT, label=kind)
        source = json.loads(
            resolve_pin(
                cfg["inputs"]["transform_controls" if kind == "han" else "adjudication"],
                ROOT,
                label="source controls",
            ).read_text()
        )
        if kind == "han":
            binding = next(iter(cfg["families"].values()))
            program = RegistryGroupedProgram.from_registry(
                registry,
                program_id=binding["program_id"],
                expected_sha256=cfg["inputs"]["registry"]["sha256"],
                bounds=RepeatBounds(**cfg["search_bounds"]),
            )
            mapping = binding["registry_to_source_roles"]
        else:
            program = RegistryAssemblyAdapter.from_registry(
                registry,
                reaction_id=cfg["reaction_id"],
                expected_sha256=cfg["inputs"]["registry"]["sha256"],
            )
            mapping = source["source_contract"]["registry_to_source_roles"]
        answer[kind] = program, mapping, source
    return answer


def trace(contracts, kind, *, parts=None, bounds=None, program=None, mapping=None):
    original, roles, source = contracts[kind]
    program = original if program is None else program
    parts = source["source_controls"][0]["components"] if parts is None else parts
    mapping = roles if mapping is None else mapping
    if kind == "han":
        if bounds is not None:
            program = replace(program, bounds=bounds)
        return trace_grouped_program(program, parts, source_roles=mapping)
    return trace_single_introduction_program(
        program, parts, source_roles=mapping, bounds=bounds or RepeatBounds()
    )


@pytest.mark.parametrize("index", [0, 1])
def test_han_stage_counts_survive_identical_tail_exchange(contracts, index):
    program, _, source = contracts["han"]
    control = source["source_controls"][index]
    result = trace(contracts, "han", parts=control["components"])
    original = program.replay(control["components"], control["expected_product"])
    assert original["computed_consistency_pass"]
    assert result.constitutional_products == tuple(original["forward_layers"][-1])
    assert result.complete_search and result.annotations is not None
    assert len(result.states_by_layer) == 5
    visits = Counter()
    for core in result.annotations.core_positions:
        if core != "exterior":
            for token in core.split("|"):
                position, count = token.rsplit(":visits_", 1)
                visits[position] += int(count)
    for depth, (stage, adapter) in enumerate(
        zip(program.specification["stages"], program.adapters, strict=True), start=1
    ):
        declared = {
            a.GetAtomMapNum()
            for template in adapter.reaction.forward.GetProducts()
            for a in template.GetAtoms()
            if a.GetAtomMapNum()
        }
        assert {k: v for k, v in visits.items() if k.startswith(f"step_{depth}:")} == {
            f"step_{depth}:map_{number}": stage["events"] for number in declared
        }
    assert any("visits_2" in core for core in result.annotations.core_positions)
    assert any("|" in core for core in result.annotations.core_positions)


def test_han_unprotected_competitor_never_acquires_labels(contracts):
    control = contracts["han"][2]["regression_controls"][0]
    result = trace(contracts, "han", parts=control["components"])
    assert result.complete_search and result.annotations is None
    assert result.disposition == "no_complete_forward_product"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_ugi3_agrees_with_independent_rdkit_origin_bookkeeping(contracts, index):
    adapter, mapping, source = contracts["ugi3"]
    parts = source["source_controls"][index]["components"]
    result = trace(contracts, "ugi3", parts=parts)
    assert result.complete_search and result.annotations is not None
    smiles = [parts[role] for role in adapter.roles]
    reactants = tuple(Chem.MolFromSmiles(smi) for smi in smiles)
    for outcome in adapter.reaction.forward.RunReactants(reactants):
        product = outcome[0]
        Chem.SanitizeMol(product)
        independent = _outcome_annotation(product, smiles, reactants, adapter.roles)
        assert independent["product_smiles"] == result.annotations.canonical_product_smiles
        assert (
            tuple(
                mapping.get(row["origin_role"], row["origin_role"])
                for row in independent["atom_records"]
            )
            == result.annotations.atom_roles
        )
        assert tuple(row["core_position"] or "exterior" for row in independent["atom_records"]) == (
            result.annotations.core_positions
        )
    assert result.annotations.atom_roles.count("assembly_introduced") == 1


@pytest.mark.parametrize("kind", ["han", "ugi3"])
@pytest.mark.parametrize("bound", ["maximum_outcomes", "maximum_states", "maximum_transitions"])
def test_truncated_search_never_assigns_origins(contracts, kind, bound):
    # Two duplicate raw outcomes make the transition bound independent of unique states.
    bounds = RepeatBounds(**{bound: 1})
    result = trace(contracts, kind, bounds=bounds)
    if kind == "ugi3" and bound == "maximum_transitions":
        assert result.complete_search  # Exactly one transition is within this bound.
    else:
        assert not result.complete_search and result.annotations is None
        assert result.disposition.endswith("bound")


@pytest.mark.parametrize("kind", ["han", "ugi3"])
def test_atom_order_does_not_choose_origins_or_event_order(contracts, kind):
    parts = contracts[kind][2]["source_controls"][0]["components"]
    permuted = {}
    for role, smiles in parts.items():
        molecule = Chem.MolFromSmiles(smiles)
        molecule = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        permuted[role] = Chem.MolToSmiles(molecule, canonical=False)
    assert trace(contracts, kind, parts=permuted) == trace(contracts, kind, parts=parts)
    assert (
        "target"
        not in inspect.signature(
            trace_grouped_program if kind == "han" else trace_single_introduction_program
        ).parameters
    )


@pytest.mark.parametrize("kind", ["han", "ugi3"])
def test_roles_cannot_collapse(contracts, kind):
    with pytest.raises(LibraryAssemblyError):
        trace(contracts, kind, mapping={role: "same" for role in contracts[kind][1]})


@pytest.mark.parametrize("mutation", ["lost_source", "false_introduction", "duplicate"])
def test_unresolved_origin_and_excess_raw_outcomes_cannot_hide(contracts, mutation):
    adapter = contracts["ugi3"][0]
    forward = adapter.reaction.forward

    def changed(reactants, **kwargs):
        outcomes = forward.RunReactants(reactants, **kwargs)
        if mutation == "duplicate":
            return outcomes + outcomes
        for outcome in outcomes:
            for atom in outcome[0].GetAtoms():
                if mutation == "lost_source" and atom.GetIsotope():
                    atom.SetIsotope(0)
                    break
                if mutation == "false_introduction" and not atom.GetIsotope():
                    atom.SetIntProp("react_atom_idx", 0)
                    break
        return outcomes

    modified = SimpleNamespace(
        roles=adapter.roles,
        _assess=adapter._assess,
        reaction=SimpleNamespace(
            forward=SimpleNamespace(
                GetProducts=forward.GetProducts,
                GetReactants=forward.GetReactants,
                RunReactants=changed,
            )
        ),
    )
    result = trace(contracts, "ugi3", program=modified, bounds=RepeatBounds(maximum_transitions=1))
    assert not result.complete_search and result.annotations is None
    assert result.disposition == (
        "semantic_transition_bound"
        if mutation == "duplicate"
        else "unresolved_introduced_atom_origin"
    )
