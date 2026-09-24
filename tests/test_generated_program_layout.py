from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.generated_program_layout import compile_generated_layouts, layout_signatures
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    semantic = json.loads(
        (ROOT / "configs/multireaction/combinatorial_program_semantics_v1.json").read_text()
    )
    dataset = json.loads(
        (ROOT / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    libraries = load_assembly_libraries(
        [
            (ROOT / semantic["inputs"][k]["path"], semantic["inputs"][k]["sha256"])
            for k in semantic["registries"]
        ],
        expected_families=semantic["programs"],
    )
    ugi = load_qualified_forward_reaction(
        ROOT / semantic["inputs"]["ugi_registry"]["path"],
        ROOT / semantic["inputs"]["ugi_variant"]["path"],
        reaction_id="ugi_3cr_agile",
    )
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        records = {
            f: cache.record(int(cache.indices(program_id=f, fold="train")[0])) for f in libraries
        }
        vocabulary, atoms = cache.vocabulary, cache.atom_vocabulary

    def compile(record, **changes):
        args = dict(
            smiles=record.graph.canonical_smiles,
            depth=record.program_depth,
            adapter=libraries[record.program_id],
            program_policy=dataset["programs"][record.program_id],
            semantic_policy=semantic["programs"][record.program_id],
            limits=dataset["limits"],
            support=semantic["support_bounds"],
            vocabulary=vocabulary,
            atoms=atoms,
            ugi_reaction=ugi,
        )
        return compile_generated_layouts(**{**args, **changes})

    return records, compile


def test_all_twelve_source_programs_reconstruct_without_source_layout_input(source):
    records, compile = source
    assert len(records) == 12
    for family, original in records.items():
        result = compile(original)
        assert result.status == "qualified", (family, result)
        assert len(result.records) == len(result.witnesses) == result.witness_count
        expected = layout_signatures(original, maximum_closures=1)
        assert expected in [layout_signatures(r, maximum_closures=1) for r in result.records]
        assert all(
            r.graph.canonical_smiles == original.graph.canonical_smiles for r in result.records
        )
        assert all(w["exact_forward_replay"] for w in result.witnesses)


def test_distinct_exact_disulfide_programs_remain_explicit_alternatives(source):
    records, compile = source
    result = compile(records["disulfide_coupling"])
    assert len(result.records) > 1
    assert len({tuple(r.role_states) for r in result.records}) > 1
    assert len({json.dumps(w["components"], sort_keys=True) for w in result.witnesses}) > 1


def test_ambiguity_in_any_single_witness_abstains_instead_of_dropping_it(source, monkeypatch):
    from forge.model import generated_program_layout as module

    records, compile = source
    original = module.trace_library_atom_semantics
    calls = []

    def ambiguous(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise LibraryAssemblyError("ambiguous atom semantics within witness")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "trace_library_atom_semantics", ambiguous)
    result = compile(records["disulfide_coupling"])
    assert result.status == "semantic_abstention" and result.records == ()
    assert "within witness" in result.reason


def test_graph_support_and_reaction_identity_remain_strict(source):
    records, compile = source
    record = records["urea_amine_isocyanate"]
    assert compile(record, smiles="C.C").status == "invalid_graph"
    assert compile(record, smiles="CCCC").status == "no_exact_program"
    assert (
        compile(record, support={"maximum_heavy_atoms": 1, "maximum_closures": 1}).status
        == "outside_graph_support"
    )


def test_input_atom_order_cannot_change_compiled_layouts(source):
    records, compile = source
    record = records["ugi_3cr_agile"]
    molecule = Chem.MolFromSmiles(record.graph.canonical_smiles)
    reordered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    alternate = Chem.MolToSmiles(reordered, canonical=False)
    a, b = compile(record), compile(record, smiles=alternate)
    assert a.status == b.status == "qualified"
    assert [layout_signatures(r, maximum_closures=1) for r in a.records] == [
        layout_signatures(r, maximum_closures=1) for r in b.records
    ]


def test_masked_atom_targets_and_ignored_morphology_do_not_create_new_model_context(source):
    records, _ = source
    record = records["amide_coupling_acid_amine"]
    expected = layout_signatures(record, maximum_closures=1)
    changed_nodes = record.graph.node_states.copy()
    changed_nodes[~record.fixed_atom_mask] += 1
    changed = replace(record, graph=replace(record.graph, node_states=changed_nodes))
    assert layout_signatures(changed, maximum_closures=1) == expected
    morphology = derive_role_morphology_states(record)
    modified = replace(record, role_morphology_states=morphology + np.ones_like(morphology))
    observed = layout_signatures(modified, maximum_closures=1)
    assert observed["sampler_context"] != expected["sampler_context"]
    assert observed["sparse_flow_context"] == expected["sparse_flow_context"]


def test_program_expansion_replays_all_families_and_abstains_on_enumeration_limit(source):
    from forge.assembly.library_programs import LibraryProgramLimits
    from forge.model.generated_program_expansion import expand_generated_program

    records, compile = source
    dataset = json.loads(
        (ROOT / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    libraries = load_assembly_libraries(
        [
            (ROOT / dataset["inputs"][k]["path"], dataset["inputs"][k]["sha256"])
            for k in dataset["registries"]
        ],
        expected_families=dataset["programs"],
    )
    for family, record in records.items():
        witness = compile(record).witnesses[0]
        policy = dataset["programs"][family]
        limits = LibraryProgramLimits(policy["maximum_steps"], **dataset["limits"])
        args = dict(depth=record.program_depth, accumulator_role=policy["accumulator_role"])
        result = expand_generated_program(
            libraries[family], witness["components"], limits=limits, **args
        )
        assert result.status == "complete"
        assert record.graph.canonical_smiles in {p[-1] for p in result.paths}
        assert all(len(p) == record.program_depth for p in result.paths)
        incomplete = expand_generated_program(
            libraries[family],
            witness["components"],
            limits=replace(limits, maximum_outcomes=1),
            **args,
        )
        assert incomplete.status == "abstain" and incomplete.reason == "outcome_limit"
        assert incomplete.paths == ()


@pytest.mark.parametrize(
    "module_name",
    [
        "combinatorial_generated_layouts",
        "combinatorial_context_expansion",
        "combinatorial_count_layout_generation",
    ],
)
def test_saved_layout_verifiers_reject_changed_policy(tmp_path, module_name):
    import importlib

    module = importlib.import_module("experiments.phase1.multireaction." + module_name)
    # Policy checking precedes all computation and input access.
    result = {
        "schema_version": module.SCHEMA,
        "status": "numerical_complete",
        "policy": {**module.POLICY, "gate_changes": True},
        "sources": [],
    }
    output = tmp_path / "result.json"
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="contract"):
        module.verify(ROOT, output)
