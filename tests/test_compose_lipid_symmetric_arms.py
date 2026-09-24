import json
from pathlib import Path

from rdkit import Chem

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.generated_source import evaluate_executor
from forge.assembly.program_atom_origins import trace_repeated_program
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import sha256_file
from forge.model.compose_lipid_symmetric_arms import (
    construct_symmetric_role_arms,
    template_automorphisms,
)
from tests.test_compose_lipid_component_constraints import layout_from_record
from tests.test_compose_lipid_generation import gold_predictions
from tests.test_source_instance_coordinates import record


def test_registry_symmetric_internal_arms_preserve_size_and_full_replay():
    path = Path("data/vendor/qualified_maleate_ester_source_program_v1.json")
    reaction = json.loads(path.read_text())["reactions"][0]
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=reaction["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    template = adapter.reaction.forward.GetReactantTemplate(1)
    assert len(template_automorphisms(template)) == 2
    assert not template_automorphisms(template, maximum=1)
    components = dict(amine_head="NCCN(C)C", maleate="CCCCOC(=O)C=CC(=O)OCCCC")
    bounds = RepeatBounds(maximum_events=1, maximum_outcomes=256)
    traced = trace_repeated_program(
        adapter,
        components,
        accumulator_role="amine_head",
        events=1,
        source_roles={r: r for r in adapter.roles},
        bounds=bounds,
    )
    assert traced.annotations is not None
    a = traced.annotations
    r, vocabulary, atoms = record(a.canonical_product_smiles, a.atom_roles, a.core_positions)
    layout = layout_from_record(r, {r: 1 for r in adapter.roles})
    pred = {k: v[0].numpy() for k, v in gold_predictions(r, atoms).items()}
    # Anonymous test labels use the same transform-map suffix as production labels.
    names = [v if ":map_" in v else "test:" + v for v in vocabulary.core_position_states]
    result = construct_symmetric_role_arms(layout, pred, atoms, adapter, "maleate", names)
    assert len(result["proposals"]) == 2, result
    e = dict(
        kind="repeated",
        adapter=adapter,
        mapping={r: r for r in adapter.roles},
        program=reaction["source_program"],
        bounds=bounds,
        full_source_contract=True,
    )
    for p in result["proposals"]:
        assert len(p["nodes"]) == r.node_count
        assert Chem.MolFromSmiles(p["smiles"]).GetNumAtoms() == r.node_count
        assert evaluate_executor(e, p["smiles"], events=1)["exact_registry_program_roundtrip"]
