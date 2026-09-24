from pathlib import Path

import numpy as np
import torch

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.generated_program_layout import layout_signatures
from forge.model.precursor_reuse_projection import TerminalTrace
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.source_core_completion import SourceCoreTrace, source_core_layout
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_core_layout_masks_every_precursor_exterior_target():
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        families = cache.vocabulary.program_states[1:]
        for family in families:
            record = cache.record(int(cache.indices(fold="train", program_id=family)[0]))
            scaffold, reason = source_core_layout(record)
            assert reason is None
            assert not scaffold.graph.canonical_smiles
            assert not np.any(scaffold.graph.node_states[~scaffold.fixed_atom_mask])
            assert not np.any(scaffold.graph.parents[~scaffold.fixed_parent_bond_mask])
            assert not np.any(scaffold.graph.parent_bonds[~scaffold.fixed_parent_bond_mask])
            assert scaffold.node_count == record.node_count
            assert scaffold.graph.closure_count == record.graph.closure_count
            exterior = record.core_position_states == 1
            assert np.array_equal(
                scaffold.fixed_atom_mask[exterior], record.fixed_atom_mask[exterior]
            )
            if family == "ugi_3cr_agile":
                assert (
                    layout_signatures(record, maximum_closures=1)["sparse_flow_context"]
                    == layout_signatures(scaffold, maximum_closures=1)["sparse_flow_context"]
                )
                before = collate_synthesis_program_layouts([record], maximum_closures=1)
                after = collate_synthesis_program_layouts([scaffold], maximum_closures=1)
                # Masking exterior graphs changes computed morphology. This field is ignored
                # by the sparse neural flow and by the base terminal decoder used here.
                assert all(
                    torch.equal(before[k], after[k])
                    for k in before
                    if k != "role_morphology_states"
                )


def test_core_trace_preserves_original_neural_outputs_and_terminal_states():
    with SynthesisProgramProductionCache(
        ROOT / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        records = cache.records(
            [
                int(cache.indices(fold="train", program_id=f)[0])
                for f in cache.vocabulary.program_states[1:]
            ]
        )
    model, _, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
        ROOT / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json", device="cpu"
    )
    plain, core = TerminalTrace(model, records, atoms), SourceCoreTrace(model, records, atoms)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        outputs = [
            sample_synthesis_program_products(
                trace,
                records,
                atoms,
                nodes,
                bonds,
                samples_per_program=1,
                sample_steps=2,
                batch_size=4,
                seed=9001,
                device="cpu",
                terminal_decode_policy="strict_valence_topology_argmax",
            )[0]
            for trace in (plain, core)
        ]
    finally:
        torch.set_num_threads(threads)
    assert outputs[0] == outputs[1]
    assert plain.terminals == core.terminals
    assert len(core.core_rows) == len(records) == 12
