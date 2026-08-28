from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior, _fixed_signature

REPO = Path(__file__).resolve().parents[1]
PROGRAM = "bl_2023_repeated_aza_michael"


def _bundle(record):
    blocks = tuple(
        sorted(
            Counter(
                (
                    block.role_state,
                    tuple(
                        sorted(
                            Counter(
                                int(value)
                                for value in record.core_position_states[block.start : block.stop]
                                if int(value) > 1
                            ).items()
                        )
                    ),
                )
                for block in record.component_blocks
            ).items()
        )
    )
    return record.program_depth, blocks, _fixed_signature(record)


def test_full_corpus_bl_core_representation_is_fail_closed() -> None:
    result = json.loads(
        (REPO / "results/phase1/bl_core_constrained_representation_v1/result.json").read_text()
    )
    bl = result["programs"][PROGRAM]
    assert result["status"] == "pass"
    assert result["gates"]["fixed_core_policy_exact"] is True
    assert bl["records_represented"] == 610
    assert bl["core_atom_rows"] == bl["fixed_atom_rows"] == 6350
    assert bl["fixed_policy_failures"] == 0
    assert result["summary"]["component_identifiers_used"] is False
    assert result["summary"]["fragment_tokens_used"] is False


def test_packed_bl_core_cache_and_factorized_layout_preserve_semantic_support() -> None:
    result = json.loads(
        (REPO / "results/phase1/bl_core_constrained_production_cache_v1/result.json").read_text()
    )
    assert result["status"] == "pass"
    assert result["fixed_state_failures"] == 0
    cache_path = REPO / "results/phase1/bl_core_constrained_production_cache_v1/cache.npz"
    with SynthesisProgramProductionCache(cache_path) as cache:
        bl_records = cache.records(cache.indices(program_id=PROGRAM, fold="train"))
        assert all(
            np.array_equal(record.fixed_atom_mask, record.core_position_states > 1)
            for record in bl_records
        )
        assert all(
            np.any(record.fixed_parent_bond_mask) or np.any(record.fixed_closure_bond_mask)
            for record in bl_records
        )
        lx_records = cache.records(
            cache.indices(program_id="lx_2024_repeated_reductive_amination", fold="train")
        )
        assert all(not np.any(record.fixed_atom_mask) for record in lx_records)

        prior = SynthesisProgramLayoutPrior(cache)
        assert prior.validate_support() == {
            "programs": 3,
            "semantic_bundles": 9,
            "component_size_support_cells": 183,
        }
        training_support = {_bundle(record) for record in bl_records}
        sampled = prior.sample(PROGRAM, sample_count=256, seed=20260824)
        assert {_bundle(record) for record in sampled}.issubset(training_support)
        failed_production_seed = 1779023141640653592
        production_draw = prior.sample(PROGRAM, sample_count=512, seed=failed_production_seed)
        assert len(production_draw) == 512
        assert all(record.graph.node_count <= prior.maximum_heavy_atoms for record in production_draw)
        assert all(
            record.graph.node_states[index]
            == prior._fixed_node_states[PROGRAM][int(record.core_position_states[index])]
            for record in sampled
            for index in np.flatnonzero(record.fixed_atom_mask)
        )
