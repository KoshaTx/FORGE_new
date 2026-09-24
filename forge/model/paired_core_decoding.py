"""Decode two terminal core conditions from one unchanged neural trajectory."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.synthesis_program_sampling import (
    _fixed_state_exact_records,
    _terminal_smiles,
    decode_synthesis_program_strict_argmax,
    sample_synthesis_program_products,
)


def paired_terminal_core_decode(
    model, baseline, scaffolded, atoms, nodes, bonds, *, seed, flow_steps
):
    """One bounded batch: capture terminal scores, verify identity control, decode added cores."""
    if not baseline or len(baseline) != len(scaffolded):
        raise ValueError("paired terminal populations differ")
    for a, b in zip(baseline, scaffolded, strict=True):
        if (
            (
                a.program_id,
                a.program_depth,
                a.graph.structure_id,
                a.node_count,
                a.graph.closure_count,
                a.component_blocks,
            )
            != (
                b.program_id,
                b.program_depth,
                b.graph.structure_id,
                b.node_count,
                b.graph.closure_count,
                b.component_blocks,
            )
            or not np.array_equal(a.role_states, b.role_states)
            or not np.array_equal(a.core_position_states, b.core_position_states)
        ):
            raise ValueError("paired core conditions must share identical layout coordinates")
        exterior = a.core_position_states == 1
        if not np.array_equal(
            a.fixed_atom_mask[exterior], b.fixed_atom_mask[exterior]
        ) or not np.array_equal(a.graph.node_states[exterior], b.graph.node_states[exterior]):
            raise ValueError("paired core condition changed exterior atom constraints")
    predictions, calls = None, 0

    def capture(_module, _args, output):
        nonlocal predictions, calls
        calls += 1
        # Only the most recent output is retained; memory is bounded by one neural batch.
        predictions = {k: v.detach().clone() for k, v in output.items()}

    hook = model.register_forward_hook(capture)
    try:
        original, sampler = sample_synthesis_program_products(
            model,
            baseline,
            atoms,
            nodes,
            bonds,
            samples_per_program=1,
            sample_steps=flow_steps,
            batch_size=len(baseline),
            seed=seed,
            device="cpu",
            terminal_decode_policy="strict_valence_topology_argmax",
        )
    finally:
        hook.remove()
    if predictions is None or calls != flow_steps + 1 or sampler["fixed_state_failures"]:
        raise ValueError("terminal capture or trajectory accounting failed")
    digest = hashlib.sha256()
    for key in sorted(predictions):
        value = predictions[key].cpu().numpy()
        digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())

    def decode(records):
        layout = collate_synthesis_program_layouts(records, maximum_closures=model.maximum_closures)
        terminal, reasons = decode_synthesis_program_strict_argmax(
            predictions, layout, records, atoms
        )
        if not _fixed_state_exact_records(terminal, records):
            raise ValueError("paired decoder changed a fixed coordinate")
        output = []
        for i, (r, reason) in enumerate(zip(records, reasons, strict=True)):
            smiles = (
                None
                if reason
                else _terminal_smiles(terminal, i, r.node_count, r.graph.closure_count, atoms)
            )
            output.append(
                {
                    "sample_index": i,
                    "program_id": r.program_id,
                    "layout_record_id": r.graph.structure_id,
                    "canonical_smiles": smiles,
                    "valid": smiles is not None,
                    "constraint_abstention_reason": reason,
                }
            )
        return output

    with torch.inference_mode():
        control = decode(baseline)
        for row, source in zip(control, original, strict=True):
            if any(source[k] != v for k, v in row.items()):
                raise ValueError("captured-score identity control differs from frozen sampler")
        treatment = decode(scaffolded)
    return (
        control,
        treatment,
        {
            "terminal_scores_sha256": digest.hexdigest(),
            "model_forward_calls": calls,
            "identical_trajectory_and_terminal_scores": True,
            "identity_control_exact": True,
            "fixed_states_exact": True,
        },
    )
