"""Replay a paired core batch with immutable-ring capacity reserved before parent selection."""

import hashlib

import numpy as np
import torch

from forge.model.fixed_closure_decoding import decode_fixed_closure_reserved_argmax
from forge.model.paired_core_decoding import paired_terminal_core_decode
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.synthesis_program_sampling import _fixed_state_exact_records, _terminal_smiles


def paired_reserved_core_decode(
    model, baseline, scaffolded, atoms, nodes, bonds, *, seed, flow_steps
):
    predictions = None

    def capture(_module, _args, output):
        nonlocal predictions
        predictions = {k: v.detach().clone() for k, v in output.items()}

    handle = model.register_forward_hook(capture)
    try:
        control, legacy, evidence = paired_terminal_core_decode(
            model, baseline, scaffolded, atoms, nodes, bonds, seed=seed, flow_steps=flow_steps
        )
    finally:
        handle.remove()
    if predictions is None:
        raise ValueError("reserved decoder did not capture terminal scores")
    digest = hashlib.sha256()
    for key in sorted(predictions):
        value = predictions[key].cpu().numpy()
        digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
        digest.update(value.tobytes())
    if digest.hexdigest() != evidence["terminal_scores_sha256"]:
        raise ValueError("reserved decoder scores differ from paired identity control")
    layout = collate_synthesis_program_layouts(scaffolded, maximum_closures=model.maximum_closures)
    with torch.inference_mode():
        terminal, reasons = decode_fixed_closure_reserved_argmax(
            predictions, layout, scaffolded, atoms
        )
    if not _fixed_state_exact_records(terminal, scaffolded):
        raise ValueError("reserved decoder altered an immutable coordinate")
    output = []
    for i, (record, reason) in enumerate(zip(scaffolded, reasons, strict=True)):
        smiles = (
            None
            if reason
            else _terminal_smiles(terminal, i, record.node_count, record.graph.closure_count, atoms)
        )
        row = {
            "sample_index": i,
            "program_id": record.program_id,
            "layout_record_id": record.graph.structure_id,
            "canonical_smiles": smiles,
            "valid": smiles is not None,
            "constraint_abstention_reason": reason,
        }
        if not np.any(record.fixed_closure_bond_mask) and row != legacy[i]:
            raise ValueError("no-fixed-closure identity control failed")
        output.append(row)
    return (
        control,
        output,
        {
            **evidence,
            "parent_selection": "greedy_original_scores_after_all_immutable_edges_reserved",
            "legacy_core_control_recomputed": True,
            "no_fixed_closure_identity_exact": True,
            "fixed_closure_attempts": sum(
                bool(np.any(r.fixed_closure_bond_mask)) for r in scaffolded
            ),
        },
    )
