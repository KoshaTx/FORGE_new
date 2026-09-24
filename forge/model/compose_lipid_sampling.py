"""Joint sparse-flow sampling shared by COMPOSE production and bounded diagnostics."""

import torch

from forge.flow import rstar_step
from forge.model.reaction_program_flow import (
    decode_synthesis_program_argmax,
    resolve_synthesis_program_source_marginals,
)
from forge.model.synthesis_program_sampling import (
    _endpoint_candidate_mask,
    _fixed_state_exact_tensor,
    _initial_state,
    _parent_candidate_mask,
    _program_conditioning,
    _restore_fixed_states_in_place,
    pointer_rstar_step,
)


def sample(model, layout, node, bond, *, steps, seed, return_predictions=False):
    """Unchanged unconstrained R-star readout, with an explicit device-local RNG."""
    device = layout["nodes"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    state = _initial_state(layout, node, bond, generator)
    sources = resolve_synthesis_program_source_marginals(layout, node, bond)
    conditioning = _program_conditioning(model, layout)
    candidates = {
        "parents": _parent_candidate_mask(layout["node_mask"]),
        "closure_left": _endpoint_candidate_mask(layout["node_mask"], model.maximum_closures),
        "closure_right": _endpoint_candidate_mask(layout["node_mask"], model.maximum_closures),
    }
    batch_size = len(layout["program_states"])
    for step in range(steps):
        assert bool(_fixed_state_exact_tensor(state, layout)), "Fixed context changed"
        t = step / steps
        output = model(**state, t=torch.full((batch_size,), t, device=device), **conditioning)
        for field, mask, source in zip(
            ("nodes", "parent_bonds", "closure_bonds"),
            ("atom_variable_mask", "parent_bond_variable_mask", "closure_bond_variable_mask"),
            sources,
            strict=True,
        ):
            state[field] = rstar_step(
                state[field],
                output[field].softmax(-1),
                source,
                t,
                1 / steps,
                layout[mask],
                generator,
            )
        for field, mask in (
            ("parents", "parent_variable_mask"),
            ("closure_left", "closure_endpoint_variable_mask"),
            ("closure_right", "closure_endpoint_variable_mask"),
        ):
            if bool(layout[mask].any()):
                state[field] = pointer_rstar_step(
                    state[field],
                    output[field],
                    candidates[field],
                    layout[mask],
                    t,
                    1 / steps,
                    generator,
                )
        _restore_fixed_states_in_place(state, layout)
    output = model(**state, t=torch.ones(batch_size, device=device), **conditioning)
    terminal = decode_synthesis_program_argmax(output, layout)
    assert bool(_fixed_state_exact_tensor(terminal, layout)), "Fixed terminal context changed"
    return (terminal, output) if return_predictions else terminal
