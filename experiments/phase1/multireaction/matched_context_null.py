"""Separate inference estimand: null-trained weights with common external structure/noise."""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from forge.model.compose_lipid_sampling import sample
from forge.model.reaction_program_flow import decode_synthesis_program_argmax
from forge.model.synthesis_program_sampling import _fixed_state_exact_tensor

SEMANTIC_FIELDS = (
    "program_states",
    "role_states",
    "core_position_states",
    "program_depths",
    "component_instance_states",
    "component_position_states",
    "repeat_group_states",
    "role_morphology_states",
    "adapter_mask",
)
FIELDS = ("nodes", "parents", "parent_bonds", "closure_left", "closure_right", "closure_bonds")


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    metadata = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":"))
    return hashlib.sha256(metadata.encode() + array.tobytes()).hexdigest()


def context_digest(batch, node, bond):
    return {
        k: tensor_digest(v)
        for k, v in sorted({**batch, "source_node_noise": node, "source_bond_noise": bond}.items())
        if torch.is_tensor(v)
    }


class MatchedContextNull(nn.Module):
    """Zero semantics only at forward; leave caller structure, noise and masks unchanged.

    This explicitly differs from CountOnlySampler and accepts no count-only admission.
    The base already has native semantic-null mode; redundant zeroing exposes a strict
    auditable zero-nine boundary before every forward and prevents cached-memory leaks.
    """

    def __init__(self, base):
        super().__init__()
        if getattr(base, "semantic_conditioning", None) != "semantic_null":
            raise ValueError("Matched-context null requires native null-trained architecture")
        self.base = base

    @property
    def maximum_closures(self):
        return self.base.maximum_closures

    def forward(self, **kwargs):
        if "program_memory" in kwargs:
            raise ValueError("Matched null forbids cached conditioned program memory")
        if any(k not in kwargs or kwargs[k] is None for k in SEMANTIC_FIELDS):
            raise ValueError("Missing semantic coordinate")
        return self.base(**{**kwargs, **{k: torch.zeros_like(kwargs[k]) for k in SEMANTIC_FIELDS}})


def equal_tensors(left, right):
    if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("Exact tensor dictionary mismatch")


def capture(model, base, batch, node, bond, *, seed, steps=64, semantic_null=False):
    """Observe initial noise and true flow endpoint without mutating sampler state."""
    before = context_digest(batch, node, bond)
    calls, initial, raw = [], [], []
    zero_checks = 0

    def trace(_module, _args, kwargs):
        nonlocal zero_checks
        for key in ("node_mask", "child_mask", "closure_mask"):
            if kwargs[key] is not batch[key]:
                raise ValueError("External physical mask replaced")
        if semantic_null:
            if "program_memory" in kwargs or any(
                bool(kwargs[k].count_nonzero()) for k in SEMANTIC_FIELDS
            ):
                raise ValueError(
                    "Nonzero semantic value or cached conditioning crossed null boundary"
                )
            zero_checks += len(SEMANTIC_FIELDS)
        else:
            for key in SEMANTIC_FIELDS:
                if kwargs[key] is not batch[key]:
                    raise ValueError("Conditioned semantics changed")
        t = float(kwargs["t"][0])
        calls.append(t)
        if t in (0.0, 1.0):
            record = {k: kwargs[k].detach().clone() for k in FIELDS}
            (initial if t == 0 else raw).append(record)

    handle = base.register_forward_pre_hook(trace, with_kwargs=True)
    try:
        with torch.inference_mode():
            terminal, predictions = sample(
                model, batch, node, bond, steps=steps, seed=seed, return_predictions=True
            )
    finally:
        handle.remove()
    if calls != [i / steps for i in range(steps + 1)] or len(initial) != 1 or len(raw) != 1:
        raise ValueError("Forward schedule changed")
    equal_tensors(terminal, decode_synthesis_program_argmax(predictions, batch))
    for value in (initial[0], raw[0], terminal):
        if not bool(_fixed_state_exact_tensor(value, batch)):
            raise ValueError("Fixed external structure changed")
    if before != context_digest(batch, node, bond):
        raise ValueError("Caller layout or source-noise tensor mutated")
    if any(
        v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in predictions.values()
    ):
        raise ValueError("Nonfinite or non-FP32 predictions")
    return {
        "raw": raw[0],
        "terminal": terminal,
        "predictions": predictions,
        "trace": {
            "context": before,
            "initial_state": {k: tensor_digest(v) for k, v in initial[0].items()},
            "forward_calls": len(calls),
            "zero_semantic_coordinate_checks": zero_checks,
            "seed": seed,
        },
    }
