"""Information-boundary tests; no fitted model or chemistry outcomes are scored."""

import pytest
import torch
from torch import nn

from experiments.phase1.multireaction.matched_context_null import (
    SEMANTIC_FIELDS,
    MatchedContextNull,
    context_digest,
)


class Recorder(nn.Module):
    semantic_conditioning = "semantic_null"
    maximum_closures = 12

    def forward(self, **kwargs):
        return kwargs


def example():
    return {
        **{k: torch.ones((2, 3), dtype=torch.long) for k in SEMANTIC_FIELDS},
        "nodes": torch.ones((2, 3), dtype=torch.long),
        "node_mask": torch.ones((2, 3), dtype=torch.bool),
    }


def test_zero_all_nine_keep_state_and_physical_mask_objects():
    values = example()
    original = {k: v.clone() for k, v in values.items()}
    result = MatchedContextNull(Recorder())(**values)
    assert all(torch.count_nonzero(result[k]) == 0 for k in SEMANTIC_FIELDS)
    for k in ("nodes", "node_mask"):
        assert result[k] is values[k]
    assert all(torch.equal(values[k], v) for k, v in original.items())


@pytest.mark.parametrize("bad", ["memory", "missing", "none", "conditioned"])
def test_forbidden_conditioning_and_wrong_base(bad):
    base = Recorder()
    values = example()
    if bad == "conditioned":
        base.semantic_conditioning = "conditioned"
    elif bad == "memory":
        values["program_memory"] = object()
    elif bad == "missing":
        values.pop("component_instance_states")
    else:
        values["component_instance_states"] = None
    with pytest.raises(ValueError):
        MatchedContextNull(base)(**values)


def test_context_digest_binds_values_shapes_and_dtype():
    values = example()
    noise = torch.tensor([0.2, 0.8])
    before = context_digest(values, noise, noise)
    values["nodes"][0, 0] = 2
    assert before != context_digest(values, noise, noise)
    values["nodes"][0, 0] = 1
    assert before == context_digest(values, noise, noise)
    assert before != context_digest(values, noise.double(), noise)
