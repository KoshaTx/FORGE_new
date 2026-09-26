"""Forward-only controls for bounded training and sampling qualification.

The clean targets, corruption masks, topology context and source marginals remain with
the caller. Semantic null therefore tests semantics conditional on the supplied core
and layout; it is deliberately not an unconditional shared-null implementation.
"""

from collections.abc import Mapping

import torch
from torch import nn

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


class ForwardControl(nn.Module):
    """Apply one declared intervention at every real model forward boundary.

    Sampling memory is intentionally rebuilt from the transformed inputs by the base
    model. Reusing untransformed cached memory would leak conditioning or violate its
    identity check. This qualification adapter does not establish production throughput
    or compatibility with the existing serialized checkpoint schema.
    """

    def __init__(self, model, mode: str, program_mapping: Mapping[int, int] | None = None):
        super().__init__()
        if mode not in {"conditioned", "label_null", "semantic_null", "cyclic"}:
            raise ValueError("Unsupported forward-only control; shared null requires a new prior")
        mapping = dict(program_mapping or {})
        states = set(range(1, len(model.vocabulary.program_states)))
        if mode == "cyclic":
            if (
                set(mapping) != states
                or set(mapping.values()) != states
                or any(type(k) is not int or type(v) is not int for k, v in mapping.items())
            ):
                raise ValueError("Cyclic map must cover the complete non-null model vocabulary")
        elif mapping:
            raise ValueError("Unexpected permutation for a noncyclic control")
        self.model = model
        self.mode = mode
        self.register_buffer(
            "program_permutation", torch.tensor([0] + [mapping.get(i, i) for i in sorted(states)])
        )

    @property
    def maximum_closures(self):
        return self.model.maximum_closures

    def forward(self, **kwargs):
        if "program_memory" in kwargs:
            raise ValueError("Precomputed conditioning memory is not admitted by this control")
        if any(key not in kwargs for key in SEMANTIC_FIELDS):
            raise ValueError("Missing semantic coordinate at the model-forward boundary")
        if self.mode == "label_null":
            kwargs["program_states"] = torch.zeros_like(kwargs["program_states"])
        elif self.mode == "semantic_null":
            kwargs.update({key: torch.zeros_like(kwargs[key]) for key in SEMANTIC_FIELDS})
        elif self.mode == "cyclic":
            states = kwargs["program_states"]
            if states.dtype != torch.long or bool((states <= 0).any()):
                raise ValueError("Cyclic control expects qualified non-null program states")
            if bool((states >= len(self.program_permutation)).any()):
                raise ValueError("Program state outside the frozen vocabulary")
            kwargs["program_states"] = self.program_permutation[states]
        return self.model(**kwargs)
