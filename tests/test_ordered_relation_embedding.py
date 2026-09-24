"""Protect explicit dispatch and unchanged parameter/checkpoint identity."""

from types import SimpleNamespace

import pytest
import torch

from forge.model.ordered_relation_embedding import (
    OrderedRelationEmbedding,
    configure_relation_embedding,
    ordered_backward,
)


def test_default_does_not_replace_original_module():
    model = SimpleNamespace(relation_bias=torch.nn.Embedding(5, 8))
    original = model.relation_bias
    configure_relation_embedding(model)
    assert model.relation_bias is original


def test_unknown_backend_and_cpu_opt_in_fail_explicitly():
    model = SimpleNamespace(relation_bias=torch.nn.Embedding(5, 8))
    with pytest.raises(ValueError, match="Unknown"):
        configure_relation_embedding(model, "unqualified")
    with pytest.raises(ValueError, match="CUDA FP32"):
        configure_relation_embedding(model, "ordered_fp32_v1")
    with pytest.raises(ValueError, match="CUDA FP32"):
        ordered_backward(torch.zeros(4000, 8), torch.zeros(4000, dtype=torch.long), 5)


def test_wrapper_preserves_parameter_optimizer_and_state_dict_identity():
    torch.manual_seed(2026092431)
    original = torch.nn.Embedding(5, 8)
    optimizer = torch.optim.AdamW(original.parameters())
    wrapper = OrderedRelationEmbedding(original.weight)
    assert wrapper.weight is original.weight is optimizer.param_groups[0]["params"][0]
    assert wrapper.weight.data_ptr() == original.weight.data_ptr()
    assert wrapper.state_dict().keys() == original.state_dict().keys()
    indices = torch.tensor([[4, 0, 4, 1], [3, 2, 2, 0]])
    torch.testing.assert_close(wrapper(indices), original(indices), atol=0, rtol=0)
    restored = torch.nn.Embedding(5, 8)
    restored.load_state_dict(wrapper.state_dict(), strict=True)
    torch.testing.assert_close(restored(indices), wrapper(indices), atol=0, rtol=0)
