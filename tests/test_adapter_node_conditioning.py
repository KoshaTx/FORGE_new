from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from forge.model.adapter_node_conditioning import (  # noqa: E402
    AdapterConditioningError,
    AdapterNodeConditioning,
)


def _inputs():
    return {
        "origin_states": torch.tensor([[1, 2, 3, 4]]),
        "core_position_states": torch.tensor([[1, 2, 0, 5]]),
        "port_states": torch.tensor([[1, 2, 0, 0]]),
        "distance_to_core": torch.tensor([[0, 0, 3, 0]]),
        "distance_to_own_port": torch.tensor([[0, 0, 3, -1]]),
        "adapter_mask": torch.tensor([[True, True, True, True]]),
    }


def test_minimal_adapter_conditioning_masks_broad_records_exactly() -> None:
    model = AdapterNodeConditioning(hidden_dim=16, maximum_distance=32)
    inputs = _inputs()
    active = model(**inputs)
    inputs["adapter_mask"] = torch.zeros_like(inputs["adapter_mask"])
    inactive = model(**inputs)

    assert active.shape == (1, 4, 16)
    assert torch.count_nonzero(active) > 0
    assert torch.count_nonzero(inactive) == 0


def test_all_port_distances_are_an_explicit_ablation() -> None:
    inputs = _inputs()
    minimal = AdapterNodeConditioning(hidden_dim=16, maximum_distance=32)
    with pytest.raises(AdapterConditioningError, match="minimal conditioner"):
        minimal(**inputs, distances_to_all_ports=torch.zeros((1, 4, 3), dtype=torch.long))

    full = AdapterNodeConditioning(
        hidden_dim=16,
        maximum_distance=32,
        use_all_port_distances=True,
    )
    output = full(
        **inputs,
        distances_to_all_ports=torch.zeros((1, 4, 3), dtype=torch.long),
    )
    assert output.shape == (1, 4, 16)
