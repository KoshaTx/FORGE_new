import numpy as np
import pytest
import torch

from experiments.phase1.synthesis_guidance.guidance.current_sampler_zero import (
    CurrentSamplerZeroObserver,
)


class ToySamplerModel(torch.nn.Module):
    def forward(self, *, nodes, **_kwargs):
        return nodes.clone()


def inputs(t):
    return {
        "t": torch.full((3,), t),
        **{
            name: torch.ones(3, 2, dtype=torch.long)
            for name in (
                "nodes",
                "parents",
                "parent_bonds",
                "closure_left",
                "closure_right",
                "closure_bonds",
            )
        },
    }


def test_zero_observation_keeps_inputs_outputs_and_random_states_exact():
    model = ToySamplerModel()
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    observer = CurrentSamplerZeroObserver(model, steps=4, checkpoints=(1, 3))
    with observer:
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            kwargs = inputs(t)
            before = {k: v.clone() for k, v in kwargs.items()}
            assert torch.equal(model(**kwargs), kwargs["nodes"])
            assert all(torch.equal(before[k], v) for k, v in kwargs.items())
    observer.require_complete()
    assert len(observer.forwards) == 5
    assert torch.equal(state, torch.get_rng_state())
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, after, strict=True))
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("strength", [0.25, -1.0, float("nan"), float("inf")])
def test_nonzero_and_invalid_strengths_are_rejected(strength):
    with pytest.raises(ValueError, match="not qualified"):
        CurrentSamplerZeroObserver(ToySamplerModel(), steps=4, checkpoints=(1,), strength=strength)


def test_incomplete_checkpoint_coverage_is_not_a_pass():
    observer = CurrentSamplerZeroObserver(ToySamplerModel(), steps=4, checkpoints=(1,))
    with observer:
        observer.model(**inputs(0.5))
    with pytest.raises(ValueError, match="coverage"):
        observer.require_complete()


def test_exception_removes_hook_and_reuse_fails():
    observer = CurrentSamplerZeroObserver(ToySamplerModel(), steps=4, checkpoints=(1,))
    with pytest.raises(RuntimeError), observer:
        raise RuntimeError("interrupted sampling")
    assert not observer.model._forward_pre_hooks
    with pytest.raises(ValueError, match="reused"):
        with observer:
            pass
