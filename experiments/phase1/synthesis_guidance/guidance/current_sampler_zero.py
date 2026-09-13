"""Bounded zero-strength observation of the current shared sampler.

This qualifies only an identity controller at declared model-call checkpoints. It does not
implement trajectory resumption, nonzero ancestry mutation, or a synthesis-value policy.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch

from experiments.phase1.synthesis_guidance.guidance.ugi_synthesis_guidance import (
    select_smc_ancestry,
)


class CurrentSamplerZeroObserver:
    def __init__(
        self, model: Any, *, steps: int, checkpoints: tuple[int, ...], strength: float = 0.0
    ):
        if not math.isfinite(strength) or strength != 0.0:
            raise ValueError("nonzero current-sampler guidance is not qualified")
        if (
            type(steps) is not int
            or steps < 2
            or not checkpoints
            or len(set(checkpoints)) != len(checkpoints)
            or any(type(s) is not int or not 0 <= s < steps for s in checkpoints)
        ):
            raise ValueError("invalid zero-observation checkpoints")
        self.model, self.steps, self.checkpoints = model, steps, checkpoints
        self.forwards: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self._handle: Any = None
        self._entered = False

    def _observe(self, _model: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        times = kwargs["t"].detach().cpu()
        if times.ndim != 1 or not times.numel() or not torch.all(times == times[0]):
            raise ValueError("observation requires one shared flow time per batch")
        time_value, size = float(times[0]), times.numel()
        digest = hashlib.sha256()
        for name in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        ):
            tensor = kwargs[name].detach().cpu().contiguous()
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.numpy().tobytes())
        self.forwards.append(
            {"time": time_value, "batch_size": size, "state_sha256": digest.hexdigest()}
        )
        selected = [s for s in self.checkpoints if time_value == s / self.steps]
        if not selected:
            return
        # Values are synthetic and do not trigger an assessor. At exactly zero strength the
        # existing controller must return identity even for unequal diagnostic values/weights.
        indices = np.arange(size)
        decision = select_smc_ancestry(
            np.linspace(-2.0, 2.0, size),
            np.linspace(-7.0, 11.0, size),
            np.linspace(3.0, -9.0, size),
            current_beta=1.0,
            previous_beta=0.0,
            guidance_strength=0.0,
            base_seed=0,
            arm="zero_preflight",
            program_index=0,
            checkpoint_index=selected[0],
            rollout_index=0,
        )
        if (
            decision.resampled
            or decision.keyed_seed is not None
            or not np.array_equal(decision.ancestors, indices)
        ):
            raise ValueError("zero guidance changed ancestry or consumed a resampling seed")
        self.decisions.append(
            {
                "step": selected[0],
                "identity": True,
                "batch_size": size,
                "resampled": False,
                "resampling_seed": None,
                "route_values_used": False,
            }
        )

    def __enter__(self) -> CurrentSamplerZeroObserver:
        if self._entered:
            raise ValueError("zero observers cannot be reused")
        self._entered = True
        self._handle = self.model.register_forward_pre_hook(self._observe, with_kwargs=True)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def require_complete(self) -> None:
        if [r["step"] for r in self.decisions] != list(self.checkpoints):
            raise ValueError("zero checkpoint coverage differs; require exactly one complete batch")
