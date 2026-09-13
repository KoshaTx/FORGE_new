"""A bounded, head-only semantic correction to frozen Ugi atom predictions.

The caller supplies the four baseline amine coordinates and fixed normalization scales.
This module does not derive chemistry annotations, own a backbone, or fit normalization data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

SEMANTIC_COORDINATES = 4
RESIDUAL_HIDDEN_DIM = 64
TRAINABLE_PARAMETER_ALLOWLIST = (
    "hidden.weight",
    "hidden.bias",
    "output.weight",
    "output.bias",
)


class UgiSemanticResidualError(ValueError):
    """The residual-head inputs or head-only parameter boundary are invalid."""


def _finite_tensor(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or value.is_complex():
        raise UgiSemanticResidualError(f"{label} must be a real tensor")
    if not bool(torch.isfinite(value).all()):
        raise UgiSemanticResidualError(f"{label} must contain only finite values")


class UgiSemanticResidualHead(nn.Module):
    """Concatenate detached node features, time, and semantic coordinates before an MLP.

    Both semantic arms have identical parameterization and seed-controlled initialization.
    ``use_semantics=False`` supplies four zeros after validating the same input contract.
    Construction and forward calls preserve the ambient random-number stream.
    """

    def __init__(
        self,
        hidden_dim: int,
        atom_classes: int,
        *,
        semantic_scales: Sequence[float],
        initialization_seed: int,
        use_semantics: bool = True,
    ) -> None:
        super().__init__()
        for label, value in (("hidden_dim", hidden_dim), ("atom_classes", atom_classes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise UgiSemanticResidualError(f"{label} must be a positive integer")
        if (
            isinstance(initialization_seed, bool)
            or not isinstance(initialization_seed, int)
            or not 0 <= initialization_seed < 2**63
        ):
            raise UgiSemanticResidualError("initialization_seed must be an integer in [0, 2**63)")
        if not isinstance(use_semantics, bool):
            raise UgiSemanticResidualError("use_semantics must be boolean")
        try:
            scales = torch.as_tensor(semantic_scales, dtype=torch.float32, device="cpu")
        except (TypeError, ValueError, RuntimeError) as error:
            raise UgiSemanticResidualError(
                "semantic_scales must contain four positive values"
            ) from error
        if (
            scales.shape != (SEMANTIC_COORDINATES,)
            or not bool(torch.isfinite(scales).all())
            or not bool((scales > 0).all())
        ):
            raise UgiSemanticResidualError(
                "semantic_scales must contain four finite positive values"
            )
        self.hidden_dim = hidden_dim
        self.atom_classes = atom_classes
        self.initialization_seed = initialization_seed
        self.use_semantics = use_semantics
        self.register_buffer("semantic_scales", scales.detach().clone())
        # Linear constructors use the ambient CPU generator. Restore that stream, then initialize
        # with a private generator so informative and zero-coordinate arms start identically.
        with torch.random.fork_rng(devices=[]):
            self.hidden = nn.Linear(
                hidden_dim + 1 + SEMANTIC_COORDINATES,
                RESIDUAL_HIDDEN_DIM,
                device="cpu",
                dtype=torch.float32,
            )
            self.output = nn.Linear(
                RESIDUAL_HIDDEN_DIM, atom_classes, device="cpu", dtype=torch.float32
            )
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        nn.init.kaiming_uniform_(self.hidden.weight, a=math.sqrt(5), generator=generator)
        bound = 1 / math.sqrt(self.hidden.in_features)
        nn.init.uniform_(self.hidden.bias, -bound, bound, generator=generator)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hidden: Tensor,
        flow_time: Tensor,
        amine_semantics: Tensor,
        variable_amine_mask: Tensor,
    ) -> Tensor:
        """Return a masked ``[batch, nodes, atom_classes]`` atom-logit correction."""

        _finite_tensor(hidden, "hidden")
        if (
            not hidden.is_floating_point()
            or hidden.ndim != 3
            or hidden.shape[-1] != self.hidden_dim
            or min(hidden.shape[:2]) < 1
        ):
            raise UgiSemanticResidualError("hidden must have shape [batch, nodes, hidden_dim]")
        batch, nodes, _ = hidden.shape
        _finite_tensor(flow_time, "flow_time")
        if flow_time.shape != (batch,) or not flow_time.is_floating_point():
            raise UgiSemanticResidualError("flow_time must be a floating tensor of shape [batch]")
        if not bool(((flow_time >= 0) & (flow_time <= 1)).all()):
            raise UgiSemanticResidualError("flow_time must lie in [0, 1]")
        _finite_tensor(amine_semantics, "amine_semantics")
        if amine_semantics.shape != (batch, SEMANTIC_COORDINATES):
            raise UgiSemanticResidualError("amine_semantics must have shape [batch, 4]")
        if amine_semantics.dtype == torch.bool or not bool((amine_semantics >= 0).all()):
            raise UgiSemanticResidualError(
                "amine_semantics must contain nonnegative numeric values"
            )
        if (
            not isinstance(variable_amine_mask, Tensor)
            or variable_amine_mask.dtype != torch.bool
            or variable_amine_mask.shape != (batch, nodes)
        ):
            raise UgiSemanticResidualError("variable_amine_mask must be boolean [batch, nodes]")
        if any(
            tensor.device != hidden.device
            for tensor in (flow_time, amine_semantics, variable_amine_mask, self.hidden.weight)
        ):
            raise UgiSemanticResidualError("head and all inputs must be on the same device")
        if hidden.dtype != self.hidden.weight.dtype:
            raise UgiSemanticResidualError("hidden dtype must match the residual head dtype")
        if (
            self.semantic_scales.shape != (SEMANTIC_COORDINATES,)
            or not bool(torch.isfinite(self.semantic_scales).all())
            or not bool((self.semantic_scales > 0).all())
        ):
            raise UgiSemanticResidualError("semantic_scales must remain finite and positive")
        semantics = amine_semantics.detach().to(dtype=hidden.dtype) / self.semantic_scales
        if not self.use_semantics:
            semantics = torch.zeros_like(semantics)
        features = torch.cat(
            (
                hidden.detach(),
                flow_time.detach().to(dtype=hidden.dtype)[:, None, None].expand(-1, nodes, -1),
                semantics[:, None, :].expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        correction = self.output(torch.relu(self.hidden(features)))
        _finite_tensor(correction, "residual correction")
        return torch.where(
            variable_amine_mask.unsqueeze(-1), correction, torch.zeros_like(correction)
        )

    def trainable_parameter_receipt(self) -> dict[str, Any]:
        """Authenticate the complete trainable parameter set against the head-only allowlist."""

        parameters = dict(self.named_parameters())
        if set(parameters) != set(TRAINABLE_PARAMETER_ALLOWLIST) or not all(
            parameter.requires_grad for parameter in parameters.values()
        ):
            raise UgiSemanticResidualError(
                "trainable parameters violate the residual-head allowlist"
            )
        return {
            "schema_version": "forge.ugi_semantic_residual_parameters.v1",
            "scope": "semantic_residual_head_only",
            "owns_backbone": False,
            "trainable_parameter_allowlist": list(TRAINABLE_PARAMETER_ALLOWLIST),
            "parameter_shapes": {
                name: list(parameters[name].shape) for name in TRAINABLE_PARAMETER_ALLOWLIST
            },
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in parameters.values()
            ),
            "hidden_dim": self.hidden_dim,
            "residual_hidden_dim": RESIDUAL_HIDDEN_DIM,
            "atom_classes": self.atom_classes,
            "initialization_seed": self.initialization_seed,
            "use_semantics": self.use_semantics,
            "semantic_scales": self.semantic_scales.detach().cpu().tolist(),
            "detached_inputs": ["hidden", "flow_time", "amine_semantics", "baseline_node_logits"],
        }


def apply_semantic_residual(
    predictions: Mapping[str, Tensor],
    head: UgiSemanticResidualHead,
    *,
    hidden: Tensor,
    flow_time: Tensor,
    amine_semantics: Tensor,
    variable_amine_mask: Tensor,
) -> dict[str, Tensor]:
    """Shallow-copy predictions, changing only masked node logits with head-only gradients."""

    if not isinstance(predictions, Mapping) or "nodes" not in predictions:
        raise UgiSemanticResidualError("predictions must contain baseline 'nodes' logits")
    baseline = predictions["nodes"]
    _finite_tensor(baseline, "baseline nodes")
    correction = head(hidden, flow_time, amine_semantics, variable_amine_mask)
    if baseline.shape != correction.shape:
        raise UgiSemanticResidualError("baseline nodes must match [batch, nodes, atom_classes]")
    if baseline.device != correction.device or baseline.dtype != correction.dtype:
        raise UgiSemanticResidualError(
            "baseline nodes and residual must have the same device and dtype"
        )
    result = dict(predictions)
    result["nodes"] = torch.where(
        variable_amine_mask.unsqueeze(-1), baseline.detach() + correction, baseline.detach()
    )
    _finite_tensor(result["nodes"], "corrected nodes")
    return result
