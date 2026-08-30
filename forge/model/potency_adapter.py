"""Authenticated lightweight potency-adapter overlays for the FORGE Transformer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from forge.model.potency_conditioning import PotencyAdapterPolicy

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


class PotencyAdapterError(ValueError):
    """A potency adapter differs from its declared frozen-backbone contract."""


POTENCY_PARAMETER_TOKEN = ".potency_adapter."


def potency_parameter_names(model: Any) -> tuple[str, ...]:
    names = tuple(name for name, _ in model.named_parameters() if POTENCY_PARAMETER_TOKEN in name)
    if not names:
        raise PotencyAdapterError("model has no potency adapters")
    return names


def initialize_potency_adapter_from_base(
    model: Any,
    base_state: Mapping[str, Any],
) -> tuple[str, ...]:
    """Load an authenticated base state and allow only new potency-adapter keys."""

    if torch is None:
        raise PotencyAdapterError("potency adapter initialization requires torch")
    incompatible = model.load_state_dict(base_state, strict=False)
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    expected = tuple(sorted(name for name in model.state_dict() if POTENCY_PARAMETER_TOKEN in name))
    if unexpected or missing != expected:
        raise PotencyAdapterError(
            "base checkpoint differs from the potency model outside the declared adapter"
        )
    return missing


def freeze_base_parameters(model: Any) -> tuple[str, ...]:
    """Freeze the complete generator and expose only potency-adapter parameters."""

    names = potency_parameter_names(model)
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    observed = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if observed != names:
        raise PotencyAdapterError("potency adapter trainable-parameter set changed")
    return names


def potency_adapter_state_dict(model: Any) -> dict[str, Any]:
    expected = set(potency_parameter_names(model))
    state = model.state_dict()
    observed = {name for name in state if POTENCY_PARAMETER_TOKEN in name}
    if observed != expected:
        raise PotencyAdapterError("potency adapter parameter and state keys disagree")
    return {name: state[name].detach().cpu().clone() for name in sorted(observed)}


def apply_potency_adapter_state(model: Any, delta: Mapping[str, Any]) -> None:
    expected = set(potency_parameter_names(model))
    if set(delta) != expected:
        raise PotencyAdapterError("potency adapter delta keys changed")
    state = model.state_dict()
    for name in sorted(expected):
        value = delta[name]
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(state[name].shape):
            raise PotencyAdapterError(f"potency adapter tensor changed shape: {name}")
        state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))


def configure_potency_adapter(model: Any, policy: PotencyAdapterPolicy) -> None:
    configure = getattr(model, "configure_potency_policy", None)
    if configure is None:
        raise PotencyAdapterError("model does not support a potency policy")
    configure(policy)


def potency_parameter_report(model: Any) -> dict[str, int | float]:
    selected = set(potency_parameter_names(model))
    total = sum(parameter.numel() for parameter in model.parameters())
    adapter = sum(
        parameter.numel() for name, parameter in model.named_parameters() if name in selected
    )
    if total < 1 or adapter < 1 or adapter >= total:
        raise PotencyAdapterError("potency adapter parameter accounting is invalid")
    return {
        "total_parameters": total,
        "base_frozen_parameters": total - adapter,
        "adapter_trainable_parameters": adapter,
        "adapter_fraction": adapter / total,
    }


def optimizer_parameters(model: Any) -> Sequence[Any]:
    values = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not values:
        raise PotencyAdapterError("potency adapter optimizer has no parameters")
    return values


def reconstruct_potency_conditioned_model(
    *,
    base_package: Mapping[str, Any],
    overlay: Mapping[str, Any],
    vocabulary: Any,
    node_classes: int,
    device: Any,
    checkpoint_schema: str,
) -> tuple[Any, PotencyAdapterPolicy]:
    """Rebuild and authenticate a base-plus-adapter model without duplicating base weights."""

    from forge.model.defog_feasibility import _model_state_sha256
    from forge.model.synthesis_program_training import build_synthesis_program_flow

    base = overlay.get("base")
    raw_model_config = overlay.get("model_config")
    raw_delta = overlay.get("adapter_state")
    if (
        overlay.get("schema_version") != checkpoint_schema
        or overlay.get("trusted_local_checkpoint") is not True
        or not isinstance(base, Mapping)
        or base.get("model_state_sha256") != base_package.get("model_state_sha256")
        or not isinstance(raw_model_config, Mapping)
        or not isinstance(raw_delta, Mapping)
    ):
        raise PotencyAdapterError("potency adapter checkpoint authentication failed")
    expected_model_config = dict(base_package["model_config"])
    for field in ("potency_adapter_dim", "potency_condition_dim"):
        if field not in raw_model_config:
            raise PotencyAdapterError(f"potency adapter model config omits {field}")
        expected_model_config[field] = raw_model_config[field]
    if dict(raw_model_config) != expected_model_config:
        raise PotencyAdapterError("potency adapter changed the authenticated base model config")
    policy = PotencyAdapterPolicy.from_mapping(overlay.get("potency_policy"))
    model = build_synthesis_program_flow(
        vocabulary=vocabulary,
        node_classes=node_classes,
        model_config=raw_model_config,
        device=device,
    )
    initialize_potency_adapter_from_base(model, base_package["model_state"])
    configure_potency_adapter(model, policy)
    apply_potency_adapter_state(model, raw_delta)
    if _model_state_sha256(model) != overlay.get("combined_model_state_sha256"):
        raise PotencyAdapterError("combined potency adapter model state changed")
    model.eval()
    return model, policy


__all__ = [
    "POTENCY_PARAMETER_TOKEN",
    "PotencyAdapterError",
    "apply_potency_adapter_state",
    "configure_potency_adapter",
    "freeze_base_parameters",
    "initialize_potency_adapter_from_base",
    "optimizer_parameters",
    "potency_adapter_state_dict",
    "potency_parameter_names",
    "potency_parameter_report",
    "reconstruct_potency_conditioned_model",
]
