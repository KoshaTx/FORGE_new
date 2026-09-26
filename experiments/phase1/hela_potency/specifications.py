"""Allow-listed HeLa potency experiment specifications."""

from __future__ import annotations

POTENCY_SPECIFICATIONS: dict[str, str] = {
    "phase1-potency-study-corpus": "experiments/phase1/hela_potency/data.json",
    "phase1-ugi-hela-potency-adapter-smoke": (
        "experiments/phase1/hela_potency/potency_adapter_smoke.json"
    ),
    "phase1-ugi-hela-potency-adapter-full-h100": (
        "experiments/phase1/hela_potency/potency_adapter_full_h100.json"
    ),
    "phase1-ugi-hela-potency-adapter-failure-attribution": (
        "experiments/phase1/hela_potency/potency_adapter_failure_attribution.json"
    ),
    "phase1-ugi-hela-potency-ordinal-adapter-smoke-v2": (
        "experiments/phase1/hela_potency/potency_adapter_ordinal_smoke.json"
    ),
    "phase1-ugi-hela-potency-ordinal-adapter-full-h100-v2": (
        "experiments/phase1/hela_potency/potency_adapter_ordinal_full_h100.json"
    ),
    "phase1-ugi-hela-partial-state-value-smoke-v1": (
        "experiments/phase1/hela_potency/partial_state_value_smoke.json"
    ),
    "phase1-ugi-hela-partial-state-value-full-h100-v1": (
        "experiments/phase1/hela_potency/partial_state_value_full_h100.json"
    ),
    "phase1-ugi-hela-partial-state-attention-value-smoke-v2": (
        "experiments/phase1/hela_potency/partial_state_attention_value_smoke.json"
    ),
    "phase1-ugi-hela-partial-state-attention-value-full-h100-v2": (
        "experiments/phase1/hela_potency/partial_state_attention_value_full_h100.json"
    ),
}


__all__ = ["POTENCY_SPECIFICATIONS"]
