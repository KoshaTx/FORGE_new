"""Verify exact Ugi topology-decoder support on the frozen production program draw."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_programs,
    project_program_for_mixed_transformer,
)
from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.synthesis_program_sampling import _terminal_role_morphology
from forge.model.ugi_transformer_topology import (
    UgiTransformerTopologyError,
    UgiTransformerTopologyPolicy,
    decode_ugi_exact_topology,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.ugi_transformer_topology_support_preflight_config.v2"
RESULT_SCHEMA = "forge.ugi_transformer_topology_support_preflight_result.v2"
INPUT_LABELS = (
    "production_cache",
    "program_draw",
    "topology_closure_config",
    "topology_morphology_config",
)


class UgiTransformerTopologySupportPreflightError(ValueError):
    """The exact decoder or its pinned production support contract changed."""


def _target_morphology(record: Any) -> dict[int, tuple[int, int, int, int]]:
    states = record.role_morphology_states
    if states is None:
        raise UgiTransformerTopologySupportPreflightError(
            "projected Ugi program has no role morphology"
        )
    output: dict[int, tuple[int, int, int, int]] = {}
    for role_state in sorted(set(int(value) for value in record.role_states if value > 0)):
        values = np.unique(states[record.role_states == role_state], axis=0)
        if values.shape != (1, 4) or np.any(values[0] < 1):
            raise UgiTransformerTopologySupportPreflightError(
                "projected Ugi role morphology is inconsistent"
            )
        output[role_state] = tuple(int(value) - 1 for value in values[0])
    return output


def run_ugi_transformer_topology_support_preflight(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Construct one exact topology for every frozen program without chemistry generation."""

    if torch is None:
        raise UgiTransformerTopologySupportPreflightError("topology preflight requires torch")
    config = read_json_object(
        config_path,
        error=UgiTransformerTopologySupportPreflightError,
        label="Ugi topology support preflight config",
    )
    inputs = config.get("inputs")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(inputs, Mapping)
        or set(inputs) != set(INPUT_LABELS)
        or int(config.get("expected_programs", 0)) < 1
        or int(config.get("maximum_children", 0)) < 1
    ):
        raise UgiTransformerTopologySupportPreflightError(
            "topology support preflight contract changed"
        )
    paths = {label: resolve_pin(pin, repo, label=label) for label, pin in inputs.items()}
    closure = read_json_object(
        paths["topology_closure_config"],
        error=UgiTransformerTopologySupportPreflightError,
        label="Ugi closure support config",
    )
    morphology = read_json_object(
        paths["topology_morphology_config"],
        error=UgiTransformerTopologySupportPreflightError,
        label="Ugi morphology support config",
    )
    try:
        policy = UgiTransformerTopologyPolicy.from_support_documents(closure, morphology)
    except UgiTransformerTopologyError as error:
        raise UgiTransformerTopologySupportPreflightError(str(error)) from error

    expected = int(config["expected_programs"])
    programs = _load_programs(paths["program_draw"], count=expected)
    cache = SynthesisProgramProductionCache(paths["production_cache"])
    failures: Counter[str] = Counter()
    mismatches = 0
    by_cycle: dict[int, Counter[str]] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        for index, program in enumerate(programs):
            record = project_program_for_mixed_transformer(prior, program, sample_index=index)
            closure_count = record.graph.closure_count
            predictions = {
                "offspring": torch.zeros(
                    (1, record.node_count, int(config["maximum_children"]) + 1)
                ),
                "closure_left": torch.zeros((1, closure_count, record.node_count)),
                "closure_right": torch.zeros((1, closure_count, record.node_count)),
            }
            total_cycles = sum(program.cycle_ranks)
            cycle_counter = by_cycle.setdefault(total_cycles, Counter())
            try:
                decoded = decode_ugi_exact_topology(
                    predictions,
                    index=0,
                    record=record,
                    policy=policy,
                    generator=generator,
                )
            except UgiTransformerTopologyError as error:
                reason = str(error)
                failures[reason] += 1
                cycle_counter["failure"] += 1
                continue
            observed = _terminal_role_morphology(
                record,
                parents=decoded.parents,
                closure_left=decoded.closure_left,
                closure_right=decoded.closure_right,
            )
            if observed != _target_morphology(record):
                mismatches += 1
                cycle_counter["mismatch"] += 1
            else:
                cycle_counter["exact"] += 1
    finally:
        cache.close()

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if not failures and mismatches == 0 else "fail",
        "programs": len(programs),
        "seed": int(config["seed"]),
        "maximum_children": int(config["maximum_children"]),
        "policy": policy.to_mapping(),
        "observed": {
            "decoded": len(programs) - sum(failures.values()),
            "exact_program_morphology": len(programs) - sum(failures.values()) - mismatches,
            "mismatches": mismatches,
            "failures": sum(failures.values()),
            "failure_reasons": dict(sorted(failures.items())),
            "by_total_cycle_rank": {
                str(cycle): dict(sorted(counts.items()))
                for cycle, counts in sorted(by_cycle.items())
            },
        },
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "config": pin_record(config_path, repo),
        "gates": {
            "attempt_denominator_exact": len(programs) == expected,
            "all_programs_decoded": not failures,
            "all_decoded_morphologies_exact": mismatches == 0,
            "component_vocabulary_absent": True,
            "chemistry_generation_calls_zero": True,
            "heldout_used_for_selection_zero": True,
            "repairs_zero": True,
        },
        "nonclaims": [
            "Topology support does not establish chemical validity or exact-L1 yield.",
            "This diagnostic does not train, select a checkpoint, or generate candidates.",
        ],
    }
    if not all(result["gates"].values()):
        result["status"] = "fail"
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_ugi_transformer_topology_support_preflight(
        arguments.config.resolve(),
        arguments.repo.resolve(),
        arguments.output.resolve(),
    )
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTransformerTopologySupportPreflightError",
    "run_ugi_transformer_topology_support_preflight",
]
