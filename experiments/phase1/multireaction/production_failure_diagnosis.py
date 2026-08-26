"""Training/calibration-only diagnosis of the frozen multi-reaction production result."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

SCHEMA = "forge.synthesis_program_production_failure_diagnosis.v1"
ARM = "shared_three_program_conditioned"
METRICS = ("raw_valid_fraction", "exact_l1_yield_per_attempt")


class ProductionFailureDiagnosisError(ValueError):
    """Frozen evidence cannot support the calibration-only diagnosis."""


def diagnose_production_failure(
    aggregate_path: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    """Summarize frozen calibration traces without using heldout data for model selection."""

    repo = repo.resolve()
    aggregate_path = aggregate_path.resolve()
    aggregate = read_json_object(
        aggregate_path, error=ProductionFailureDiagnosisError, label="production aggregate"
    )
    if aggregate.get("schema_version") != "forge.synthesis_program_production_adjudication.v1":
        raise ProductionFailureDiagnosisError("unsupported production aggregate schema")
    evidence = aggregate.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ProductionFailureDiagnosisError("diagnosis requires all three frozen replicates")

    cells: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    evidence_pins: list[dict[str, Any]] = []
    for record in evidence:
        evaluation_pin = {key: record["evaluation_result"][key] for key in ("path", "sha256")}
        training_pin = {key: record["training_result"][key] for key in ("path", "sha256")}
        evaluation_path = resolve_pin(evaluation_pin, repo, label="frozen evaluation result")
        training_path = resolve_pin(training_pin, repo, label="frozen training result")
        evaluation = read_json_object(
            evaluation_path, error=ProductionFailureDiagnosisError, label="evaluation result"
        )
        checkpoints = evaluation.get("checkpoint_metrics", {}).get(ARM)
        if not isinstance(checkpoints, dict):
            raise ProductionFailureDiagnosisError("conditioned calibration trace is missing")
        for step_text, checkpoint in checkpoints.items():
            calibration = checkpoint.get("calibration")
            if not isinstance(calibration, dict):
                raise ProductionFailureDiagnosisError("checkpoint omits calibration metrics")
            for program_id, row in calibration.items():
                for metric in METRICS:
                    value = row.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ProductionFailureDiagnosisError(
                            f"calibration metric {program_id}.{metric} is not numeric"
                        )
                    cells[str(program_id)][int(step_text)][metric].append(float(value))
        evidence_pins.append(
            {
                "replicate": record["replicate"],
                "training_result": pin_record(training_path, repo),
                "evaluation_result": pin_record(evaluation_path, repo),
            }
        )

    programs: dict[str, Any] = {}
    for program_id, checkpoints in sorted(cells.items()):
        summaries: dict[str, Any] = {}
        for step, metrics in sorted(checkpoints.items()):
            summaries[str(step)] = {
                metric: {
                    "by_seed": values,
                    "mean": float(np.mean(values)),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for metric, values in sorted(metrics.items())
            }
        yields = {
            step: row["exact_l1_yield_per_attempt"]["mean"] for step, row in summaries.items()
        }
        peak_step = max(yields, key=yields.get)
        final_step = str(max(int(step) for step in yields))
        programs[program_id] = {
            "calibration_checkpoints": summaries,
            "peak_calibration_exact_l1_step": int(peak_step),
            "peak_calibration_exact_l1_yield": yields[peak_step],
            "final_calibration_exact_l1_yield": yields[final_step],
            "zero_exact_l1_at_every_checkpoint": all(value == 0.0 for value in yields.values()),
        }

    ugi = programs["ugi_3cr_agile"]
    bl = programs["bl_2023_repeated_aza_michael"]
    result = {
        "schema_version": SCHEMA,
        "status": "calibration_failure_localized",
        "aggregate": pin_record(aggregate_path, repo),
        "evidence": evidence_pins,
        "programs": programs,
        "findings": {
            "bl_failure_precedes_heldout_evaluation": bl["zero_exact_l1_at_every_checkpoint"],
            "ugi_calibration_degrades_after_its_peak": (
                ugi["final_calibration_exact_l1_yield"] < ugi["peak_calibration_exact_l1_yield"]
            ),
            "fixed_final_checkpoint_was_not_selected_on_calibration": True,
        },
        "interpretation": (
            "The BL arm produces zero exact-L1 calibration samples at every frozen checkpoint, "
            "and shared-model Ugi yield falls after an early calibration peak. This localizes the "
            "failure to shared training/representation before heldout evaluation. It does not by "
            "itself distinguish optimization interference from insufficient program-specific "
            "capacity."
        ),
        "decision": (
            "Preserve the primary production comparison as a negative result. Any program-specific "
            "adapter or routing-head follow-up must be a newly frozen experiment and cannot replace "
            "the completed primary comparison."
        ),
        "heldout_used_for_model_or_threshold_selection": False,
        "candidate_selection": False,
        "calls": {"route": 0, "oracle": 0},
        "implementation": pin_record(Path(__file__), repo),
    }
    write_json(output_path.resolve(), result)
    return result


__all__ = ["ProductionFailureDiagnosisError", "SCHEMA", "diagnose_production_failure"]
