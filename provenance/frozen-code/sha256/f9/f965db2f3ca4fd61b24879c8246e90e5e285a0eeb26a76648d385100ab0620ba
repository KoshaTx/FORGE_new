"""Adopt verified stage receipts after an orchestration-only import-format revision."""

import json
from pathlib import Path

from experiments.phase1.multireaction import combinatorial_generation_pipeline as pipeline
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "results/phase1/combinatorial_generation_pipeline_replay_v1"
NEW = ROOT / "results/phase1/combinatorial_generation_pipeline_replay_v2"
CONFIG = ROOT / "configs/multireaction/combinatorial_generation_pipeline_replay_v2.json"


def main():
    if NEW.exists() or (OLD / "result.json").exists():
        raise ValueError("recovery requires a fresh destination and an incomplete original pipeline")
    config, templates = pipeline.contract(ROOT, CONFIG)
    prior, receipts, inputs = None, {}, [pipeline._pin(CONFIG, ROOT)]
    for stage in pipeline.STAGES:
        path = OLD / "stages" / stage / "complete.json"
        receipt = pipeline._read(path)
        stage_config = resolve_pin(receipt["config"], ROOT, label="recovered stage config")
        result = resolve_pin(receipt["result"], ROOT, label="recovered stage result")
        if pipeline._read(stage_config) != pipeline.stage_config(stage, config, templates, prior):
            raise ValueError("revised request changes a completed stage")
        if pipeline._read(result)["config"] != receipt["config"]:
            raise ValueError("recovered result belongs to another stage config")
        inputs.extend([pipeline._pin(path, ROOT), receipt["config"], receipt["result"]])
        receipts[stage], prior = receipt, receipt["result"]
    pipeline._write(NEW / "request.json", pipeline._pin(CONFIG, ROOT))
    for stage, receipt in receipts.items():
        pipeline._write(NEW / "stages" / stage / "complete.json", receipt)
    pipeline._write(
        NEW / "recovery.json",
        {
            "status": "stage_configs_identical_under_revised_orchestration_request",
            "inputs": inputs,
            "source": pipeline._pin(Path(__file__), ROOT),
            "next_command": "Resume the revised request; the pipeline verifies every adopted stage before publishing its result.",
        },
    )
    print(json.dumps({"status": "prepared_verified_stage_reuse", "stages": list(receipts)}))


if __name__ == "__main__":
    main()
