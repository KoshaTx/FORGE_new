"""Rebuild the frozen development manifest and compare all original fields."""

import hashlib
import json
from pathlib import Path

from forge.synthesis.engine.single_step_benchmark_manifest import (
    OUTPUT_FILENAMES,
    build_manifest_payloads,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/route/single_step_proposal_lane_qualification_manifest_v1.json"


def pin(path):
    path = (ROOT / path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    config = json.loads(CONFIG.read_text())
    frozen = ROOT / config["output_directory"]
    payloads = build_manifest_payloads(ROOT, CONFIG)
    fresh_result = json.loads(payloads["result"])
    original_result = json.loads((frozen / OUTPUT_FILENAMES["result"]).read_text())
    source_bindings = fresh_result["implementation_source_bindings"]
    old_fields = {
        k: v
        for k, v in fresh_result.items()
        if k not in {"implementation_source_bindings", "execution_provenance"}
    }
    checks = {
        name: payload == (frozen / OUTPUT_FILENAMES[name]).read_bytes()
        for name, payload in payloads.items()
        if name != "result"
    }
    checks["all_original_result_fields"] = old_fields == original_result
    assert all(checks.values()), checks
    for name, payload in payloads.items():
        (OUTPUT / ("replayed-" + OUTPUT_FILENAMES[name])).write_bytes(payload)
    inputs = {CONFIG, Path(__file__), ROOT / "provenance/frozen-code/manifest.json"}
    inputs.update(ROOT / r["path"] for r in config["inputs"].values())
    inputs.update(ROOT / r["path"] for r in source_bindings.values())
    inputs.update(
        ROOT / r["path"] for r in fresh_result["execution_provenance"]["current_modules"].values()
    )
    inputs.update(frozen / OUTPUT_FILENAMES[name] for name in payloads)
    receipt = {
        "schema_version": "forge.compose_lipid_manifest_recovery_replay.v1",
        "status": "all_original_payloads_and_result_fields_reproduced",
        "inputs": {pin(p)["path"]: pin(p) for p in sorted(inputs)},
        "artifacts": {
            name: pin(OUTPUT / ("replayed-" + OUTPUT_FILENAMES[name])) for name in payloads
        },
        "checks": checks,
        "original_implementation_pins": config["implementation"],
        "authenticated_source_bindings": source_bindings,
        "execution_provenance": fresh_result["execution_provenance"],
        "training_calls": 0,
        "generation_calls": 0,
        "proposal_calls": 0,
        "sealed_holdout_accessed": False,
        "original_artifacts_changed": False,
        "scope": "Development benchmark manifest reconstruction only; no benchmark, guidance or training execution.",
    }
    (OUTPUT / "replay.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"], checks)


if __name__ == "__main__":
    main()
