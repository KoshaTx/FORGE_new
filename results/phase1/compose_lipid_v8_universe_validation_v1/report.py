"""Summarize full-universe accounting, replay, and completed repository validation."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import sha256_file  # noqa: E402

HELPER = ROOT / "results/phase1/compose_lipid_v8_source_recovery_v1/report.py"
spec = importlib.util.spec_from_file_location("source_recovery_report", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    path = (ROOT / path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    checks = read(OUTPUT / "checks.json")
    assert "full-tests" in checks, "full repository test run must finish"
    assert all(
        check["exit_code"] == 0
        for name, check in checks.items()
        if name not in {"full-tests", "historical-benchmark-tests"}
    )
    replay = read(OUTPUT / "universe-replay.json")
    assert replay["exit_code"] == 0
    universe_path = Path("results/phase1/compose_lipid_v8_universe_readiness_v1/result.json")
    universe = read(universe_path)
    assert replay["input"] == pin(universe_path)
    snapshot = read(OUTPUT / "source-snapshot.json")
    assert all(str(sha256_file(ROOT / path)) == digest for path, digest in snapshot.items())
    assert all(read(OUTPUT / "snapshot-comparison.json").values())
    for name in ("full-tests.log", "full-tests.xml"):
        (OUTPUT / (name + ".gz")).write_bytes(gzip.compress((OUTPUT / name).read_bytes(), mtime=0))
    focused = helper.tests(OUTPUT / "focused-tests.xml")
    full = helper.tests(OUTPUT / "full-tests.xml.gz")
    assert focused["counts"]["passed"] == focused["counts"]["tests"]
    baseline_path = Path("results/phase1/compose_lipid_v8_source_recovery_v1/result.json")
    baseline = read(baseline_path)
    old_bad = set(baseline["full"]["failure_or_error_details"])
    new_bad = set(full["failure_or_error_details"])
    inputs = {
        universe_path,
        baseline_path,
        HELPER.relative_to(ROOT),
        Path(__file__).relative_to(ROOT),
        Path("docs/COMPOSE_LIPID_V8_PRETRAINING.md"),
        Path("docs/COMPOSE_LIPID_V8_SOURCE_CONTRACT_REQUEST.md"),
    }
    inputs.update(Path(pin["path"]) for pin in universe["artifacts"].values())
    inputs.add(Path(universe["config"]["path"]))
    result = {
        "schema_version": "forge.compose_lipid_universe_validation.v1",
        "status": "full_universe_accounted_blocked_on_components_programs_representation_and_repository_inputs",
        "training_ready": False,
        "training_calls": 0,
        "phase1_definition_of_done_met": False,
        "seed": 0,
        "random_sampling_used": False,
        "universe_summary": universe["summary"],
        "full_universe_replay_passed": True,
        "focused": focused,
        "full": full,
        "new_failure_or_error_ids": sorted(new_bad - old_bad),
        "resolved_failure_or_error_ids": sorted(old_bad - new_bad),
        "inputs": {str(path): pin(path) for path in sorted(inputs)},
        "artifacts": {
            path.name: pin(path)
            for path in sorted(OUTPUT.iterdir())
            if path.is_file()
            and path.name not in {"result.json", "full-tests.log", "full-tests.xml"}
        },
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "focused": focused["counts"],
                "full": full["counts"],
                "new_failures": len(new_bad - old_bad),
            }
        )
    )


if __name__ == "__main__":
    main()
