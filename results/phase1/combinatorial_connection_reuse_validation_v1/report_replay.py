"""Authenticate the frozen connection protocol and compare discovery with fresh replay."""

import json
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def main():
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": str(sha256_file(path)),
        }
        return json.loads(path.read_text())

    protocol = read(ROOT / "configs/multireaction/combinatorial_connection_reuse_protocol_v1.json")
    for pin in [*protocol["source_pins"], *protocol["runs"]]:
        resolve_pin(pin, ROOT, label="frozen connection protocol")
        inputs[pin["path"]] = pin
    discovery = read(
        ROOT / "results/phase1/combinatorial_connection_reuse_discovery_v1/result.json"
    )
    replay = read(ROOT / "results/phase1/combinatorial_connection_reuse_replay_v1/result.json")
    excluded = {"created_at_utc", "duration_seconds", "artifacts"}
    keys = set(discovery) - excluded
    assert set(replay) - excluded == keys
    assert all(discovery[key] == replay[key] for key in keys)
    assert set(discovery["artifacts"]) == set(replay["artifacts"])
    for name in discovery["artifacts"]:
        assert discovery["artifacts"][name]["sha256"] == replay["artifacts"][name]["sha256"]
    for result in (discovery, replay):
        assert result["sources"] == protocol["source_pins"]
        assert result["config"] == protocol["runs"][0]
        for pin in [
            result["config"],
            result["baseline_result"],
            *result["sources"],
            *result["artifacts"].values(),
        ]:
            resolve_pin(pin, ROOT, label="connection replay input")
            inputs[pin["path"]] = pin
    inputs[str(Path(__file__).relative_to(ROOT))] = {
        "path": str(Path(__file__).relative_to(ROOT)),
        "sha256": str(sha256_file(Path(__file__))),
    }
    output = {
        "schema_version": "forge.combinatorial_connection_reuse_replay_equivalence.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified_scientific_payload_and_artifact_equivalence",
        "inputs": inputs,
        "compared_fields": sorted(keys),
        "compared_artifacts": sorted(discovery["artifacts"]),
        "excluded": ["execution timestamp", "duration", "output-directory paths"],
        "new_random_sampling": False,
    }
    path = Path(__file__).with_name("replay_equivalence.json")
    if path.exists():
        raise ValueError("replay report already exists")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(output["status"])


if __name__ == "__main__":
    main()
