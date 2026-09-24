"""Verify the frozen occurrence experiment and summarize all retained populations."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1.multireaction.combinatorial_occurrence_reuse import verify
from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def main():
    protocol_path = ROOT / "configs/multireaction/combinatorial_occurrence_reuse_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    inputs = [pin(Path(__file__)), pin(protocol_path)]
    for value in protocol["source_pins"]:
        resolve_pin(value, ROOT, label="frozen occurrence source")
        inputs.append(value)
    populations = {}
    for label, frozen in zip(
        ("discovery", "replication_1", "replication_2"), protocol["runs"], strict=True
    ):
        config_pin = {"path": frozen["config"], "sha256": frozen["sha256"]}
        resolve_pin(config_pin, ROOT, label="frozen occurrence config")
        inputs.append(config_pin)
        path = ROOT / f"results/phase1/combinatorial_occurrence_reuse_{label}_v1/result.json"
        result = json.loads(path.read_text())
        if result["config"] != config_pin or result["sources"] != protocol["source_pins"]:
            raise ValueError("occurrence experiment differs from frozen protocol")
        inputs.append(pin(path))
        receipt = verify(ROOT, path)
        deep = [
            r
            for r in result["by_depth"]
            if r["family"] == "urea_amine_isocyanate" and r["depth"] > 2
        ]
        gain = sum(r["exact_program"] * (1 if r["arm"] == "graph_reuse" else -1) for r in deep)
        totals = {}
        all_arms = {
            "frozen_model": result["comparison_vs_frozen_model"]["per_arm"]["baseline"],
            "previous_completion": result["per_arm"]["baseline"],
            "occurrence_completion": result["per_arm"]["graph_reuse"],
        }
        for arm, families in all_arms.items():
            totals[arm] = {
                key: sum(row[key] for row in families.values())
                for key in (
                    "attempts",
                    "valid_connected",
                    "exact_program",
                    "novel_valid_attempts_vs_all_cache_train",
                    "unique_valid_products",
                    "unique_novel_exact_products",
                )
            }
        populations[label] = {
            "verification": receipt,
            "totals": totals,
            "qualification": result["qualification"],
            "additional_exact_gain": result["total_exact_gain"],
            "deeper_urea_exact_gain": gain,
            "urea_depths": [
                r for r in result["by_depth"] if r["family"] == "urea_amine_isocyanate"
            ],
            "per_family_deltas": result["per_family_deltas"],
            "acceptance_passed": result["total_exact_gain"] > 0
            and gain > 0
            and result["all_family_preservation_screen_passed"],
            "proposal_dispositions": result["proposal_dispositions"],
            "graph_copy_proposals": result["graph_copy_proposals"],
            "additional_assembly_checks": result["additional_assembly_checks"],
        }
        print(label, "verified", flush=True)
    discovery_path = ROOT / "results/phase1/combinatorial_occurrence_reuse_discovery_v1/result.json"
    replay_path = ROOT / "results/phase1/combinatorial_occurrence_reuse_replay_v1/result.json"
    inputs.append(pin(replay_path))
    replay_receipt = verify(ROOT, replay_path)

    def stable(value):
        value = copy.deepcopy(value)
        value["artifact_hashes"] = {k: p["sha256"] for k, p in value["artifacts"].items()}
        for key in ("created_at_utc", "duration_seconds", "artifacts"):
            value.pop(key)
        return value

    equal = stable(json.loads(discovery_path.read_text())) == stable(
        json.loads(replay_path.read_text())
    )
    if not equal:
        raise ValueError("occurrence replay scientific payload differs")
    result = {
        "schema_version": "forge.combinatorial_occurrence_reuse_evidence.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "populations": populations,
        "replay_verification": replay_receipt,
        "scientific_payload_replay_identical": equal,
        "all_populations_pass": all(p["acceptance_passed"] for p in populations.values()),
        "limits": [
            "These are previously saved TRAIN layout/noise populations. They are not new heldout evidence or independent model training.",
            "The new relation is exact source-derived occurrence context. New-layout construction remains unproven.",
            "All original denominators and unsuccessful outputs remain; count preservation still uses explicit TRAIN identity and current-family multiplicities.",
            "Summed within-family distinct counts do not deduplicate across families. No chemical realism or L2/L3 closure is established.",
        ],
    }
    (OUT / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
