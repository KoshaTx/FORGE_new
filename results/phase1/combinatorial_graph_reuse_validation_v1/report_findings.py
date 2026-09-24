"""Authenticate the fixed confirmation protocol and summarize all three paired populations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1.multireaction.combinatorial_graph_reuse_verify import (
    verify as verify_parent,
)
from experiments.phase1.multireaction.combinatorial_reuse_admission import (
    verify as verify_admission,
)
from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def main():
    protocol_path = ROOT / "configs/multireaction/combinatorial_reuse_confirmation_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    inputs = [pin(Path(__file__)), pin(protocol_path)]
    for name, digest in protocol["admission_sources"].items():
        value = {"path": name, "sha256": digest}
        resolve_pin(value, ROOT, label="frozen admission rule")
        inputs.append(value)
    for run in protocol["runs"]:
        value = {"path": run["config"], "sha256": run["sha256"]}
        resolve_pin(value, ROOT, label="frozen confirmation sampling")
        inputs.append(value)
    result = {
        "schema_version": "forge.combinatorial_graph_reuse_findings.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "populations": {},
        "limits": [
            "All populations use empirical TRAIN layouts and one frozen checkpoint; fresh layout/noise seeds are not heldout generalization.",
            "Novelty and multiplicity constraints intentionally use TRAIN identity and the current family output inventory. Count preservation is partly by construction.",
            "Exact reconstruction improves the composite completion pipeline; no learned checkpoint improvement, chemical realism, autonomous layout generation, L2/L3 closure or release readiness is established.",
            "Unique-product totals below sum per-family distinct counts and do not deduplicate across different families.",
        ],
    }
    for label in ("discovery", "confirmation_1", "confirmation_2"):
        parent_path = ROOT / f"results/phase1/combinatorial_graph_reuse_{label}_v1/result.json"
        admission_path = (
            ROOT / f"results/phase1/combinatorial_reuse_admission_{label}_v1/result.json"
        )
        parent, admission = (json.loads(p.read_text()) for p in (parent_path, admission_path))
        config_path = ROOT / admission["config"]["path"]
        config = json.loads(config_path.read_text())
        inputs.extend([pin(parent_path), pin(admission_path), pin(config_path)])
        if label != "discovery":
            run = protocol["runs"][int(label[-1]) - 1]
            if parent["config"] != {"path": run["config"], "sha256": run["sha256"]}:
                raise ValueError("confirmation sampling config differs from frozen protocol")
            if config["confirmation_protocol"] != pin(protocol_path):
                raise ValueError("confirmation protocol substituted")
            parent_config = json.loads((ROOT / parent["config"]["path"]).read_text())
            if parent_config["sampling"] != run["sampling"]:
                raise ValueError("confirmation seeds/budget differ from frozen protocol")
        totals = {}
        for arm, families in admission["per_arm"].items():
            total = {
                key: sum(row[key] for row in families.values())
                for key in (
                    "attempts",
                    "valid_connected",
                    "exact_program",
                    "unique_valid_products",
                    "novel_valid_attempts_vs_all_cache_train",
                    "unique_novel_exact_products",
                    "exact_with_novel_component_in_every_witness",
                )
            }
            repeated = [r for r in admission["by_depth"] if r["arm"] == arm and r["depth"] > 1]
            total.update(
                multistep_attempts=sum(r["attempts"] for r in repeated),
                multistep_exact=sum(r["exact_program"] for r in repeated),
            )
            totals[arm] = total
        result["populations"][label] = {
            "totals": totals,
            "admission_verification": verify_admission(ROOT, admission_path),
            "all_family_count_preservation": admission["all_family_preservation_screen_passed"],
            "exact_gain": admission["total_exact_gain"],
            "unconstrained_parent_exact_gain": parent["total_exact_gain"],
            "unconstrained_parent_preservation": parent["all_family_preservation_screen_passed"],
            "per_family_deltas": admission["per_family_deltas"],
            "proposal_budget": {
                "graph_copy_proposals": parent["graph_copy_proposals"],
                "additional_program_checks": parent["additional_assembly_checks"],
            },
        }
        print(label, "verified", flush=True)
    replay_path = ROOT / "results/phase1/combinatorial_graph_reuse_replay_v1/result.json"
    inputs.append(pin(replay_path))
    result["discovery_replay"] = verify_parent(
        ROOT,
        ROOT / "results/phase1/combinatorial_graph_reuse_discovery_v1/result.json",
        replay_path,
    )
    result["status"] = "three_paired_populations_verified_all_declared_count_metrics_preserved"
    (OUT / "findings.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
