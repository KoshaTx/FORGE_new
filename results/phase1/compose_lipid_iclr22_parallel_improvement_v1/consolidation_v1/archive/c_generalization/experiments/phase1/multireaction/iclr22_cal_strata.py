"""Audit existing CAL component-generalization strata; never admits final evaluation."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def classify(row):
    identities = {value[1] for value in row["component_instances"]}
    overlap = set(row["component_identity_overlap_TRAIN"])
    if not identities or not overlap <= identities:
        raise ValueError("Component identity/overlap metadata is incomplete or inconsistent")
    novel = identities - overlap
    label = (
        "all_components_seen"
        if not novel
        else "all_components_unseen" if not overlap else "some_components_unseen"
    )
    roles = sorted({value[0] for value in row["component_instances"] if value[1] in novel})
    if bool(row["component_disjoint_reference_eligible"]) != (not overlap):
        raise ValueError("Existing wholly-disjoint predicate differs")
    return label, roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    base = args.root / "results/phase1/compose_lipid_iclr22_research_v1"
    reference_path = base / "routes/cal_reference.json"
    representation_path = (
        base
        / "parallel_completion_v1/training_evaluation/cal_bridge_full_v1/namespace_rollup_v1/result.json"
    )
    reference = json.loads(reference_path.read_text())
    representation = json.loads(representation_path.read_text())
    sources = {r["target_id"]: r for r in reference["rows"]}
    assert len(sources) == len(reference["rows"]) == 2059
    assert len(representation["rows"]) == 2059
    rows = []
    families = defaultdict(Counter)
    strata = Counter()
    roles = defaultdict(Counter)
    for variant in representation["rows"]:
        row = sources[variant["target_id"]]
        assert row["family"] == variant["family"]
        source_admitted = variant["status"] in {
            "admitted_unique_physical",
            "admitted_exact_role_alias",
        }
        label, unseen_roles = classify(row)
        key = label if source_admitted else "not_source_qualified"
        families[row["family"]][key] += 1
        strata[key] += 1
        if source_admitted:
            roles[row["family"]].update(unseen_roles)
        rows.append(
            {
                "target_id": row["target_id"],
                "family": row["family"],
                "source_qualified_representation": source_admitted,
                "component_stratum": label,
                "unseen_roles": unseen_roles,
                "combination_group": row["combination_group"],
                "morphology_group": row["morphology_group"],
                "source_pmids": row["source_pmids"],
                "source_independence_admitted": False,
                "representation_artifact": variant["artifact"],
            }
        )
    assert sum(strata.values()) == 2059
    assert strata["not_source_qualified"] == 200
    assert strata["all_components_unseen"] == 12
    args.output.mkdir(parents=True, exist_ok=False)
    producer_copy = args.output / "producer.py"
    producer_copy.write_bytes(Path(__file__).read_bytes())
    (args.output / "strata.json").write_text(json.dumps(rows, indent=2) + "\n")
    result = {
        "schema": "forge.iclr22.CAL_component_generalization_strata.v1",
        "hypothesis": "A wholly-component-disjoint requirement may hide usable partial held-component diagnostic strata.",
        "alternative_explanation": "Shared or source-unqualified CAL cases cannot establish all-family component generalization.",
        "seed": 2026092682,
        "inputs": [pin(reference_path), pin(representation_path)],
        "producer": pin(producer_copy),
        "population": {
            "original_CAL": 2059,
            "source_qualified": 1859,
            "families": 22,
            "development_only": True,
        },
        "strata": dict(strata),
        "per_family": {k: dict(v) for k, v in sorted(families.items())},
        "unseen_roles_by_family": {k: dict(v) for k, v in sorted(roles.items())},
        "rows": pin(args.output / "strata.json"),
        "checks": {
            "complete_denominator": True,
            "all_wholly_disjoint_metadata_reproduced": True,
            "source_qualification_preserved": True,
        },
        "cost": {
            "CPU_seconds": time.process_time() - started,
            "model_calls": 0,
            "network_calls": 0,
        },
        "final_evaluation_admitted": False,
        "TEST_graphs_loaded": 0,
        "interpretation": "Partial held-component diagnostics and wholly-disjoint tests answer different questions. Existing CAL exposure prevents relabeling these strata as fresh final confirmation.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "strata": result["strata"],
                "per_family": result["per_family"],
                "CPU_seconds": result["cost"]["CPU_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
