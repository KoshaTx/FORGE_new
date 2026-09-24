"""Read-only verification of the completed v1 dataset; print a hash-pinned JSON receipt."""

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from forge.assembly.families import load_assembly_libraries
from forge.assembly.library_programs import replay_library_program
from forge.core.hashing import resolve_pin, sha256_file

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "results/phase1/library_program_dataset_v1/result.json"


def main():
    result = json.loads(RESULT.read_text())
    config_path = resolve_pin(result["config"], REPO, label="config")
    config = json.loads(config_path.read_text())
    paths = {k: resolve_pin(v, REPO, label=k) for k, v in result["inputs"].items()}
    artifacts = {k: resolve_pin(v, REPO, label=k) for k, v in result["artifacts"].items()}
    for path, digest in result["sources"].items():
        assert sha256_file(REPO / path) == digest, path
    libraries = load_assembly_libraries(
        [(paths[k], config["inputs"][k]["sha256"]) for k in config["registries"]],
        expected_families=config["programs"],
    )
    seen, folds, masses = set(), Counter(), Counter()
    with gzip.open(artifacts["products.csv.gz"], "rt") as handle:
        for row in csv.DictReader(handle):
            identity = row["product_id"]
            assert identity not in seen
            assert identity == hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            seen.add(identity)
            folds[row["fold"]] += 1
            mass = float(row["training_sampling_weight"])
            assert math.isfinite(mass) and mass >= 0
            if mass:
                assert row["fold"] == "train"
                assert row["unique_policy_qualified_program_available"] == "1"
                assert float(row["mean_realism_weight"]) > 0
                masses[row["reaction_family"]] += mass
    assert len(seen) == result["summary"]["unique_products"]
    assert dict(folds) == result["summary"]["fold_counts"]
    assert set(masses) == set(result["summary"]["train_family_masses"])
    assert all(math.isclose(mass, 1 / len(masses), abs_tol=1e-12) for mass in masses.values())
    counts, qualified, exact = Counter(), Counter(), Counter()
    with (
        paths["corpus"].open() as source,
        gzip.open(artifacts["row_partitions.csv.gz"], "rt") as output,
    ):
        for index, (raw, row) in enumerate(
            zip(csv.DictReader(source), csv.DictReader(output), strict=True), 1
        ):
            assert int(row["source_row"]) == index
            assert raw["reaction_family"] == row["reaction_family"]
            assert float(raw["realism_weight"]) == float(row["original_realism_weight"])
            assert row["product_id"] in seen
            family = row["reaction_family"]
            counts[family] += 1
            qualified[family] += int(row["unique_policy_qualified_program"])
            exact[family] += int(row["exact_program_recovered"])
    for family, summary in result["summary"]["families"].items():
        assert counts[family] == summary["source_rows"]
        assert qualified[family] == summary["qualified_program_rows"]
        assert exact[family] == summary["exact_program_rows"]
    samples, sampled = [], Counter()
    groups = 0
    with gzip.open(artifacts["programs.jsonl.gz"], "rt") as handle:
        for line in handle:
            group = json.loads(line)
            groups += 1
            family = group["reaction_id"]
            for target in group["targets"]:
                if target["minimum_steps"] is None:
                    continue
                policy = target["policy_path_count_capped_at_two"]
                disposition = (
                    "qualified" if policy == 1 else "policy_rejected" if not policy else "ambiguous"
                )
                kind = (
                    family,
                    disposition,
                    "repeated" if target["minimum_steps"] > 1 else "single",
                )
                if sampled[kind] >= 4:
                    continue
                witness = target["policy_intermediate_products"] or target["intermediate_products"]
                replayed = replay_library_program(
                    libraries[family],
                    dict(group["components"]),
                    witness,
                    accumulator_role=group["accumulator_role"],
                    maximum_outcomes=config["limits"]["maximum_outcomes"],
                )
                assert replayed == bool(policy)
                sampled[kind] += 1
                samples.append(
                    {
                        "group_id": group["group_id"],
                        "target_sha256": hashlib.sha256(
                            target["product_smiles"].encode()
                        ).hexdigest(),
                        "kind": kind,
                        "strict_replay": replayed,
                    }
                )
    assert groups == result["programs"]["groups"]
    print(
        json.dumps(
            {
                "schema_version": "forge.library_program_artifact_checks.v1",
                "status": "pass",
                "inputs": {
                    "result": {
                        "path": str(RESULT.relative_to(REPO)),
                        "sha256": str(sha256_file(RESULT)),
                    },
                    "verifier": {
                        "path": str(Path(__file__).relative_to(REPO)),
                        "sha256": str(sha256_file(Path(__file__))),
                    },
                    "config": result["config"],
                    **result["inputs"],
                },
                "verified_artifacts": result["artifacts"],
                "verified_sources": result["sources"],
                "all_original_rows_and_weights_preserved": sum(counts.values()),
                "unique_products": len(seen),
                "program_groups": groups,
                "training_mass": dict(masses),
                "replay_sample_selection": "First four targets per family, disposition and depth class in deterministic group-hash order.",
                "replay_samples": samples,
                "limitations": [
                    "Exact strict replay is sampled; full program and partition qualification is reported by the pinned dataset builder.",
                    "Artifact integrity does not override failed dataset admission gates.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
