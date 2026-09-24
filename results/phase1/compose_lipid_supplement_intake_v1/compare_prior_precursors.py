"""Compare prior TRAIN-only inferred precursor sets with supplied source components."""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import resolve_pin, sha256_file  # noqa: E402


def read(path):
    with gzip.open(path, "rt") as stream:
        yield from map(json.loads, stream)


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def main():
    audit_path = ROOT / "results/phase1/compose_lipid_v8_precursor_audit_v2/result.json"
    audit = json.loads(audit_path.read_text())
    identities_path = resolve_pin(
        audit["artifacts"]["identities.jsonl.gz"], ROOT, label="prior inferred identities"
    )
    rows_path = resolve_pin(audit["artifacts"]["rows.jsonl.gz"], ROOT, label="prior TRAIN audit")
    source_check_path = OUT / "precursor_split_check.json"
    source_check = json.loads(source_check_path.read_text())
    manifest_path = resolve_pin(
        source_check["inputs"]["components"], ROOT, label="supplied component manifest"
    )
    identities = {row["constitution_id"]: row["canonical_smiles"] for row in read(identities_path)}
    inferred = {row["target_id"]: row for row in read(rows_path) if row["exact_program_evidence"]}
    counts = defaultdict(Counter)
    disagreements = []
    seen = set()
    for row in read(manifest_path):
        target = row["target_id"]
        if target not in inferred:
            continue
        if target in seen:
            raise ValueError("Duplicate supplied target")
        seen.add(target)
        prior = inferred[target]
        if prior["family"] != row["family"]:
            raise ValueError("Family mismatch")
        expected = {item["constitution"] for item in row["component_instances"]}
        observed = {identities[identity] for identity in prior["component_ids"]}
        family = counts[row["family"]]
        family["exact_replay_targets_compared"] += 1
        same = expected == observed
        family[
            "matching_precursor_structure_sets" if same else "different_precursor_structure_sets"
        ] += 1
        if not same:
            disagreements.append(
                {
                    "target_id": target,
                    "family": row["family"],
                    "inferred_only": sorted(observed - expected),
                    "supplied_only": sorted(expected - observed),
                }
            )
    if seen != inferred.keys():
        raise ValueError("Supplied manifest misses prior exact-replay targets")
    totals = Counter()
    for family in counts.values():
        totals.update(family)
    result = {
        "schema_version": "forge.compose_lipid_prior_precursor_concordance.v1",
        "inputs": {
            "prior_audit": pin(audit_path),
            "prior_identities": pin(identities_path),
            "prior_rows": pin(rows_path),
            "supplied_manifest": pin(manifest_path),
            "supplied_identity_check": pin(source_check_path),
        },
        "implementation": pin(Path(__file__)),
        "totals": dict(totals),
        "by_family": {key: dict(value) for key, value in sorted(counts.items())},
        "scope": "Exact precursor structure SET agreement on previously inspected TRAIN records only; role, multiplicity and historical-route precision are not established by this comparison.",
        "training_admitted": False,
        "seed": 0,
    }
    with (OUT / "prior_precursor_disagreements.jsonl").open("w") as stream:
        for row in disagreements:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result["disagreements"] = pin(OUT / "prior_precursor_disagreements.jsonl")
    (OUT / "prior_precursor_concordance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
