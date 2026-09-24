"""Test whether the v5 dictionary can resolve v8 labels; never admit identities."""

import gzip
import json
from collections import Counter
from pathlib import Path

from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    inputs = {
        "prior_audit": {
            "path": "results/phase1/compose_lipid_v8_precursor_audit_v1/result.json",
            "sha256": "371cf8c61c9a631f364672e4162d22f3725cfd2b8bf111d33bd69873ced3812a",
        },
        "dictionary": {
            "path": "data/source_cache/compose_lipid_v5_program_evidence_2026-09-18/precursors.jsonl.gz",
            "sha256": "71b43339935347062fee1e05f0eb3adf8acc3b9efc4d980a9cadacfd69f2c0e0",
        },
    }
    paths = {name: resolve_pin(record, ROOT, label=name) for name, record in inputs.items()}
    audit = json.loads(paths["prior_audit"].read_text())
    with gzip.open(paths["dictionary"], "rt") as stream:
        dictionary = {}
        for row in map(json.loads, stream):
            if row["precursor_id"] in dictionary:
                raise ValueError("duplicate dictionary label")
            dictionary[row["precursor_id"]] = row
    inputs["identities"] = audit["artifacts"]["identities.jsonl.gz"]
    identities = resolve_pin(inputs["identities"], ROOT, label="identities")
    counts = Counter()
    conflicts = []
    with gzip.open(identities, "rt") as stream:
        for row in map(json.loads, stream):
            for binding in row["source_labels"]:
                family, role = binding["family"], binding["role"]
                metadata = json.loads(binding["label"])
                counts[family, role, "scoped_labels"] += 1
                values = list(metadata.values())
                other = (
                    dictionary.get(values[0])
                    if len(values) == 1 and isinstance(values[0], str)
                    else None
                )
                if other is None:
                    counts[family, role, "absent"] += 1
                    continue
                equal = constitutional_molecule(other["constitution"])[0] == row["canonical_smiles"]
                counts[
                    family, role, "same_constitution" if equal else "conflicting_constitution"
                ] += 1
                counts[
                    family,
                    role,
                    "family_declared" if family in other["families"] else "family_not_declared",
                ] += 1
                if not equal:
                    conflicts.append(
                        {"v8": binding, "v8_constitution": row["canonical_smiles"], "v5": other}
                    )
    result = {
        "schema_version": "forge.prior_dictionary_concordance.v1",
        "inputs": inputs,
        "implementation": pin(Path(__file__).resolve()),
        "seed": 0,
        "provider_evaluation_graphs_parsed": False,
        "dictionary_labels_admitted": 0,
        "training_calls": 0,
        "by_family_role_status": {"/".join(k): v for k, v in sorted(counts.items())},
        "conflicts": conflicts,
        "decision": "The prior dictionary is incomplete for the already reconstructed v8 roles. Matching labels are a concordance diagnostic only; no role aliases, new holdout identities or split qualification are inferred.",
    }
    (HERE / "prior_dictionary_concordance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
