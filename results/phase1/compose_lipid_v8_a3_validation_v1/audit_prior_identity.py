"""Find certain prior-product overlaps by exact constitutional strings, without graph parsing.

Source IDs differ across releases. String matches prove overlap; nonmatches cannot prove
graph disjointness across canonicalization versions. This audit grants no training admission.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
IMPORT_PIN = {
    "path": "results/phase1/compose_lipid_v8_import_v1/result.json",
    "sha256": "a5c9ec8e95542d227eaf2ed3f9ab99e3d51e91c93f28cec1d9227cb7612befae",
}
PRIOR_PIN = {
    "path": "data/source_cache/compose_lipid_v5_program_evidence_2026-09-18/construction_supervision.jsonl.gz",
    "sha256": "860744d5dbda12bc13968ebe907f99a6889b6ad0b130fea5a629fabdb0e7fb6f",
}


def main() -> None:
    imported = json.loads(resolve_pin(IMPORT_PIN, REPO, label="import").read_text())
    prior_path = resolve_pin(PRIOR_PIN, REPO, label="prior construction")
    db_path = resolve_pin(imported["artifacts"]["corpus.sqlite"], REPO, label="protected TRAIN")
    train = defaultdict(list)
    family_counts = Counter()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        for identity, family, constitution in db.execute(
            "SELECT t.target_id,a.family,t.constitution FROM assignments a JOIN targets t USING(target_id) "
            "WHERE a.forge_split='train' ORDER BY a.family,t.target_id"
        ):
            train[constitution].append((identity, family))
            family_counts[family] += 1
    matches = defaultdict(set)
    identities = {}
    with gzip.open(prior_path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            for identity, family in train.get(row["constitution"], []):
                matches[identity].add(
                    (row["target_id"], row.get("prior_development_split") or "unassigned")
                )
                identities[identity] = (
                    family,
                    hashlib.sha256(row["constitution"].encode()).hexdigest(),
                )
    ledger = []
    by_family = {
        name: {
            "inspected_train": count,
            "exact_string_overlap": 0,
            "protected_prior_product_overlap": 0,
            "source_id_equal": 0,
        }
        for name, count in sorted(family_counts.items())
    }
    for identity, prior in sorted(matches.items()):
        family, digest = identities[identity]
        protected = any(
            fold in ("validation", "test", "calibration", "heldout") for _, fold in prior
        )
        equal_id = any(old_id == identity for old_id, _ in prior)
        ledger.append(
            {
                "target_id": identity,
                "family": family,
                "constitution_string_sha256": digest,
                "prior_sources": [
                    {"target_id": old_id, "prior_development_split": fold}
                    for old_id, fold in sorted(prior)
                ],
                "protected_prior_product_overlap": protected,
                "same_source_id": equal_id,
                "training_admitted": False,
            }
        )
        by_family[family]["exact_string_overlap"] += 1
        by_family[family]["protected_prior_product_overlap"] += protected
        by_family[family]["source_id_equal"] += equal_id
    result = {
        "schema_version": "forge.compose_lipid_prior_identity_audit.v1",
        "status": "certain_overlaps_found_nonmatches_unresolved",
        "inputs": {
            "v8_import": IMPORT_PIN,
            "v5_construction": PRIOR_PIN,
            "v8_database": imported["artifacts"]["corpus.sqlite"],
        },
        "implementation": {
            "path": Path(__file__).relative_to(REPO).as_posix(),
            "sha256": str(sha256_file(Path(__file__))),
        },
        "by_family": by_family,
        "overlaps": ledger,
        "matching": "literal stereo-free constitutional string equality; neither source target IDs nor nonmatches establish chemical disjointness",
        "policy": {
            "seed": 0,
            "random_sampling_used": False,
            "molecular_graphs_parsed": 0,
            "training_calls": 0,
            "training_admitted": 0,
        },
        "remaining_gate": "Exclude these protected overlaps in the final TRAIN view; finish graph-identity and precursor protections for all other records before admission.",
    }
    (OUT / "prior_identity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
