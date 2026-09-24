"""Inventory authenticated eligible B5 source metadata without changing admission."""

from collections import Counter
from pathlib import Path

from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    prep = ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
    reader = FullPreparationCorpus(ROOT, prep)
    structures = {r["component_id"]: r["constitution"] for r in rows(reader.precursors)}
    counts = Counter()
    examples = {}
    total = 0
    for item in reader.iter_preparation_records(family="vitamin_b5_multistep"):
        metadata = item["source"].get("primary_metadata", {})
        instances = item["preparation"]["component_instances"]
        key = compact(
            {
                "metadata": {
                    k: metadata.get(k) for k in ("series", "design_lane", "head_axis", "tail_axis")
                },
                "roles": [[r, q] for r, _, q in instances],
            }
        )
        counts[key] += 1
        total += 1
        if key not in examples:
            examples[key] = {
                "target_id": item["preparation"]["target_id"],
                "metadata": metadata,
                "component_instances": instances,
                "component_structures": {i: structures[i] for _, i, _ in instances},
            }
    expected = reader.result["summary"]["by_family"]["vitamin_b5_multistep"][
        "eligible_for_program_preparation"
    ]
    if total != expected:
        raise ValueError("Incomplete B5 eligible inventory")
    dump(
        HERE / "eligible-bindings.json",
        {
            "schema_version": "forge.b5_source_binding_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {"preparation": pin(ROOT, prep), "precursors": pin(ROOT, reader.precursors)},
            "population": "current_full_preparation_eligible_only",
            "target_molecular_payload_not_used_for_profile_selection": True,
            "rows": total,
            "groups": [
                {"profile": k, "rows": counts[k], "example": examples[k]} for k in sorted(counts)
            ],
            "training_admitted": False,
        },
    )


if __name__ == "__main__":
    main()
