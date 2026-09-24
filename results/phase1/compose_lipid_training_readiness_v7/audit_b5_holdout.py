"""Explain the all-family coverage conflict without changing frozen exclusions."""

import json
import sqlite3
from pathlib import Path

from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    partition_path = ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json"
    mapping_path = (
        ROOT
        / "results/phase1/compose_lipid_split_source_recovery_v2/historical-global-component-exclusions.json"
    )
    registry_path = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
    partition = json.loads(partition_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    registry = json.loads(registry_path.read_text())
    if pin(ROOT, registry_path) != mapping["inputs"]["b5_recipe_registry"]:
        raise ValueError("B5 holdout audit registry differs from the component binding")
    canonical = constitutional_molecule(registry["original_task_contract"]["source_core_smiles"])[0]
    catalogue = resolve_pin(mapping["inputs"]["catalogue"], ROOT, label="precursor catalogue")
    matches = [r for r in rows(catalogue) if r["constitution"] == canonical]
    if len(matches) != 1:
        raise ValueError("B5 core has no unique complete catalogue identity")
    core = matches[0]
    bindings = [b for b in mapping["bindings"] if core["component_id"] in b["component_ids"]]
    if not bindings:
        raise ValueError("Claimed B5 core has no recovered historical holdout")
    held = {b["legacy_id"] for b in bindings}
    records_path = resolve_pin(
        mapping["inputs"]["legacy_records"], ROOT, label="original source records"
    )
    original_witnesses = []
    for row in rows(records_path):
        instances = [v for v in row["precursor_instances"] if v["precursor_id"] in held]
        if instances:
            original_witnesses.append(
                {
                    "target_id": row["target_id"],
                    "family": row["family"],
                    "held_instances": instances,
                }
            )
    database = resolve_pin(partition["artifact"], ROOT, label="current partition")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        core_rows = db.execute(
            "SELECT count(*) FROM partition p,json_each(p.protected_component_ids) j WHERE p.family=? AND j.value=?",
            ("vitamin_b5_multistep", core["component_id"]),
        ).fetchone()[0]
        others = [
            dict(
                zip(
                    (
                        "target_id",
                        "old_projection",
                        "corrected_projection",
                        "prior_protected",
                        "canonical_protected",
                        "disposition",
                    ),
                    r,
                )
            )
            for r in db.execute(
                "SELECT target_id,old_projection,corrected_projection,prior_protected,canonical_protected,disposition FROM partition p WHERE family='vitamin_b5_multistep' AND NOT EXISTS (SELECT 1 FROM json_each(p.protected_component_ids) j WHERE j.value=?)",
                (core["component_id"],),
            )
        ]
    family = partition["summary"]["by_family"]["vitamin_b5_multistep"]
    if (
        core_rows + len(others) != family["rows"]
        or family.get("eligible_for_program_preparation", 0) != 0
    ):
        raise ValueError("B5 coverage conflict does not reproduce")
    dump(
        HERE / "b5-frozen-holdout-conflict.json",
        {
            "schema_version": "forge.compose_lipid_frozen_family_coverage_conflict.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "partition": pin(ROOT, partition_path),
                "held_component_bindings": pin(ROOT, mapping_path),
                "recipe_registry": pin(ROOT, registry_path),
                "catalogue": mapping["inputs"]["catalogue"],
                "original_source_records": mapping["inputs"]["legacy_records"],
            },
            "family": "vitamin_b5_multistep",
            "source_rows": family["rows"],
            "core_constitution": canonical,
            "global_core_id": core["component_id"],
            "historical_bindings": bindings,
            "original_held_component_witnesses": original_witnesses,
            "rows_with_held_core": core_rows,
            "remaining_protected_rows": others,
            "eligible_rows": 0,
            "all_23_families_ready": False,
            "conclusion": "Preserving the recovered historical global precursor holdout excludes the common vitamin-B5 core. Every supplied B5 row is protected. All-family training is incompatible with these frozen exclusions; this audit does not authorize changing them.",
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
