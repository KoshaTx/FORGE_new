"""Authenticate original recipe audit, repeatability and validation without admitting training."""

import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_original_binding import FAMILIES, POLICY, RESULT_SCHEMA
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    paths = {
        "original_binding": ROOT / "results/phase1/compose_lipid_original_binding_v1/result.json",
        "replay": ROOT / "results/phase1/compose_lipid_original_binding_replay_v1/result.json",
        "validation": OUT / "validation_report.json",
        "source_abstention": ROOT
        / "results/phase1/compose_lipid_acid_epoxide_source_v1/result.json",
        "previous_science": ROOT
        / "results/phase1/compose_lipid_full_preparation_validation_v3/result.json",
        "task_intake": ROOT / "results/phase1/compose_lipid_original_tasks_intake_v1/result.json",
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    for name in ("original_binding", "replay"):
        receipt = data[name]
        assert receipt["schema_version"] == RESULT_SCHEMA and receipt["policy"] == POLICY
        resolve_pin(receipt["config"], ROOT, label="config")
        for group in ("inputs", "implementation", "artifacts"):
            for key, value in receipt[group].items():
                resolve_pin(value, ROOT, label=key)
    first, second = data["original_binding"], data["replay"]
    for field in ("config", "inputs", "implementation", "policy", "summary"):
        assert first[field] == second[field], field
    assert (
        first["artifacts"]["bindings.jsonl.gz"]["sha256"]
        == second["artifacts"]["bindings.jsonl.gz"]["sha256"]
    )
    assert set(first["summary"]) == FAMILIES
    selected = sum(row["rows"] for row in first["summary"].values())
    verified = sum(row["verified_recipe_rows"] for row in first["summary"].values())
    validation = data["validation"]
    for name, value in validation["inputs"].items():
        resolve_pin(value, ROOT, label=name)
    assert not validation["new_failures"] and not validation["new_errors"]
    assert validation["source_snapshot_unchanged"]
    assert validation["vendor_verify_exit_code"] == 0
    assert validation["focused"]["failures"] == validation["focused"]["errors"] == 0
    summary = {
        "selected_recipes": selected,
        "verified_recipes": verified,
        "disagreements_or_unresolved": selected - verified,
        "by_family": first["summary"],
        "reaction_calls": 0,
        "new_reaction_qualified_records": 0,
        "training_calls": 0,
        "training_admitted": False,
        "training_ready": False,
    }
    comparison = {
        "inputs": {k: pin(ROOT, paths[k]) for k in ("original_binding", "replay")},
        "summary_identical": True,
        "ledger_byte_identical": True,
        "ledger_sha256": first["artifacts"]["bindings.jsonl.gz"]["sha256"],
    }
    dump(OUT / "replay_comparison.json", comparison)
    result = {
        "schema_version": "forge.compose_lipid_original_binding_closeout.v1",
        "status": "recipe_provenance_verified_training_unqualified",
        "date": "2026-09-20",
        "seed": 0,
        "inputs": {k: pin(ROOT, p) for k, p in paths.items()},
        "implementation": pin(ROOT, Path(__file__)),
        "summary": summary,
        "validation": validation,
        "replay_comparison": pin(ROOT, OUT / "replay_comparison.json"),
        "phase1_definition_of_done_met": validation["phase1_definition_of_done_met"],
        "open_gates": [
            "all_family_source_program_qualification",
            "full_universe_partition_and_study_isolation",
            "final_population_representation_including_large_molecules",
            "global_constitutional_deduplication",
            "balanced_training_weights",
            "repository_wide_test_gate",
        ],
    }
    dump(OUT / "result.json", result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
