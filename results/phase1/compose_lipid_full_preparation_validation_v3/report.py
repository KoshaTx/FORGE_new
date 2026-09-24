"""Independently join full protection and supplied replay, preserving all unresolved rows."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.core.hashing import resolve_pin, sha256_file  # noqa: E402
from forge.corpus.compose_lipid_source_view import dump, pin  # noqa: E402
from forge.corpus.compose_lipid_supplement import compact, rows  # noqa: E402


def authenticated(name):
    path = ROOT / name
    result = json.loads(path.read_text())
    for label, value in result["implementation"].items():
        resolve_pin(value, ROOT, label=label)
    config = json.loads(resolve_pin(result["config"], ROOT, label="config").read_text())
    if result["inputs"] != config["inputs"]:
        raise ValueError("Receipt/config input disagreement")
    for label, value in result["inputs"].items():
        resolve_pin(value, ROOT, label=label)
    for label, value in result.get("artifacts", {}).items():
        resolve_pin(value, ROOT, label=label)
    artifact = result.get("artifact", result.get("artifacts", {}).get("replay.jsonl.gz"))
    return result, resolve_pin(artifact, ROOT, label="ledger")


def main():
    view_path = "results/phase1/compose_lipid_full_preparation_v1/result.json"
    replay_path = "results/phase1/compose_lipid_family_replay_v1/result.json"
    view, full_ledger = authenticated(view_path)
    replay, replay_ledger = authenticated(replay_path)
    if replay["inputs"]["preparation"] != pin(ROOT, ROOT / view_path):
        raise ValueError("Replay uses another protection view")
    source_view_path = resolve_pin(view["inputs"]["source_view"], ROOT, label="previous view")
    original_view = json.loads(source_view_path.read_text())
    original = {r["target_id"]: r for r in rows(resolve_pin(original_view["artifact"], ROOT, label="original preparation")) if r["eligible_for_program_preparation"]}
    allowed, counts, reasons = {}, defaultdict(Counter), Counter()
    full_counts, dispositions = Counter(), Counter()
    maximum = 0
    eligible_identities = defaultdict(list)
    for row in rows(full_ledger):
        family = row["family"]
        full_counts[family] += 1
        dispositions[row["disposition"]] += 1
        maximum = max(maximum, row["source_declared_heavy_atoms"])
        if row["training_admitted"]:
            raise ValueError("Full preparation admitted training")
        if row["eligible_for_program_preparation"]:
            target = row["target_id"]
            if (target in allowed or target not in original or row["old_split"] != "train"
                or row["corrected_split"] != "train" or row["exclusion_reasons"] or row["pending_reasons"]):
                raise ValueError("Protected/unassigned/duplicate target is eligible")
            if any(row[k] != original[target][k] for k in ("family", "constitution_id", "component_instances")):
                raise ValueError("Preparation changed an authenticated selected recipe")
            allowed[target] = row
            eligible_identities[row["constitution_id"]].append(target)
            counts[family]["eligible_preparation_rows"] += 1
        elif row["target_id"] in original:
            reasons.update(row["exclusion_reasons"] + row["pending_reasons"])
    expected = {k: v["rows"] for k, v in view["summary"]["by_family"].items()}
    if dict(full_counts) != expected or dict(dispositions) != view["summary"]["dispositions"]:
        raise ValueError("Full ledger coverage differs from its receipt")
    seen, outcomes, exact = set(), {}, set()
    for row in rows(replay_ledger):
        target = row["target_id"]
        if target in seen or target not in allowed:
            raise ValueError("Replay duplicated or released a target")
        seen.add(target)
        if any(row[k] != allowed[target][k] for k in ("family", "constitution_id", "component_instances", "construction_basis")):
            raise ValueError("Replay altered source recipe, quantities or identity")
        if row["training_admitted"] or row["experimental_execution_admitted"]:
            raise ValueError("Computed replay was promoted to unsupported supervision")
        value = row["replay"]
        family = row["family"]
        outcomes[target] = value["disposition"]
        counts[family][value["disposition"]] += 1
        if value["computed_consistency_pass"]:
            if not value["checks"] or not all(value["checks"].values()) or value["disposition"] != "exact_computed_reconstruction":
                raise ValueError("Incomplete check promoted to exact")
            if value["verified_target_constitution_id"] != row["constitution_id"]:
                raise ValueError("Exact replay has the wrong target identity")
            products = value.get("forward_products")
            if "forward_layers" in value:
                products = value["forward_layers"][-1]
            if products is not None and (len(products) != 1 or hashlib.sha256(products[0].encode()).hexdigest() != row["constitution_id"]):
                raise ValueError("Stored forward product does not reconstruct the target")
            exact.add(target)
    if seen != set(allowed):
        raise ValueError("Replay did not account for every preparation record")
    binding_path = "results/phase1/compose_lipid_role_binding_v2/result.json"
    binding_result, binding_ledger = authenticated(binding_path)
    if binding_result["inputs"]["preparation"] != pin(ROOT, ROOT / view_path):
        raise ValueError("Role concordance uses another protected population")
    translated = set()
    for row in rows(binding_ledger):
        target = row["target_id"]
        if target not in allowed or target in translated or row["family"] != "aldehyde_ugi3":
            raise ValueError("Role-concordance replay changed its population")
        if any(row[k] != allowed[target][k] for k in ("family", "constitution_id", "component_instances", "construction_basis")):
            raise ValueError("Role concordance altered supplied precursor instances")
        if row["training_admitted"] or row["experimental_execution_admitted"]:
            raise ValueError("Role concordance promoted experimental or training evidence")
        value = row["replay"]
        if outcomes[target] != "unsupported_source_role_tuple":
            raise ValueError("Role-concordance overlay must resolve only the recorded namespace gap")
        translated.add(target)
        counts[row["family"]][outcomes[target]] -= 1
        outcomes[target] = value["disposition"]
        counts[row["family"]][outcomes[target]] += 1
        if value["computed_consistency_pass"]:
            if (not value["checks"] or not all(value["checks"].values())
                or value["verified_target_constitution_id"] != row["constitution_id"]
                or len(value["forward_products"]) != 1
                or hashlib.sha256(value["forward_products"][0].encode()).hexdigest() != row["constitution_id"]):
                raise ValueError("Role-concordance replay is incomplete or has the wrong product")
            exact.add(target)
    if translated != {target for target, row in allowed.items() if row["family"] == "aldehyde_ugi3"}:
        raise ValueError("Role-concordance replay omitted a protected preparation row")
    config = json.loads(resolve_pin(replay["config"], ROOT, label="replay config").read_text())
    implemented = set(config["qualified_executor_families"])
    definitions_path = resolve_pin(view["inputs"]["family_definitions"], ROOT, label="family definitions")
    definitions = json.loads(definitions_path.read_text())
    missing_path = ROOT / "results/phase1/compose_lipid_supplement_intake_v1/missing_source_files.json"
    missing = json.loads(missing_path.read_text())
    missing_by_family = {
        "aldehyde_ugi3": [p for p in missing["missing_paths"] if "/family_05_" in p],
        "aldehyde_ugi4": [p for p in missing["missing_paths"] if "/family_03_" in p],
        "thiolactone_aminolysis_michael": [p for p in missing["missing_paths"] if "/family_18_" in p],
        "acid_epoxide_diester_multistep": [p for p in missing["missing_paths"] if "/family_21_" in p],
    }
    family_report = {}
    for family in sorted(expected):
        family_report[family] = {
            "preparation": view["summary"]["by_family"][family],
            "supplied_replay": dict(counts[family]),
            "family_definition_present": family in definitions,
            "qualified_executor_available": family in implemented,
            "missing_original_package_files": missing_by_family.get(family, []),
            "missing_originals_are_blanket_replay_requirement": False,
            "remaining_program_work": (
                "source_specific_scope_and_any_unresolved_recipes_remain_separate"
                if family in implemented else "qualify_source_program_sites_stages_and_controls"
                if family in definitions else "reference_category_not_program_supervision"
            ),
        }
    exact_classes = {identity: sorted(t for t in targets if t in exact)
                     for identity, targets in eligible_identities.items() if any(t in exact for t in targets)}
    dedup = {
        "schema_version": "forge.compose_lipid_preparation_identity_classes.v1",
        "scope": "eligible_preparation_only_not_full_universe_deduplication",
        "inputs": {"preparation": pin(ROOT, ROOT / view_path), "replay": pin(ROOT, ROOT / replay_path), "role_binding": pin(ROOT, ROOT / binding_path)},
        "implementation": pin(ROOT, Path(__file__)), "seed": 0,
        "eligible_source_rows": len(allowed), "eligible_constitutional_classes": len(eligible_identities),
        "exact_source_rows": len(exact), "exact_constitutional_classes": len(exact_classes),
        "duplicate_eligible_classes": {k: sorted(v) for k, v in sorted(eligible_identities.items()) if len(v) > 1},
        "duplicate_exact_classes": {k: v for k, v in sorted(exact_classes.items()) if len(v) > 1},
        "training_weights_fitted": False, "training_admitted": False,
    }
    dump(OUT / "constitutional_deduplication.json", dedup)
    final = OUT / "readiness.jsonl.gz"
    with tempfile.NamedTemporaryFile(dir=OUT, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
            for row in rows(full_ledger):
                target = row["target_id"]
                assessment = outcomes.get(target, "not_run_protected_or_unassigned")
                stream.write((compact({
                    "target_id": target, "family": row["family"],
                    "preparation_disposition": row["disposition"],
                    "reconstruction_disposition": assessment,
                    "exclusion_reasons": row["exclusion_reasons"], "pending_reasons": row["pending_reasons"],
                    "passes_completed_recipe_checks": target in exact,
                    "training_admitted": False,
                }) + "\n").encode())
    if final.exists():
        if sha256_file(temporary) != sha256_file(final):
            raise ValueError("Readiness ledger did not reproduce byte-for-byte")
        temporary.unlink()
    else:
        os.replace(temporary, final)
    report = {
        "schema_version": "forge.compose_lipid_full_preparation_report.v1", "seed": 0,
        "implementation": pin(ROOT, Path(__file__)),
        "inputs": {"preparation": pin(ROOT, ROOT / view_path), "replay": pin(ROOT, ROOT / replay_path), "role_binding": pin(ROOT, ROOT / binding_path), "missing_source_inventory": pin(ROOT, missing_path)},
        "artifacts": {"readiness.jsonl.gz": pin(ROOT, final), "constitutional_deduplication.json": pin(ROOT, OUT / "constitutional_deduplication.json")},
        "totals": {"source_records": sum(full_counts.values()), "preparation_dispositions": dict(dispositions),
            "eligible_preparation_rows": len(allowed), "supplied_exact_reconstructions": len(exact),
            "eligible_rows_without_exact_supplied_reconstruction": len(allowed) - len(exact),
            "newly_excluded_previous_preparation_rows": len(original) - len(allowed),
            "new_exclusion_reason_counts_overlap": dict(reasons), "maximum_source_declared_heavy_atoms": maximum},
        "by_family": family_report,
        "checks": {"all_source_records_accounted": True, "replay_target_coverage_exact": True,
            "protected_or_unassigned_replay_targets": 0, "recipe_role_quantity_changes": 0,
            "forward_product_identity_mismatches": 0, "training_calls": 0},
        "remaining_gates": ["qualified_full_universe_partition_and_constitutional_overlap",
            "full_source_study_and_morphology_protection", "remaining_source_program_qualification",
            "full_size_representation_on_the_final_eligible_population", "final_constitutional_deduplication_and_balanced_weights",
            "repository_wide_tests"],
        "training_admitted": False,
    }
    dump(OUT / "reconstruction_report.json", report)
    print(report["totals"], flush=True)


if __name__ == "__main__":
    main()
