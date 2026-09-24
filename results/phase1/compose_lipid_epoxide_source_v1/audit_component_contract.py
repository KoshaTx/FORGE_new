"""Check original occupancy and complete precursor binding; do not qualify chemistry."""

import json
from collections import Counter
from pathlib import Path

from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    extraction_path = ROOT / "results/phase1/compose_lipid_epoxide_original_tasks_v1/result.json"
    extraction = json.loads(extraction_path.read_text())
    for group in ("inputs", "implementation", "task_shards"):
        for key, value in extraction[group].items():
            resolve_pin(value, ROOT, label=key)
    ledger = resolve_pin(extraction["artifact"], ROOT, label="original tasks")
    intake_path = resolve_pin(extraction["inputs"]["intake"], ROOT, label="intake")
    intake = json.loads(intake_path.read_text())
    catalogue = resolve_pin(intake["inputs"]["precursors"], ROOT, label="precursors")
    selected = list(rows(ledger))
    wanted = {c[1] for row in selected for c in row["preparation"]["component_instances"]}
    structures = {
        row["component_id"]: row["constitution"]
        for row in rows(catalogue)
        if row["component_id"] in wanted
    }
    if set(structures) != wanted:
        raise ComposeLipidError("Eligible epoxide component IDs missing from global catalogue")
    registry_path = ROOT / "data/vendor/qualified_han_db_grouped_source_program_v1.json"
    registry = json.loads(registry_path.read_text())
    reaction = next(
        r for r in registry["reactions"] if r["reaction_id"] == "source_han_db_amine_epoxide"
    )
    queries = {
        r["name"]: Chem.MolFromSmarts(r["required_handle_smarts"])
        for r in reaction["reactant_roles"]
    }
    if any(q is None for q in queries.values()):
        raise ComposeLipidError("Registry structural handle query did not parse")
    counts, occupancy, lanes, heads, failures = Counter(), Counter(), Counter(), {}, Counter()
    records = []
    for row in selected:
        p, task, construction = row["preparation"], row["original_task"], row["construction"]
        if (
            p["eligible_for_program_preparation"] is not True
            or p["old_split"] != "train"
            or p["corrected_split"] != "train"
            or p["exclusion_reasons"]
            or p["pending_reasons"]
        ):
            raise ComposeLipidError("Protected epoxide record reached molecular inspection")
        instances = p["component_instances"]
        checks = {
            "original_task_identity": task["task_id"] == construction["task_id"]
            and task["family"] == p["family"] == "amine_epoxide_opening",
            "no_original_training_admission": task["training_admissible"] is False,
            "complete_roles": len(instances) == 2
            and {r for r, _, _ in instances} == {"amine_head", "epoxide_tail"},
            "metadata_agrees": all(
                row["primary_metadata"][k] == task.get(k) for k in row["primary_metadata"]
            ),
            "declared_repeated_complete_tail": task["one_complete_epoxide_repeated_per_nh"] is True
            and task["mixed_epoxide_identities"] is False,
            "declared_terminal_regioisomer": task["terminal_epoxide_regioisomer_fixed"] is True,
        }
        if not checks["complete_roles"]:
            raise ComposeLipidError("Unexpected epoxide component role inventory")
        ids = {r: identity for r, identity, _ in instances}
        quantities = {r: q for r, _, q in instances}
        declared = task["occupancy"]
        checks["exact_declared_quantities"] = (
            type(declared) is int
            and declared > 0
            and all(type(q) is int for q in quantities.values())
            and quantities == {"amine_head": 1, "epoxide_tail": declared}
        )
        checks["exported_events_agree"] = construction["reaction_steps_sites"] == {
            "task": {"occupancy": declared, "terminal_epoxide_regioisomer_fixed": True},
            "result": {"accounting_verified": True, "reaction_events": declared},
        }
        molecules, identities = {}, {}
        for role, field in (("amine_head", "head_smiles"), ("epoxide_tail", "tail_smiles")):
            identity, mol = constitutional_molecule(structures[ids[role]])
            identities[role], molecules[role] = identity, mol
            checks[role + "_complete_structure_binding"] = (
                identity == constitutional_molecule(task[field])[0]
            )
        matches = {
            r: molecules[r].GetSubstructMatches(q, uniquify=True, maxMatches=1024)
            for r, q in queries.items()
        }
        checks["complete_handle_search"] = all(len(m) < 1024 for m in matches.values())
        hydrogen_count = sum(
            molecules["amine_head"].GetAtomWithIdx(m[0]).GetTotalNumHs()
            for m in matches["amine_head"]
        )
        checks["declared_occupancy_equals_eligible_nh_inventory"] = hydrogen_count == declared
        checks["single_terminal_epoxide_handle"] = len(matches["epoxide_tail"]) == 1
        counts["rows"] += 1
        counts["input_contract_pass"] += all(checks.values())
        occupancy[str(declared)] += 1
        lanes[task["head_lane"]] += 1
        head = heads.setdefault(
            identities["amine_head"],
            {
                "rows": 0,
                "occupancy": declared,
                "eligible_nh_inventory": hydrogen_count,
                "source_subseries": task["source_subseries"],
                "head_lane": task["head_lane"],
            },
        )
        head["rows"] += 1
        failures.update(k for k, value in checks.items() if not value)
        records.append(
            {
                "target_id": p["target_id"],
                "component_instances": instances,
                "checks": checks,
                "chemistry_qualified": False,
                "training_admitted": False,
            }
        )
    if counts["rows"] != extraction["rows"]:
        raise ComposeLipidError("Epoxide input audit lost eligible records")
    dump(
        HERE / "component-contract-audit.json",
        {
            "schema_version": "forge.epoxide_original_component_contract_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "original_extraction": pin(ROOT, extraction_path),
                "selected_task_ledger": pin(ROOT, ledger),
                "global_precursors": pin(ROOT, catalogue),
                "structural_query_registry": pin(ROOT, registry_path),
            },
            "summary": dict(counts),
            "declared_occupancy": dict(sorted(occupancy.items())),
            "head_lanes": dict(lanes),
            "heads": dict(sorted(heads.items())),
            "failed_checks": dict(failures),
            "records": records,
            "source_subseries": sorted({r["original_task"]["source_subseries"] for r in selected}),
            "scope": "Structural queries from the qualified Han registry are used only to audit original input inventories. Anderson source-program qualification is not transferred from Han, and no target is read to choose sites or quantities.",
            "next_dependency": "Recover Anderson DOI 10.1073/pnas.0910603106 supplementary exact component/product controls and applicable occupancy/regioselectivity evidence.",
            "chemistry_qualified": False,
            "training_admitted": False,
            "training_calls": 0,
        },
    )
    print(json.dumps({"summary": dict(counts), "failed_checks": dict(failures)}, sort_keys=True))


if __name__ == "__main__":
    main()
