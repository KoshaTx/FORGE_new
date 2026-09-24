"""Bind recovered legacy holdout identifiers to existing complete global precursors.

Exact B5 head inputs are resolved through the original task recipe. Synthetic
tail-design groups retain their original group protection without inventing a molecule.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    recovery_path = HERE / "historical-frozen-group-recovery.json"
    recovery = json.loads(recovery_path.read_text())
    if recovery["pass"] is not True:
        raise ValueError("Historical frozen groups are unqualified")
    group_path = resolve_pin(recovery["artifact"], ROOT, label="historical frozen groups")
    selected = set(json.loads(group_path.read_text())["selected_precursors"])
    cache_path = resolve_pin(
        recovery["inputs"]["original_replay"], ROOT, label="original precursor replay"
    )
    cache = json.loads(cache_path.read_text())
    intake_path = resolve_pin(cache["inputs"]["intake"], ROOT, label="supplement intake")
    intake = json.loads(intake_path.read_text())
    catalog_path = resolve_pin(
        intake["inputs"]["precursors"], ROOT, label="complete precursor catalogue"
    )
    history_path = resolve_pin(cache["inputs"]["historical_policy"], ROOT, label="legacy policy")
    history = json.loads(history_path.read_text())
    driver_path = resolve_pin(
        history["source_assets"]["scripts/build_post_instruction_generator_splits_v8.py"]["file"],
        ROOT,
        label="source aliases",
    )
    tree = ast.parse(driver_path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "virtual_instances"
    )
    prefixes = {}
    for branch in function.body:
        if not isinstance(branch, ast.If):
            continue
        families = ast.literal_eval(branch.test.comparators[0])
        families = [families] if isinstance(families, str) else sorted(families)
        for node in ast.walk(branch):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "p"
            ):
                continue
            value = node.args[1]
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == "metadata"
            ):
                for family in families:
                    prefixes[(family, ast.literal_eval(value.slice))] = ast.literal_eval(
                        node.args[0]
                    )
    mapping, bases, raw_ids = defaultdict(set), defaultdict(set), {}
    catalogs = list(rows(catalog_path))
    by_smiles = {row["constitution"]: row["component_id"] for row in catalogs}
    if len(by_smiles) != len(catalogs):
        raise ValueError("Complete precursor graph identity is not globally unique")
    for row in catalogs:
        for alias in row["scoped_aliases"]:
            identities = [(alias["value"], "supplied_scoped_alias")]
            prefix = prefixes.get((alias["family"], alias["field"]))
            if prefix is not None:
                identities.append(
                    (f"{alias['family']}:{prefix}:{alias['value']}", "source_namespaced_alias")
                )
            for identity, basis in identities:
                if identity in selected:
                    mapping[identity].add(row["component_id"])
                    bases[identity].add(basis)
    # Exact old role/string IDs and complete graphs for named MUSCLE contexts were
    # jointly authenticated by the 200k combination replay.
    muscle_path = resolve_pin(
        cache["inputs"]["muscle_contexts"], ROOT, label="original named contexts"
    )
    records_path = resolve_pin(
        cache["artifacts"]["records.jsonl.gz"], ROOT, label="reproduced legacy component records"
    )
    contexts = {r["graph"]["identity_id"]: r for r in rows(muscle_path)}
    for row in rows(records_path):
        for item in row["precursor_instances"]:
            if item["precursor_id"] in selected:
                raw_ids.setdefault(item["precursor_id"], set()).add((row["family"], item["role"]))
        if row["target_id"] not in contexts:
            continue
        context = contexts[row["target_id"]]
        smiles = {p["role"]: p["smiles"] for p in context["precursors"]}
        if len(smiles) != len(context["precursors"]):
            raise ValueError("Named context roles are not unique")
        for item in row["precursor_instances"]:
            identity = item["precursor_id"]
            if identity in selected:
                canonical, _ = constitutional_molecule(smiles[item["role"]])
                mapping[identity].add(by_smiles[canonical])
                bases[identity].add("authenticated_exact_named_context_graph")
    b5_result_path = ROOT / "results/phase1/compose_lipid_b5_original_tasks_v1/result.json"
    b5_result = json.loads(b5_result_path.read_text())
    b5_registry_path = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
    profiles = json.loads(b5_registry_path.read_text())["original_task_contract"]["profiles"]
    missing_heads = {
        identity
        for identity in selected - set(mapping)
        if raw_ids[identity] == {("vitamin_b5_multistep", "series_head")}
    }
    head_graphs, head_evidence = {}, defaultdict(list)
    for shard, shard_pin in sorted(b5_result["task_shards"].items()):
        shard_path = resolve_pin(shard_pin, ROOT, label=shard)
        for line, task in enumerate(rows(shard_path), 1):
            identity = task["head_id"]
            if identity not in missing_heads:
                continue
            profile = profiles[task["design_lane"]]
            order = profile["reagent_order"]
            if (
                len(task["reagents"]) != len(order)
                or [r["role"] for r in task["reagents"]] != [r[1] for r in order]
                or task["events"] != profile["events"]
                or any(task[k] != v for k, v in profile["selector"].items())
            ):
                raise ValueError("Original B5 head task disagrees with its pinned recipe")
            positions = [i for i, role in enumerate(order) if role[0] == profile["head_role"]]
            if len(positions) != 1:
                raise ValueError("B5 head role is not unique")
            raw = task["reagents"][positions[0]]["smiles"]
            if raw not in head_graphs:
                head_graphs[raw] = constitutional_molecule(raw)[0]
            component = by_smiles[head_graphs[raw]]
            if component not in mapping[identity]:
                head_evidence[identity].append(
                    {
                        "component_id": component,
                        "task_shard": shard,
                        "line": line,
                        "task_id": task["task_id"],
                        "head_role": profile["head_role"],
                        "reagent_index": positions[0],
                    }
                )
            mapping[identity].add(component)
            bases[identity].add("original_B5_task_exact_head_input")
    unresolved = selected - set(mapping)
    for identity in unresolved:
        if raw_ids[identity] != {("vitamin_b5_multistep", "series_tail_design")}:
            raise ValueError("A complete historical held component remains unresolved")
    descriptor_calls = [
        ast.unparse(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "p"
        and isinstance(node.args[1], ast.BinOp)
    ]
    if unresolved and descriptor_calls != [
        "p('tail', metadata['series'] + ':' + metadata['tail_axis'])"
    ]:
        raise ValueError("Legacy B5 design descriptor construction changed")
    output = {
        "schema_version": "forge.compose_lipid_historical_global_component_exclusions.v1",
        "seed": 0,
        "implementation": pin(ROOT, Path(__file__).resolve()),
        "inputs": {
            "group_recovery": pin(ROOT, recovery_path),
            "selected_groups": pin(ROOT, group_path),
            "precursor_replay": pin(ROOT, cache_path),
            "catalogue": pin(ROOT, catalog_path),
            "intake": pin(ROOT, intake_path),
            "historical_policy": pin(ROOT, history_path),
            "source_driver": pin(ROOT, driver_path),
            "muscle_contexts": pin(ROOT, muscle_path),
            "legacy_records": pin(ROOT, records_path),
            "b5_original_tasks": pin(ROOT, b5_result_path),
            "b5_recipe_registry": pin(ROOT, b5_registry_path),
        },
        "b5_task_shards": b5_result["task_shards"],
        "b5_head_evidence": dict(head_evidence),
        "bindings": [
            {
                "legacy_id": identity,
                "component_ids": sorted(values),
                "binding_basis": sorted(bases[identity]),
            }
            for identity, values in sorted(mapping.items())
        ],
        "global_protected_components": sorted(set().union(*mapping.values())),
        "legacy_descriptor_groups": [
            {
                "legacy_id": identity,
                "family": "vitamin_b5_multistep",
                "role": "series_tail_design",
                "source_expression": descriptor_calls[0],
                "protection": "Retain exact historical descriptor exclusion through full legacy group projection; this descriptor is not a fabricated complete precursor.",
            }
            for identity in sorted(unresolved)
        ],
        "summary": {
            "selected_legacy_ids": len(selected),
            "complete_component_bindings": len(mapping),
            "legacy_descriptor_groups": len(unresolved),
            "global_protected_components": len(set().union(*mapping.values())),
            "ambiguous_complete_bindings": sum(len(v) != 1 for v in mapping.values()),
        },
        "training_admitted": False,
        "source_component_ids_modified": False,
    }
    dump(HERE / "historical-global-component-exclusions.json", output)


if __name__ == "__main__":
    main()
