"""Account for source-label disagreements without altering tasks or admitting recipes."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    receipt = ROOT / "results/phase1/compose_lipid_supplied_b5_v1/result.json"
    report = json.loads(receipt.read_text())
    registry_path = resolve_pin(report["inputs"]["registry"], ROOT, label="registry")
    registry = json.loads(registry_path.read_text())
    extraction = json.loads(
        resolve_pin(report["inputs"]["original_tasks"], ROOT, label="original tasks").read_text()
    )
    originals = {
        r["preparation"]["target_id"]: r
        for r in rows(resolve_pin(extraction["artifact"], ROOT, label="selected original tasks"))
    }
    source_path = resolve_pin(report["inputs"]["source_controls"], ROOT, label="source controls")
    source = json.loads(source_path.read_text())
    control = next(c for c in source["controls"] if c["label"] == "I91")
    labels = {
        constitutional_molecule(control["components"]["tail_acid_1"])[
            0
        ]: "source_compound_18_malonate",
        constitutional_molecule(control["components"]["tail_acid_2"])[
            0
        ]: "source_compound_11_glutarate",
    }
    counts, groups, examples, failed_ids = Counter(), Counter(), {}, set()
    stage_widths = defaultdict(Counter)
    ledger = resolve_pin(report["artifact"], ROOT, label="replay ledger")
    for row in rows(ledger):
        result, target = row["result"], row["target_id"]
        binding = result["binding"]
        counts["eligible_rows"] += 1
        counts["exact_rows"] += result["computed_consistency_pass"]
        if result["computed_consistency_pass"]:
            stage_widths[binding["program_id"]][
                compact(list(map(len, result["replay"]["forward_layers"])))
            ] += 1
        original = originals[target]
        profile = registry["original_task_contract"]["profiles"][binding["profile_id"]]
        axes = original["original_task"]["tail_axis"].split("__")
        for role, axis in zip(profile["tail_roles"], axes, strict=True):
            if binding["tail_axis_checks"][role]:
                continue
            failed_ids.add(target)
            counts["failed_tail_claims"] += 1
            rule = profile["tail_rules"][role].get(axis)
            expected = (
                [constitutional_molecule(s)[0] for s in rule.get("smiles", [])] if rule else []
            )
            actual = binding["components"][role]
            key = compact(
                {
                    "profile": binding["profile_id"],
                    "role": role,
                    "axis": axis,
                    "rule_kind": rule.get("kind") if rule else None,
                    "actual_source_identity": labels.get(actual, actual),
                    "expected_source_identities": [labels.get(s, s) for s in expected],
                }
            )
            groups[key] += 1
            examples.setdefault(
                key,
                {
                    "target_id": target,
                    "original_task_reference": row["original_task_reference"],
                    "actual_component_smiles": actual,
                    "expected_component_smiles": expected,
                    "component_instances": row["component_instances"],
                },
            )
    counts["pending_rows"] = len(failed_ids)
    if (
        counts["eligible_rows"] != report["summary"]["rows"]
        or counts["exact_rows"] != report["summary"]["exact"]
    ):
        raise ValueError("B5 source-scope audit lost eligible rows")
    dump(
        HERE / "source-scope-audit.json",
        {
            "schema_version": "forge.b5_source_scope_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "replay": pin(ROOT, receipt),
                "replay_ledger": pin(ROOT, ledger),
                "registry": pin(ROOT, registry_path),
                "original_tasks": report["inputs"]["original_tasks"],
                "source_controls": pin(ROOT, source_path),
            },
            "counts": dict(counts),
            "source_stage_widths": {k: dict(v) for k, v in sorted(stage_widths.items())},
            "source_claim_disagreements": [
                {"claim": json.loads(k), "rows": groups[k], "example": examples[k]}
                for k in sorted(groups)
            ],
            "source_files_changed": False,
            "product_matching_used_to_resolve_source_claims": False,
            "reported_label_interpretation": "The current conservative contract binds reported_tail_acid to the source series and attachment site. The task string itself is not site-namespaced. These are disagreements with that contract, not a conclusion that the supplied structures were never reported anywhere in the B5 study.",
            "training_admitted": False,
            "next_adjudication": "All source-label conflicts stay pending. Preserve original task and exported metadata; any separate computed-variant qualification must explicitly reject the incorrect reported identity and independently justify its same-series precursor and attachment scope.",
        },
    )
    print(json.dumps(dict(counts), sort_keys=True))


if __name__ == "__main__":
    main()
