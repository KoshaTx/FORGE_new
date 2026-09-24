"""Authenticate fixed-closure correction and bounded source-core completion evidence."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    inputs = {}

    def pin(path):
        value = {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}
        inputs[value["path"]] = value
        return value

    def load(path):
        pin(path)
        return json.loads(path.read_text())

    def authenticate(value):
        return resolve_pin(value, ROOT, label="reserved core evidence")

    def result(name):
        r = load(ROOT / f"results/phase1/{name}/result.json")
        config = load(authenticate(r["config"]))
        assert r["sources"] == config["source_pins"]
        for p in [*r["sources"], *r["artifacts"].values()]:
            pin(authenticate(p))
        if "baseline_result" in r:
            assert r["baseline_result"] == config["baseline_result"]
            pin(authenticate(r["baseline_result"]))
        return r

    def artifact(r, name):
        content = authenticate(r["artifacts"][name]).read_text()
        if name.endswith("jsonl"):
            return [json.loads(line) for line in content.splitlines()]
        return json.loads(content)

    def equivalent(a, b):
        ignored = {"created_at_utc", "duration_seconds", "artifacts"}
        assert {k: v for k, v in a.items() if k not in ignored} == {
            k: v for k, v in b.items() if k not in ignored
        }
        assert {k: p["sha256"] for k, p in a["artifacts"].items()} == {
            k: p["sha256"] for k, p in b["artifacts"].items()
        }
        return True

    def totals(comparison):
        return {
            arm: {
                metric: sum(f[metric] for f in families.values())
                for metric in (
                    "attempts",
                    "valid_connected",
                    "exact_program",
                    "unique_valid_products",
                    "novel_valid_attempts_vs_all_cache_train",
                    "unique_novel_exact_products",
                )
            }
            for arm, families in comparison["per_arm"].items()
        }

    ordered = result("combinatorial_ordered_core_decoding_v1")
    reserved = result("combinatorial_reserved_core_decoding_v1")
    reserved_replay = result("combinatorial_reserved_core_decoding_replay_v1")
    assert artifact(ordered, "layouts.json") == artifact(reserved, "layouts.json")
    for a, b in zip(ordered["pairing"], reserved["pairing"], strict=True):
        assert all(a[k] == b[k] for k in a)
        assert b["fixed_states_exact"] and b["no_fixed_closure_identity_exact"]
    original_rows = artifact(ordered, "attempts.jsonl")
    reserved_rows = artifact(reserved, "attempts.jsonl")
    changed = Counter()
    restored = []
    for a, b in zip(original_rows, reserved_rows, strict=True):
        if a != b:
            assert a["arm"] == b["arm"] == "graph_reuse"
            assert a["program_id"] == b["program_id"] == "acetal_aldehyde_diol"
            changed[b["program_id"]] += 1
        if not a["valid_connected"] and b["valid_connected"]:
            assert a["constraint_abstention_reason"] == "fixed_closure_valence_exceeds_support"
            restored.append(b["sample_index"])
    assert len(restored) == 32
    correction = {
        "totals": totals(reserved["comparison"]),
        "per_family": reserved["comparison"]["per_arm"],
        "per_family_deltas": reserved["comparison"]["per_family_deltas"],
        "layouts_and_original_neural_scores_identical_to_ordered_run": True,
        "all_other_eleven_families_identical_to_ordered_run": True,
        "changed_attempts_by_family": dict(changed),
        "restored_fixed_closure_attempts": restored,
        "paired_preservation_pass": reserved["comparison"]["all_family_preservation_screen_passed"],
        "fresh_replay_identical": equivalent(reserved, reserved_replay),
    }
    assert not correction["paired_preservation_pass"]
    protocol = load(
        ROOT / "configs/multireaction/combinatorial_source_core_completion_protocol_v3.json"
    )
    runs = {}
    for label, expected_count in (
        ("discovery", 768),
        ("replication_1", 1536),
        ("replication_2", 1536),
    ):
        r = result(f"combinatorial_source_core_completion_{label}_v1")
        assert r["config"] == protocol["runs"][label]
        assert r["policy"] == protocol["policy"]
        assert r["neural_attempts_replayed"] == expected_count
        assert r["frozen_neural_terminal_states_identical"]
        assert not r["new_neural_noise_or_layout_draws"]
        assert r["all_family_preservation_screen_passed"] and r["total_exact_gain"] > 0
        rows = artifact(r, "attempts.jsonl")
        before = [row for row in rows if row["arm"] == "baseline"]
        after = [row for row in rows if row["arm"] == "graph_reuse"]
        decisions = artifact(r, "admissions.jsonl")
        proposals = artifact(r, "proposals.jsonl")
        assert len(before) == len(after) == expected_count
        changes, unique = [], set()
        for a, b, d, p in zip(before, after, decisions, proposals, strict=True):
            assert all(a[k] == b[k] for k in ("sample_index", "program_id", "requested_depth"))
            if not a["valid_connected"] or a["assembly"]["status"] == "exact_computed_program":
                assert a["canonical_smiles"] == b["canonical_smiles"]
                assert a["assembly"] == b["assembly"]
            if d["disposition"] == "admitted_exact_completion":
                assert a["canonical_smiles"] != b["canonical_smiles"]
                assert b["valid_connected"] and b["assembly"]["status"] == "exact_computed_program"
                assert b["novel_vs_train"] and b["all_witnesses_have_novel_component"]
                assert p["proposals"][0]["source_fixed_graph_preserved"]
                assert p["proposals"][0]["original_atom_and_edge_counts_preserved"]
                changes.append(b["sample_index"])
                unique.add((b["program_id"], b["canonical_smiles"]))
            else:
                assert a["canonical_smiles"] == b["canonical_smiles"]
        assert len(changes) == len(unique) == r["total_exact_gain"]
        runs[label] = {
            "totals": totals(r),
            "per_family": r["per_arm"],
            "per_family_deltas": r["per_family_deltas"],
            "all_eight_metrics_preserved_in_every_family": r[
                "all_family_preservation_screen_passed"
            ],
            "exact_gain": r["total_exact_gain"],
            "changed_attempts": changes,
            "new_exact_distinct_train_novel_products": len(unique),
            "all_new_exact_outputs_have_novel_component_in_every_witness": True,
            "original_invalid_and_exact_outputs_unchanged": True,
            "original_atom_edge_counts_and_fixed_graphs_preserved": True,
            "all_neural_terminal_states_reproduced": True,
            "admission_dispositions": r["admission_dispositions"],
            "proposal_statuses": r["proposal_statuses"],
            "qualification": r["qualification"],
        }
        if label == "discovery":
            runs[label]["fresh_replay_identical"] = equivalent(
                r, result("combinatorial_source_core_completion_replay_v1")
            )
    pin(Path(__file__))
    payload = {
        "schema_version": "forge.combinatorial_reserved_core_findings.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "closure_correction": correction,
        "source_core_completion": runs,
        "scientific_preservation_screen_passed_for_admitted_source_core_stage": True,
        "promoted": False,
        "goal_complete": False,
        "decision": "Retain the admitted source-core stage as a bounded pipeline improvement, with all negative raw-proposal results and execution failures preserved.",
        "limits": [
            "Source-core completion supplies source-derived typed core atoms and bonds from existing TRAIN layouts. It does not establish learned or source-independent core reconstruction.",
            "The three saved populations are previously inspected. Replication does not constitute heldout confirmation or across-training uncertainty.",
            "Novelty refers to full TRAIN product identity and the declared component partitions; it is not evidence of chemical realism, synthesis feasibility or efficacy.",
            "The eight per-family metrics include full requested-depth exact reconstruction under each family's own program. The unchanged Ugi metric alone is insufficient.",
            "Unique-product totals are sums of within-family counts. All twelve library scopes are preserved; some supply neutral products or precursors rather than complete ionizable lipids.",
        ],
    }
    for p in inputs.values():
        authenticate(p)
    target = OUT / "findings.json"
    if target.exists():
        raise ValueError("findings result already exists")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: r["totals"] for label, r in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
