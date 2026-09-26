"""Summarize complete saved stream-D experiments without recomputation or promotion."""

import hashlib
import json
import time
from pathlib import Path

from audit_new_leaves import MAIN, OUT, ROOT, pin, read, write


def run():
    start = time.process_time()
    dest = OUT / "closeout_v1"
    assert not dest.exists()
    inputs = {}

    def load(key, relative):
        path = OUT / relative
        inputs[key] = pin(path)
        return read(path)

    early = load("audit", "result.json")
    d = load("D_pair", "full_pair_v2/treatment/validation.json")
    b = load("B_pair", "joint_b_d_v1/assessment/validation.json")
    e = load("E_pair", "e_routes_v1/assessment/validation.json")
    joint = load("joint_binding_review", "joint_binding_review_v1/result.json")
    lookup = load("lookup", "targeted_lookup_v1/result.json")
    load("lookup_authorization", "targeted_lookup_v1/authorization_history.json")
    load("new_listing_admission", "targeted_lookup_v1/root_admission.json")
    load("new_route_admission", "new_clock_routes_v1/root_admission.json")
    newest = load("new_clock_completion", "new_clock_recount_v1/completion.json")
    assert newest["all_completed"]
    panels = {
        arm: load("new_clock_" + arm, f"new_clock_recount_v1/{arm}/validation.json")
        for arm in ["original", "B", "joint"]
    }
    assert all(
        r["passed"]
        and r["control_exactly_reproduces_saved_records"]
        and r["full_denominator"] == 1408
        and r["branches"] == 3723
        for r in panels.values()
    )
    assert d["passed"] and b["passed"] and e["passed"] and joint["passed"]
    costs = []
    for name, path in [
        ("four_leaf_searches", "leaf_search_v1/completion.json"),
        ("failed_pair_startup", "full_pair_v1/completion.json"),
        ("original_D_pair", "full_pair_v2/completion.json"),
        ("B_routing", "joint_b_d_v1/assessment/exit.json"),
        ("E_routing", "e_routes_v1/assessment/exit.json"),
        ("B258_head_search", "b258_search_v1/completion.json"),
        ("three_new_clock_pairs", "new_clock_recount_v1/completion.json"),
    ]:
        data = load("cost_" + name, path)
        costs.append(
            {
                "stage": name,
                "children_CPU_seconds": data["children_CPU_seconds"],
                "supervisor_CPU_seconds": data["supervisor_CPU_seconds"],
            }
        )
    measured = sum(r["children_CPU_seconds"] + r["supervisor_CPU_seconds"] for r in costs)
    manifest = (
        MAIN
        / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/baseline/source_manifest.json"
    )
    source = read(manifest)
    differences = [
        ref["path"]
        for ref in source["files"]
        if hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() != ref["sha256"]
    ]
    assert not differences
    inputs["baseline_source_manifest"] = pin(manifest)
    inputs["producer"] = pin(Path(__file__))
    result = {
        "schema": "forge.iclr22.stream_d.closed_bounded_pass.v1",
        "complete_saved_data_pass": True,
        "official_cohort_promoted": False,
        "evaluation_scope": "One original trained checkpoint, TRAIN-derived development requests; no independent held-out or multi-seed promotion.",
        "inputs": inputs,
        "requests_per_panel": 1408,
        "families": 22,
        "requests_per_family": 64,
        "source_invariance": {
            "files": len(source["files"]),
            "digest": source["source_digest"],
            "differences": differences,
            "candidate_worktree": str(ROOT),
        },
        "initial_saved_cache_fixed_point_gain": early["cached_closed_root_fixed_point_additions"],
        "D_at_original_clock": {
            "totals": d["totals"],
            "transitions": d["transitions"],
            "by_family": d["by_family"],
        },
        "B_at_original_clock": {
            "totals": b["totals"],
            "transitions_against_D": b["against_D_only"],
            "by_family": b["by_family"],
        },
        "E_at_original_clock": e,
        "joint_at_original_clock": {
            "totals": joint["totals"],
            "transitions": joint["transitions_against_D"],
            "source_counts": joint["source_counts"],
        },
        "new_clock_panels": panels,
        "external_lookup": {
            "requests": lookup["HTTP_requests"],
            "positive_identities": lookup["candidate_positive_identities"],
            "observations": lookup["observations"],
            "completion_time": lookup["completed_at_utc"],
            "current_listing_not_stock": True,
        },
        "bounded_search_denominators": {
            "four_targets": 4,
            "additional_B_head": 1,
            "retained_trees": 15,
            "all_failed_or_unknown_trees_retained": True,
        },
        "negative_results": [
            "Exact cached-route fixed-point reuse added no closures.",
            "Initial paired supervisor failed before evaluation because inherited hard CPU limit blocked child setup; preserved as an operational failure and corrected in a versioned runner.",
            "Ketone Ugi4 and aryl reductive-amination target searches did not close their terminal evidence.",
            "The unrestricted B repaired panel lost one primary and 12 L2 positives before new evidence; preserved separately.",
            "The unrestricted E diversity panel lost 127 primary positives net against D, and 11 changed strict dossiers are unestablished; preserved separately.",
        ],
        "scientific_limits": [
            "Computational paths and vendor-directory listings do not establish executed chemistry, stock, price, pKa, delivery or comprehensive molecular quality.",
            "Missing paths/listings remain unknown, not proven unmakeable.",
            "The fixed joint selector was not rerun after new listings; B258 gain belongs only to the B repaired identity when that exact head is present.",
        ],
        "no_GPU_training_TEST_production_edits": True,
        "costs": {
            "major_experiment_receipts": costs,
            "major_experiment_CPU_seconds": measured,
            "accounting_scope": "Direct measured child+supervisor totals for searches and full real-evaluator runs, excluding separately pinned small preparation/review/test costs; not total interactive shell CPU.",
            "authorized_aggregate_CPU_seconds": 1800,
        },
        "closeout_CPU_seconds": time.process_time() - start,
    }
    dest.mkdir()
    write(dest / "result.json", result)
    lines = [
        "# Stream D bounded route-evidence pass",
        "",
        "All results use complete development cohorts of 1,408 requests (64 per family), with 1,387 exact L1 and 3,723 required branches. No held-out or multi-seed admission is claimed.",
        "",
        "| Fixed cohort | Old-clock primary | New-clock control | New-clock augmented primary | L2-ready | Direct-only | Strict dossiers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in panels.items():
        a, c = row["after"], row["before"]
        lines.append(
            f"| {arm} | {c['combined_primary']} | {c['combined_primary']} | {a['combined_primary']} | {a['L2_ready']} | {a['L3_direct_only']} | {a['strict_secondary']} |"
        )
    lines += [
        "",
        "The D-only intervention originally increased 616→621 by closing five reductive-amination requests. All old-clock per-request control records reproduce exactly at the new common clock. Every new clock arm uses the unchanged real evaluator and actual paired typed classifications; no old route verdict transfers to a changed molecular identity.",
        "",
        "B and E were not blindly promoted: full B originally measured 620 primary / 1,177 L2-ready; full E+D measured 494 primary / 996 L2-ready / 86 direct-only, with 16 unchanged strict dossiers. Its 11 changed old strict molecules have unestablished new dossiers. The fixed joint selector retains 7 B and 29 E alternatives, measured 623 primary before these new observations, and preserves all original positive route endpoints.",
        "",
        "Two exact public PubChem queries required explicit user approval after two automatic-review rejections. No requests were sent before approval. The executed lookup used 4 HTTP calls and returned two exact positive vendor-directory observations. These are listings, not stock. New observations retain their true timestamps; all panels use 2026-09-26T06:22:31.247475+00:00, with unchanged 30-day policy.",
        "",
        "The new head route is an intact saved tree. The disulfide route substitutes an entire saved donor tree at one exact original leaf, preserving all reaction children. Independent root admission and 14 focused tamper/time controls passed. Three counterfactual controls verify precise removal of only newly introduced evidence.",
        "",
        'Reproduce the readout from saved results with `PYTHONPATH="$PWD" .venv/bin/python experiments/phase1/route_improvement/closeout.py` in the D worktree, using a new output directory for a fresh execution. Run commands, seeds, input and code SHA256s, raw response receipts, full per-family rows, complete transitions and costs are in result.json and its pinned inputs. Searches/full evaluations are one-shot; do not rerun them implicitly.',
        "",
        f'Major measured searches/evaluations consumed {measured:.6f} aggregate CPU seconds; small preparation, review and test receipts are separately preserved. The source snapshot remains byte-identical across {len(source["files"])} files. Root owns the consolidated decision-log entry and any later manuscript promotion.',
        "",
    ]
    (dest / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "result": pin(dest / "result.json"),
                "README": pin(dest / "README.md"),
                "major_CPU_seconds": measured,
                "panels": {k: v["after"] for k, v in panels.items()},
            }
        )
    )


if __name__ == "__main__":
    run()
