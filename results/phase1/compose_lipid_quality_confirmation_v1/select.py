"""Fresh-layout orchestration; selection, references and assessment remain frozen.

This version adds explicit layout input pins to the development runner. It imports
its unchanged candidate preparation and assessment functions, and cannot mutate
its reference cache. The optional equivalence mode checks all selected rows on
the old ledger without repeating fingerprint/descriptor assessments.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from collections import Counter
from pathlib import Path

import torch
from rdkit import rdBase

from forge.model.compose_lipid_quality_selection import first_exact, select_reference_supported
from results.phase1.compose_lipid_component_decoder_v1.contracts import load_all
from results.phase1.compose_lipid_quality_selection_v2.run import (
    candidate_pools,
    novel_component_summary,
    pin,
    read,
    transitions,
    write,
)
from results.phase1.compose_lipid_quality_v1.run import assess_panel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = HERE.with_name("compose_lipid_quality_selection_v2")
QUALITY = HERE.with_name("compose_lipid_quality_v1")
CACHE = HERE.with_name("compose_lipid_mapped_preparation_v3") / "cache"


def reference_support_read_only(pins):
    cached = FROZEN / "reference_support.pkl"
    receipt = read(FROZEN / "reference_support.json")
    inputs = {
        key: pins[key]
        for key in ("reference", "quality_module", "quality_runner", "quality_policy")
    }
    if receipt["inputs"] != inputs or receipt["artifact"] != pin(cached):
        raise ValueError("Frozen local reference cache does not match its inputs/artifact")
    with cached.open("rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--producer-protocol", type=Path, required=True)
    parser.add_argument("--layout-payload", type=Path, required=True)
    parser.add_argument("--layout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "assessment")
    parser.add_argument("--verify-selected", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    out = args.output.resolve()
    if not out.is_relative_to(HERE / "assessment"):
        raise ValueError("Confirmation output must stay under assessment/")
    out.mkdir(exist_ok=True, parents=True)
    if (out / "result.json").exists() or (out / "equivalence.json").exists():
        raise ValueError("Preserve completed results; choose a new output directory")
    torch.set_num_threads(1)
    paths = {
        "ledger": args.ledger.resolve(),
        "producer_protocol": args.producer_protocol.resolve(),
        "policy": FROZEN / "policy.json",
        "prespecification": FROZEN / "prespecification.json",
        "selector": ROOT / "forge/model/compose_lipid_quality_selection.py",
        "runner": Path(__file__),
        "frozen_runner": FROZEN / "run.py",
        "frozen_selection_protocol": FROZEN / "aema_domain/protocol.json",
        "reference": QUALITY / "reference.json",
        "quality_policy": QUALITY / "policy.json",
        "quality_controls": QUALITY / "positive_controls.json",
        "quality_module": ROOT / "forge/model/compose_lipid_quality.py",
        "quality_runner": QUALITY / "run.py",
        "membership": CACHE / "sources.sqlite",
        "manifest": CACHE / "manifest.json",
        "layout_payload": args.layout_payload.resolve(),
        "layout_protocol": args.layout_protocol.resolve(),
        "role_contract_loader": HERE.with_name("compose_lipid_component_decoder_v1")
        / "contracts.py",
    }
    pins = {name: pin(path) for name, path in paths.items()}
    frozen_pins = read(paths["frozen_selection_protocol"])["inputs"]
    for key in (
        "policy",
        "prespecification",
        "selector",
        "reference",
        "quality_policy",
        "quality_controls",
        "quality_module",
        "quality_runner",
        "membership",
        "manifest",
        "role_contract_loader",
    ):
        if pins[key] != frozen_pins[key]:
            raise ValueError(f"Frozen selection/reference input changed: {key}")
    if pins["frozen_runner"] != frozen_pins["runner"]:
        raise ValueError("Frozen candidate-preparation/evaluation orchestration changed")
    policy, quality_policy, reference = (
        read(paths["policy"]),
        read(paths["quality_policy"]),
        read(paths["reference"]),
    )
    assert read(paths["prespecification"])["policy"] == pins["policy"]
    assert reference["inputs"]["quality_code"] == pins["quality_module"]
    assert reference["inputs"]["runner"] == pins["quality_runner"]
    assert pins["membership"] == read(paths["manifest"])["artifacts"]["sources.sqlite"]
    assert pins["layout_payload"] == read(paths["layout_protocol"])["inputs"]["payload"]
    rows = read(paths["ledger"])
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Duplicate requested output IDs")
    if any(set(row["branches"]) != {"d0", "d1"} for row in rows):
        raise ValueError("Expected exactly the frozen D0 and D1 branches")
    layouts = torch.load(paths["layout_payload"], weights_only=False, map_location="cpu")["layouts"]
    if sorted(row["index"] for row in rows) != list(range(len(layouts))):
        raise ValueError("Every pinned layout must retain one candidate pool")
    if len(layouts) != 1408 or set(Counter(row["family"] for row in rows).values()) != {64}:
        raise ValueError("Expected the frozen 22-family, 64-request-per-family design")
    if any(layouts[row["index"]].family != row["family"] for row in rows):
        raise ValueError("Candidate requests differ from pinned TRAIN layouts")
    with rdBase.BlockLogs():
        executors, source_pins, _ = load_all()
    pins.update({"source_contract:" + name: value for name, value in source_pins.items()})
    write(
        out / "protocol.json",
        {"inputs": pins, "policy": policy, "seed": policy["seed"], "requests": len(rows)},
    )
    print("Loading pinned TRAIN feature reference", flush=True)
    product_support, component_support, _ = reference_support_read_only(pins)
    panels, per_attempt, selection_reports, novelty, costs = {}, {}, {}, {}, {}
    equivalence = {}
    with sqlite3.connect(paths["membership"].as_uri() + "?mode=ro", uri=True) as db:
        for branch in ("d0", "d1"):
            print(f"Assessing complete-component candidates: {branch}", flush=True)
            pools, diagnostics, originals = candidate_pools(
                rows, branch, product_support, component_support, reference, layouts, executors, db
            )
            write(
                out / f"{branch}_candidate_diagnostics.json",
                {"inputs": pins, "candidates": diagnostics},
            )
            baseline, supported, reports = {}, {}, {}
            for family, family_pools in sorted(pools.items()):
                baseline[family] = [first_exact(pool) for pool in family_pools]
                supported[family], reports[family] = select_reference_supported(
                    family_pools,
                    node_limit=policy["support_aware"]["solver"]["node_limit"],
                    time_limit=policy["support_aware"]["solver"]["time_limit_seconds_per_stage"],
                )
            selection_reports[branch] = reports
            costs[branch] = {
                "requests": len(rows),
                "producer_proposals": sum(
                    len(row["branches"][branch].get("proposals", [])) for row in rows
                ),
                "actual_costs_by_request": [
                    {
                        "index": row["index"],
                        "costs": row["branches"][branch].get("costs"),
                        "construction": row["branches"][branch].get("construction"),
                    }
                    for row in rows
                ],
            }
            for kind, selection in (("exact_only", baseline), ("support_aware", supported)):
                label = branch + "_" + kind
                selected = [
                    originals[c.request, c.ordinal]
                    for family in sorted(selection)
                    for c in selection[family]
                ]
                selected.sort(key=lambda row: row["index"])
                write(out / f"{label}_selected.json", {"inputs": pins, "attempts": selected})
                if args.verify_selected is not None:
                    expected_path = args.verify_selected.resolve() / f"{label}_selected.json"
                    expected = read(expected_path)["attempts"]
                    if selected != expected:
                        raise ValueError(f"Input-plumbing equivalence failed: {label}")
                    equivalence[label] = {
                        "reference": pin(expected_path),
                        "identical_complete_selected_rows": True,
                        "requests": len(selected),
                    }
                    continue
                panels[label], detail = assess_panel(
                    selected,
                    "selected",
                    product_support,
                    component_support,
                    reference,
                    quality_policy,
                    db,
                    executors,
                    layouts,
                )
                per_attempt[label] = detail
                novelty[label] = novel_component_summary(selection)
                write(out / f"{label}_assessment.json", {"inputs": pins, "attempts": detail})
    if args.verify_selected is not None:
        write(
            out / "equivalence.json",
            {
                "schema_version": "forge.compose_lipid_confirmation_selection_equivalence.v1",
                "inputs": pins,
                "comparisons": equivalence,
                "selection_reports": selection_reports,
                "runtime_seconds": time.monotonic() - started,
                "scope": "Candidate preparation and all four selected panels; no repeated fingerprint assessment.",
            },
        )
        print(json.dumps({"equivalence": True, "panels": len(equivalence)}), flush=True)
        return
    comparisons = {
        name: transitions(per_attempt[a], per_attempt[b])
        for name, a, b in [
            ("d0_selection", "d0_exact_only", "d0_support_aware"),
            ("d1_selection", "d1_exact_only", "d1_support_aware"),
            ("decoder_exact_only", "d0_exact_only", "d1_exact_only"),
            ("decoder_support_aware", "d0_support_aware", "d1_support_aware"),
        ]
    }
    result = {
        "schema_version": "forge.compose_lipid_quality_confirmation_selection_result.v1",
        "inputs": pins,
        "requests": len(rows),
        "by_panel": panels,
        "component_novelty": novelty,
        "selection_reports": selection_reports,
        "paired_transitions": comparisons,
        "costs": costs,
        "runtime_seconds": time.monotonic() - started,
        "nonclaims": [
            "Observed neighborhoods/rings do not certify chemical quality.",
            "Unknown structures remain in the candidate ledger and attempt denominators.",
            "Broad observed-lipid TRAIN comparison is not family-matched or independent generalization.",
            "This selection/evaluation stage makes no model calls and accesses no sealed TEST structures.",
        ],
    }
    write(out / "result.json", result)
    print(
        json.dumps(
            {
                name: {
                    "exact": sum(r["exact_l1"] for r in families.values()),
                    "observed": sum(
                        r["exact_all_local_features_observed"] for r in families.values()
                    ),
                }
                for name, families in panels.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
