"""Frozen saved-state quality lineage census; never runs a model or selects products."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import time
from collections import Counter
from pathlib import Path

import torch
from rdkit import rdBase

from forge.model import compose_lipid_structural_audit
from forge.model.precursor_reuse_projection import state_graph
from results.phase1.compose_lipid_structure_repair_v1.evaluation.gates import (
    assess_design,
    load_gate_context,
)

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
OUT = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/b_quality"
FRESH = ROOT / "results/phase1/compose_lipid_quality_confirmation_v1"
QUALITY = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/quality"
REPLAY = (
    ROOT
    / "results/phase1/compose_lipid_iclr22_table_completion_v1/assembly/readout_recovery_v4/conditioned_v1"
)
FAMILIES = ("a3_amine_aldehyde_alkyne", "aldehyde_ugi4", "ketone_ugi4", "aryl_reductive_amination")
PATHS = {
    "producer": Path(__file__),
    "baseline": ROOT
    / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/baseline/source_manifest.json",
    "selected": QUALITY / "all22_context_preserving/result.json",
    "gates": QUALITY / "all22_original_reselection/gates.json",
    "candidates": FRESH / "assessment/d1_candidate_diagnostics.json",
    "compact": FRESH / "compact_attempts.json",
    "layouts": FRESH / "input.pt",
    "gate_policy": ROOT
    / "results/phase1/compose_lipid_structure_repair_v1/evaluation/gate_policy_v3.json",
    "gate_code": ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation/gates.py",
    "census": ROOT
    / "results/phase1/compose_lipid_iclr22_research_v1/evidence_completion_v1/quality/weak_family_census/result.json",
    "replay_result": REPLAY / "result.json",
}


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def digest_assessment(value):
    return {
        "assessed": True,
        "exact": value["source_exact"],
        "design": value["qualified_design_pass"],
        "axes": value["status_by_axis"],
        "context_codes": sorted(
            {f["code"] for f in value["chemical"]["flags"] if f["tier"] == "context_required"}
        ),
        "ring_failures": value["ring"].get("fundamental_size_mismatch"),
    }


def assess(value, layout, context):
    if not value.get("smiles"):
        return {
            "assessed": False,
            "exact": False,
            "design": False,
            "reason": value.get("reason", "invalid_or_refused"),
            "axes": {},
            "context_codes": [],
        }
    nodes, edges = state_graph(value["state"])
    result = assess_design(
        layout,
        nodes,
        edges,
        value["state"],
        "observed_saved_raw_tree",
        value["smiles"],
        value["check"]["exact"],
        context,
    )
    return digest_assessment(result)


def summarize(values):
    return {
        "requests": len(values),
        "assessed": sum(v["assessed"] for v in values),
        "exact": sum(v["exact"] for v in values),
        "design": sum(v["design"] for v in values),
        "context": sum(bool(v["context_codes"]) for v in values),
        "axes": {
            a: dict(Counter(v["axes"].get(a, "unassessed_invalid") for v in values))
            for a in ["ring", "head", "chemical"]
        },
    }


def prepare():
    selected = read(PATHS["selected"])["by_family"]
    ids = sorted(c["request"] for f in FAMILIES for c in selected[f]["selections"])
    assert len(ids) == 256
    shards = [
        REPLAY / f"shards/draw-{d}-{o:04d}.json"
        for d in range(5)
        for o in sorted({i // 8 * 8 for i in ids})
    ]
    sources = [p for d in ("forge", "experiments") for p in (WORKTREE / d).rglob("*.py")]
    write(
        OUT / "lineage_protocol.json",
        {
            "schema": "forge.quality.saved_lineage.v1",
            "inputs": {k: pin(p) for k, p in PATHS.items()},
            "source_files": [pin(p) for p in sources],
            "shards": [pin(p) for p in shards],
            "families": FAMILIES,
            "requests": ids,
            "draws": list(range(5)),
            "CPU_cap_seconds": 180,
            "threads": 1,
            "seed": None,
            "analysis": "All256 target requests and all5 draws: true endpoint, terminal argmax, constrained d1 raw, every saved candidate assessment, and selected identity. Defect introduction only asserted between two assessed valid adjacent states; invalid predecessors stay unlocalized.",
            "selection": "Unchanged saved selection; no new proposals or selection.",
            "chemistry": "Unchanged qualified graph/head/ring predicates. Context flags descriptive only; no TRAIN unsupported rejection.",
            "model_TEST_network_paid_calls": 0,
        },
    )


def run():
    start = time.process_time()
    protocol = read(OUT / "lineage_protocol.json")
    for p in [*protocol["inputs"].values(), *protocol["source_files"], *protocol["shards"]]:
        if pin(ROOT / p["path"]) != p:
            raise ValueError("Pinned input changed: " + p["path"])
    assert Path(compose_lipid_structural_audit.__file__).is_relative_to(WORKTREE)
    torch.set_num_threads(1)

    def timeout(*_):
        raise TimeoutError("Lineage CPU budget reached; no complete result admitted")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, 175)
    payload = torch.load(PATHS["layouts"], weights_only=False, map_location="cpu")
    layouts = payload["layouts"]
    selected = read(PATHS["selected"])["by_family"]
    compact = {r["index"]: r for r in read(PATHS["compact"])}
    census = {r["request"]: r for r in read(PATHS["census"])["all1408_rows"]}
    gates = {(r["index"], r["ordinal"]): r["assessment"] for r in read(PATHS["gates"])["attempts"]}
    candidates = {(r["index"], r["ordinal"]): r for r in read(PATHS["candidates"])["candidates"]}
    replays = {}
    for p in protocol["shards"]:
        for r in read(ROOT / p["path"])["rows"]:
            replays[r["index"], r["draw"]] = r
    rows = []
    with rdBase.BlockLogs():
        context = load_gate_context(ROOT, PATHS["gate_policy"])
        for family in FAMILIES:
            for chosen in selected[family]["selections"]:
                idx, ordinal = chosen["request"], chosen["ordinal"]
                branch = compact[idx]["branches"]["d1"]
                pool = [dict(branch["raw"], kind="raw"), *branch["proposals"]]
                kind = pool[ordinal]["kind"]
                draw = int(m.group(1)) if (m := re.match(r"draw(\d+):", kind)) else 0
                assert pool[ordinal]["smiles"] == chosen["smiles"]
                final = census[idx]
                chosen_quality = {
                    "assessed": True,
                    "exact": final["source_exact"],
                    "design": final["qualified_design_pass"],
                    "axes": final["status_by_axis"],
                    "context_codes": final["context_codes"],
                }
                draws = []
                for d in range(5):
                    replay = replays[idx, d]
                    constrained = (
                        branch["raw"]
                        if d == 0
                        else next(v["raw"] for v in branch["additional_draws"] if v["draw"] == d)
                    )
                    stages = {
                        name: assess(v, layouts[idx], context) for name, v in replay["arms"].items()
                    }
                    stages["constrained_d1"] = assess(constrained, layouts[idx], context)
                    draws.append({"draw": d, "stages": stages})
                predecessor = draws[draw]["stages"]["constrained_d1"]
                introduced = []
                if predecessor["assessed"]:
                    introduced += [
                        a
                        for a, s in chosen_quality["axes"].items()
                        if s == "fail" and predecessor["axes"].get(a) != "fail"
                    ]
                    introduced += [
                        "context:" + c
                        for c in set(chosen_quality["context_codes"])
                        - set(predecessor["context_codes"])
                    ]
                pool_rows = []
                for k, p in enumerate(pool):
                    a = gates.get((idx, k))
                    pool_rows.append(
                        {
                            "ordinal": k,
                            "kind": p["kind"],
                            "smiles": p.get("smiles"),
                            "exact": p["check"]["exact"],
                            "candidate_status": candidates.get((idx, k), {}).get("status"),
                            "quality": digest_assessment(a) if a else None,
                        }
                    )
                rows.append(
                    {
                        "request": idx,
                        "family": family,
                        "selected_ordinal": ordinal,
                        "selected_kind": kind,
                        "selected_draw": draw,
                        "selected_quality": chosen_quality,
                        "selected_smiles": chosen["smiles"],
                        "construction_selected": not (
                            kind == "raw" or re.fullmatch(r"draw\d+:raw", kind)
                        ),
                        "selected_parent_assessed": predecessor["assessed"],
                        "construction_new_failure_axes_or_context": introduced,
                        "draws": draws,
                        "candidate_pool": pool_rows,
                        "selected_adverse": not chosen_quality["design"]
                        or bool(chosen_quality["context_codes"]),
                    }
                )
    summaries = {}
    for family in FAMILIES:
        local = [r for r in rows if r["family"] == family]
        summaries[family] = {
            "selected": summarize([r["selected_quality"] for r in local]),
            "all5": {
                stage: summarize([d["stages"][stage] for r in local for d in r["draws"]])
                for stage in rows[0]["draws"][0]["stages"]
            },
            "adverse_selected": sum(r["selected_adverse"] for r in local),
            "construction_selected": sum(r["construction_selected"] for r in local),
            "introduced_into_selected_from_assessed_constrained_parent": dict(
                Counter(a for r in local for a in r["construction_new_failure_axes_or_context"])
            ),
        }
    elapsed = time.process_time() - start
    write(
        OUT / "lineage_result.json",
        {
            "complete": True,
            "protocol": pin(OUT / "lineage_protocol.json"),
            "rows": rows,
            "summaries": summaries,
            "requests": len(rows),
            "all5_draws": len(rows) * 5,
            "CPU_seconds": elapsed,
            "imported_module": pin(Path(compose_lipid_structural_audit.__file__)),
            "new_model_source_executor_selection_calls": 0,
        },
    )
    print(json.dumps({"CPU_seconds": elapsed, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
