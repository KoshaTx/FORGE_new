"""One-shot paired closure-only repair with full saved-family selection floors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from experiments.phase1.multireaction.quality_lineage_diagnostic import (
    FAMILIES,
    FRESH,
    OUT,
    PATHS,
    ROOT,
    WORKTREE,
    pin,
    read,
    write,
)
from forge.model import compose_lipid_closure_repair as candidate_module
from forge.model.compose_lipid_component_diversity import accepted_components
from forge.model.compose_lipid_quality import StructuralSupport, source_role_mapping
from forge.model.precursor_reuse_projection import state_graph
from results.phase1.compose_lipid_component_decoder_v1.contracts import assess, load_all, matching
from results.phase1.compose_lipid_structure_repair_v1.evaluation.gates import (
    assess_design,
    load_gate_context,
)
from results.phase1.compose_lipid_structure_repair_v1.evaluation.replay import load_candidate_graphs

HERE = OUT / "closure_v1"
REFERENCE = ROOT / "results/phase1/compose_lipid_quality_v1/reference.json"
SELECTOR = ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation/design_selector.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contexts(assessment):
    return {f["code"] for f in assessment["chemical"]["flags"] if f["tier"] == "context_required"}


def prepare():
    HERE.mkdir(exist_ok=True)
    lineage = read(OUT / "lineage_result.json")
    selected = {
        c["request"]: c
        for f in FAMILIES
        for c in read(PATHS["selected"])["by_family"][f]["selections"]
    }
    target = [
        r["request"] for r in lineage["rows"] if r["selected_quality"]["axes"]["ring"] == "fail"
    ]
    controls = [
        min(
            r["request"]
            for r in lineage["rows"]
            if r["family"] == f
            and r["selected_quality"]["design"]
            and not r["selected_quality"]["context_codes"]
        )
        for f in FAMILIES
    ]
    wanted = {(i, selected[i]["ordinal"]) for i in target + controls}
    payload = torch.load(PATHS["layouts"], map_location="cpu", weights_only=False)
    rows = read(PATHS["compact"])
    graphs = []
    with rdBase.BlockLogs():
        for i, o, n, e, s, b, p in load_candidate_graphs(rows, "d1", wanted, payload["layouts"]):
            assert (
                selected[i]["smiles"]
                == next(r for r in lineage["rows"] if r["request"] == i)["selected_smiles"]
            )
            graphs.append(
                {
                    "request": i,
                    "ordinal": o,
                    "nodes": n.tolist(),
                    "edges": e.tolist(),
                    "tree_state": s,
                    "tree_basis": b,
                    "smiles": selected[i]["smiles"],
                    "source": p,
                    "control": i in controls,
                }
            )
    write(HERE / "parents_input.json", {"graphs": graphs})
    source = read(
        ROOT
        / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1/quality/repair_experiment_readiness_v1/protocol.json"
    )
    logits = {}
    for r in lineage["rows"]:
        if r["request"] not in target:
            continue
        idx, draw = r["request"], r["selected_draw"]
        path = FRESH / f"logits/draw-{draw}-{idx//8*8:04d}.json"
        rec = read(path)
        logits[idx] = {
            "receipt": pin(path),
            "predictions": rec["predictions"],
            "offset": idx // 8 * 8,
        }
    reference = read(REFERENCE)
    paths = {
        **PATHS,
        "lineage": OUT / "lineage_result.json",
        "parents": HERE / "parents_input.json",
        "runner": Path(__file__),
        "candidate": Path(candidate_module.__file__),
        "tests": WORKTREE / "tests/test_compose_lipid_closure_repair.py",
        "test_receipt": OUT / "closure_tests_v2.xml",
        "reference": REFERENCE,
        "selector": SELECTOR,
        "membership": ROOT / reference["inputs"]["membership"]["path"],
        "prior_ring_result": ROOT
        / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1/quality/repair_experiment_readiness_v1/full_run_v1/result.json",
    }
    write(
        HERE / "protocol.json",
        {
            "schema": "forge.quality.closure_only_diagnostic.v1",
            "inputs": {k: pin(v) for k, v in paths.items()},
            "prior_ring_input_pins": source["inputs"],
            "source_files": [
                pin(p) for d in ("forge", "experiments") for p in (WORKTREE / d).rglob("*.py")
            ],
            "logits": logits,
            "target_requests": sorted(target),
            "source_positive_control_requests": controls,
            "family_denominator": 64,
            "target_family_total": 256,
            "full_cohort_denominator": 1408,
            "other_1152_requests": "Unchanged original selections; no intervention.",
            "intervention": "Only variable closure endpoints on the saved spanning tree; preserve node states, tree parents and bonds, closure bond states, counts, component membership and fixed core. Derive allowed sizes from sampled source morphology, coordinate aligned repeated components, no atom caps or TRAIN identity rejection.",
            "bounds": {
                "CPU_seconds": 600,
                "threads": 1,
                "maximum_proposals_per_parent": 2,
                "beam_width": 16,
                "maximum_pairs_per_group": 4096,
                "source_calls_per_parent": 2,
                "selector_seconds_per_objective": 5,
                "selector_nodes": 10000,
            },
            "arms": ["unchanged_selected", "closure_endpoints"],
            "selection": "Original unchanged selector; only full-design candidates with no new context codes and no per-request observed-feature loss eligible. Preserve source-exact statuses, fullN, product and per-role unique/Shannon/Simpson, and novelty observations/distinct identities. Original always retained.",
            "novelty": "Exact pinned TRAIN membership and complete source-role component identities; no copying whole TRAIN components.",
            "source_and_design": "Unchanged full source executors and gates; preserve rejected/invalid/abstained proposals and sources. Four exact/design positive saved controls must remain noops.",
            "comparison_to_prior": "Prior 192-request ring-motif experiment is historical with different proposal search; no matched total-compute causal claim.",
            "attribution": "Decoder construction only; no retraining or new model sampling. Local ring-size compliance not independent realism or synthesis execution.",
            "seed": None,
            "deterministic": True,
            "model_TEST_network_GPU_calls": 0,
            "changed_identity": "Invalidate inherited likelihood, trajectory probability and route verdict.",
        },
    )


def run():
    start = time.process_time()
    torch.set_num_threads(1)
    protocol = read(HERE / "protocol.json")
    for p in [*protocol["inputs"].values(), *protocol["source_files"]]:
        if pin(ROOT / p["path"]) != p:
            raise ValueError("Changed pin " + p["path"])
    for entry in protocol["logits"].values():
        for k in ("receipt", "predictions"):
            if pin(ROOT / entry[k]["path"]) != entry[k]:
                raise ValueError("Changed logit input")
    assert Path(candidate_module.__file__).is_relative_to(WORKTREE)

    def timeout(*_):
        raise TimeoutError("Closure diagnostic CPU cap exceeded")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, 590)
    payload = torch.load(PATHS["layouts"], weights_only=False, map_location="cpu")
    layouts, atoms = payload["layouts"], payload["atoms"]
    selector = load(SELECTOR, "b_frozen_design_selector")
    census = {r["request"]: r for r in read(PATHS["census"])["all1408_rows"]}
    before = {}
    for f in FAMILIES:
        for c in read(PATHS["selected"])["by_family"][f]["selections"]:
            before[c["request"]] = selector.SelectionCandidate(
                **{
                    **c,
                    "components": tuple(selector.ComponentIdentity(**p) for p in c["components"]),
                }
            )
    pools = {i: [c] for i, c in before.items()}
    gates = {(i, c.ordinal): census[i]["qualified_design_pass"] for i, c in before.items()}
    contexts_by_key = {(i, c.ordinal): set(census[i]["context_codes"]) for i, c in before.items()}
    reference = read(REFERENCE)
    db = sqlite3.connect(f"file:{ROOT/reference['inputs']['membership']['path']}?mode=ro", uri=True)
    all_train = set(reference["component_smiles_by_identity"].values())
    support = {}
    with rdBase.BlockLogs():
        executors, source_pins, source_controls = load_all()
        context = load_gate_context(ROOT, PATHS["gate_policy"])
        for f in FAMILIES:
            products = StructuralSupport()
            for s in reference["product_support"][f]:
                products.add(s)
            parts = {}
            for role, ids in reference["component_ids_by_family_role"][f].items():
                parts[role] = StructuralSupport()
                for identity in ids:
                    parts[role].add(reference["component_smiles_by_identity"][identity])
            support[f] = (products, parts)
    records = []
    (HERE / "parents").mkdir()
    with rdBase.BlockLogs():
        for graph in sorted(
            read(HERE / "parents_input.json")["graphs"], key=lambda r: r["request"]
        ):
            i = graph["request"]
            layout = layouts[i]
            checked = assess(executors, layout, graph["smiles"])
            assert checked["exact"] == before[i].exact
            if graph["tree_state"] is None:
                report = {
                    "status": "no_saved_ordered_tree_abstention",
                    "proposals": [],
                    "costs": {},
                }
            else:
                if graph["control"]:
                    predictions = {}
                else:
                    entry = protocol["logits"][str(i)]
                    saved = torch.load(
                        ROOT / entry["predictions"]["path"], weights_only=False, map_location="cpu"
                    )
                    predictions = {k: v[i - entry["offset"]].numpy() for k, v in saved.items()}
                report = candidate_module.propose_closure_endpoints(
                    layout,
                    graph["tree_state"],
                    predictions,
                    atoms,
                    maximum_proposals=2,
                    beam_width=16,
                    maximum_pairs_per_group=4096,
                )
            if graph["control"]:
                assert (
                    checked["exact"]
                    and not report["proposals"]
                    and report["status"] == "already_matches_request_noop"
                )
            assessed = []
            for slot, p in enumerate(report["proposals"]):
                n, e = state_graph(p["tree_state"])
                check = assess(executors, layout, p["smiles"])
                # The parent edges are literally unchanged. The new closure endpoints
                # define the generated fundamental cycles; this is not an SSSR swap.
                audit = assess_design(
                    layout,
                    n,
                    e,
                    p["tree_state"],
                    p["tree_basis"],
                    p["smiles"],
                    check["exact"],
                    context,
                )
                parts = accepted_components(check) if check["exact"] else {}
                mapping = (
                    source_role_mapping(parts, check, matching(executors, layout)) if parts else {}
                )
                product_support, part_support = support[layout.family]
                components = tuple(
                    selector.ComponentIdentity(
                        role, s, s not in all_train, s not in part_support[mapping[role]].identities
                    )
                    for role, s in sorted(parts.items())
                )
                observed = bool(
                    check["exact"]
                    and product_support.assess(Chem.MolFromSmiles(p["smiles"]))[
                        "all_local_features_observed"
                    ]
                    and all(
                        part_support[mapping[role]].assess(Chem.MolFromSmiles(s))[
                            "all_local_features_observed"
                        ]
                        for role, s in parts.items()
                    )
                )
                import hashlib

                novel = (
                    db.execute(
                        "select 1 from sources where constitution_id=?",
                        (hashlib.sha256(p["smiles"].encode()).hexdigest(),),
                    ).fetchone()
                    is None
                )
                c = selector.SelectionCandidate(
                    i, 400000 + slot, p["smiles"], True, check["exact"], observed, novel, components
                )
                eligible = bool(
                    c.exact == before[i].exact
                    and audit["qualified_design_pass"]
                    and contexts(audit) <= set(census[i]["context_codes"])
                    and (not before[i].local_features_observed or observed)
                )
                row = {
                    "candidate": asdict(c),
                    "source_check": check,
                    "assessment": audit,
                    "eligible": eligible,
                    "graph": p,
                    "tree_basis_interpretation": "Unchanged parent tree with constructed endpoints. Same mathematical fundamental-cycle predicate; not original sampled provenance.",
                    "inherited_route_verdict": None,
                    "inherited_model_likelihood": None,
                }
                assessed.append(row)
                if eligible:
                    pools[i].append(c)
                    gates[i, c.ordinal] = True
                    contexts_by_key[i, c.ordinal] = contexts(audit)
            entry = {
                "request": i,
                "family": layout.family,
                "parent": graph,
                "baseline_source_check": checked,
                "report": report,
                "assessed_proposals": assessed,
            }
            write(HERE / "parents" / f"{i:04d}.json", entry)
            records.append(entry)
    families = {}
    all_after = {}
    for f in FAMILIES:
        baseline = [c for i, c in before.items() if layouts[i].family == f]
        selected, solver = selector.select_design_supported(
            [pools[c.request] for c in baseline],
            baseline=baseline,
            design_pass=frozenset(k for k, v in gates.items() if v and layouts[k[0]].family == f),
            time_limit=5,
            node_limit=10000,
        )
        selector.check_strict_noninferiority(baseline, selected)
        assert all(
            contexts_by_key[c.request, c.ordinal] <= set(census[c.request]["context_codes"])
            for c in selected
        )
        assert sum(c.local_features_observed for c in selected) >= sum(
            c.local_features_observed for c in baseline
        )
        families[f] = {
            "requests": 64,
            "exact_before": sum(c.exact for c in baseline),
            "exact_after": sum(c.exact for c in selected),
            "design_before": sum(gates[c.request, c.ordinal] for c in baseline),
            "design_after": sum(gates[c.request, c.ordinal] for c in selected),
            "context_before": sum(bool(contexts_by_key[c.request, c.ordinal]) for c in baseline),
            "context_after": sum(bool(contexts_by_key[c.request, c.ordinal]) for c in selected),
            "changed_requests": [c.request for c in selected if c != before[c.request]],
            "selections": [asdict(c) for c in selected],
            "solver": solver,
            "strict_integer_diversity_and_novelty_checks_passed": True,
        }
        all_after.update({c.request: c for c in selected})
    total_delta = sum(f["design_after"] - f["design_before"] for f in families.values())
    write(
        HERE / "result.json",
        {
            "complete": True,
            "protocol": pin(HERE / "protocol.json"),
            "source_inputs": source_pins,
            "source_controls": source_controls,
            "families": families,
            "target_requests": len(protocol["target_requests"]),
            "full_family_requests": 256,
            "full_cohort_requests": 1408,
            "full_cohort_design_before": 1324,
            "full_cohort_design_after": 1324 + total_delta,
            "full_cohort_exact_before_after": 1387,
            "source_calls_proposals": sum(len(r["assessed_proposals"]) for r in records),
            "source_calls_baseline_and_controls": len(records),
            "all_proposal_ledger": [
                {
                    "request": r["request"],
                    "ordinal": p["candidate"]["ordinal"],
                    "source_exact": p["source_check"]["exact"],
                    "design": p["assessment"]["qualified_design_pass"],
                    "eligible": p["eligible"],
                    "selected": all_after[r["request"]].ordinal == p["candidate"]["ordinal"],
                }
                for r in records
                for p in r["assessed_proposals"]
            ],
            "parent_receipts": [pin(p) for p in sorted((HERE / "parents").glob("*.json"))],
            "statuses": dict(
                __import__("collections").Counter(r["report"]["status"] for r in records)
            ),
            "CPU_seconds": time.process_time() - start,
            "new_model_TEST_network_GPU_calls": 0,
            "official_cohort_changed": False,
            "attribution": "Additional bounded decoder construction and unchanged constrained selection; not a model-learning gain or independent realism validation.",
        },
    )
    print(
        json.dumps(
            {
                "CPU_seconds": time.process_time() - start,
                "families": {
                    f: {k: v for k, v in d.items() if k not in ("selections", "solver")}
                    for f, d in families.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
