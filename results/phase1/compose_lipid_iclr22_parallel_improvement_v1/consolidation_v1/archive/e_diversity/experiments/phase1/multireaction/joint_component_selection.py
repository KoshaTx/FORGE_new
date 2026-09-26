"""Conservative joint choice among already source- and route-assessed alternatives."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from experiments.phase1.multireaction import component_concentration_helpers as h
from experiments.phase1.multireaction import component_concentration_selector as selector
from experiments.phase1.multireaction.verify_component_concentration import counts, violations

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = WORKTREE.parent.parent
N = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1"
OUT = N / "integration/joint_selection_v1"
ROUTE_KEYS = ("L2_ready", "combined_primary", "L3_direct_only", "strict_secondary")


def eligibility(before, after, old_gate, new_gate, old_route, new_route):
    reasons = []
    if before.exact != after.exact:
        reasons.append("changed_exact_status")
    if old_gate["qualified_design_pass"] and not new_gate["qualified_design_pass"]:
        reasons.append("lost_current_design_pass")
    if not h.codes(new_gate) <= h.codes(old_gate):
        reasons.append("new_context_code")
    for key in ROUTE_KEYS:
        if type(old_route[key]) is not bool or type(new_route[key]) is not bool:
            raise ValueError("Untyped route " + key)
        if old_route[key] and not new_route[key]:
            reasons.append("lost_" + key)
    if {p.role for p in before.components} != {p.role for p in after.components}:
        reasons.append("changed_role_set")
    return reasons


def bind_route(candidate, family, request, verdict):
    if (
        request["index"] != candidate.request
        or verdict["index"] != candidate.request
        or request["family"] != family
        or verdict["family"] != family
        or request["canonical_product"] != candidate.smiles
        or request["selected_ordinal"] != candidate.ordinal
        or request["exact_L1"] != candidate.exact
        or verdict["exact_L1"] != candidate.exact
    ):
        raise ValueError("Route request/product identity mismatch")
    if candidate.exact:
        parts = {p.role: p.smiles for p in candidate.components}
        requirements = {r["role"]: r["identity"] for r in request["requirements"]}
        if parts != requirements:
            raise ValueError("Route component identities mismatch")
        declared = {r["branch_id"]: r["identity"] for r in request["requirements"]}
        observed = {r["branch_id"]: r["identity"] for r in verdict["branches"]}
        if declared != observed:
            raise ValueError("Classifier branches mismatch")
    return True


def routing_summary(rows, routes, gates):
    return {
        "requests": len(rows),
        **{k: sum(routes[c.request, c.ordinal][k] for c in rows) for k in ROUTE_KEYS},
        "makeable_design": sum(
            routes[c.request, c.ordinal]["combined_primary"]
            and gates[c.request, c.ordinal]["qualified_design_pass"]
            for c in rows
        ),
        "makeable_design_no_context": sum(
            routes[c.request, c.ordinal]["combined_primary"]
            and gates[c.request, c.ordinal]["qualified_design_pass"]
            and not h.codes(gates[c.request, c.ordinal])
            for c in rows
        ),
    }


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    route_dir = N / "d_routes"
    b_records = N / "b_quality/closure_v1"
    e_records = N / "e_diversity/concentration_v1"
    files = {
        "producer": Path(__file__),
        "selector": Path(selector.__file__),
        "helpers": Path(h.__file__),
        "independent_counter": WORKTREE
        / "experiments/phase1/multireaction/verify_component_concentration.py",
        "tests": N / "e_diversity/joint_tests_v1.xml",
        "test_source": WORKTREE / "tests/test_joint_component_selection.py",
        "current": ROOT
        / "results/phase1/compose_lipid_iclr22_research_v1/quality/all22_context_preserving/result.json",
        "gates": N / "a_attribution/common_gate_ledger_v1/gates.json",
        "B_selection": b_records / "result.json",
        "B_witness": b_records / "changed_products.json",
        "B_verification": b_records / "verification.json",
        "E_selection": e_records / "candidate_cohort_1408.json",
        "E_witness": e_records / "source_witnesses.json",
        "E_verification": e_records / "independent_verification.json",
        "D_protocol": route_dir / "full_pair_v2/treatment/protocol.json",
        "D_products": route_dir / "full_pair_v2/treatment/products.json",
        "D_validation": route_dir / "full_pair_v2/treatment/validation.json",
        "B_protocol": route_dir / "joint_b_d_v1/assessment/protocol.json",
        "B_products": route_dir / "joint_b_d_v1/assessment/products.json",
        "B_validation": route_dir / "joint_b_d_v1/assessment/validation.json",
        "E_protocol": route_dir / "e_routes_v1/assessment/protocol.json",
        "E_products": route_dir / "e_routes_v1/assessment/products.json",
        "E_validation": route_dir / "e_routes_v1/assessment/validation.json",
    }
    clocks = set()
    for arm in ("D", "B", "E"):
        protocol = h.read(files[arm + "_protocol"])
        pop = ROOT / protocol["population"]["path"]
        assert h.digest(pop) == protocol["population"]["sha256"]
        files[arm + "_requests"] = pop
        clocks.add(protocol["as_of_utc"])
        assert h.read(files[arm + "_validation"])["passed"]
    assert len(clocks) == 1
    h.write(
        OUT / "protocol.json",
        {
            "schema": "forge.route_preserving_joint_selection.v1",
            "inputs": {k: h.pin(v) for k, v in files.items()},
            "population": 1408,
            "per_family": 64,
            "families": 22,
            "candidate_scope": "original1408 plus E245 selected identity alternatives plus B19 selected repair alternatives; no additional prior pool candidates or proposals",
            "route_clock": next(iter(clocks)),
            "baseline": "D-only621, B+D620 and E+D494 independently classified identities; preserve exact corresponding provenance",
            "eligibility": "Per-request exact status and roles unchanged; retain every old design pass; new context codes subset; retain each old positive L2_ready, computational makeability, direct-onlyL3 and strict dossier. Unknown new strict proof cannot inherit old verdict. No identity transfer.",
            "objectives": [
                "maximize_limited_design",
                "maximize_distinct_supported",
                "maximize_supported",
                "minimize_component_role_sum_squared",
                "original_ordinal_tie",
            ],
            "floors": "Original complete current panel per-family product/component observation/unique/integerShannon/Simpson/novelty+supported floors; independent pooled same and global role-agnostic checks; failed whole-panel audit retains entire original panel, never select posthoc family subsets.",
            "outcome_policy": "Report attempted and admitted panels separately with all denied alternatives and reasons. New identity routes are copied only from their exact already-assessed request ledger. No new route classifier claims.",
            "bounds": {
                "CPU_seconds": 300,
                "threads": 1,
                "solver_seconds_per_stage": 5,
                "node_limit": 10000,
            },
            "new_model_source_route_classifier_TEST_network_GPU_calls": 0,
        },
    )
    print(json.dumps({"protocol": h.pin(OUT / "protocol.json")}), flush=True)


def global_component_violations(before, after):
    a = Counter(p.smiles for c in before for p in c.components)
    b = Counter(p.smiles for c in after for p in c.components)
    import math

    bad = []
    if sum(a.values()) != sum(b.values()):
        bad.append("global_component_observations")
    if len(b) < len(a):
        bad.append("global_component_unique")
    if sum(v * v for v in b.values()) > sum(v * v for v in a.values()):
        bad.append("global_component_Simpson")
    if math.prod(v**v for v in b.values()) > math.prod(v**v for v in a.values()):
        bad.append("global_component_Shannon")
    for suffix in ("global_train_novel", "role_train_novel"):
        aa = [p.smiles for c in before for p in c.components if getattr(p, suffix)]
        bb = [p.smiles for c in after for p in c.components if getattr(p, suffix)]
        if len(bb) < len(aa) or len(set(bb)) < len(set(aa)):
            bad.append("global_component_" + suffix)
    return bad


def run():
    start = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (300, 305))
    p = h.read(OUT / "protocol.json")
    for value in p["inputs"].values():
        assert h.digest(Path(value["path"])) == value["sha256"]
    docs = {
        k: h.read(Path(v["path"]))
        for k, v in p["inputs"].items()
        if Path(v["path"]).suffix == ".json"
    }
    current = docs["current"]["by_family"]
    gates = {(r["index"], r["ordinal"]): r["assessment"] for r in docs["gates"]["attempts"]}
    candidates = {}
    family_of = {}
    routes = {}
    requests = {}
    source_arm = {}
    pool = defaultdict(list)
    baseline = []
    indexed = {
        arm: (
            {r["index"]: r for r in docs[arm + "_requests"]["requests"]},
            {r["index"]: r for r in docs[arm + "_products"]["products"]},
        )
        for arm in ("D", "B", "E")
    }
    for family, rec in current.items():
        for raw in rec["selections"]:
            c = h.convert(selector, raw)
            key = (c.request, c.ordinal)
            baseline.append(c)
            pool[c.request].append(c)
            candidates[key] = c
            family_of[c.request] = family
            source_arm[key] = "D"
            request, verdict = (mapping[c.request] for mapping in indexed["D"])
            bind_route(c, family, request, verdict)
            routes[key] = verdict
            requests[key] = request
    assert len(baseline) == 1408
    e_records = {r["request"]: r for r in docs["E_selection"]["selections"]}
    b_records = {
        r["request"]: r for f in docs["B_selection"]["families"].values() for r in f["selections"]
    }
    b_witnesses = {r["index"]: r for r in docs["B_witness"]["rows"]}
    baseline_by = {c.request: c for c in baseline}
    selected_alternatives = {
        "E": [e_records[c.request] for c in baseline if e_records[c.request]["smiles"] != c.smiles],
        "B": [b_records[i] for i in sorted(b_witnesses)],
    }
    assert len(selected_alternatives["E"]) == 245 and len(selected_alternatives["B"]) == 19
    dispositions = []
    for arm, alternatives in selected_alternatives.items():
        for raw in alternatives:
            c = h.convert(selector, raw)
            old = baseline_by[c.request]
            key = (c.request, c.ordinal)
            assert key not in candidates
            if arm == "B":
                gates[key] = b_witnesses[c.request]["design_assessment"]
                assert b_witnesses[c.request]["smiles"] == c.smiles
            oldkey = (old.request, old.ordinal)
            family = family_of[c.request]
            request, verdict = (mapping[c.request] for mapping in indexed[arm])
            bind_route(c, family, request, verdict)
            oldrole = {
                r["role"]: (r["quantity"], r["stages"]) for r in requests[oldkey]["requirements"]
            }
            newrole = {r["role"]: (r["quantity"], r["stages"]) for r in request["requirements"]}
            assert oldrole == newrole
            rejected = eligibility(old, c, gates[oldkey], gates[key], routes[oldkey], verdict)
            dispositions.append(
                {
                    "request": c.request,
                    "ordinal": c.ordinal,
                    "arm": arm,
                    "smiles": c.smiles,
                    "eligible": not rejected,
                    "reasons": rejected,
                }
            )
            candidates[key] = c
            routes[key] = verdict
            requests[key] = request
            source_arm[key] = arm
            if not rejected:
                pool[c.request].append(c)
    h.single_thread_solver(selector)
    all_selected = []
    families = {}
    for family in sorted(current):
        base = sorted(
            [c for c in baseline if family_of[c.request] == family], key=lambda c: c.request
        )
        pools = [pool[c.request] for c in base]
        clean = frozenset(
            (c.request, c.ordinal)
            for group in pools
            for c in group
            if gates[c.request, c.ordinal]["qualified_design_pass"]
        )
        chosen, solver = selector.select_design_supported(
            pools,
            baseline=base,
            design_pass=clean,
            minimize_concentration=True,
            node_limit=10000,
            time_limit=5,
        )
        h.verify_floors(base, chosen, gates)
        for a, b in zip(base, chosen, strict=True):
            assert not eligibility(
                a,
                b,
                gates[a.request, a.ordinal],
                gates[b.request, b.ordinal],
                routes[a.request, a.ordinal],
                routes[b.request, b.ordinal],
            )
        families[family] = {
            "before": h.summary(base, gates),
            "after": h.summary(chosen, gates),
            "routing_before": routing_summary(base, routes, gates),
            "routing_after": routing_summary(chosen, routes, gates),
            "solver": solver,
            "selection": [asdict(c) for c in chosen],
        }
        all_selected.extend(chosen)
    baseline.sort(key=lambda c: c.request)
    all_selected.sort(key=lambda c: c.request)
    floor_bad = violations(
        counts([asdict(c) for c in baseline], gates),
        counts([asdict(c) for c in all_selected], gates),
    ) + global_component_violations(baseline, all_selected)
    final = all_selected if not floor_bad else baseline
    record = {
        "schema": p["schema"],
        "complete": True,
        "protocol": h.pin(OUT / "protocol.json"),
        "candidate_count": len(candidates),
        "eligible_alternative_count": sum(d["eligible"] for d in dispositions),
        "dispositions": dispositions,
        "families": families,
        "pooled_violations": floor_bad,
        "whole_panel_fallback": bool(floor_bad),
        "current": h.summary(baseline, gates),
        "attempted": h.summary(all_selected, gates),
        "final": h.summary(final, gates),
        "routing_current": routing_summary(baseline, routes, gates),
        "routing_attempted": routing_summary(all_selected, routes, gates),
        "routing_final": routing_summary(final, routes, gates),
        "transitions": h.transitions(baseline, final, gates),
        "chosen_source_counts": dict(Counter(source_arm[c.request, c.ordinal] for c in final)),
        "selection": [asdict(c) for c in final],
        "selected_evidence": [
            {
                "request": c.request,
                "ordinal": c.ordinal,
                "source_arm": source_arm[c.request, c.ordinal],
                "route_request": requests[c.request, c.ordinal],
                "route_verdict": routes[c.request, c.ordinal],
                "design_assessment": gates[c.request, c.ordinal],
            }
            for c in final
        ],
        "CPU_seconds": time.process_time() - start,
        "official_cohort_promotion": False,
        "new_model_source_route_classifier_TEST_network_GPU_calls": 0,
    }
    h.write(OUT / "result.json", record)
    print(
        json.dumps(
            {
                "result": h.pin(OUT / "result.json"),
                "design": record["final"]["design"],
                "routing": record["routing_final"],
                "source_counts": record["chosen_source_counts"],
                "violations": floor_bad,
                "CPU_seconds": record["CPU_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.action == "freeze" else run)()
