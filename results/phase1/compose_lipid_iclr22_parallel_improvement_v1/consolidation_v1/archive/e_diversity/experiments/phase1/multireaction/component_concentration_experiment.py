"""Hash-frozen paired selection experiment on a saved complete development pool."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import forge
from experiments.phase1.multireaction import component_concentration_helpers as h
from experiments.phase1.multireaction import component_concentration_selector as candidate

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = WORKTREE.parent.parent
OUT = WORKTREE / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/e_diversity"
RUN = OUT / "concentration_v1"
N = OUT.parent
A_SOURCE = (
    ROOT
    / ".worktrees/iclr22-improve-a_attribution/experiments/phase1/multireaction/saved_pool_attribution.py"
)


def freeze():
    assert Path(forge.__file__).resolve().is_relative_to(WORKTREE)
    RUN.mkdir(parents=True, exist_ok=False)
    results = WORKTREE / "results/phase1"
    files = {
        "producer": Path(__file__),
        "selector": Path(candidate.__file__),
        "helpers": Path(h.__file__),
        "helper_original": A_SOURCE,
        "selector_original": results
        / "compose_lipid_structure_repair_v1/evaluation/design_selector.py",
        "candidate_ledger": results
        / "compose_lipid_quality_confirmation_v1/assessment/d1_candidate_diagnostics.json",
        "current": results
        / "compose_lipid_iclr22_research_v1/quality/all22_context_preserving/result.json",
        "current_protocol": results
        / "compose_lipid_iclr22_research_v1/quality/all22_context_preserving/protocol.json",
        "current_verification": results
        / "compose_lipid_iclr22_research_v1/quality/all22_context_preserving/verification.json",
        "common_gates": N / "a_attribution/common_gate_ledger_v1/gates.json",
        "common_gate_verification": N / "a_attribution/common_gate_ledger_v1/verification.json",
        "common_gate_protocol": N / "a_attribution/common_gate_ledger_v1/protocol.json",
        "source_baseline": N / "baseline/source_manifest.json",
        "focused_tests": OUT / "tests_v1.xml",
        "test_source": WORKTREE / "tests/test_component_concentration_selector.py",
    }
    pins = {k: h.pin(v) for k, v in files.items()}
    assert (
        pins["common_gates"]["sha256"]
        == "fea87d4001b7629b26104f07a8f3c8b1bad5e70c8f0e809e3515d5d6da846315"
    )
    assert (
        pins["selector_original"]["sha256"]
        == "30b16af0171ade303c17ff18a6af1621d43a3ea45059d1c2bdb20b8985a09d6f"
    )
    assert h.read(files["common_gate_verification"])["passed"] is True
    h.write(
        RUN / "protocol.json",
        {
            "schema": "forge.saved_pool_component_concentration.v1",
            "inputs": pins,
            "worktree": str(WORKTREE),
            "forge_import": str(Path(forge.__file__).resolve()),
            "requests": 1408,
            "families": 22,
            "requests_per_family": 64,
            "assessed_candidate_entries": 11331,
            "arms": ["ordinal_control", "component_concentration"],
            "common_objectives": [
                "maximize_limited_design",
                "maximize_distinct_supported_products",
                "maximize_supported_requests",
            ],
            "control_tail": ["minimize_ordinal"],
            "treatment_tail": [
                "minimize_sum_over_component_roles_and_identities_of_squared_multiplicity",
                "minimize_ordinal",
            ],
            "weighting": "Unweighted integer role-count sum. Observation count is fixed for each role within family; no fitted weights or outcome-dependent normalizations.",
            "floors": "Both arms retain current per-request exact statuses, role observations, product and each-role unique/Simpson/Shannon, TRAIN novelty observation and distinct counts; current per-request context-code subset; additionally hard-floor supported requests and distinct supported products equally in both arms.",
            "global_check": "Independently audit pooled product and component-role floors against current panel. A pooled regression prevents overall promotion; no post-hoc family subset selection.",
            "ledger_scope": "All 11331 assessed pool entries; only 15 identity-authenticated common gate replacements. No new candidate or generalized basis admission.",
            "determinism": {
                "solver_threads": 1,
                "solver_random_seed": 0,
                "family_order": "lexicographic",
                "final_tie": "original_ordinal_sum",
            },
            "limits": {
                "total_cpu_seconds": 1700,
                "solver_seconds_per_stage": 5,
                "node_limit": 10000,
            },
            "failure_policy": "Any source/hash/independent floor failure stops; censored solve retains full current family baseline. No auto retry. Complete result requires all22 families.",
            "quality_scope": "Limited current design/context checks and TRAIN support; no independent realism or new routing evidence.",
            "changed_identity_policy": "Old route closure and likelihood verdicts do not transfer; each changed identity needs reassessment.",
            "new_model_chemistry_TEST_network_GPU_calls": 0,
        },
    )
    print(json.dumps({"protocol": h.pin(RUN / "protocol.json")}), flush=True)


def support_check(before, after):
    assert sum(c.local_features_observed for c in after) >= sum(
        c.local_features_observed for c in before
    )
    assert len({c.smiles for c in after if c.local_features_observed}) >= len(
        {c.smiles for c in before if c.local_features_observed}
    )


def compare_floors(before, after, gates):
    try:
        h.verify_floors(before, after, gates)
        support_check(before, after)
    except AssertionError:
        return False
    return True


def collision_summary(rows):
    groups, _ = candidate._counts(rows)
    return {
        "sum_component_role_squared_multiplicity": sum(
            v * v
            for k, counts in groups.items()
            if k.startswith("component:")
            for v in counts.values()
        ),
        "per_role": {
            k: {
                "observations": sum(c.values()),
                "sum_squared": sum(v * v for v in c.values()),
                "maximum_multiplicity": max(c.values()),
                "maximum_share": max(c.values()) / sum(c.values()),
            }
            for k, c in groups.items()
            if k.startswith("component:")
        },
    }


def run():
    started = time.process_time()
    wall = time.monotonic()
    sys.set_int_max_str_digits(0)
    p = h.read(RUN / "protocol.json")
    assert str(WORKTREE) == p["worktree"]
    assert Path(forge.__file__).resolve().is_relative_to(WORKTREE)
    assert not (RUN / "result.json").exists()
    for record in p["inputs"].values():
        assert h.digest(Path(record["path"])) == record["sha256"], record["path"]
    resource.setrlimit(resource.RLIMIT_CPU, (1700, 1705))
    rows = [
        r
        for r in h.read(Path(p["inputs"]["candidate_ledger"]["path"]))["candidates"]
        if r["status"] == "assessed"
    ]
    assert len(rows) == 11331
    gates = {
        (r["index"], r["ordinal"]): r["assessment"]
        for r in h.read(Path(p["inputs"]["common_gates"]["path"]))["attempts"]
    }
    current = h.read(Path(p["inputs"]["current"]["path"]))["by_family"]
    assert len(current) == 22
    h.single_thread_solver(candidate)
    families, aggregate = {}, {"current": [], "ordinal_control": [], "component_concentration": []}
    for family in sorted(current):
        family_start = time.process_time()
        subset = [r for r in rows if r["family"] == family]
        pools, baseline, clean, context_excluded = h.build(
            candidate, subset, current[family]["selections"], gates
        )
        assert len(baseline) == 64
        report = {
            "family": family,
            "candidate_entries": len(subset),
            "context_excluded": context_excluded,
            "current": h.summary(baseline, gates),
            "arms": {},
        }
        aggregate["current"].extend(baseline)
        primary = []
        for arm, enabled in [("ordinal_control", False), ("component_concentration", True)]:
            start = time.process_time()
            chosen, status = candidate.select_design_supported(
                pools,
                baseline=baseline,
                design_pass=clean,
                minimize_concentration=enabled,
                node_limit=10000,
                time_limit=5,
            )
            assert compare_floors(baseline, chosen, gates), (family, arm)
            arm_data = {
                "selection": [asdict(c) for c in chosen],
                "summary": h.summary(chosen, gates),
                "collision": collision_summary(chosen),
                "solver": status,
                "changes_from_current": h.transitions(baseline, chosen, gates),
                "cpu_seconds": time.process_time() - start,
                "family_floors_passed": True,
            }
            report["arms"][arm] = arm_data
            aggregate[arm].extend(chosen)
            if status["status"] == "optimal_independently_verified":
                primary.append([s["objective_value"] for s in status["stages"][:3]])
        if len(primary) == 2:
            assert primary[0] == primary[1], family
        report["primary_objectives_match_when_both_optimal"] = (
            len(primary) == 2 and primary[0] == primary[1]
        )
        report["cpu_seconds"] = time.process_time() - family_start
        h.write(RUN / f"family-{family}.json", report)
        families[family] = h.pin(RUN / f"family-{family}.json")
        progress = {
            "complete": False,
            "families_completed": len(families),
            "last_family": family,
            "cpu_seconds": time.process_time() - started,
        }
        temp = RUN / "progress.tmp"
        temp.write_text(json.dumps(progress))
        os.replace(temp, RUN / "progress.json")
        print(json.dumps(progress), flush=True)
    for arm in aggregate:
        aggregate[arm].sort(key=lambda c: c.request)
        assert len(aggregate[arm]) == 1408
    summary = {
        arm: {
            "summary": h.summary(selected, gates),
            "collision": collision_summary(selected),
            "pooled_floors_passed": compare_floors(aggregate["current"], selected, gates),
            "changes_from_current": h.transitions(aggregate["current"], selected, gates),
        }
        for arm, selected in aggregate.items()
    }
    assert summary["current"]["summary"]["design"] == 1324
    assert summary["current"]["summary"]["context"] == 41
    result = {
        "schema": p["schema"],
        "complete": True,
        "protocol": h.pin(RUN / "protocol.json"),
        "family_results": families,
        "pooled": summary,
        "paired_control_to_concentration": h.transitions(
            aggregate["ordinal_control"], aggregate["component_concentration"], gates
        ),
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
        "new_model_chemistry_TEST_network_GPU_calls": 0,
        "scientific_promotion": False,
        "promotion_scope": "Pooled/family safeguards are necessary; independent route and realism comparison remain open. All changed identities are candidates only.",
    }
    h.write(RUN / "result.json", result)
    print(
        json.dumps(
            {
                "complete": True,
                "result": h.pin(RUN / "result.json"),
                "cpu_seconds": result["cpu_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)()
