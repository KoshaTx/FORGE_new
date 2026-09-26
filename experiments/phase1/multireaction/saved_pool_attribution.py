"""Frozen-pool ordering controls with unchanged chemistry and diversity constraints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import forge
from experiments.phase1.multireaction import saved_pool_selector as candidate

WORKTREE = Path(__file__).resolve().parents[3]
OUTPUT = WORKTREE / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution"
RUN = OUTPUT / "ordering_v2"
FRESH = Path("results/phase1/compose_lipid_quality_confirmation_v1")
QUALITY = Path("results/phase1/compose_lipid_iclr22_research_v1/quality")
SELECTOR = Path("results/phase1/compose_lipid_structure_repair_v1/evaluation/design_selector.py")
SEEDS = (2026092671, 2026092672, 2026092673)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return {"path": str(path.absolute()), "sha256": digest(path)}


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_frozen():
    spec = importlib.util.spec_from_file_location(
        "attribution_frozen_selector", WORKTREE / SELECTOR
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def single_thread_solver(module):
    original = module.milp

    def solve(*args, **kwargs):
        kwargs["options"] = {**kwargs.get("options", {}), "threads": 1, "parallel": False}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            return original(*args, **kwargs)

    module.milp = solve


def convert(module, row):
    return module.SelectionCandidate(
        **{**row, "components": tuple(module.ComponentIdentity(**p) for p in row["components"])}
    )


def codes(gate):
    return frozenset(
        f["code"] for f in gate["chemical"]["flags"] if f["tier"] == "context_required"
    )


def tie_costs(pools, arm):
    costs = {}
    for pool in pools:
        if arm == "ordinal":
            costs.update({(c.request, c.ordinal): c.ordinal for c in pool})
            continue
        if arm == "canonical":
            ordered = sorted(pool, key=lambda c: (c.smiles or "", c.ordinal))
        elif arm == "reverse":
            ordered = sorted(pool, key=lambda c: -c.ordinal)
        else:
            seed = int(arm.removeprefix("permuted_"))
            ordered = sorted(
                pool,
                key=lambda c: hashlib.sha256(f"{seed}:{c.request}:{c.ordinal}".encode()).digest(),
            )
        costs.update({(c.request, c.ordinal): rank for rank, c in enumerate(ordered)})
    return costs


def summary(rows, gates):
    groups, novelty, unique_novel = defaultdict(Counter), Counter(), defaultdict(set)
    for c in rows:
        if c.connected:
            groups["product"][c.smiles] += 1
            if c.product_train_novel:
                novelty["product"] += 1
                unique_novel["product"].add(c.smiles)
        for part in c.components:
            key = "component:" + part.role
            groups[key][part.smiles] += 1
            for suffix, flag in (
                ("global", part.global_train_novel),
                ("role", part.role_train_novel),
            ):
                if flag:
                    novelty[key + ":" + suffix] += 1
                    unique_novel[key + ":" + suffix].add(part.smiles)
    diversity = {}
    for key, counts in groups.items():
        n = sum(counts.values())
        pp = math.prod(v**v for v in counts.values())
        square = sum(v * v for v in counts.values())
        diversity[key] = {
            "observations": n,
            "unique": len(counts),
            "sum_squared": square,
            "multiplicity_power_product": str(pp),
            "shannon_effective": math.exp(
                math.log(n) - sum(v * math.log(v) for v in counts.values()) / n
            ),
            "simpson_effective": n * n / square,
            "identities": dict(counts),
        }
    audits = [gates[c.request, c.ordinal] for c in rows]
    return {
        "requests": len(rows),
        "exact": sum(c.exact for c in rows),
        "connected": sum(c.connected for c in rows),
        "design": sum(a["qualified_design_pass"] for a in audits),
        "context": sum(bool(codes(a)) for a in audits),
        "design_no_context": sum(a["qualified_design_pass"] and not codes(a) for a in audits),
        "observed_TRAIN_features": sum(c.local_features_observed for c in rows),
        "diversity": diversity,
        "novelty_observations": dict(novelty),
        "novelty_unique": {k: len(v) for k, v in unique_novel.items()},
    }


def verify_floors(before, after, gates):
    assert [c.request for c in before] == [c.request for c in after]
    for a, b in zip(before, after, strict=True):
        assert a.exact == b.exact
        assert codes(gates[b.request, b.ordinal]) <= codes(gates[a.request, a.ordinal])
    left, right = summary(before, gates), summary(after, gates)
    assert right["design"] >= left["design"]
    assert set(left["diversity"]) == set(right["diversity"])
    for key, old in left["diversity"].items():
        new = right["diversity"][key]
        assert new["observations"] == old["observations"]
        assert new["unique"] >= old["unique"]
        assert new["sum_squared"] <= old["sum_squared"]
        assert int(new["multiplicity_power_product"]) <= int(old["multiplicity_power_product"])
    for metric in ("novelty_observations", "novelty_unique"):
        assert all(right[metric].get(k, 0) >= v for k, v in left[metric].items())
    return True


def transitions(before, after, gates):
    result = {k: {"gains": [], "losses": []} for k in ("exact", "design", "no_context", "observed")}
    changed = []
    for a, b in zip(before, after, strict=True):
        assert a.request == b.request
        if a.smiles != b.smiles:
            changed.append(a.request)
        aa, bb = gates[a.request, a.ordinal], gates[b.request, b.ordinal]
        left = (a.exact, aa["qualified_design_pass"], not codes(aa), a.local_features_observed)
        right = (b.exact, bb["qualified_design_pass"], not codes(bb), b.local_features_observed)
        for key, x, y in zip(result, left, right, strict=True):
            if bool(x) != bool(y):
                result[key]["gains" if y else "losses"].append(a.request)
    return {"changed_product_requests": changed, "paired": result}


def freeze():
    assert Path(forge.__file__).resolve().is_relative_to(WORKTREE)
    RUN.mkdir(exist_ok=False, parents=True)
    files = {
        "producer": Path(__file__),
        "candidate_selector": Path(candidate.__file__),
        "baseline_selector": WORKTREE / SELECTOR,
        "candidate_ledger": WORKTREE / FRESH / "assessment/d1_candidate_diagnostics.json",
        "compact_ledger": WORKTREE / FRESH / "compact_attempts.json",
        "gates": WORKTREE / QUALITY / "all22_original_reselection/gates.json",
        "original_selections": WORKTREE / QUALITY / "all22_original_reselection/result.json",
        "current_selections": WORKTREE / QUALITY / "all22_context_preserving/result.json",
        "current_protocol": WORKTREE / QUALITY / "all22_context_preserving/protocol.json",
        "current_verification": WORKTREE / QUALITY / "all22_context_preserving/verification.json",
        "candidate_builder": WORKTREE / "results/phase1/compose_lipid_quality_selection_v2/run.py",
        "construction": WORKTREE / FRESH / "construct.py",
        "construction_protocol": WORKTREE / FRESH / "construction-protocol.json",
        "baseline_manifest": OUTPUT.parent / "baseline/source_manifest.json",
    }
    manifest = read(files["baseline_manifest"])
    assert (
        manifest["source_digest"]
        == "497bf6b57586f25a76d8380cda7d272dc83be73ed74130dca0f3eb30b20597e0"
    )
    source_trace = [
        "compose_lipid_component_constraints.py",
        "compose_lipid_scaffold_construction.py",
        "compose_lipid_symmetric_arms.py",
        "compose_lipid_generation.py",
        "source_core_completion.py",
    ]
    for name in source_trace:
        files["trace_" + name] = WORKTREE / "forge/model" / name
    pins = {k: pin(v) for k, v in files.items()}
    expected = {x["path"]: x["sha256"] for x in manifest["files"]}
    for name in source_trace:
        rel = "forge/model/" + name
        assert digest(WORKTREE / rel) == expected[rel]
    protocol = {
        "schema": "forge.saved_pool_ordering_controls.v1",
        "inputs": pins,
        "worktree": str(WORKTREE),
        "forge_import": str(Path(forge.__file__).resolve()),
        "seeds": list(SEEDS),
        "requests": 1408,
        "requests_per_family": 64,
        "families": 22,
        "arms": ["ordinal", "canonical", *[f"permuted_{s}" for s in SEEDS], "reverse"],
        "qualification": "Pinned historical selector reproduces all saved identities; candidate default reproduces frozen selector.",
        "pools": "All 11331 assessed unique candidate entries retained; exclusions are separate eligibility masks.",
        "intervention": "Only final lexicographic tie costs; no model score exists in final selector inputs.",
        "treatment_floors": "Current selected per-request exact status, all product/component observation/unique/Shannon/Simpson and novelty floors; context code subset of current selection; maximize unchanged frozen limited design first.",
        "counterfactual_control": "Ordinal arm under the same current-panel floors as all treatments; separate exact historical replay precedes it.",
        "quality_scope": "Original frozen all-candidate gates, predating later selected-only ring receipts. No new molecule or current-quality promotion.",
        "cost_scope": "Incremental replay/selection CPU only; sunk generation and constructor budgets identical and retained.",
        "maximum_CPU_seconds": 1800,
        "solver_threads": 1,
        "solver_seconds_per_stage": 5,
        "node_limit": 10000,
        "new_model_chemistry_TEST_network_calls": 0,
        "no_causal_claim": "Tests ordering dependence of a model-generated fixed pool, not removal of learning or a rules-only generator.",
        "early_stop": "Stop on hash/source/default-replay/invariant failure; preserve solver censoring and baseline fallback; no automatic retry.",
    }
    write(RUN / "protocol.json", protocol)
    print(json.dumps({"protocol": pin(RUN / "protocol.json")}))


def load_inputs():
    p = read(RUN / "protocol.json")
    assert str(WORKTREE) == p["worktree"]
    assert Path(forge.__file__).resolve().is_relative_to(WORKTREE)
    for record in p["inputs"].values():
        assert digest(Path(record["path"])) == record["sha256"], record["path"]
    doc = read(Path(p["inputs"]["candidate_ledger"]["path"]))
    rows = [r for r in doc["candidates"] if r["status"] == "assessed"]
    assert len(rows) == 11331
    gates = {
        (r["index"], r["ordinal"]): r["assessment"]
        for r in read(Path(p["inputs"]["gates"]["path"]))["attempts"]
    }
    original = read(Path(p["inputs"]["original_selections"]["path"]))["selections"]
    current = read(Path(p["inputs"]["current_selections"]["path"]))["by_family"]
    return p, rows, gates, original, current


def build(module, rows, baseline_rows, gates):
    all_pools = defaultdict(list)
    for row in rows:
        c = convert(module, row["candidate"])
        all_pools[c.request].append(c)
    baseline = sorted([convert(module, r) for r in baseline_rows], key=lambda c: c.request)
    pools, excluded = [], []
    for base in baseline:
        pool = []
        for c in all_pools[base.request]:
            gate = gates.get((c.request, c.ordinal))
            if gate is not None and not codes(gate) <= codes(gates[base.request, base.ordinal]):
                excluded.append((c.request, c.ordinal))
            else:
                pool.append(c)
        assert base in pool
        pools.append(pool)
    clean = frozenset(
        (c.request, c.ordinal)
        for pool in pools
        for c in pool
        if gates.get((c.request, c.ordinal), {}).get("qualified_design_pass")
    )
    return pools, baseline, clean, excluded


def qualify():
    started = time.process_time()
    p, rows, gates, original, current = load_inputs()
    frozen = load_frozen()
    single_thread_solver(frozen)
    single_thread_solver(candidate)
    results = {}
    for family in sorted(current):
        saved = original[family]["saved"]
        pools, baseline, clean, excluded = build(frozen, rows, saved, gates)
        chosen, report = frozen.select_design_supported(pools, baseline=baseline, design_pass=clean)
        assert (
            json.loads(json.dumps([asdict(c) for c in chosen])) == current[family]["selections"]
        ), family
        cp, cb, cc, _ = build(candidate, rows, saved, gates)
        chosen2, report2 = candidate.select_design_supported(cp, baseline=cb, design_pass=cc)
        assert [asdict(c) for c in chosen2] == [asdict(c) for c in chosen], family
        assert report == report2, family
        results[family] = {
            "requests": len(chosen),
            "frozen_identity_exact": True,
            "default_candidate_identity_exact": True,
            "status": report["status"],
        }
    write(
        RUN / "qualification.json",
        {
            "passed": True,
            "protocol": pin(RUN / "protocol.json"),
            "by_family": results,
            "CPU_seconds": time.process_time() - started,
            "requests": sum(x["requests"] for x in results.values()),
            "profile": "44 family solver calls including loading/authentication",
        },
    )
    print(json.dumps(read(RUN / "qualification.json")))


def run_controls():
    started = time.process_time()
    p, rows, gates, original, current = load_inputs()
    qualification = read(RUN / "qualification.json")
    assert qualification["passed"] and qualification["requests"] == 1408
    single_thread_solver(candidate)
    out = RUN / "controls"
    out.mkdir(exist_ok=False)
    by_arm, references = {}, {}
    for family in sorted(current):
        references[family] = build(candidate, rows, current[family]["selections"], gates)
    for arm in p["arms"]:
        armstart = time.process_time()
        family_results, all_values = {}, []
        for family, (pools, baseline, clean, excluded) in references.items():
            chosen, report = candidate.select_design_supported(
                pools, baseline=baseline, design_pass=clean, tie_costs=tie_costs(pools, arm)
            )
            assert verify_floors(baseline, chosen, gates)
            family_results[family] = {
                "summary": summary(chosen, gates),
                "paired_vs_current": transitions(baseline, chosen, gates),
                "selected": [asdict(c) for c in chosen],
                "report": report,
                "context_exclusions": excluded,
                "all_floors_pass": True,
            }
            all_values.extend(chosen)
            write(out / f"{arm}__{family}.json", family_results[family])
        by_arm[arm] = {
            "by_family": family_results,
            "summary": summary(all_values, gates),
            "CPU_seconds": time.process_time() - armstart,
        }
        assert by_arm[arm]["summary"]["requests"] == 1408
    ordinal = by_arm["ordinal"]
    for arm, report in by_arm.items():
        for family, value in report["by_family"].items():
            before = [convert(candidate, c) for c in ordinal["by_family"][family]["selected"]]
            after = [convert(candidate, c) for c in value["selected"]]
            value["paired_vs_ordinal_control"] = transitions(before, after, gates)
    result = {
        "schema": "forge.saved_pool_ordering_results.v1",
        "complete": True,
        "protocol": pin(RUN / "protocol.json"),
        "qualification": pin(RUN / "qualification.json"),
        "arms": by_arm,
        "candidate_entries": len(rows),
        "requests": 1408,
        "CPU_seconds": time.process_time() - started,
        "quality_promotion": False,
        "learned_generator_ablation": False,
        "new_model_chemistry_TEST_network_calls": 0,
    }
    write(RUN / "result.json", result)
    print(
        json.dumps(
            {
                a: {k: v for k, v in r["summary"].items() if isinstance(v, int)}
                for a, r in by_arm.items()
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "qualify", "run"))
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert int(os.environ.get(variable, "999")) <= 3, variable
    {"freeze": freeze, "qualify": qualify, "run": run_controls}[args.stage]()


if __name__ == "__main__":
    main()
