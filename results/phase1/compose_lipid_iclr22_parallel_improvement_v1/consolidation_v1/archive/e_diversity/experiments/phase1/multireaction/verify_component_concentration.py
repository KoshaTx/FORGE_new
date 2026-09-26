"""Independent saved-row recount; no solver, model, RDKit, or chemistry calls."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
OUT = WORKTREE / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/e_diversity"
RUN = OUT / "concentration_v1"


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.absolute()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def context(a):
    return {r["code"] for r in a["chemical"]["flags"] if r["tier"] == "context_required"}


def counts(rows, gates):
    groups = defaultdict(Counter)
    novel = Counter()
    novel_set = defaultdict(set)
    for r in rows:
        parts = [("product", r["smiles"], r["product_train_novel"], None)] if r["connected"] else []
        parts += [
            ("component:" + c["role"], c["smiles"], c["global_train_novel"], c["role_train_novel"])
            for c in r["components"]
        ]
        for role, identity, global_novel, role_novel in parts:
            groups[role][identity] += 1
            for key, flag in [
                (role if role == "product" else role + ":global", global_novel),
                (role + ":role", role_novel),
            ]:
                if flag:
                    novel[key] += 1
                    novel_set[key].add(identity)
    audits = [gates[r["request"], r["ordinal"]] for r in rows]
    return {
        "requests": len(rows),
        "exact": sum(r["exact"] for r in rows),
        "connected": sum(r["connected"] for r in rows),
        "design": sum(a["qualified_design_pass"] for a in audits),
        "context": sum(bool(context(a)) for a in audits),
        "design_no_context": sum(a["qualified_design_pass"] and not context(a) for a in audits),
        "supported": sum(r["local_features_observed"] for r in rows),
        "distinct_supported": len({r["smiles"] for r in rows if r["local_features_observed"]}),
        "groups": dict(groups),
        "novel": dict(novel),
        "novel_unique": {k: len(v) for k, v in novel_set.items()},
    }


def violations(left, right):
    bad = []
    for key in ("requests", "exact", "connected"):
        if left[key] != right[key]:
            bad.append(key)
    for key in ("design", "supported", "distinct_supported"):
        if right[key] < left[key]:
            bad.append(key)
    if set(left["groups"]) != set(right["groups"]):
        bad.append("role_space")
    for group, c in left["groups"].items():
        new = right["groups"].get(group, {})
        if sum(c.values()) != sum(new.values()):
            bad.append(group + ":denominator")
        if len(new) < len(c):
            bad.append(group + ":unique")
        if sum(v * v for v in new.values()) > sum(v * v for v in c.values()):
            bad.append(group + ":Simpson")
        if math.prod(v**v for v in new.values()) > math.prod(v**v for v in c.values()):
            bad.append(group + ":Shannon")
    for metric in ("novel", "novel_unique"):
        for key, value in left[metric].items():
            if right[metric].get(key, 0) < value:
                bad.append(metric + ":" + key)
    return bad


def compact(c):
    return {k: v for k, v in c.items() if k not in ("groups", "novel", "novel_unique")}


def main():
    started = time.process_time()
    p = read(RUN / "protocol.json")
    for record in p["inputs"].values():
        assert pin(Path(record["path"]))["sha256"] == record["sha256"]
    r = read(RUN / "result.json")
    assert r["complete"] and r["protocol"] == pin(RUN / "protocol.json")
    records = [
        x
        for x in read(Path(p["inputs"]["candidate_ledger"]["path"]))["candidates"]
        if x["status"] == "assessed"
    ]
    pool = {(x["index"], x["ordinal"]): x for x in records}
    assert len(pool) == len(records) == 11331
    gates = {
        (x["index"], x["ordinal"]): x["assessment"]
        for x in read(Path(p["inputs"]["common_gates"]["path"]))["attempts"]
    }
    original = read(Path(p["inputs"]["current"]["path"]))["by_family"]
    complete = {"current": [], "ordinal_control": [], "component_concentration": []}
    family_checks, ceilings, handoff = {}, {}, []
    for family, artifact in sorted(r["family_results"].items()):
        path = Path(artifact["path"])
        assert pin(path) == artifact
        fr = read(path)
        baseline = original[family]["selections"]
        old = counts(baseline, gates)
        assert old["requests"] == 64
        complete["current"].extend(baseline)
        details = {}
        for arm in ("ordinal_control", "component_concentration"):
            rows = fr["arms"][arm]["selection"]
            assert [c["request"] for c in rows] == [c["request"] for c in baseline]
            for before, after in zip(baseline, rows, strict=True):
                assert after == pool[after["request"], after["ordinal"]]["candidate"]
                assert pool[after["request"], after["ordinal"]]["family"] == family
                assert before["exact"] == after["exact"]
                assert context(gates[after["request"], after["ordinal"]]) <= context(
                    gates[before["request"], before["ordinal"]]
                )
                if arm == "component_concentration" and before["smiles"] != after["smiles"]:
                    handoff.append(
                        {
                            "family": family,
                            "request": after["request"],
                            "previous": before,
                            "candidate": after,
                            "assessment": gates[after["request"], after["ordinal"]],
                            "candidate_kind": pool[after["request"], after["ordinal"]]["kind"],
                            "prior_route_verdict": None,
                            "prior_likelihood": None,
                        }
                    )
            new = counts(rows, gates)
            assert not violations(old, new), (family, arm, violations(old, new))
            for key in ("requests", "exact", "connected", "design", "context", "design_no_context"):
                assert new[key] == fr["arms"][arm]["summary"][key]
            for group, c in new["groups"].items():
                saved = fr["arms"][arm]["summary"]["diversity"][group]
                assert saved["identities"] == c
                assert saved["unique"] == len(c) and saved["sum_squared"] == sum(
                    v * v for v in c.values()
                )
            assert fr["arms"][arm]["solver"]["status"] == "optimal_independently_verified"
            details[arm] = compact(new)
            complete[arm].extend(rows)
        assert [
            s["objective_value"] for s in fr["arms"]["ordinal_control"]["solver"]["stages"][:3]
        ] == [
            s["objective_value"]
            for s in fr["arms"]["component_concentration"]["solver"]["stages"][:3]
        ]
        family_checks[family] = details
        # Diagnostic pool ceiling only: no additional solve and no quality-rule extrapolation.
        iso = {c["role"] for b in baseline for c in b["components"] if "isocyanide" in c["role"]}
        for role in sorted(iso):
            options = {}
            for base in baseline:
                if not base["exact"]:
                    continue
                values = set()
                for rec in records:
                    x = rec["candidate"]
                    if x["request"] != base["request"] or not x["exact"]:
                        continue
                    if {c["role"] for c in x["components"]} != {
                        c["role"] for c in base["components"]
                    }:
                        continue
                    if not context(gates[x["request"], x["ordinal"]]) <= context(
                        gates[base["request"], base["ordinal"]]
                    ):
                        continue
                    values.update(c["smiles"] for c in x["components"] if c["role"] == role)
                options[base["request"]] = sorted(values)
            union = set().union(*(set(v) for v in options.values()))
            forced = Counter(v[0] for v in options.values() if len(v) == 1)
            ceilings[family + ":" + role] = {
                "requests": len(options),
                "eligible_pool_unique": len(union),
                "requests_with_single_identity_option": sum(len(v) == 1 for v in options.values()),
                "forced_identity_counts": dict(forced),
                "per_request_options": options,
                "scope": "Necessary pool support ceiling under exact role/context eligibility; does not assert simultaneous quality/floor feasibility.",
            }
    totals = {
        arm: counts(sorted(rows, key=lambda c: c["request"]), gates)
        for arm, rows in complete.items()
    }
    pooled = {
        arm: {**compact(value), "violations_vs_current": violations(totals["current"], value)}
        for arm, value in totals.items()
    }
    assert totals["current"]["requests"] == 1408 and totals["current"]["design"] == 1324
    assert not pooled["component_concentration"]["violations_vs_current"]
    assert len(handoff) == 245
    write(
        RUN / "changed_products.json",
        {
            "population": 1408,
            "changed": 245,
            "protocol": pin(RUN / "protocol.json"),
            "result": pin(RUN / "result.json"),
            "records": handoff,
            "route_policy": "Old route closure and likelihood evidence cleared for changed identity. Exact-source component identities copied from their saved candidate witness, never inferred from current incumbent.",
        },
    )
    write(
        RUN / "candidate_cohort_1408.json",
        {
            "requests": 1408,
            "selections": sorted(complete["component_concentration"], key=lambda c: c["request"]),
            "result": pin(RUN / "result.json"),
        },
    )
    write(
        RUN / "independent_verification.json",
        {
            "passed": True,
            "complete": True,
            "source": pin(Path(__file__)),
            "protocol": pin(RUN / "protocol.json"),
            "result": pin(RUN / "result.json"),
            "family_checks": family_checks,
            "pooled": pooled,
            "isocyanide_pool_limits": ceilings,
            "changed_products": pin(RUN / "changed_products.json"),
            "candidate_cohort": pin(RUN / "candidate_cohort_1408.json"),
            "cpu_seconds": time.process_time() - started,
            "new_solver_model_chemistry_network_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "verification": pin(RUN / "independent_verification.json"),
                "pooled": pooled,
                "cpu_seconds": time.process_time() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
