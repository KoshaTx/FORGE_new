"""Independent stdlib recount of immutable saved-pool controls; no selector imports."""

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return {"path": str(Path(path).absolute()), "sha256": digest(path)}


def codes(g):
    return {f["code"] for f in g["chemical"]["flags"] if f["tier"] == "context_required"}


def recount(rows, gates):
    counts = defaultdict(Counter)
    novel, novel_unique = Counter(), defaultdict(set)
    result = dict.fromkeys(
        (
            "requests",
            "exact",
            "connected",
            "design",
            "context",
            "design_no_context",
            "observed_TRAIN_features",
        ),
        0,
    )
    for c in rows:
        g = gates[c["request"], c["ordinal"]]
        flags = codes(g)
        for name, value in (
            ("requests", 1),
            ("exact", c["exact"]),
            ("connected", c["connected"]),
            ("design", g["qualified_design_pass"]),
            ("context", bool(flags)),
            ("design_no_context", g["qualified_design_pass"] and not flags),
            ("observed_TRAIN_features", c["local_features_observed"]),
        ):
            result[name] += int(value)
        if c["connected"]:
            counts["product"][c["smiles"]] += 1
            if c["product_train_novel"]:
                novel["product"] += 1
                novel_unique["product"].add(c["smiles"])
        for part in c["components"]:
            group = "component:" + part["role"]
            counts[group][part["smiles"]] += 1
            for tag, field in [("global", "global_train_novel"), ("role", "role_train_novel")]:
                if part[field]:
                    key = group + ":" + tag
                    novel[key] += 1
                    novel_unique[key].add(part["smiles"])
    result["diversity"] = {}
    for group, counter in counts.items():
        n = counter.total()
        ss = sum(x * x for x in counter.values())
        entropy = -sum((v / n) * math.log(v / n) for v in counter.values())
        result["diversity"][group] = {
            "observations": n,
            "unique": len(counter),
            "sum_squared": ss,
            "multiplicity_power_product": str(math.prod(v**v for v in counter.values())),
            "shannon_effective": math.exp(entropy),
            "simpson_effective": n * n / ss,
            "identities": dict(counter),
        }
    result["novelty_observations"] = dict(novel)
    result["novelty_unique"] = {k: len(v) for k, v in novel_unique.items()}
    return result


def equal(left, right):
    if isinstance(left, float):
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), (left.keys(), right.keys())
        for k in left:
            equal(left[k], right[k])
    else:
        assert left == right, (left, right)


def paired(before, after, gates):
    out = {
        "changed_product_requests": [],
        "paired": {
            k: {"gains": [], "losses": []} for k in ("exact", "design", "no_context", "observed")
        },
    }
    for a, b in zip(before, after, strict=True):
        assert a["request"] == b["request"]
        if a["smiles"] != b["smiles"]:
            out["changed_product_requests"].append(a["request"])
        ga, gb = (gates[x["request"], x["ordinal"]] for x in (a, b))
        va = (a["exact"], ga["qualified_design_pass"], not codes(ga), a["local_features_observed"])
        vb = (b["exact"], gb["qualified_design_pass"], not codes(gb), b["local_features_observed"])
        for key, x, y in zip(out["paired"], va, vb, strict=True):
            if bool(x) != bool(y):
                out["paired"][key]["gains" if y else "losses"].append(a["request"])
    return out


def floors(base, value):
    for key in ("requests", "exact", "connected"):
        assert value[key] == base[key]
    assert value["design"] >= base["design"]
    assert base["diversity"].keys() == value["diversity"].keys()
    for key, old in base["diversity"].items():
        new = value["diversity"][key]
        assert new["observations"] == old["observations"]
        assert new["unique"] >= old["unique"]
        assert new["sum_squared"] <= old["sum_squared"]
        assert int(new["multiplicity_power_product"]) <= int(old["multiplicity_power_product"])
    for key in ("novelty_observations", "novelty_unique"):
        assert all(value[key].get(k, 0) >= v for k, v in base[key].items())


def verify(run):
    start = time.process_time()
    protocol, result = read(run / "protocol.json"), read(run / "result.json")
    assert result["complete"] and result["requests"] == 1408
    assert result["protocol"] == pin(run / "protocol.json")
    assert result["qualification"] == pin(run / "qualification.json")
    for p in protocol["inputs"].values():
        assert digest(p["path"]) == p["sha256"], p
    rows = read(protocol["inputs"]["candidate_ledger"]["path"])["candidates"]
    pool = {(r["index"], r["ordinal"]): r for r in rows if r["status"] == "assessed"}
    assert len(pool) == result["candidate_entries"] == 11331
    gates = {
        (r["index"], r["ordinal"]): r["assessment"]
        for r in read(protocol["inputs"]["gates"]["path"])["attempts"]
    }
    current = read(protocol["inputs"]["current_selections"]["path"])["by_family"]
    assert len(current) == 22
    counts, comparisons, summary_rows = 0, 0, 0
    totals = {}
    for arm, values in result["arms"].items():
        collected = []
        for family, value in values["by_family"].items():
            selected = value["selected"]
            baseline = sorted(current[family]["selections"], key=lambda x: x["request"])
            assert len(selected) == len(baseline) == 64
            assert len({c["request"] for c in selected}) == 64
            for c, b in zip(selected, baseline, strict=True):
                assert c == pool[c["request"], c["ordinal"]]["candidate"]
                assert family == pool[c["request"], c["ordinal"]]["family"]
                assert c["request"] == b["request"] and c["exact"] == b["exact"]
                assert sorted(x["role"] for x in c["components"]) == sorted(
                    x["role"] for x in b["components"]
                )
                assert codes(gates[c["request"], c["ordinal"]]) <= codes(
                    gates[b["request"], b["ordinal"]]
                )
            aggregate = recount(selected, gates)
            equal(aggregate, value["summary"])
            floors(recount(baseline, gates), aggregate)
            equal(paired(baseline, selected, gates), value["paired_vs_current"])
            ordinal = result["arms"]["ordinal"]["by_family"][family]["selected"]
            equal(paired(ordinal, selected, gates), value["paired_vs_ordinal_control"])
            shard = read(run / "controls" / f"{arm}__{family}.json")
            equal(shard, {k: v for k, v in value.items() if k != "paired_vs_ordinal_control"})
            assert value["all_floors_pass"]
            collected += selected
            counts += len(selected)
            comparisons += 2
            summary_rows += 1
        assert len(collected) == 1408 and len({c["request"] for c in collected}) == 1408
        equal(recount(collected, gates), values["summary"])
        summary_rows += 1
        totals[arm] = {k: v for k, v in values["summary"].items() if isinstance(v, int)}
    record = {
        "passed": True,
        "producer": pin(Path(__file__)),
        "protocol": pin(run / "protocol.json"),
        "result": pin(run / "result.json"),
        "selection_identities": counts,
        "summary_recounts": summary_rows,
        "paired_family_recounts": comparisons,
        "global": totals,
        "all_per_request_exact_context_roles_and_family_diversity_novelty_floors": True,
        "independent_implementation": "stdlib only; no experiment, selector, chemistry or model imports",
        "scope": "Authenticates numerical outputs and eligibility/floors; does not independently prove MILP global optimality or molecular realism",
        "CPU_seconds": time.process_time() - start,
    }
    with (run / "independent_recount_v1.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "review": pin(run / "independent_recount_v1.json"),
                "CPU_seconds": record["CPU_seconds"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    verify(parser.parse_args().run)
