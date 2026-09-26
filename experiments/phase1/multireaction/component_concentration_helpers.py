"""Unmodified saved-ledger helpers copied from the source-pinned stream A producer."""

import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict


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
