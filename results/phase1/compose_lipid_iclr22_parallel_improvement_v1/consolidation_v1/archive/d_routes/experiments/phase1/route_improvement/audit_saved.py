"""Classify the fixed 1408-product route frontier and rank exact cached-evidence gaps."""

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

import forge

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
PAR = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1"
BASE = (
    PAR
    / "routes/next_bottlenecks_v3/new_leaf_listing_diagnostic_v1/paired_evaluation_preparation_v1/paired_run_v1/treatment"
)
WEAK = {"reductive_amination", "ketone_ugi4", "disulfide_michael", "aryl_reductive_amination"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def freeze():
    manifest = (
        ROOT
        / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/baseline/source_manifest.json"
    )
    assert (
        read(manifest)["source_digest"]
        == "497bf6b57586f25a76d8380cda7d272dc83be73ed74130dca0f3eb30b20597e0"
    )
    sources = {
        name: BASE / name
        for name in ["result.json", "protocol.json", "components.json", "products.json"]
    }
    sources["quality"] = PAR / "routes/recount_closeout_v3/result.json"
    sources["previous_frontier"] = PAR / "routes/next_bottlenecks_v3/frontier.json"
    sources["baseline_manifest"] = manifest
    previous = read(PAR / "routes/next_bottlenecks_v3/next_action.json")
    for key, value in previous["inputs"].items():
        if key.startswith("prior_search_protocol_"):
            sources[key] = ROOT / value["path"]
    sources["root_search_v2"] = PAR / "routes/next_bottlenecks_v3/root_search_v2/protocol.json"
    write(
        OUT / "audit_protocol_v1.json",
        {
            "schema": "forge.route_gap_audit.v1",
            "inputs": {key: pin(path) for key, path in sources.items()},
            "producer": pin(Path(__file__)),
            "seed": 0,
            "maximum_CPU_seconds": 90,
            "threads": 1,
            "population": 1408,
            "scope": "Saved component and product exact-identity census only; fixed-point and single-leaf closure are explicitly counterfactual diagnoses, not admitted outcomes",
            "new_HTTP_search_model_TEST_calls": 0,
        },
    )


def run():
    start = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (90, 91))
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    protocol = read(OUT / "audit_protocol_v1.json")
    assert pin(Path(__file__)) == protocol["producer"]
    for value in protocol["inputs"].values():
        assert digest(ROOT / value["path"]) == value["sha256"]
    inputs = {
        key: read(ROOT / value["path"])
        for key, value in protocol["inputs"].items()
        if key != "baseline_manifest"
    }
    products = inputs["products.json"]["products"]
    components = inputs["components.json"]["components"]
    assert len(products) == 1408 and sum(row["combined_primary"] for row in products) == 616
    assert inputs["result.json"]["components"] == protocol["inputs"]["components.json"]
    assert inputs["result.json"]["products"] == protocol["inputs"]["products.json"]
    population = read(ROOT / inputs["protocol.json"]["population"]["path"])
    assert (
        digest(ROOT / inputs["protocol.json"]["population"]["path"])
        == inputs["protocol.json"]["population"]["sha256"]
    )
    requirements = {r["index"]: r for r in population["requests"]}
    quality = {r["index"]: r for r in inputs["quality"]["rows"]}
    attempts = defaultdict(list)
    for key, value in inputs.items():
        if not (key.startswith("prior_search_protocol_") or key == "root_search_v2"):
            continue
        for ordinal, row in enumerate(value.get("panel", [])):
            identity = row.get("identity", row.get("canonical_smiles"))
            if identity:
                attempts[identity].append(
                    {"protocol": protocol["inputs"][key], "index": row.get("index", ordinal)}
                )
    by_identity = {row["identity"]: row for row in components}
    known = {b["identity"] for row in products for b in row["branches"] if b["makeable"]}
    initially_known = set(known)
    root_products = defaultdict(set)
    families = defaultdict(Counter)
    requests = []
    for row in products:
        source = requirements[row["index"]]
        assert Counter((b["branch_id"], b["identity"]) for b in row["branches"]) == Counter(
            (b["branch_id"], b["identity"]) for b in source["requirements"]
        )
        q = quality[row["index"]]
        assert (
            q["smiles"] == source["selected_smiles"] and q["ordinal"] == source["selected_ordinal"]
        )
        gaps = sorted({b["identity"] for b in row["branches"] if not b["makeable"]})
        families[row["family"]].update(
            {
                "requests": 1,
                "exact_L1": row["exact_L1"],
                "primary": row["combined_primary"],
                "L2_ready": row["L2_ready"],
                "L3_direct_only": row["L3_direct_only"],
                "strict_secondary": row["strict_secondary"],
            }
        )
        for b in row["branches"]:
            if not b["makeable"]:
                families[row["family"]]["unresolved_branch_" + b["state"]] += 1
        for identity in gaps:
            root_products[identity].add(row["index"])
        requests.append(
            {
                "index": row["index"],
                "family": row["family"],
                "exact_L1": row["exact_L1"],
                "primary": row["combined_primary"],
                "unresolved_roots": gaps,
            }
        )
    expansions = {}
    while True:
        added = {}
        for row in components:
            if row["identity"] in known:
                continue
            for path in row["paths"]:
                if path["path_supported"] and all(
                    leaf["listed"] or leaf["identity"] in known for leaf in path["leaves"]
                ):
                    added[row["identity"]] = path
                    break
        if not added:
            break
        known.update(added)
        expansions.update(added)
    leaf_roots = defaultdict(set)
    leaves = defaultdict(list)
    for identity, indices in root_products.items():
        row = by_identity[identity]
        for path in row["paths"]:
            if not path["path_supported"]:
                continue
            missing = sorted({x["identity"] for x in path["leaves"] if not x["listed"]})
            for leaf in missing:
                leaves[leaf].append(
                    {
                        "root": identity,
                        "route_id": path["route_id"],
                        "receipt": path["receipt"],
                        "all_unknown_leaves": missing,
                        "affected_products": sorted(indices),
                    }
                )
            if len(missing) == 1:
                leaf_roots[missing[0]].add(identity)
    ranked = []
    for leaf, traces in leaves.items():
        closes = leaf_roots[leaf]
        gains = [
            row["index"]
            for row in requests
            if row["exact_L1"] and not row["primary"] and set(row["unresolved_roots"]) <= closes
        ]
        affected = sorted({i for trace in traces for i in trace["affected_products"]})
        family_gains = Counter(requests[i]["family"] for i in gains)
        mol = Chem.MolFromSmiles(leaf)
        ranked.append(
            {
                "identity": leaf,
                "heavy_atoms": mol.GetNumHeavyAtoms() if mol else None,
                "affected_products": affected,
                "single_leaf_conditional_gain_indices": gains,
                "conditional_gain_by_family": dict(family_gains),
                "weak_family_conditional_gains": sum(
                    n for f, n in family_gains.items() if f in WEAK
                ),
                "roots_closed_if_leaf_supported": sorted(closes),
                "previous_declared_searches": attempts.get(leaf, []),
                "eligible_new_search": leaf not in attempts,
                "traces": traces,
            }
        )
    ranked.sort(
        key=lambda r: (
            -r["weak_family_conditional_gains"],
            -len(r["single_leaf_conditional_gain_indices"]),
            -len(r["affected_products"]),
            r["heavy_atoms"],
            r["identity"],
        )
    )
    write(
        OUT / "leaf_frontier_v1.json",
        {"rows": ranked, "protocol": pin(OUT / "audit_protocol_v1.json")},
    )
    unresolved = []
    for identity, indices in sorted(root_products.items()):
        row = by_identity[identity]
        supported = [p for p in row["paths"] if p["path_supported"]]
        unresolved.append(
            {
                "identity": identity,
                "state": row["state"],
                "affected_products": sorted(indices),
                "families": dict(Counter(requests[i]["family"] for i in indices)),
                "category": (
                    "supported_path_missing_listing"
                    if supported
                    else ("unsupported_saved_paths" if row["paths"] else "no_admitted_path")
                ),
                "path_count": len(row["paths"]),
                "supported_paths": len(supported),
                "path_reasons": dict(Counter(x for p in row["paths"] for x in p["reasons"])),
                "prior_searches": attempts.get(identity, []),
                "direct_listing_outcomes": row["direct"],
            }
        )
    write(
        OUT / "result.json",
        {
            "schema": "forge.route_gap_audit_result.v1",
            "protocol": pin(OUT / "audit_protocol_v1.json"),
            "imported_forge": str(Path(forge.__file__).resolve()),
            "input_pins": protocol["inputs"],
            "full_denominators": inputs["result.json"]["totals"],
            "by_family": {k: dict(v) for k, v in families.items()},
            "component_states": dict(Counter(r["state"] for r in components)),
            "makeable_components": len(initially_known),
            "unresolved_component_roots": len(root_products),
            "unresolved_categories": dict(Counter(r["category"] for r in unresolved)),
            "unresolved_roots": unresolved,
            "requests": requests,
            "unique_unknown_leaves": len(leaves),
            "cached_closed_root_fixed_point_additions": len(expansions),
            "attempted_or_declared_identities": len(attempts),
            "new_measured_successes": 0,
            "CPU_seconds": time.process_time() - start,
            "HTTP_GPU_TEST_model_calls": 0,
            "listing_unknown_not_unmakeable": True,
            "all_source_molecules_and_denominators_unchanged": True,
        },
    )
    print(
        json.dumps(
            {
                "CPU_seconds": time.process_time() - start,
                "new_leaf_targets_with_gain": [
                    (
                        r["identity"],
                        len(r["single_leaf_conditional_gain_indices"]),
                        r["conditional_gain_by_family"],
                    )
                    for r in ranked
                    if r["eligible_new_search"] and r["single_leaf_conditional_gain_indices"]
                ][:20],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
