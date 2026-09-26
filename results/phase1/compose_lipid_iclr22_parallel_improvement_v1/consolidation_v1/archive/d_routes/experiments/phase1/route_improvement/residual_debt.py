"""Exact saved-evidence debt for the fixed 624/1408 joint panel; no new evidence."""

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

MAIN = Path(__file__).resolve().parents[5]
ROOT = Path(__file__).resolve().parents[3]
OUT = MAIN / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
DEST = OUT / "residual_debt_v1"
JOINT = OUT / "new_clock_recount_v1/joint"
KEYS = ("exact_L1", "combined_primary", "L2_ready", "L3_direct_only", "strict_secondary")
CLOSED = {"buy", "buy_and_make", "make_from_vendor_listed"}


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {
        "path": str(path.resolve().relative_to(MAIN)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def minimal_sets(values, cap=20000):
    ordered = sorted(set(values), key=lambda x: (len(x), tuple(sorted(x))))
    kept = []
    for value in ordered:
        if not any(previous <= value for previous in kept):
            kept.append(value)
        if len(kept) > cap:
            raise ValueError("Exact antichain exceeds declared cap; no truncated result admitted")
    return kept


def product_frontier(branch_options, cap=20000):
    result = [frozenset()]
    for options in branch_options:
        if not options:
            return []
        if len(result) * len(options) > 1000000:
            raise ValueError("Exact intermediate enumeration cap exceeded; no result admitted")
        result = minimal_sets((left | right for left in result for right in options), cap)
    return result


def closes_with_listings(product, components, granted):
    if not product["exact_L1"]:
        return False
    for branch in product["branches"]:
        if branch["makeable"]:
            continue
        valid = False
        for path in components[branch["identity"]]["paths"]:
            if path["path_supported"] and all(
                leaf["listed"] or leaf["identity"] in granted for leaf in path["leaves"]
            ):
                valid = True
                break
        if not valid:
            return False
    return True


def freeze():
    assert not DEST.exists()
    protocol = read(JOINT / "protocol.json")
    sources = {
        name: JOINT / name
        for name in [
            "result.json",
            "products.json",
            "components.json",
            "validation.json",
            "protocol.json",
            "control_products.json",
        ]
    }
    sources.update(
        {
            "population": MAIN / protocol["population"]["path"],
            "joint_selection": OUT.parent / "integration/joint_selection_v1/run_v2/result.json",
            "joint_binding": OUT / "joint_binding_review_v1/result.json",
            "recount_review": OUT / "new_clock_recount_v1/root_review.json",
            "recount_completion": OUT / "new_clock_recount_v1/completion.json",
            "prior_audit": OUT / "result.json",
            "prior_audit_protocol": OUT / "audit_protocol_v1.json",
            "new_four_searches": OUT / "leaf_search_v1/protocol.json",
            "new_head_search": OUT / "b258_search_v1/protocol.json",
        }
    )
    for key, ref in read(OUT / "audit_protocol_v1.json")["inputs"].items():
        if key.startswith("prior_search_protocol_") or key == "root_search_v2":
            sources[key] = MAIN / ref["path"]
    DEST.mkdir()
    write(
        DEST / "protocol.json",
        {
            "schema": "forge.fixed_joint_residual_route_debt.v1",
            "producer": pin(Path(__file__)),
            "inputs": {k: pin(p) for k, p in sources.items()},
            "population": 1408,
            "families": 22,
            "expected_primary": 624,
            "expected_exact": 1387,
            "expected_L2": 1189,
            "expected_direct": 121,
            "expected_strict": 27,
            "clock": protocol["as_of_utc"],
            "maximum_CPU_seconds": 150,
            "total_stage_budget_CPU_seconds": 180,
            "threads": 1,
            "seed": 0,
            "new_HTTP_search_planner_paths_model_GPU_TEST_calls": 0,
            "reuse_decision": "Reuse authenticated current component/product classifications. Earlier complete route frontier is for a different616-product population and cannot answer fixed624joint debt. No classifier rerun or new route construction.",
            "enumeration": {
                "exact_per_product_inclusion_minimal_leaf_sets": True,
                "maximum_antichain_per_product": 20000,
                "maximum_intermediate_combinations": 1000000,
                "on_cap": "Fail audit; never silently truncate.",
                "ranked_explicit_batches": "All observed inclusion-minimal product sets of up to4identities, all singleton leaves, and top4singleton-ranked nested prefixes. Larger exact sets retained in per-product ledger. This is not global optimality over every possible4-identity combination.",
                "maximum_ranked_sets": 100000,
            },
            "interpretation": "Conditional full-product closure only if each exact missing leaf acquires admissible fresh positive listing evidence. No guarantee any query succeeds; missing/zero/failed listing evidence is not unmakeability. Whole-root closure and new searches remain separate hypothetical evidence events.",
        },
    )
    print(json.dumps(pin(DEST / "protocol.json")))


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (150, 150))
    start = time.process_time()
    protocol = read(DEST / "protocol.json")
    assert pin(Path(__file__)) == protocol["producer"]
    data = {}
    for key, ref in protocol["inputs"].items():
        path = MAIN / ref["path"]
        assert pin(path) == ref
        data[key] = read(path)
    result = data["result.json"]
    products = data["products.json"]["products"]
    components = {r["identity"]: r for r in data["components.json"]["components"]}
    requests = data["population"]["requests"]
    assert (
        result["products"] == protocol["inputs"]["products.json"]
        and result["components"] == protocol["inputs"]["components.json"]
    )
    assert (
        result["as_of_utc"] == protocol["clock"]
        and data["validation.json"]["passed"]
        and data["recount_completion"]["all_completed"]
    )
    assert data["recount_review"]["passed"] and data["joint_binding"]["passed"]
    selected = data["joint_selection"]["selected_evidence"]
    assert [r["route_request"] for r in selected] == requests
    assert [r["route_verdict"] for r in selected] == data["control_products.json"]["products"]
    assert len(products) == len(requests) == 1408 and len(components) == len(
        data["components.json"]["components"]
    )
    expected = dict(zip(KEYS, [1387, 624, 1189, 121, 27], strict=True))
    assert {k: sum(r[k] for r in products) for k in KEYS} == expected
    assert sum(len(r["branches"]) for r in products) == 3723
    pending = defaultdict(list)
    for row in result["pending_or_unsolved_paths"]:
        pending[row["target_identity"]].append(row)
    attempts = defaultdict(list)
    for key, value in data.items():
        if key.startswith("prior_search_protocol_") or key in {
            "root_search_v2",
            "new_four_searches",
            "new_head_search",
        }:
            for ordinal, row in enumerate(value.get("panel", [])):
                identity = row.get("identity", row.get("canonical_smiles"))
                if identity:
                    attempts[identity].append(
                        {
                            "protocol": protocol["inputs"][key],
                            "declared_index": row.get("index", ordinal),
                        }
                    )
    family = defaultdict(Counter)
    roots_to_products = defaultdict(set)
    leaf_to_products = defaultdict(set)
    leaf_to_roots = defaultdict(set)
    leaf_states = defaultdict(set)
    root_options = {}
    root_records = []
    all_invalid = []
    for identity, component in components.items():
        if component["state"] in CLOSED:
            continue
        supported = [p for p in component["paths"] if p["path_supported"]]
        options = []
        paths = []
        for path in component["paths"]:
            missing = frozenset(x["identity"] for x in path["leaves"] if not x["listed"])
            paths.append(
                {
                    "route_id": path["route_id"],
                    "supported": path["path_supported"],
                    "complete": path["complete"],
                    "receipt": path["receipt"],
                    "missing_exact_leaf_identities": sorted(missing),
                    "reasons": path["reasons"],
                }
            )
            if path["path_supported"]:
                assert missing and not path["complete"]
                options.append(missing)
                for leaf in path["leaves"]:
                    if not leaf["listed"]:
                        leaf_to_roots[leaf["identity"]].add(identity)
                        leaf_states[leaf["identity"]].update(
                            leaf["listing_states"] or ["unassessed_no_lookup"]
                        )
            else:
                all_invalid.append({"root": identity, **paths[-1]})
        root_options[identity] = minimal_sets(options)
        status = (
            "supported_paths_missing_exact_listings"
            if supported
            else (
                "invalid_or_unsupported_assessed_paths_only"
                if component["paths"]
                else (
                    "no_admitted_path_saved_pending_trees"
                    if pending[identity]
                    else "no_admitted_or_pending_path"
                )
            )
        )
        root_records.append(
            {
                "identity": identity,
                "state": component["state"],
                "debt": status,
                "assessed_paths": paths,
                "minimal_listing_sets": [sorted(x) for x in root_options[identity]],
                "saved_pending_trees": pending[identity],
                "prior_declared_or_attempted_searches": attempts[identity],
            }
        )
    product_records = []
    frontiers = {}
    root_sets = {}
    requirement_products = defaultdict(set)
    for index, (product, request) in enumerate(zip(products, requests, strict=True)):
        assert (
            index == product["index"] == request["index"] and product["family"] == request["family"]
        )
        assert (
            product["exact_L1"] == request["exact_L1"]
            and product["strict_secondary"] == request["strict_complete"]
        )
        assert [(r["branch_id"], r["identity"]) for r in product["branches"]] == [
            (r["branch_id"], r["identity"]) for r in request["requirements"]
        ]
        for branch in product["branches"]:
            component = components[branch["identity"]]
            assert branch["state"] == component["state"]
            assert branch["makeable"] == (component["state"] in CLOSED)
            assert branch["direct_listed"] == any(
                v["state"] == "listed" for v in component["direct"]
            )
            assert branch["computational_path"] == any(
                v["path_supported"] for v in component["paths"]
            )
            assert branch["listed_leaf_path"] == any(v["complete"] for v in component["paths"])
        assert product["combined_primary"] == (
            product["exact_L1"] and all(b["makeable"] for b in product["branches"])
        )
        assert product["L2_ready"] == (
            product["exact_L1"]
            and all(b["direct_listed"] or b["computational_path"] for b in product["branches"])
        )
        assert product["L3_direct_only"] == (
            product["exact_L1"] and all(b["direct_listed"] for b in product["branches"])
        )
        f = family[product["family"]]
        f.update({"requests": 1, **{k: product[k] for k in KEYS}})
        missing_roots = sorted({b["identity"] for b in product["branches"] if not b["makeable"]})
        missing_paths = [identity for identity in missing_roots if not root_options[identity]]
        if not product["exact_L1"]:
            category = "nonexact_L1_no_invented_components"
        elif product["combined_primary"]:
            category = "currently_makeable"
        elif missing_paths:
            category = "at_least_one_component_without_supported_path"
        else:
            category = "supported_paths_only_missing_exact_listing_evidence"
        f[category] += 1
        f["unclosed_component_occurrences"] += sum(not b["makeable"] for b in product["branches"])
        sets = []
        if product["exact_L1"] and not product["combined_primary"]:
            root_sets[index] = frozenset(missing_roots)
            for identity in missing_roots:
                roots_to_products[identity].add(index)
                for option in root_options[identity]:
                    for leaf in option:
                        leaf_to_products[leaf].add(index)
            if not missing_paths:
                sets = product_frontier([root_options[identity] for identity in missing_roots])
                assert sets and all(sets)
                frontiers[index] = sets
                for subset in sets:
                    requirement_products[subset].add(index)
        product_records.append(
            {
                "index": index,
                "family": product["family"],
                "canonical_product": request.get("canonical_product", request["selected_smiles"]),
                "selected_ordinal": request["selected_ordinal"],
                "source_arm": selected[index]["source_arm"],
                "category": category,
                "unclosed_component_identities": missing_roots,
                "component_identities_without_supported_path": missing_paths,
                "minimal_listing_identity_sets": [sorted(x) for x in sets],
                "conditional_leaf_set_enumeration_complete": True,
            }
        )
    assert len(frontiers) == 565 and len(root_sets) == 763
    anchored = defaultdict(list)
    for subset, indices in requirement_products.items():
        anchored[min(subset)].append((subset, indices))

    def closes(granted):
        return sorted(
            set().union(
                *(
                    indices
                    for anchor in granted
                    for subset, indices in anchored.get(anchor, [])
                    if subset <= granted
                )
            )
        )

    for leaf, roots in leaf_to_roots.items():
        leaf_to_products[leaf] = set().union(*(roots_to_products[root] for root in roots))
    leaf_rows = []
    for leaf in sorted(leaf_to_roots):
        indices = closes(frozenset([leaf]))
        leaf_rows.append(
            {
                "identity": leaf,
                "distinct_unclosed_product_coverage": len(leaf_to_products[leaf]),
                "coverage_product_indices": sorted(leaf_to_products[leaf]),
                "distinct_unclosed_roots": len(leaf_to_roots[leaf]),
                "root_identities": sorted(leaf_to_roots[leaf]),
                "currently_observed_listing_states": sorted(leaf_states[leaf]),
                "conditional_products_closed_alone": len(indices),
                "conditional_product_indices_alone": indices,
                "prior_declared_or_attempted_searches": attempts[leaf],
            }
        )
    leaf_rows.sort(
        key=lambda r: (
            -r["conditional_products_closed_alone"],
            -r["distinct_unclosed_product_coverage"],
            r["identity"],
        )
    )
    coverage_rows = sorted(
        leaf_rows,
        key=lambda r: (
            -r["distinct_unclosed_product_coverage"],
            -r["conditional_products_closed_alone"],
            r["identity"],
        ),
    )
    candidate_sets = {subset for subset in requirement_products if len(subset) <= 4}
    candidate_sets.update(frozenset([r["identity"]]) for r in leaf_rows)
    for size in range(2, 5):
        candidate_sets.add(frozenset(r["identity"] for r in leaf_rows[:size]))
    assert len(candidate_sets) <= protocol["enumeration"]["maximum_ranked_sets"]
    batches = [
        {
            "identities": sorted(subset),
            "identity_count": len(subset),
            "conditional_closed_products": len(closes(subset)),
            "conditional_product_indices": closes(subset),
        }
        for subset in candidate_sets
    ]
    batches.sort(
        key=lambda r: (r["identity_count"], -r["conditional_closed_products"], r["identities"])
    )
    root_rows = []
    for record in root_records:
        identity = record["identity"]
        indices = sorted(roots_to_products[identity])
        alone = sorted(i for i, roots in root_sets.items() if roots == {identity})
        root_rows.append(
            {
                **record,
                "distinct_unclosed_product_coverage": len(indices),
                "coverage_product_indices": indices,
                "conditional_products_closed_if_this_root_fully_qualified": len(alone),
                "conditional_product_indices_if_this_root_fully_qualified": alone,
            }
        )
    root_rows.sort(
        key=lambda r: (
            -r["conditional_products_closed_if_this_root_fully_qualified"],
            -r["distinct_unclosed_product_coverage"],
            r["identity"],
        )
    )
    root_batches = defaultdict(list)
    for index, required in root_sets.items():
        root_batches[required].append(index)
    root_batch_rows = [
        {
            "identities": sorted(required),
            "count": len(required),
            "products_with_exactly_these_unclosed_roots": indices,
            "conditional_closed_product_indices_if_all_roots_fully_qualified": sorted(
                i for i, needs in root_sets.items() if needs <= required
            ),
        }
        for required, indices in root_batches.items()
    ]
    root_batch_rows.sort(
        key=lambda r: (
            r["count"],
            -len(r["conditional_closed_product_indices_if_all_roots_fully_qualified"]),
            r["identities"],
        )
    )
    # Independent direct boolean witness checks bypass antichain/enumeration code.
    verify_batches = [
        frozenset(row["identities"]) for row in batches if row["conditional_closed_products"]
    ][:100]
    assert all(
        closes(batch)
        == [
            r["index"]
            for r in products
            if not r["combined_primary"] and closes_with_listings(r, components, batch)
        ]
        for batch in verify_batches
    )
    for index, subsets in frontiers.items():
        assert all(closes_with_listings(products[index], components, s) for s in subsets)
    totals = dict(sum(family.values(), Counter()))
    next_listing = next((r for r in batches if r["conditional_closed_products"]), None)
    next_root = next(
        (
            r
            for r in root_rows
            if not r["minimal_listing_sets"]
            and not r["prior_declared_or_attempted_searches"]
            and r["conditional_products_closed_if_this_root_fully_qualified"]
        ),
        None,
    )
    result = {
        "schema": "forge.fixed_joint_residual_route_debt_result.v1",
        "passed": True,
        "protocol": pin(DEST / "protocol.json"),
        "full_population": 1408,
        "totals": totals,
        "by_family": [{"family": name, **dict(values)} for name, values in sorted(family.items())],
        "distinct_required_components": len(components),
        "distinct_unclosed_components": len(root_records),
        "component_debt_counts": dict(Counter(r["debt"] for r in root_records)),
        "component_state_counts": dict(Counter(r["state"] for r in components.values())),
        "assessed_invalid_paths": all_invalid,
        "saved_pending_tree_reason_counts": dict(
            Counter(row["reason"] for records in pending.values() for row in records)
        ),
        "required_root_pending_tree_count": sum(len(pending[r["identity"]]) for r in root_records),
        "leaf_identity_count": len(leaf_rows),
        "listing_state_identity_counts": dict(
            Counter(
                state for row in leaf_rows for state in row["currently_observed_listing_states"]
            )
        ),
        "exact_listing_closable_products": len(frontiers),
        "minimal_product_leaf_set_count": sum(map(len, frontiers.values())),
        "distinct_minimal_product_leaf_sets": len(requirement_products),
        "largest_minimal_product_leaf_set": max(map(len, requirement_products)),
        "products_with_listing_only_singleton_solution": sum(
            any(len(s) == 1 for s in subsets) for subsets in frontiers.values()
        ),
        "products_without_supported_complete_path": sum(
            row["category"] == "at_least_one_component_without_supported_path"
            for row in product_records
        ),
        "enumeration_censored": False,
        "independent_checks": {
            "request_objects_match_fixed_joint": 1408,
            "control_verdicts_match_original_joint": 1408,
            "branch_component_identity_checks": 3723,
            "full_metric_recount_matches": expected,
            "explicit_batch_direct_boolean_checks": len(verify_batches),
            "every_product_minimal_set_directly_closes": True,
        },
        "next_smallest_listing_batch": next_listing,
        "next_unattempted_component_search_hypothesis": next_root,
        "next_batch_not_authorized_or_executed": True,
        "new_observed_gains": 0,
        "new_HTTP_search_planner_model_GPU_TEST_calls": 0,
        "claim_scope": "Exact conditional maxima over already-supported saved paths at fixed as-of; all granted identities must obtain fresh exact positive evidence and pass future admission/recount. Ranking is no guarantee availability or successful search. No changed cohort identities or new constructed paths.",
        "CPU_seconds": time.process_time() - start,
    }
    artifacts = {
        "products": product_records,
        "roots": root_rows,
        "leaf_rank_by_singleton_closure": leaf_rows,
        "leaf_rank_by_coverage": coverage_rows,
        "explicit_listing_batches": batches,
        "explicit_component_batches": root_batch_rows,
    }
    result["artifacts"] = {}
    for name, value in artifacts.items():
        path = DEST / (name + ".json")
        write(path, {"protocol": pin(DEST / "protocol.json"), "records": value})
        result["artifacts"][name] = pin(path)
    write(DEST / "result.json", result)
    print(
        json.dumps(
            {
                "result": pin(DEST / "result.json"),
                "totals": totals,
                "debt": result["component_debt_counts"],
                "next_listing": next_listing,
                "next_unattempted_root": (
                    None
                    if next_root is None
                    else {
                        k: next_root[k]
                        for k in [
                            "identity",
                            "conditional_products_closed_if_this_root_fully_qualified",
                            "conditional_product_indices_if_this_root_fully_qualified",
                        ]
                    }
                ),
                "CPU_seconds": result["CPU_seconds"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    globals()[args.action]()
