"""All-request makeability rollup from one immutable snapshot of saved evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import process_time

import legacy_equivalence_v1 as equivalence
from adapters_v9 import add_listings, configure_documentary, external_routes, strict_routes, time
from common_v2 import GOAL, OUT, ROOT, authenticate, pin, read_pin, write
from computational_makeability_v2 import (
    BranchEvidence,
    BranchRequirement,
    ComponentEvidence,
    LookupOutcome,
    MakeabilityPolicy,
    ProductState,
    Receipt,
    VendorListing,
    classify_component,
    classify_product,
)
from source_relocation import CATALOGS, RESOLUTIONS


def listings_from_envelope(envelope, *, original45=False):
    result = defaultdict(list)
    for row in envelope["listings"]:
        source = read_pin(row["receipt"])
        if original45:
            assert source["observation"]["canonical_smiles"] == row["identity"]
            assert source["accessed_at_utc"].replace("Z", "+00:00") == row["observed_at"]
            assert source["admission"]["admitted"] is True and row["vendor_count"] == 1
        else:
            for key in ("identity", "outcome", "vendor_count", "observed_at"):
                assert source[key] == row[key], ("vendor scalar/source mismatch", key)
        location = row["receipt"].get("path", row["receipt"].get("location"))
        result[row["identity"]].append(
            VendorListing(
                row["identity"],
                LookupOutcome(row["outcome"]),
                row["vendor_count"],
                time(row["observed_at"]) if row["observed_at"] else None,
                Receipt(location, row["receipt"]["sha256"]),
            )
        )
    return result


def make_protocol(destination, route_files, vendor_files, *, overrides=None):
    protocol = {
        "schema": "forge.makeability_evaluation_protocol.v2",
        "classifier_predecessor": pin(
            ROOT / "forge/synthesis/assessment/computational_makeability.py"
        ),
        "new_basis": "planner_vendor_closed requires separately authenticated original whole tree and all current exact leaf listings; original planner_solved flag remains false",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "policy": pin(ROOT / "configs/route/compose_lipid_computational_makeability_v1.json"),
        "population": pin(OUT / "requests.json"),
        "admission": pin(OUT / "admission.json"),
        "baseline_listings": pin(OUT / "baseline_vendor_listings.json"),
        "route_envelopes": [pin(path) for path in route_files],
        "vendor_envelopes": [pin(path) for path in vendor_files],
        "bulk_admissions": [],
        "overall_protocol": pin(GOAL / "computational_makeability_v1/protocol.json"),
        "source_relocation_catalogs": [pin(p) for p in CATALOGS],
        "implementation": [
            pin(__file__),
            pin(OUT / "adapters_v9.py"),
            pin(OUT.parent / "routes/planner_vendor_closed_v1/documentary_stream_v1.py"),
            pin(OUT / "bulk_supplier_intervention_v1/stream_evidence_v1.py"),
            pin(OUT / "common_v2.py"),
            pin(OUT / "common.py"),
            pin(OUT / "source_relocation.py"),
            pin(OUT / "computational_makeability_v2.py"),
            pin(OUT / "legacy_equivalence_v1.py"),
            pin(ROOT / "forge/synthesis/assessment/computational_makeability.py"),
        ],
        "metrics": read_pin(pin(OUT / "admission.json"))["metrics"],
        "all_requests": 1408,
        "sampling_or_selection_changes": False,
        "strict_secondary_source": pin(GOAL / "planner/combined_v5/result.json"),
        "execution_scope": "Saved-body classification only; no models, network, chemistry replay or TEST",
        "missing_assessment_is_failure_or_zero": False,
    }
    if overrides:
        protocol.update(overrides)
    write(destination / "protocol.json", protocol)
    return protocol


def run(destination, protocol):
    start = process_time()
    for item in protocol["implementation"]:
        authenticate(item["path"], item["sha256"])
    configure_documentary(protocol.get("bulk_admissions", []), protocol["vendor_envelopes"])
    population = read_pin(protocol["population"])
    policy = MakeabilityPolicy.from_mapping(read_pin(protocol["policy"]))
    as_of = time(protocol["as_of_utc"])
    baseline_listings = read_pin(protocol["baseline_listings"])
    listings = listings_from_envelope(baseline_listings, original45=True)
    original_keys = {
        (
            row["identity"],
            row["receipt"]["sha256"],
            row["outcome"],
            row["vendor_count"],
            row["observed_at"],
        )
        for row in baseline_listings["listings"]
    }
    for item in protocol["vendor_envelopes"]:
        envelope = read_pin(item)
        # The final vendor envelope explicitly reuses the 45 pinned initial records.
        # Do not broaden their stronger-source adapter to arbitrary new admission booleans.
        additions = {
            **envelope,
            "listings": [
                row
                for row in envelope["listings"]
                if (
                    row["identity"],
                    row["receipt"]["sha256"],
                    row["outcome"],
                    row["vendor_count"],
                    row["observed_at"],
                )
                not in original_keys
            ],
        }
        for identity, values in listings_from_envelope(additions).items():
            listings[identity].extend(values)
    routes, source_origins, source_exclusions = strict_routes(population["inputs"]["families"])
    pending = []
    for item in protocol["route_envelopes"]:
        extra, missed = external_routes(
            read_pin(item),
            policy=policy,
            as_of=as_of,
            vendor_snapshots=protocol["vendor_envelopes"]
            + protocol.get("proof_vendor_snapshots", []),
            policy_pin=protocol["policy"],
        )
        pending.extend(missed)
        for identity, values in extra.items():
            routes[identity].extend(values)
    all_roots = {b["identity"] for r in population["requests"] for b in r["requirements"]}
    evidence = {}
    assessments = {}
    original_evidence = {}
    old_evidence = {}
    for identity in sorted(all_roots):
        paths = tuple(
            replace(path, root=add_listings(path.root, listings))
            for path in routes.get(identity, ())
        )
        evidence[identity] = ComponentEvidence(identity, tuple(listings.get(identity, ())), paths)
        assessments[identity] = classify_component(evidence[identity], policy=policy, as_of=as_of)
        original_evidence[identity], old_evidence[identity] = equivalence.component(
            evidence[identity], policy, as_of
        )
    by_family = defaultdict(Counter)
    rows = []
    family_role_sets = defaultdict(set)
    leaf_sets = defaultdict(set)
    for request in population["requests"]:
        family = request["family"]
        counts = by_family[family]
        counts["requests"] += 1
        counts["exact_L1"] += request["exact_L1"]
        counts["nonexact_L1"] += not request["exact_L1"]
        counts["strict_secondary_products"] += request["strict_complete"]
        requirements = tuple(
            BranchRequirement(
                b["branch_id"], b["role"], b["identity"], b["quantity"], tuple(b["stages"])
            )
            for b in request["requirements"]
        )
        supplied = tuple(BranchEvidence(b, evidence[b.identity]) for b in requirements)
        product_arguments = dict(
            request_id=str(request["index"]),
            l1_exact=request["exact_L1"],
            l1_receipt=Receipt(
                population["inputs"]["assemblies"]["path"],
                population["inputs"]["assemblies"]["sha256"],
            ),
            required_branches=requirements,
            evidence=supplied,
            policy=policy,
            as_of=as_of,
        )
        result = classify_product(**product_arguments)
        equivalence.product(product_arguments, original_evidence, old_evidence)
        branch_rows = []
        for branch in requirements:
            assessed = assessments[branch.identity]
            direct = any(item.state.value == "listed" for item in assessed.direct)
            path = any(item.path_supported for item in assessed.routes)
            complete = any(item.complete for item in assessed.routes)
            family_role_sets[family].add((branch.role, branch.identity))
            counts["required_branches"] += 1
            counts["incorporated_component_occurrences"] += branch.quantity
            counts["direct_listed_branches"] += direct
            counts["computational_path_branches"] += path
            counts["listed_leaf_path_branches"] += complete
            counts["L2_ready_branches"] += direct or path
            counts["makeable_branches"] += assessed.makeable
            counts["not_direct_listed_branches"] += not direct
            counts["not_direct_listed_with_path"] += not direct and path
            counts["no_direct_lookup_record_branches"] += not assessed.direct
            counts["no_accepted_computational_path_branches"] += not path
            for path_result in assessed.routes:
                if path_result.path_supported:
                    leaf_sets[family].update(leaf.identity for leaf in path_result.leaves)
            branch_rows.append(
                {
                    "branch_id": branch.branch_id,
                    "identity": branch.identity,
                    "state": assessed.state.value,
                    "direct_listed": direct,
                    "computational_path": path,
                    "listed_leaf_path": complete,
                    "makeable": assessed.makeable,
                }
            )
        l2_ready = request["exact_L1"] and all(
            row["direct_listed"] or row["computational_path"] for row in branch_rows
        )
        direct_only = request["exact_L1"] and all(row["direct_listed"] for row in branch_rows)
        primary = result.state is ProductState.MAKEABLE
        counts["L2_ready_products"] += l2_ready
        counts["L3_direct_only_products"] += direct_only
        counts["combined_primary_products"] += primary
        rows.append(
            {
                "index": request["index"],
                "family": family,
                "exact_L1": request["exact_L1"],
                "state": result.state.value,
                "L2_ready": l2_ready,
                "L3_direct_only": direct_only,
                "combined_primary": primary,
                "strict_secondary": request["strict_complete"],
                "branches": branch_rows,
                "reasons": list(result.reasons),
            }
        )
    component_rows = []
    for identity, assessed in assessments.items():
        component_rows.append(
            {
                "identity": identity,
                "state": assessed.state.value,
                "direct": [
                    {
                        "state": x.state.value,
                        "receipt": (
                            {
                                "location": x.evidence.receipt.location,
                                "sha256": x.evidence.receipt.sha256,
                            }
                            if x.evidence.receipt
                            else None
                        ),
                    }
                    for x in assessed.direct
                ],
                "paths": [
                    {
                        "route_id": p.evidence.route_id,
                        "basis": p.evidence.basis.value,
                        "receipt": {
                            "location": p.evidence.receipt.location,
                            "sha256": p.evidence.receipt.sha256,
                        },
                        "path_supported": p.path_supported,
                        "complete": p.complete,
                        "reasons": p.reasons,
                        "leaves": [
                            {
                                "identity": leaf.identity,
                                "listed": leaf.listed,
                                "listing_states": [x.state.value for x in leaf.listings],
                            }
                            for leaf in p.leaves
                        ],
                    }
                    for p in assessed.routes
                ],
            }
        )
    from computational_makeability_v2 import classify_listing

    def leaf_coverage(identities):
        counter = Counter({"distinct_encountered_leaf_identities": len(identities)})
        for identity in identities:
            states = [
                classify_listing(identity, item, policy=policy, as_of=as_of).state.value
                for item in listings.get(identity, ())
            ]
            if "listed" in states:
                counter["positive"] += 1
            elif not states:
                counter["unassessed_no_lookup"] += 1
            elif all(state == "zero_listings" for state in states):
                counter["observed_zero"] += 1
            else:
                counter["unresolved_lookup"] += 1
            counter["identities_with_any_lookup"] += bool(states)
            for state in set(states):
                counter["with_" + state] += 1
        return dict(counter)

    for family, counts in by_family.items():
        counts["distinct_family_role_components"] = len(family_role_sets[family])
        for _, identity in family_role_sets[family]:
            item = assessments[identity]
            direct = any(x.state.value == "listed" for x in item.direct)
            path = any(x.path_supported for x in item.routes)
            counts["distinct_family_role_direct_listed"] += direct
            counts["distinct_family_role_path_covered"] += path
            counts["distinct_family_role_makeable"] += item.makeable
    totals = dict(sum(by_family.values(), Counter()))
    assert totals["requests"] == 1408 and totals["exact_L1"] == 1387 and totals["nonexact_L1"] == 21
    assert totals["required_branches"] == 3723
    assert totals["strict_secondary_products"] == 16
    assert all(
        not row["strict_secondary"] or row["combined_primary"] for row in rows
    ), "Existing fresh strict evidence was lost by the primary adapter"
    result = {
        "schema": "forge.computational_makeability_full_cohort.v1",
        "protocol": pin(destination / "protocol.json"),
        "population_admission": protocol["admission"],
        "as_of_utc": protocol["as_of_utc"],
        "all_request_census_complete": True,
        "all_target_route_searches_complete": False,
        "all_target_vendor_lookups_complete": False,
        "claim_scope": "Dated development-cohort evidence coverage, not exhaustive search or independent held-out evaluation",
        "totals": totals,
        "by_family": dict(by_family),
        "distinct_component_constitutions": len(assessments),
        "root_identity_coverage": {
            "universe": len(assessments),
            "with_any_listing_observation": sum(bool(x.direct) for x in assessments.values()),
            "positive_listed": sum(
                any(y.state.value == "listed" for y in x.direct) for x in assessments.values()
            ),
            "unassessed_no_lookup": sum(not x.direct for x in assessments.values()),
            "with_computational_path": sum(
                any(y.path_supported for y in x.routes) for x in assessments.values()
            ),
            "makeable": sum(x.makeable for x in assessments.values()),
            "search_completeness": "Path absence is not a failed exhaustive search; see input per-target attempt ledgers",
        },
        "leaf_coverage": leaf_coverage(set().union(*leaf_sets.values())),
        "leaf_coverage_by_family": {
            family: leaf_coverage(leaf_sets[family]) for family in by_family
        },
        "complete_product_indices": [r["index"] for r in rows if r["combined_primary"]],
        "strict_complete_product_indices": [r["index"] for r in rows if r["strict_secondary"]],
        "L2_ready_product_indices": [r["index"] for r in rows if r["L2_ready"]],
        "pending_or_unsolved_paths": pending,
        "source_route_exclusions": source_exclusions,
        "source_path_origins": source_origins,
        "CPU_seconds": process_time() - start,
        "source_relocations": list(RESOLUTIONS.values()),
        "new_model_network_chemistry_TEST_calls": 0,
        "legacy_verdict_equivalence": {
            "passed": True,
            "component_constitutions": len(assessments),
            "all_requests": len(rows),
            "scope": "Identical current evidence restricted to the two original route bases; full typed assessment equality against unchanged canonical classifier. New vendor-closed proofs excluded only from this equivalence lane.",
        },
    }
    write(
        destination / "products.json",
        {"protocol": pin(destination / "protocol.json"), "products": rows},
    )
    write(
        destination / "components.json",
        {"protocol": pin(destination / "protocol.json"), "components": component_rows},
    )
    result["products"] = pin(destination / "products.json")
    result["components"] = pin(destination / "components.json")
    write(destination / "result.json", result)
    with (destination / "family_counts.csv").open("w") as handle:
        fields = [
            "family",
            "requests",
            "exact_L1",
            "L2_ready_products",
            "L3_direct_only_products",
            "combined_primary_products",
            "strict_secondary_products",
            "required_branches",
            "computational_path_branches",
            "direct_listed_branches",
            "makeable_branches",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for family in sorted(by_family):
            writer.writerow({"family": family, **by_family[family]})
    print(
        {
            "totals": totals,
            "source_exclusions": len(source_exclusions),
            "CPU_seconds": result["CPU_seconds"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--routes", action="append", default=[])
    parser.add_argument("--vendors", action="append", default=[])
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    protocol = make_protocol(
        destination,
        [Path(p).resolve() for p in args.routes],
        [Path(p).resolve() for p in args.vendors],
    )
    run(destination, protocol)
