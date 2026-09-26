"""Exact intact/grafted trees at a new clock; all historical admission stays frozen."""

import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import candidate_routes as previous
from audit_new_leaves import MAIN, OUT, leaves, pin, read, wrapper, write

prior, old = previous.prior, previous.old
DEST = OUT / "new_clock_routes_v1"
LISTINGS = OUT / "targeted_lookup_v1/admitted_vendor_listings.json"
AS_OF = "2026-09-26T06:22:31.247475+00:00"
SCHEMA = "forge.stream_d.new_clock_exact_routes.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def listings():
    envelope = read(LISTINGS)
    require(
        pin(LISTINGS)["sha256"]
        == "e546a8f4f0b52bc6c73a71362dc092602930a7b8b9e26a295becc9acff9b0b08",
        "listing envelope changed",
    )
    require(
        envelope["candidate_only"] is False and envelope["independently_admitted"] is True,
        "listing not admitted",
    )
    require(
        read(OUT / "targeted_lookup_v1/result.json")["completed_at_utc"] == AS_OF,
        "clock precedes completed lookup",
    )
    return previous.snapshot_view()["listings"] + envelope["listings"]


def clock_check(policy, as_of, policy_pin):
    wrapper.check_policy_clock(policy, old.time(wrapper.COMMON_TIME), policy_pin)
    require(as_of == old.time(AS_OF), "unreviewed new evaluation clock")
    old_rows = previous.snapshot_view()["listings"]
    for row in old_rows:
        observed = wrapper.typed_listing(row)
        before = old.classify_listing(
            row["identity"], observed, policy=policy, as_of=old.time(wrapper.COMMON_TIME)
        ).state
        after = old.classify_listing(row["identity"], observed, policy=policy, as_of=as_of).state
        require(before is after, "old listing state changed at common clock")
    return {
        "observations": len(old_rows),
        "changed_listing_states": 0,
        "previous_as_of": wrapper.COMMON_TIME,
        "common_as_of": AS_OF,
    }


def head_census():
    folder = OUT / "b258_search_v1"
    protocol, completion = read(folder / "protocol.json"), read(folder / "completion.json")
    require(
        pin(folder / "protocol.json")["sha256"]
        == "5ed3db7a080e7e35b768616d593b8dc2b58dc555c722035ec50b44eaf7b7d929",
        "head protocol changed",
    )
    require(
        completion["all_completed"]
        and not completion["timeout"]
        and completion["exit_code"] == 0
        and completion["declared_target_count"] == 1
        and completion["children_CPU_seconds"] <= 140,
        "head completion changed",
    )
    for ref in protocol["inputs"] + protocol["asset_pins"]:
        require(pin(MAIN / ref["path"]) == ref, "head input or asset changed")
    result = prior.read_pin(completion["result"])
    usage = read(folder / "target_0_usage.json")
    require(
        result["seed"] == protocol["panel"][0]["seed"]
        and result["canonical_smiles"] == protocol["panel"][0]["identity"],
        "head seed or target changed",
    )
    require(
        prior.same_pin(result["protocol"], pin(folder / "protocol.json"))
        and prior.same_pin(usage["protocol"], pin(folder / "protocol.json")),
        "head result protocol changed",
    )
    require(
        result["verified_assets"] == protocol["asset_pins"][:-1]
        and result["runtime_config"] == protocol["runtime_config"],
        "head runtime changed",
    )
    require(
        usage["CPU_seconds"] <= protocol["child_CPU_seconds"]
        and usage["Python_network_guard_installed"]
        and usage["onnx_telemetry_disabled"],
        "head search boundary changed",
    )
    require(len(result["top_route_hypotheses"]) == 3, "head retained denominator changed")
    census = []
    for i, raw in enumerate(result["top_route_hypotheses"]):
        ref = {
            "location": completion["result"]["path"] + f"#/top_route_hypotheses/{i}",
            "sha256": completion["result"]["sha256"],
        }
        root = prior.normalized_original(raw, ref)
        old._original_leaves(raw, root, ref)
        require(root["identity"] == protocol["panel"][0]["identity"], "head original root mismatch")
        census.append(
            {"receipt": ref, "root": root, "historical_stock_solved": raw["metadata"]["is_solved"]}
        )
    return census


def replace_exact_leaf(node, donor):
    result = copy.deepcopy(node)
    if node["step"] is None:
        return copy.deepcopy(donor) if node["identity"] == donor["identity"] else result
    result["step"]["reactants"] = [
        replace_exact_leaf(child, donor) for child in node["step"]["reactants"]
    ]
    return result


def validate_path(row, *, policy, as_of):
    require(
        row["experimental_execution"] is None and row["stock"] is None,
        "unsupported scientific axes",
    )
    if row["kind"] == "intact_saved_head_tree":
        census = head_census()
        source = census[1]
        require(source["historical_stock_solved"] is True, "original planner-solved basis absent")
        require(
            row["source"] == source["receipt"] and row["root"] == source["root"],
            "intact head tree changed",
        )
        require(
            row["basis"] == "planner_solved" and row["planner_solved"] is True, "head basis changed"
        )
    elif row["kind"] == "exact_saved_leaf_graft":
        ctx = previous.context()
        source = ctx.original_segment(row["original_source"])
        receipt = row["donor_source"]
        require(
            (receipt["location"], receipt["sha256"]) in ctx.saved_tree_targets,
            "donor outside authenticated complete census",
        )
        raw = old.pointer(receipt)
        donor = prior.normalized_original(raw, receipt)
        old._original_leaves(raw, donor, receipt)
        require(donor["identity"] in leaves(source), "graft identity is not an original leaf")
        require(
            row["root"] == replace_exact_leaf(source, donor),
            "changed step, reactant, identity or graft",
        )
        require(
            row["basis"] == prior.BASIS and row["planner_solved"] is None,
            "graft must preserve derived basis",
        )
    else:
        raise ValueError("unknown route kind")
    require(
        row["root"]["identity"] == row["target_identity"] and row["root"]["step"] is not None,
        "nontrivial target required",
    )
    terminals = leaves(row["root"])
    require(terminals == row["terminal_identities"], "terminal census changed")
    all_rows = listings()
    exact = set()
    for observation in all_rows:
        if (
            old.classify_listing(
                observation["identity"],
                wrapper.typed_listing(observation),
                policy=policy,
                as_of=as_of,
            ).state
            is old.ListingState.LISTED
        ):
            exact.add(observation["identity"])
    require(set(terminals) <= exact, "unknown, future or expired terminal listing")
    return {
        "target": row["target_identity"],
        "terminal_occurrences": len(terminals),
        "terminals": terminals,
    }


def prepare():
    started = time.process_time()
    require(not DEST.exists(), "one-shot directory exists")
    old_protocol = prior.read_pin(prior.pin(prior.ASSESS / "protocol.json"))
    old_components_pin = prior.pin(prior.ASSESS / "components.json")
    source_verdicts = {
        path["route_id"]: {
            "location": old_components_pin["path"] + f"#/components/{i}/paths/{j}",
            "sha256": old_components_pin["sha256"],
        }
        for i, component in enumerate(prior.read_pin(old_components_pin)["components"])
        for j, path in enumerate(component["paths"])
    }
    sources = {}
    for ref in old_protocol["route_envelopes"]:
        for i, row in enumerate(prior.read_pin(ref)["paths"]):
            sources[row["route_id"]] = (
                row,
                {"location": prior.location(ref) + f"#/paths/{i}", "sha256": ref["sha256"]},
            )
    donor = next(
        row
        for row in read(OUT / "leaf_search_v1/tree_audit_v1.json")["trees"]
        if row["target_index"] == 2 and row["route_ordinal"] == 2
    )
    eligible = []
    for component in read(OUT / "full_pair_v2/treatment/components.json")["components"]:
        for path in component["paths"]:
            if (
                path["path_supported"]
                and any(r["identity"] == donor["identity"] for r in path["leaves"])
                and all(r["listed"] or r["identity"] == donor["identity"] for r in path["leaves"])
            ):
                eligible.append({"target": component["identity"], "route_id": path["route_id"]})
    require(
        len(eligible) == 10 and len({r["target"] for r in eligible}) == 1,
        "changed eligible old path census",
    )
    source, ref = sources[eligible[0]["route_id"]]
    head = head_census()[1]
    rows = [
        {
            "route_id": "stream_d_new_clock:head258",
            "kind": "intact_saved_head_tree",
            "basis": "planner_solved",
            "planner_solved": True,
            "source": head["receipt"],
            "target_identity": head["root"]["identity"],
            "root": head["root"],
        },
        {
            "route_id": "stream_d_new_clock:disulfide747",
            "kind": "exact_saved_leaf_graft",
            "basis": prior.BASIS,
            "planner_solved": None,
            "original_source": {
                "kind": "admitted_original_path",
                "basis": source["basis"],
                "source_path_row": ref,
                "source_verdict": source_verdicts[source["route_id"]],
            },
            "donor_source": donor["receipt"],
            "target_identity": source["target_identity"],
            "root": replace_exact_leaf(source["root"], donor["normalized_original"]),
        },
    ]
    for row in rows:
        row.update(
            {
                "terminal_identities": leaves(row["root"]),
                "experimental_execution": None,
                "stock": None,
            }
        )
    base = read(previous.BASE / "protocol.json")
    policy = prior.MakeabilityPolicy.from_mapping(prior.read_pin(base["policy"]))
    protocol = {
        "schema": "forge.stream_d.new_clock_route_preparation.v1",
        "inputs": {
            "producer": pin(Path(__file__)),
            "old_protocol": pin(previous.BASE / "protocol.json"),
            "lookup_result": pin(OUT / "targeted_lookup_v1/result.json"),
            "admitted_listings": pin(LISTINGS),
            "root_listing_admission": pin(OUT / "targeted_lookup_v1/root_admission.json"),
            "prior_route_validator": pin(Path(previous.__file__)),
            "head_source": pin(OUT / "b258_search_v1/tree_audit_v1.json"),
            "four_target_source": pin(OUT / "leaf_search_v1/tree_audit_v1.json"),
        },
        "as_of_utc": AS_OF,
        "eligible_disulfide_paths": eligible,
        "choice": "Head first saved route with only queried new leaf; first eligible full disulfide root path. Every original tree and eligible root path retained.",
        "new_search_HTTP_calls": 0,
        "independent_admission_required": True,
        "CPU_cap_seconds": 30,
    }
    clock = clock_check(policy, old.time(AS_OF), base["policy"])
    validation = [validate_path(row, policy=policy, as_of=old.time(AS_OF)) for row in rows]
    DEST.mkdir()
    write(DEST / "protocol.json", protocol)
    write(
        DEST / "candidate_paths.json",
        {
            "schema": SCHEMA,
            "candidate_only": True,
            "paths": rows,
            "as_of_utc": AS_OF,
            "protocol": pin(DEST / "protocol.json"),
        },
    )
    write(
        DEST / "preparation.json",
        {
            "passed": True,
            "protocol": pin(DEST / "protocol.json"),
            "candidates": pin(DEST / "candidate_paths.json"),
            "validation": validation,
            "clock_transition": clock,
            "source_census": {"four_targets": previous.source_census()[1], "head": head_census()},
            "new_measured_successes": None,
            "CPU_seconds": time.process_time() - started,
        },
    )
    print(
        json.dumps(
            {
                "preparation": pin(DEST / "preparation.json"),
                "CPU_seconds": time.process_time() - started,
            }
        )
    )


class Bridge:
    def __init__(self, external):
        self.previous = previous.Bridge(external)

    def external_routes(self, envelope, *, policy, as_of, vendor_snapshots, policy_pin):
        clock_check(policy, as_of, policy_pin)
        if envelope.get("schema") != SCHEMA:
            routes, pending = self.previous.external_routes(
                envelope,
                policy=policy,
                as_of=old.time(wrapper.COMMON_TIME),
                vendor_snapshots=vendor_snapshots,
                policy_pin=policy_pin,
            )
            if envelope.get("schema") in {previous.SCHEMA, wrapper.ADMITTED_SCHEMA, prior.SCHEMA}:
                wrapper.check_terminals(routes, previous.snapshot_view()["listings"], policy, as_of)
            return routes, pending
        admission = prior.read_pin(envelope["independent_admission"])
        require(
            envelope.get("candidate_only") is False
            and admission.get("passed") is True
            and admission.get("reviewer") == "root",
            "independent admission required",
        )
        require(
            admission["validator"] == pin(Path(__file__))
            and admission["paths_sha256"] == prior.content_sha(envelope["paths"]),
            "admitted paths or validator changed",
        )
        require(
            envelope["as_of_utc"] == AS_OF
            and any(prior.same_pin(ref, pin(LISTINGS)) for ref in vendor_snapshots),
            "clock or declared new listing input missing",
        )
        routes = defaultdict(list)
        for row in envelope["paths"]:
            validate_path(row, policy=policy, as_of=as_of)
            basis = (
                prior.RouteBasis.PLANNER_SOLVED
                if row["kind"] == "intact_saved_head_tree"
                else prior.RouteBasis.PLANNER_VENDOR_CLOSED
            )
            routes[row["target_identity"]].append(
                prior.ComputationalRoute(
                    row["route_id"],
                    basis,
                    old._external_node(row["root"], basis, independently_bind=False),
                    old.receipt(envelope["independent_admission"]),
                )
            )
        return routes, []


if __name__ == "__main__":
    prepare()
