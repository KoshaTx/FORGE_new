"""Exact saved-step graft candidates; no change to historical validators or evidence tiers."""

import copy
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from audit_new_leaves import MAIN, OUT, PREP, SEARCH, leaves, pin, read, wrapper, write

sys.path.insert(0, str(PREP))

prior = wrapper.historical
old = wrapper.old
BASE = PREP / "paired_run_v1/treatment"
CAND = OUT / "candidate_compositions_v1"
SCHEMA = "forge.stream_d.admitted_exact_route_grafts.v1"


def source_census():
    protocol_ref, completion_ref = pin(SEARCH / "protocol.json"), pin(SEARCH / "completion.json")
    protocol, completion = read(SEARCH / "protocol.json"), read(SEARCH / "completion.json")
    if protocol_ref["sha256"] != "1608e97ca109ce51dfe280dd23501361f585cc11b42d52558cb951b05e724e71":
        raise ValueError("Unreviewed search protocol")
    if (
        not completion["all_completed"]
        or completion["declared_indices"] != [0, 1, 2, 3]
        or completion["unstarted_indices"]
    ):
        raise ValueError("Search denominator incomplete")
    for ref in protocol["inputs"] + protocol["asset_pins"]:
        if pin(MAIN / ref["path"]) != ref:
            raise ValueError("Search input or engine asset changed")
    targets, records = {}, []
    for declaration, exit_row in zip(protocol["panel"], completion["exits"], strict=True):
        i = declaration["index"]
        usage = read(SEARCH / f"target_{i}_usage.json")
        result = prior.read_pin(exit_row["result"])
        if (
            exit_row != read(SEARCH / f"target_{i}.exit.json")
            or exit_row["index"] != i
            or not exit_row["completed"]
            or exit_row["timeout"]
            or exit_row["exit_code"] != 0
            or result["status"] != "completed"
            or result["seed"] != declaration["seed"]
            or result["canonical_smiles"] != declaration["identity"]
            or not prior.same_pin(result["protocol"], protocol_ref)
            or not prior.same_pin(usage["protocol"], protocol_ref)
            or result["verified_assets"] != protocol["asset_pins"][:-1]
            or result["runtime_config"] != protocol["runtime_config"]
            or usage["CPU_seconds"] > protocol["child_CPU_seconds"]
            or not usage["Python_network_guard_installed"]
            or not usage["onnx_telemetry_disabled"]
        ):
            raise ValueError("Search target/seed/runtime/exit mismatch")
        for j, raw in enumerate(result["top_route_hypotheses"]):
            ref = {
                "location": exit_row["result"]["path"] + f"#/top_route_hypotheses/{j}",
                "sha256": exit_row["result"]["sha256"],
            }
            node = prior.normalized_original(raw, ref)
            old._original_leaves(raw, node, ref)
            if node["identity"] != declaration["identity"]:
                raise ValueError("Changed target identity")
            targets[(ref["location"], ref["sha256"])] = declaration["identity"]
            records.append(
                {
                    "receipt": ref,
                    "target": declaration["identity"],
                    "original_historical_stock_solved": raw.get("metadata", {}).get("is_solved"),
                }
            )
    if len(records) != 12 or completion["children_CPU_seconds"] > 610:
        raise ValueError("Changed retained-tree population or search budget")
    return targets, {"protocol": protocol_ref, "completion": completion_ref, "trees": records}


def snapshot_view():
    protocol = read(BASE / "protocol.json")
    source_refs = [protocol["baseline_listings"]] + protocol["vendor_envelopes"]
    rows = {}
    for ref in source_refs:
        for row in prior.read_pin(ref)["listings"]:
            rows[json.dumps(row, sort_keys=True)] = row
    return {
        "schema": "forge.existing_listing_union_view.v1",
        "listings": [rows[k] for k in sorted(rows)],
        "source_envelopes": source_refs,
        "new_observations": 0,
    }


class Context(prior.AdmissionContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        targets, self.stream_d_census = source_census()
        if set(targets) & set(self.saved_tree_targets):
            raise ValueError("New tree replaces an old source")
        self.saved_tree_targets.update(targets)
        if read(CAND / "listing_view.json") != snapshot_view():
            raise ValueError("Listing union differs from exact existing observations")


def context():
    protocol = read(BASE / "protocol.json")
    policy = prior.MakeabilityPolicy.from_mapping(prior.read_pin(protocol["policy"]))
    return Context(
        policy=policy,
        as_of=old.time(prior.FROZEN_TIME),
        vendor_snapshots=protocol["vendor_envelopes"]
        + protocol["proof_vendor_snapshots"]
        + [pin(CAND / "listing_view.json")],
        policy_pin=protocol["policy"],
    )


def graft(original, donor, donor_ref, *, pointer="", donor_pointer="", segments=None):
    result = copy.deepcopy(original)
    if original["step"] is None:
        if original["identity"] != donor["identity"]:
            return result
        return add_donor(
            donor, donor_ref, pointer=pointer, donor_pointer=donor_pointer, segments=segments
        )
    result["step"]["reactants"] = [
        graft(child, donor, donor_ref, pointer=pointer + f"/step/reactants/{i}", segments=segments)
        for i, child in enumerate(original["step"]["reactants"])
    ]
    return result


def add_donor(node, donor_ref, *, pointer, donor_pointer, segments):
    result = copy.deepcopy(node)
    if node["step"] is not None:
        segments.append(
            {
                "pointer": pointer,
                "target_identity": node["identity"],
                "source": {
                    "kind": "authenticated_original_reaction_node",
                    "original_tree": donor_ref,
                    "original_tree_target": old.pointer(donor_ref)["smiles"],
                    "normalized_node_pointer": donor_pointer,
                    "reaction_node": node["step"]["receipt"],
                    "original_tree_auditor": pin(Path(__file__)),
                },
            }
        )
        # Canonicalize the declared root using the unchanged receipt parser.
        segments[-1]["source"]["original_tree_target"] = old.canonical(
            segments[-1]["source"]["original_tree_target"]
        )
        result["step"]["reactants"] = [
            add_donor(
                child,
                donor_ref,
                pointer=pointer + f"/step/reactants/{i}",
                donor_pointer=donor_pointer + f"/step/reactants/{i}",
                segments=segments,
            )
            for i, child in enumerate(node["step"]["reactants"])
        ]
    return result


def prepare():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started = time.process_time()
    audit = read(SEARCH / "tree_audit_v1.json")
    closed = [r for r in audit["trees"] if r["currently_closed_computational_tree"]]
    assert len(closed) == 1 and closed[0]["target_index"] == 0 and closed[0]["route_ordinal"] == 2
    donor = closed[0]
    protocol = read(BASE / "protocol.json")
    old_protocol = prior.read_pin(prior.pin(prior.ASSESS / "protocol.json"))
    old_components_pin = prior.pin(prior.ASSESS / "components.json")
    old_components = prior.read_pin(old_components_pin)["components"]
    old_verdicts = {
        path["route_id"]: {
            "location": old_components_pin["path"] + f"#/components/{i}/paths/{j}",
            "sha256": old_components_pin["sha256"],
        }
        for i, row in enumerate(old_components)
        for j, path in enumerate(row["paths"])
    }
    sources = {}
    for ref in old_protocol["route_envelopes"]:
        for i, row in enumerate(prior.read_pin(ref)["paths"]):
            sources[row["route_id"]] = row, {
                "location": prior.location(ref) + f"#/paths/{i}",
                "sha256": ref["sha256"],
            }
    components = read(BASE / "components.json")["components"]
    eligible, chosen = [], {}
    for component in components:
        if component["state"] in {"make_from_vendor_listed", "buy_and_make", "buy"}:
            continue
        for path in component["paths"]:
            if (
                path["path_supported"]
                and any(r["identity"] == donor["identity"] for r in path["leaves"])
                and all(r["listed"] or r["identity"] == donor["identity"] for r in path["leaves"])
            ):
                eligible.append({"target": component["identity"], "route_id": path["route_id"]})
                chosen.setdefault(component["identity"], path["route_id"])
    CAND.mkdir(exist_ok=False)
    (CAND / "proofs").mkdir()
    write(CAND / "listing_view.json", snapshot_view())
    write(
        CAND / "protocol.json",
        {
            "schema": "forge.stream_d.exact_graft_protocol.v1",
            "inputs": {
                "producer": pin(Path(__file__)),
                "audit": pin(SEARCH / "tree_audit_v1.json"),
                "baseline_result": pin(BASE / "result.json"),
                "baseline_components": pin(BASE / "components.json"),
                "old_components": old_components_pin,
                "old_assessment": prior.pin(prior.ASSESS / "result.json"),
                "current_policy": protocol["policy"],
            },
            "selection": "All unresolved component roots with a supported path whose only unlisted identity is the sole newly closed leaf; first eligible saved path per root, retaining all eligible paths in this ledger",
            "eligible_paths": eligible,
            "selected_paths": chosen,
            "new_vendor_observations": 0,
            "independent_admission_required": True,
        },
    )
    rows = []
    for ordinal, (identity, route_id) in enumerate(sorted(chosen.items())):
        source, ref = sources[route_id]
        segments = [
            {
                "pointer": "",
                "target_identity": identity,
                "source": {
                    "kind": "admitted_original_path",
                    "basis": source["basis"],
                    "source_path_row": ref,
                    "source_verdict": old_verdicts[route_id],
                },
            }
        ]
        root = graft(
            source["root"], donor["normalized_original"], donor["receipt"], segments=segments
        )
        proof_path = CAND / "proofs" / f"root_{ordinal}.json"
        write(
            proof_path,
            {
                "schema": "forge.authenticated_saved_route_composition_candidate.v1",
                "candidate_only": True,
                "target_identity": identity,
                "root": root,
                "segments": segments,
                "as_of_utc": prior.FROZEN_TIME,
                "evaluation_clock": protocol["as_of_utc"],
                "clock_note": "The legacy proof authenticates old listings at its fixed historical clock. The new adapter separately reclassifies every terminal at the unchanged current cohort clock.",
                "vendor_snapshot": pin(CAND / "listing_view.json"),
                "policy": protocol["policy"],
                "original_assessment": prior.pin(prior.ASSESS / "result.json"),
                "terminal_identities": leaves(root),
                "axes": {
                    "experimental_execution": None,
                    "stock": None,
                    "independent_forward_replay": None,
                },
            },
        )
        rows.append(
            {
                "route_id": "stream_d_graft:" + str(ordinal),
                "target_identity": identity,
                "basis": prior.BASIS,
                "root": root,
                "receipt": pin(proof_path),
                "planner_solved": None,
                "status": "exact_saved_step_composition_to_current_listed_terminals",
            }
        )
    write(
        CAND / "candidate_paths.json",
        {
            "schema": SCHEMA,
            "candidate_only": True,
            "paths": rows,
            "protocol": pin(CAND / "protocol.json"),
        },
    )
    ctx = context()
    validation = [ctx.composition(row) for row in rows]
    write(
        CAND / "preparation_result.json",
        {
            "protocol": pin(CAND / "protocol.json"),
            "candidates": pin(CAND / "candidate_paths.json"),
            "validator": pin(Path(__file__)),
            "source_census": ctx.stream_d_census,
            "validation": validation,
            "candidate_only": True,
            "new_measured_makeability": None,
            "CPU_seconds": time.process_time() - started,
        },
    )
    print(
        json.dumps(
            {
                "result": pin(CAND / "preparation_result.json"),
                "roots": len(rows),
                "CPU_seconds": time.process_time() - started,
            }
        )
    )


class Bridge:
    def __init__(self, existing_external):
        self.previous = wrapper.Bridge(existing_external)

    def external_routes(self, envelope, *, policy, as_of, vendor_snapshots, policy_pin):
        if envelope.get("schema") != SCHEMA:
            return self.previous.external_routes(
                envelope,
                policy=policy,
                as_of=as_of,
                vendor_snapshots=vendor_snapshots,
                policy_pin=policy_pin,
            )
        wrapper.check_policy_clock(policy, as_of, policy_pin)
        admission = prior.read_pin(envelope["independent_admission"])
        if (
            envelope.get("candidate_only") is not False
            or admission.get("passed") is not True
            or admission.get("reviewer") != "root"
            or admission["validator"] != pin(Path(__file__))
            or admission["paths_sha256"] != prior.content_sha(envelope["paths"])
        ):
            raise ValueError("Independent exact candidate admission missing or changed")
        ctx = context()
        if ctx.stream_d_census != admission["source_census"]:
            raise ValueError("Search census changed after review")
        routes = defaultdict(list)
        for row in envelope["paths"]:
            ctx.composition(row)
            routes[row["target_identity"]].append(
                prior.ComputationalRoute(
                    row["route_id"],
                    prior.RouteBasis.PLANNER_VENDOR_CLOSED,
                    old._external_node(
                        row["root"],
                        prior.RouteBasis.PLANNER_VENDOR_CLOSED,
                        independently_bind=False,
                    ),
                    old.receipt(row["receipt"]),
                )
            )
        wrapper.check_terminals(routes, snapshot_view()["listings"], policy, as_of)
        return routes, []


if __name__ == "__main__":
    prepare()
