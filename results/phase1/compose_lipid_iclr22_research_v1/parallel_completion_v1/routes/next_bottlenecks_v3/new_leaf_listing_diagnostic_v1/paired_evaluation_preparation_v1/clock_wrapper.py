"""Authenticate old proofs at their original clock; classify terminals at the new clock."""

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").exists())
PAR = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1"
SEARCH = HERE.parents[1] / "root_search_v2"
sys.path.insert(0, str(PAR / "route_admission"))
import validator_v3 as historical  # noqa: E402

old = historical.old
COMMON_TIME = "2026-09-26T01:56:45.654730+00:00"
OLD_PROTOCOL = PAR / "route_evaluation_v2/protocol.json"
OLD_PIN = {
    "path": str(OLD_PROTOCOL.relative_to(ROOT)),
    "sha256": "a2a056f851c86d29c5dc8703bedd42adbe9bfe1e9f2bafb68441ffa15fdfb2c1",
}
NEW_LISTINGS = PAR / "listing_admission_v1/admitted_vendor_listings.json"
NEW_LISTINGS_PIN = historical.pin(NEW_LISTINGS)
ADMITTED_SCHEMA = "forge.independently_admitted_original_planner_routes.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def original_protocol():
    return historical.read_pin(OLD_PIN)


def check_policy_clock(policy, as_of, policy_pin):
    protocol = original_protocol()
    require(historical.same_pin(policy_pin, protocol["policy"]), "policy pin changed")
    require(
        policy == historical.MakeabilityPolicy.from_mapping(historical.read_pin(policy_pin)),
        "policy value changed",
    )
    require(as_of == old.time(COMMON_TIME), "unreviewed common clock")


def typed_listing(row):
    return old.VendorListing(
        row["identity"],
        old.LookupOutcome(row["outcome"]),
        row["vendor_count"],
        old.time(row["observed_at"]) if row["observed_at"] else None,
        old.Receipt(historical.location(row["receipt"]), row["receipt"]["sha256"]),
    )


def require_positive(identity, row, policy, as_of):
    require(
        old.classify_listing(identity, typed_listing(row), policy=policy, as_of=as_of).state
        is old.ListingState.LISTED,
        "terminal lacks exact current listing",
    )


@lru_cache(maxsize=1)
def original_rows():
    protocol = original_protocol()
    references = (
        [protocol["baseline_listings"]]
        + protocol["vendor_envelopes"]
        + protocol["proof_vendor_snapshots"]
    )
    records = {}
    for reference in references:
        for row in historical.read_pin(reference)["listings"]:
            records[json.dumps(row, sort_keys=True)] = row
    return list(records.values())


def prove_clock_transition(policy, as_of):
    checked = []
    for row in original_rows():
        listing = typed_listing(row)
        before = old.classify_listing(
            row["identity"], listing, policy=policy, as_of=old.time(historical.FROZEN_TIME)
        ).state
        after = old.classify_listing(row["identity"], listing, policy=policy, as_of=as_of).state
        require(before is after, "existing listing changed state at new common clock")
        checked.append((row["identity"], before.value))
    return {
        "existing_observations_checked": len(checked),
        "changed_states": 0,
        "old_time": historical.FROZEN_TIME,
        "new_time": as_of.isoformat(),
    }


def terminal_identities(node):
    if node.step is None:
        return [node.identity]
    return [identity for child in node.step.reactants for identity in terminal_identities(child)]


def check_terminals(routes, rows, policy, as_of):
    listed = {
        row["identity"]
        for row in rows
        if old.classify_listing(
            row["identity"], typed_listing(row), policy=policy, as_of=as_of
        ).state
        is old.ListingState.LISTED
    }
    terminals = [
        identity
        for paths in routes.values()
        for path in paths
        for identity in terminal_identities(path.root)
    ]
    require(bool(terminals), "empty route terminal census")
    require(set(terminals) <= listed, "route terminal lost exact current listing")
    return {
        "terminal_occurrences": len(terminals),
        "distinct_terminal_identities": len(set(terminals)),
        "all_terminals_listed_at_new_clock": True,
    }


def validate_original_path(row):
    require(
        row["basis"] == "planner_solved" and row["planner_solved"] is True,
        "original planner basis/flag changed",
    )
    original = old.pointer(row["receipt"])
    require(
        original.get("metadata", {}).get("is_solved") is True,
        "original historical stock solution absent",
    )
    normalized = historical.normalized_original(original, row["receipt"])
    require(normalized == row["root"], "original complete tree changed")
    require(normalized["identity"] == row["target_identity"], "original target changed")
    old._original_leaves(original, normalized, row["receipt"])
    return normalized


def new_source_census():
    protocol_pin, completion_pin = historical.pin(SEARCH / "protocol.json"), historical.pin(
        SEARCH / "completion.json"
    )
    protocol, completion = historical.read_pin(protocol_pin), historical.read_pin(completion_pin)
    require(
        protocol_pin["sha256"]
        == "1dcf0733b68608b41b34fb18e5df146f2b793a4a318474021c82f058f2c225c4",
        "new source protocol changed",
    )
    require(
        completion_pin["sha256"]
        == "1857ed40595476f04daa6838c6d13dd0b635aaac98886db5e73f11fa6c7fe469",
        "new source completion changed",
    )
    for ref in protocol["inputs"] + protocol["asset_pins"]:
        require(
            historical.pin(ROOT / historical.location(ref)) == ref, "source or engine asset changed"
        )
    review = historical.read_pin(historical.pin(SEARCH / "root_review.json"))
    require(
        review["passed"] is True and historical.same_pin(review["protocol"], protocol_pin),
        "source execution review absent",
    )
    require(
        completion["all_completed"] is True
        and completion["declared_indices"] == [0, 1, 2, 3]
        and not completion["unstarted_indices"],
        "incomplete source search population",
    )
    trees, results = [], []
    for index, declared in enumerate(protocol["panel"]):
        exit_row = completion["exits"][index]
        saved_exit = historical.read_pin(historical.pin(SEARCH / f"target_{index}.exit.json"))
        usage = historical.read_pin(historical.pin(SEARCH / f"target_{index}_usage.json"))
        result = historical.read_pin(exit_row["result"])
        require(
            saved_exit == exit_row
            and exit_row["completed"]
            and exit_row["exit_code"] == 0
            and not exit_row["timeout"],
            "invalid search exit",
        )
        require(
            result["status"] == "completed"
            and result["seed"] == declared["seed"]
            and result["canonical_smiles"] == declared["identity"],
            "target/seed mismatch",
        )
        require(
            historical.same_pin(result["protocol"], protocol_pin)
            and historical.same_pin(usage["protocol"], protocol_pin),
            "result protocol mismatch",
        )
        require(
            result["runtime_config"] == protocol["runtime_config"]
            and result["verified_assets"] == protocol["asset_pins"][:-1],
            "runtime assets changed",
        )
        require(
            usage["CPU_seconds"] <= protocol["child_CPU_seconds"]
            and usage["Python_network_guard_installed"]
            and usage["onnx_telemetry_disabled"],
            "search boundary changed",
        )
        require(len(result["top_route_hypotheses"]) <= 3, "source tree cap exceeded")
        results.append(exit_row["result"])
        for ordinal, raw in enumerate(result["top_route_hypotheses"]):
            ref = {
                "location": historical.location(exit_row["result"])
                + f"#/top_route_hypotheses/{ordinal}",
                "sha256": exit_row["result"]["sha256"],
            }
            normalized = historical.normalized_original(raw, ref)
            require(normalized["identity"] == declared["identity"], "source root changed")
            old._original_leaves(raw, normalized, ref)
            trees.append(
                {
                    "receipt": ref,
                    "target": declared["identity"],
                    "historical_solved": raw.get("metadata", {}).get("is_solved"),
                }
            )
    require(len(trees) == 12, "complete retained tree denominator changed")
    return {
        "protocol": protocol_pin,
        "completion": completion_pin,
        "results": results,
        "trees": trees,
        "tree_count": len(trees),
    }


class Bridge:
    def __init__(self, original_external):
        self.original_external = original_external
        self.audit = []

    def external_routes(self, envelope, *, policy, as_of, vendor_snapshots, policy_pin):
        check_policy_clock(policy, as_of, policy_pin)
        if envelope.get("schema") == historical.SCHEMA:
            clock = prove_clock_transition(policy, as_of)
            protocol = original_protocol()
            rows, pending = historical.validated_routes(
                envelope,
                policy=policy,
                as_of=old.time(historical.FROZEN_TIME),
                vendor_snapshots=protocol["vendor_envelopes"] + protocol["proof_vendor_snapshots"],
                policy_pin=policy_pin,
            )
            require(sum(len(v) for v in rows.values()) == 40, "historic composition count changed")
            self.audit.append(
                {
                    "kind": "historical_compositions",
                    "clock": clock,
                    "routes": 40,
                    **check_terminals(rows, original_rows(), policy, as_of),
                }
            )
            return rows, pending
        if envelope.get("schema") == ADMITTED_SCHEMA:
            require(
                envelope.get("candidate_only") is False
                and envelope.get("independently_admitted") is True,
                "new path is not independently admitted",
            )
            admission = historical.read_pin(envelope["independent_admission"])
            require(
                admission["passed"] is True and admission["reviewer"] == "root",
                "new route admission failed",
            )
            require(
                admission["wrapper"] == historical.pin(Path(__file__))
                and admission["paths_sha256"] == historical.content_sha(envelope["paths"]),
                "new route/wrapper admission pin changed",
            )
            require(new_source_census() == admission["source_census"], "new source census changed")
            require(len(envelope["paths"]) == 1, "new route scope changed")
            for row in envelope["paths"]:
                validate_original_path(row)
            require(
                any(historical.same_pin(ref, NEW_LISTINGS_PIN) for ref in vendor_snapshots),
                "new listing envelope is not a declared evaluation input",
            )
            listings = historical.read_pin(NEW_LISTINGS_PIN)
            require(
                listings["independently_admitted"] is True and listings["candidate_only"] is False,
                "new listing admission missing",
            )
            rows, pending = self.original_external(
                envelope,
                policy=policy,
                as_of=as_of,
                vendor_snapshots=vendor_snapshots,
                policy_pin=policy_pin,
            )
            self.audit.append(
                {
                    "kind": "new_intact_tree",
                    "source_tree_count": 12,
                    **check_terminals(rows, original_rows() + listings["listings"], policy, as_of),
                }
            )
            return rows, pending
        return self.original_external(
            envelope,
            policy=policy,
            as_of=as_of,
            vendor_snapshots=vendor_snapshots,
            policy_pin=policy_pin,
        )


def candidate_envelope():
    return copy.deepcopy(historical.read_pin(historical.pin(HERE / "candidate_routes.json")))
