"""Extend the saved-source census only by the frozen completed four-target search."""

import copy
from collections import defaultdict
from pathlib import Path

import validator_v2 as prior
from validator_v2 import (  # noqa: F401
    ASSESS,
    BASIS,
    EVAL,
    FROZEN_TIME,
    HERE,
    ROOT,
    SCHEMA,
    ComputationalRoute,
    MakeabilityPolicy,
    RouteBasis,
    content_sha,
    location,
    normalized_original,
    old,
    pin,
    read_pin,
    same_pin,
)

SEARCH = HERE.parent / "routes/next_priorities_v2/leaf_search_v1"
PROTOCOL = {
    "path": str((SEARCH / "protocol.json").relative_to(ROOT)),
    "sha256": "dcc17135418a92efc7844847dcc3a517aa5d22ad0873c69167e2968aa18c389c",
}
COMPLETION = {
    "path": str((SEARCH / "completion.json").relative_to(ROOT)),
    "sha256": "0237e5536226acb9195e9a66f584349e9747a3a9bf5477373702d40dcedebaa3",
}


def new_search_population():
    """Authenticate every result, including zero-gain searches, before adding raw trees."""
    protocol, completion = read_pin(PROTOCOL), read_pin(COMPLETION)
    review = read_pin(pin(SEARCH / "root_review.json"))
    if not review["passed"] or not same_pin(review["protocol"], PROTOCOL):
        raise ValueError("new search lacks exact independent execution review")
    for receipt in protocol["inputs"]:
        old.authenticate(location(receipt), receipt["sha256"])
    old.authenticate(location(review["producer"]), review["producer"]["sha256"])
    if (
        not same_pin(completion["protocol"], PROTOCOL)
        or completion["all_completed"] is not True
        or completion["declared_indices"] != [0, 1, 2, 3]
        or completion["unstarted_indices"]
        or len(completion["exits"]) != 4
    ):
        raise ValueError("incomplete or changed new-search population")
    expected = {item["index"]: item for item in protocol["panel"]}
    if set(expected) != {0, 1, 2, 3}:
        raise ValueError("new-search target denominator changed")
    targets, trees, exclusions, receipts, total_cpu = {}, [], [], [], 0.0
    for exit_row in completion["exits"]:
        index = exit_row["index"]
        declared = expected.pop(index)
        saved_exit = read_pin(pin(SEARCH / f"target_{index}.exit.json"))
        usage_pin = pin(SEARCH / f"target_{index}_usage.json")
        usage = read_pin(usage_pin)
        result = read_pin(exit_row["result"])
        if (
            saved_exit != exit_row
            or not exit_row["completed"]
            or exit_row["exit_code"] != 0
            or exit_row["timeout"]
            or result["status"] != "completed"
            or result["seed"] != declared["seed"]
            or result["canonical_smiles"] != declared["identity"]
            or not same_pin(result["protocol"], PROTOCOL)
            or not same_pin(usage["protocol"], PROTOCOL)
            or result["runtime_config"] != protocol["runtime_config"]
            or usage["CPU_seconds"] > protocol["child_CPU_seconds"]
            or not usage["Python_network_guard_installed"]
            or not usage["onnx_telemetry_disabled"]
            or len(result["top_route_hypotheses"]) > protocol["search"]["maximum_routes_stored"]
        ):
            raise ValueError("new-search result target/seed/runtime/exit binding failed")
        total_cpu += usage["CPU_seconds"]
        receipts.append(
            {
                "index": index,
                "result": exit_row["result"],
                "exit": pin(SEARCH / f"target_{index}.exit.json"),
                "usage": usage_pin,
                "seed": declared["seed"],
                "target": declared["identity"],
            }
        )
        for tree_index, raw in enumerate(result["top_route_hypotheses"]):
            receipt = {
                "location": location(exit_row["result"]) + f"#/top_route_hypotheses/{tree_index}",
                "sha256": exit_row["result"]["sha256"],
            }
            try:
                normalized = normalized_original(raw, receipt)
                if normalized["identity"] != declared["identity"]:
                    raise ValueError("new original tree root differs from its declared target")
                old._original_leaves(raw, normalized, receipt)
            except (ValueError, KeyError) as error:
                exclusions.append({"receipt": receipt, "reason": str(error)})
                continue
            targets[(location(receipt), receipt["sha256"])] = declared["identity"]
            trees.append({"receipt": receipt, "target_identity": declared["identity"]})
    if expected or total_cpu > protocol["maximum_authorized_CPU_seconds"]:
        raise ValueError("incomplete or over-budget new-search population")
    return targets, {
        "protocol": PROTOCOL,
        "completion": COMPLETION,
        "root_review": pin(SEARCH / "root_review.json"),
        "results": receipts,
        "searches": 4,
        "trees": trees,
        "exclusions": exclusions,
        "recorded_child_CPU_seconds": total_cpu,
        "producer": pin(Path(__file__)),
    }


class AdmissionContext(prior.AdmissionContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.old_saved_tree_count = len(self.saved_tree_targets)
        self.new_tree_targets, self.new_tree_census = new_search_population()
        overlap = set(self.saved_tree_targets) & set(self.new_tree_targets)
        if overlap:
            raise ValueError("new search may not replace an old saved source")
        self.saved_tree_targets.update(self.new_tree_targets)


def validated_routes(envelope, *, policy, as_of, vendor_snapshots, policy_pin):
    if envelope.get("schema") != SCHEMA or envelope.get("candidate_only") is not False:
        raise ValueError("unrecognized admitted composition envelope")
    admission = read_pin(envelope["independent_admission"])
    if (
        admission["passed"] is not True
        or content_sha(envelope["paths"]) != admission["paths_sha256"]
        or pin(Path(__file__)) != admission["validator"]
        or pin(Path(prior.__file__)) != admission["validator_predecessor"]
    ):
        raise ValueError("admitted envelope or complete validator source changed")
    context = AdmissionContext(
        policy=policy, as_of=as_of, vendor_snapshots=vendor_snapshots, policy_pin=policy_pin
    )
    if context.new_tree_census != admission["new_tree_census"]:
        raise ValueError("new-search population or supporting receipts changed")
    result = defaultdict(list)
    for row in envelope["paths"]:
        context.composition(row)
        result[row["target_identity"]].append(
            ComputationalRoute(
                row["route_id"],
                RouteBasis.PLANNER_VENDOR_CLOSED,
                old._external_node(
                    row["root"], RouteBasis.PLANNER_VENDOR_CLOSED, independently_bind=False
                ),
                old.receipt(row["receipt"]),
            )
        )
    return result, []
