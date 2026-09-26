"""Independently authenticate exact saved-step composition and current terminals."""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EVAL = (
    ROOT / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1"
    "/computational_makeability_v1/evaluation"
)
sys.path.insert(0, str(EVAL))
import adapters_v9 as old  # noqa: E402
from common_v2 import content_sha, pin, read_pin  # noqa: E402
from computational_makeability_v2 import (  # noqa: E402
    ComputationalRoute,
    EvidenceAxes,
    ListingState,
    MakeabilityPolicy,
    RouteBasis,
    classify_listing,
)

ASSESS = EVAL / "bulk_supplier_intervention_v1/proof_increment_v1/assessment"
FROZEN_TIME = "2026-09-25T23:48:12.899654+00:00"
SCHEMA = "forge.authenticated_saved_route_compositions.v1"
BASIS = "authenticated_saved_route_composition_v1"


def location(item):
    return item.get("path", item.get("location"))


def same_pin(left, right):
    return location(left) == location(right) and left["sha256"] == right["sha256"]


def at(root, pointer):
    if pointer == "":
        return root
    if not pointer.startswith("/"):
        raise ValueError("nonlocal normalized-tree pointer")
    value = root
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def normalized_original(raw, receipt, suffix=""):
    """Build the sole original tree shape independently of the candidate producer."""
    identity = old.canonical(raw["smiles"])
    children = raw.get("children", [])
    if not children:
        return {"identity": identity, "listings": [], "step": None}
    if len(children) != 1 or not children[0].get("children"):
        raise ValueError("original alternative or empty reaction")
    step = {
        "receipt": {
            "location": location(receipt) + suffix + "/children/0",
            "sha256": receipt["sha256"],
        },
        "forward_products": None,
        "transform_receipt": None,
        "axes": asdict(EvidenceAxes()),
        "reactants": [
            normalized_original(child, receipt, suffix + f"/children/0/children/{i}")
            for i, child in enumerate(children[0]["children"])
        ],
    }
    return {"identity": identity, "listings": [], "step": step}


def step_body(node):
    return {key: value for key, value in node["step"].items() if key != "reactants"}


def require_same_step(source, actual):
    if source["identity"] != actual["identity"]:
        raise ValueError("changed graft identity")
    if source["step"] is None or actual["step"] is None:
        raise ValueError("required complete step omitted or falsely expanded")
    if step_body(source) != step_body(actual):
        raise ValueError("changed original step or unsupported evidence/forward claim")
    if len(source["step"]["reactants"]) != len(actual["step"]["reactants"]):
        raise ValueError("complete required child set changed")


class AdmissionContext:
    def __init__(self, *, policy, as_of, vendor_snapshots, policy_pin):
        self.protocol = read_pin(pin(ASSESS / "protocol.json"))
        self.result_pin = pin(ASSESS / "result.json")
        self.components_pin = pin(ASSESS / "components.json")
        self.policy, self.as_of, self.policy_pin = policy, as_of, policy_pin
        if as_of != old.time(FROZEN_TIME) or not same_pin(policy_pin, self.protocol["policy"]):
            raise ValueError("composition requires exact frozen time and policy")
        if policy != MakeabilityPolicy.from_mapping(read_pin(policy_pin)):
            raise ValueError("policy object differs from frozen contract")
        self.snapshots = vendor_snapshots
        self.snapshot_cache = {}
        self.prefix_cache = {}
        self.segment_cache = {}
        self.raw_tree_cache = {}
        self.listing_cache = {}
        self.raw_attestations_reused = []
        self.original_rows = {
            old._listing_key(row): row
            for row in read_pin(pin(EVAL / "baseline_vendor_listings.json"))["listings"]
        }
        source_admission = read_pin(self.protocol["bulk_proof_intervention"]["admission"])
        if source_admission["passed"] is not True or source_admission["as_of_utc"] != FROZEN_TIME:
            raise ValueError("saved-tree population lacks the frozen independent admission")
        self.saved_tree_census_pin = source_admission["source_result"]
        census = read_pin(self.saved_tree_census_pin)
        self.saved_tree_targets = {}
        for row in census["decisions"]:
            key = (location(row["receipt"]), row["receipt"]["sha256"])
            if (
                key in self.saved_tree_targets
                and self.saved_tree_targets[key] != row["target_identity"]
            ):
                raise ValueError("saved tree has conflicting target bindings")
            self.saved_tree_targets[key] = row["target_identity"]
        old.configure_documentary(
            self.protocol["bulk_admissions"], self.protocol["vendor_envelopes"]
        )
        # Reuse only the separately admitted complete streaming hashes, with every
        # filesystem identity/time/size field unchanged. No large file is read here.
        for receipt in self.protocol["bulk_admissions"]:
            admitted = read_pin(receipt)
            if admitted["passed"] is not True:
                raise ValueError("bulk evidence has no independent admission")
            for evidence in admitted["compressed_source_attestations"]:
                path = (ROOT / evidence["path"]).resolve()
                stat = path.stat()
                signature = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                if (
                    list(signature) != evidence["stat_signature"]
                    or evidence["verification"] != "streamed_SHA256_1MiB_chunks"
                ):
                    raise ValueError("bulk source changed since independent streaming attestation")
                old.DOCUMENTARY_AUTH.cache[(str(path), evidence["sha256"], signature)] = evidence
                self.raw_attestations_reused.append(evidence)

    def snapshot(self, receipt):
        if not any(same_pin(receipt, supplied) for supplied in self.snapshots):
            raise ValueError("undeclared vendor snapshot")
        key = (location(receipt), receipt["sha256"])
        if key not in self.snapshot_cache:
            rows = read_pin(receipt)["listings"]
            self.snapshot_cache[key] = {old._listing_key(row): row for row in rows}
        return self.snapshot_cache[key]

    def require_saved_tree(self, receipt, target):
        if self.saved_tree_targets.get((location(receipt), receipt["sha256"])) != target:
            raise ValueError(
                "original tree outside frozen saved-source population or target changed"
            )

    def positive(self, identity, row, snapshot):
        if self.snapshot(snapshot).get(old._listing_key(row)) != row:
            raise ValueError("listing absent from exact authenticated snapshot")
        key = old._listing_key(row)
        if key not in self.listing_cache:
            self.listing_cache[key] = old._bound_listing(row, self.original_rows)
        listing = self.listing_cache[key]
        if (
            classify_listing(identity, listing, policy=self.policy, as_of=self.as_of).state
            is not ListingState.LISTED
        ):
            raise ValueError("terminal lacks current exact positive listing")

    def listed_terminal(self, identity, snapshot):
        for row in self.snapshot(snapshot).values():
            if row["identity"] == identity:
                try:
                    self.positive(identity, row, snapshot)
                    return
                except ValueError:
                    continue
        raise ValueError("terminal lacks admitted listed identity")

    def prefix(self, receipt):
        key = (location(receipt), receipt["sha256"])
        if key in self.prefix_cache:
            return self.prefix_cache[key]
        proof = old.pointer(receipt)
        if (
            proof["schema"] != "forge.saved_tree_listed_prefix_candidate.v1"
            or proof["candidate_only"] is not True
            or proof["as_of_utc"] != FROZEN_TIME
            or not same_pin(proof["policy"], self.policy_pin)
            or proof["claims"]
            != {"forward_replay": None, "stock": None, "experimental_success": None}
        ):
            raise ValueError("prefix time, policy or authority changed")
        original_pin = proof["original_tree"]
        self.require_saved_tree(original_pin, proof["original_tree_target"])
        original_key = (location(original_pin), original_pin["sha256"])
        if original_key not in self.raw_tree_cache:
            raw = old.pointer(original_pin)
            normalized = normalized_original(raw, original_pin)
            old._original_leaves(raw, normalized, original_pin)
            self.raw_tree_cache[original_key] = raw, normalized
        raw, whole = self.raw_tree_cache[original_key]
        if proof["original_tree_target"] != whole["identity"] or proof[
            "original_tree_planner_solved"
        ] is not raw.get("metadata", {}).get("is_solved"):
            raise ValueError("original target or historical solved flag relabeled")
        original = at(whole, proof["normalized_subtree_pointer"])
        if original != proof["original_normalized_subtree"]:
            raise ValueError("prefix does not retain exact original subtree")
        actual_cuts = []

        def compare(source, retained, pointer):
            if source["identity"] != retained["identity"]:
                raise ValueError("changed prefix identity")
            if retained["step"] is None:
                if len(retained["listings"]) != 1:
                    raise ValueError("terminal cut requires one actual listing witness")
                row = retained["listings"][0]
                self.positive(retained["identity"], row, proof["vendor_snapshot"])
                actual_cuts.append(
                    {
                        "original_node_pointer": pointer,
                        "identity": retained["identity"],
                        "had_reaction_below": source["step"] is not None,
                        "listing": row,
                    }
                )
                return
            if retained["listings"]:
                raise ValueError("nonterminal listing injection")
            require_same_step(source, retained)
            for index, (left, right) in enumerate(
                zip(source["step"]["reactants"], retained["step"]["reactants"], strict=True)
            ):
                compare(left, right, pointer + f"/step/reactants/{index}")

        compare(original, proof["retained_root"], proof["normalized_subtree_pointer"])
        if actual_cuts != proof["terminal_cuts"] or proof["retained_root"]["step"] is None:
            raise ValueError("terminal cut ledger differs from retained nonzero tree")
        self.prefix_cache[key] = proof["retained_root"]
        return proof["retained_root"]

    def original_segment(self, source):
        receipt = source["source_path_row"]
        key = (location(receipt), receipt["sha256"], content_sha(source))
        if key in self.segment_cache:
            return self.segment_cache[key]
        base = location(receipt).split("#", 1)[0]
        if not any(
            base == location(item) and receipt["sha256"] == item["sha256"]
            for item in self.protocol["route_envelopes"]
        ):
            raise ValueError("source route not in original admitted envelopes")
        row = old.pointer(receipt)
        verdict_pin = source["source_verdict"]
        if (
            location(verdict_pin).split("#", 1)[0] != location(self.components_pin)
            or verdict_pin["sha256"] != self.components_pin["sha256"]
        ):
            raise ValueError("route verdict outside original full assessment")
        verdict = old.pointer(verdict_pin)
        parent_location, separator, _ = location(verdict_pin).rpartition("/paths/")
        if not separator:
            raise ValueError("source verdict requires an exact component/path location")
        component = old.pointer({"location": parent_location, "sha256": verdict_pin["sha256"]})
        expected_receipt = row.get("completion_receipt", row["receipt"])
        if (
            verdict["route_id"] != row["route_id"]
            or verdict["path_supported"] is not True
            or source["basis"] != row["basis"]
            or verdict["basis"] != row["basis"]
            or component["identity"] != row["target_identity"]
            or not same_pin(verdict["receipt"], expected_receipt)
        ):
            raise ValueError("route was not supported under its unchanged original basis")
        routes, pending = old.external_routes(
            {"paths": [row]},
            policy=self.policy,
            as_of=self.as_of,
            vendor_snapshots=self.snapshots,
            policy_pin=self.policy_pin,
        )
        if pending or len(routes.get(row["target_identity"], [])) != 1:
            raise ValueError("original independent adapter rejects source segment")
        if row["basis"] in {
            RouteBasis.PLANNER_SOLVED.value,
            RouteBasis.PLANNER_VENDOR_CLOSED.value,
        }:
            old._original_leaves(old.pointer(row["receipt"]), row["root"], row["receipt"])
        self.segment_cache[key] = row["root"]
        return row["root"]

    def reaction_node(self, source):
        """Retain one entire authenticated reaction; never select a reactant subset."""
        receipt = source["original_tree"]
        self.require_saved_tree(receipt, source["original_tree_target"])
        key = (location(receipt), receipt["sha256"])
        if key not in self.raw_tree_cache:
            raw = old.pointer(receipt)
            normalized = normalized_original(raw, receipt)
            old._original_leaves(raw, normalized, receipt)
            self.raw_tree_cache[key] = raw, normalized
        _, whole = self.raw_tree_cache[key]
        if source["original_tree_target"] != whole["identity"]:
            raise ValueError("reaction-node original whole-tree target changed")
        original = at(whole, source["normalized_node_pointer"])
        if original["step"] is None or not same_pin(
            original["step"]["receipt"], source["reaction_node"]
        ):
            raise ValueError("reaction-node pointer is not the exact original reaction")
        old.receipt(source["original_tree_auditor"])
        frontier = copy.deepcopy(original)
        frontier["step"]["reactants"] = [
            {"identity": child["identity"], "listings": [], "step": None}
            for child in original["step"]["reactants"]
        ]
        return frontier

    def composition(self, row):
        if (
            row["basis"] != BASIS
            or row["planner_solved"] is not None
            or row["status"] != "exact_saved_step_composition_to_current_listed_terminals"
        ):
            raise ValueError("composition must preserve distinct derived basis and no solved claim")
        proof = old.pointer(row["receipt"])
        if (
            proof["schema"] != "forge.authenticated_saved_route_composition_candidate.v1"
            or proof["candidate_only"] is not True
            or proof["target_identity"] != row["target_identity"]
            or proof["root"] != row["root"]
            or proof["as_of_utc"] != FROZEN_TIME
            or not same_pin(proof["policy"], self.policy_pin)
            or not same_pin(proof["original_assessment"], self.result_pin)
            or proof["axes"]
            != {"experimental_execution": None, "stock": None, "independent_forward_replay": None}
        ):
            raise ValueError("derived proof identity/time/authority/assessment mismatch")
        segments = {segment["pointer"]: segment for segment in proof["segments"]}
        if len(segments) != len(proof["segments"]) or "" not in segments:
            raise ValueError("unique root and graft segment pointers required")
        consumed, terminals = set(), []

        def segment(pointer, actual, ancestors):
            declaration = segments[pointer]
            consumed.add(pointer)
            source = declaration["source"]
            if source["kind"] == "admitted_original_path":
                original = self.original_segment(source)
            elif source["kind"] == "new_exact_prefix_candidate":
                original = self.prefix(source["prefix_candidate"])
            elif source["kind"] == "authenticated_original_reaction_node":
                original = self.reaction_node(source)
            else:
                raise ValueError("unsupported source segment kind")
            if (
                declaration["target_identity"] != actual["identity"]
                or original["identity"] != actual["identity"]
            ):
                raise ValueError("graft source target identity changed")
            compare(original, actual, pointer, ancestors)

        def compare(source, actual, pointer, ancestors):
            identity = actual["identity"]
            if (
                set(actual) != {"identity", "listings", "step"}
                or identity != source["identity"]
                or identity in ancestors
                or actual.get("listings")
            ):
                raise ValueError("changed identity, ancestor repeat or injected listing")
            if source["step"] is None:
                if actual["step"] is None:
                    self.listed_terminal(identity, proof["vendor_snapshot"])
                    terminals.append(identity)
                elif pointer in segments and pointer not in consumed:
                    segment(pointer, actual, ancestors)
                else:
                    raise ValueError("new reaction at original leaf lacks exact graft source")
                return
            require_same_step(source, actual)
            for index, (left, right) in enumerate(
                zip(source["step"]["reactants"], actual["step"]["reactants"], strict=True)
            ):
                compare(left, right, pointer + f"/step/reactants/{index}", ancestors | {identity})

        segment("", row["root"], frozenset())
        if consumed != set(segments) or terminals != proof["terminal_identities"]:
            raise ValueError("unused graft or incomplete terminal occurrence ledger")
        return {
            "route_id": row["route_id"],
            "target_identity": row["target_identity"],
            "segments": len(consumed),
            "terminal_occurrences": len(terminals),
            "proof": row["receipt"],
        }


def validated_routes(envelope, *, policy, as_of, vendor_snapshots, policy_pin):
    """Adapter for independently admitted derived envelopes; original adapters stay intact."""
    if envelope.get("schema") != SCHEMA:
        raise ValueError("unrecognized admitted composition envelope")
    admission = read_pin(envelope["independent_admission"])
    if (
        admission["passed"] is not True
        or content_sha(envelope["paths"]) != admission["paths_sha256"]
    ):
        raise ValueError("envelope differs from independent complete-tree admission")
    if pin(Path(__file__)) != admission["validator"]:
        raise ValueError("composition validator source changed after admission")
    context = AdmissionContext(
        policy=policy, as_of=as_of, vendor_snapshots=vendor_snapshots, policy_pin=policy_pin
    )
    result = defaultdict(list)
    for row in envelope["paths"]:
        context.composition(row)
        # This classifier basis requires all exact terminal listings. The receipt
        # retains the distinct derived-composition origin; original flags are untouched.
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
