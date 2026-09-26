"""Authenticate saved source bodies before constructing pure-classifier inputs."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime

from common_v2 import OUT, ROOT, authenticate, canonical, content_sha, pin, read_pin
from computational_makeability_v2 import (
    ComputationalRoute,
    EvidenceAxes,
    ListingState,
    LookupOutcome,
    MakeabilityPolicy,
    Receipt,
    RouteBasis,
    RouteNode,
    RouteStep,
    VendorListing,
    classify_listing,
)
from rdkit import Chem
from source_relocation import resolve


def time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def receipt(value):
    location = value.get("location", value.get("path"))
    authenticate(location, value["sha256"])
    return Receipt(location, value["sha256"])


def pointer(value):
    data = read_pin(value)
    location = value.get("location", value.get("path"))
    if "#" not in location:
        return data
    pointer = location.split("#", 1)[1]
    if not pointer.startswith("/"):
        raise ValueError("expected JSON pointer")
    for key in pointer[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


DOCUMENTARY = OUT.parent / "routes/planner_vendor_closed_v1"
sys.path.insert(0, str(DOCUMENTARY))
from documentary_stream_v1 import DocumentaryAuthenticator  # noqa: E402

DOCUMENTARY_AUTH = DocumentaryAuthenticator(ROOT, [])


def configure_documentary(admission_pins, vendor_snapshots):
    """Only an explicit independently admitted listing envelope can enable raw-file access."""
    global DOCUMENTARY_AUTH
    allowed = {}
    for admission_pin in admission_pins:
        admission = read_pin(admission_pin)
        if admission.get("passed") is not True:
            raise ValueError("independent bulk admission required")
        if not any(_same_pin(admission["listings"], item) for item in vendor_snapshots):
            raise ValueError("admitted bulk envelope is not an explicit listing input")
        envelope = read_pin(admission["listings"])
        if envelope.get("schema") != "forge.vendor_listings.v1":
            raise ValueError("unrecognized admitted listing envelope")
        sources = admission["allowlisted_sources"]
        attestations = admission["compressed_source_attestations"]
        if len(sources) != 3 or len(attestations) != 3:
            raise ValueError("all three admitted complete source descriptors required")
        for source in sources:
            matched = [item for item in attestations if _same_pin(item, source)]
            if (
                len(matched) != 1
                or matched[0]["bytes"] != source["bytes"]
                or source.get("complete") is not True
            ):
                raise ValueError("raw-source descriptor differs from independent attestation")
            key = _location(source)
            if key in allowed and allowed[key] != source:
                raise ValueError("conflicting admitted raw-source descriptors")
            allowed[key] = source
    DOCUMENTARY_AUTH = DocumentaryAuthenticator(ROOT, list(allowed.values()))


def embedded_pins(value):
    """Keep ordinary source-pin traversal; deepen only explicit vendor evidence bodies."""
    if isinstance(value, dict):
        if value.get("schema") == "forge.vendor_listing_binding.v1":
            DOCUMENTARY_AUTH.authenticate(value)
            return
        location = value.get("path", value.get("location"))
        if isinstance(location, str) and isinstance(value.get("sha256"), str):
            if not location.startswith(("https://", "http://")):
                DOCUMENTARY_AUTH.verify(value)
        for item in value.values():
            embedded_pins(item)
    elif isinstance(value, list):
        for item in value:
            embedded_pins(item)


def strict_listings(manifest):
    """One observed supplier counts as a conservative lower bound of one vendor."""
    listings = defaultdict(list)
    rows = []
    for entry in manifest["entries"]:
        record = read_pin(entry["record"])
        embedded_pins(record)
        observation = record["observation"]
        identity = canonical(observation["canonical_smiles"])
        assert identity == canonical(entry["declared_target"]["canonical_smiles"])
        assert record["admission"]["admitted"] is True
        assert observation["identity_and_required_chemical_form_qualified"] is True
        assert observation.get("primary_page_fields") is True
        assert (
            record.get("url")
            or record.get("source_url")
            or observation.get("url")
            or observation.get("source_url")
        )
        observed = time(record["accessed_at_utc"])
        listing = VendorListing(
            identity, LookupOutcome.SUCCESS, 1, observed, receipt(entry["record"])
        )
        listings[identity].append(listing)
        rows.append(
            {
                "identity": identity,
                "outcome": "success",
                "vendor_count": 1,
                "vendor_count_semantics": "one individually authenticated supplier, not an exhaustive vendor census",
                "observed_at": observed.isoformat(),
                "receipt": entry["record"],
                "strict_expires_at_utc": record["expires_at_utc"],
            }
        )
    return listings, rows


def _strict_node(node, body_pin, location):
    target = canonical(node["target"]["canonical_smiles"])
    step = node["step"]
    children = node["children"]
    if step is None:
        if children:
            raise ValueError("saved terminal with children")
        return RouteNode(target)
    if [child["target"] for child in children] != step["reactants"]:
        raise ValueError("saved exact-route children do not match declared reactants")
    exact = [
        item
        for item in step["evidence"]
        if item["tier"] == "exact_source"
        and item["exact_substrate"] is True
        and item["forward_verification"] == "verified_exact_product_unique"
    ]
    if not exact or step["forward_product_count"] != 1:
        raise ValueError("missing recorded exact forward application")
    sources = [resolve(item["source_locator"], item["source_sha256"]) for item in exact]
    # The resolver is frozen for canonical-classifier consumers. Transport only
    # its authenticated scalar fields into this versioned local dataclass.
    transform = Receipt(sources[0].location, sources[0].sha256)
    child_nodes = tuple(
        _strict_node(child, body_pin, f"{location}/children/{i}")
        for i, child in enumerate(children)
    )
    return RouteNode(
        target,
        RouteStep(
            child_nodes,
            Receipt(body_pin["path"] + "#" + location + "/step", body_pin["sha256"]),
            (target,),
            transform,
            EvidenceAxes(
                exact_source_execution=True,
                independent_forward_replay=True,
                unique_forward_product=True,
            ),
        ),
    )


def strict_routes(family_files):
    """Retain actual complete step trees, even when their terminal evidence is missing."""
    routes = defaultdict(list)
    origins = defaultdict(list)
    exclusions = []
    seen = set()
    for body_pin in family_files:
        body = read_pin(body_pin)
        for row_number, product in enumerate(body["products"]):
            if not product["exact_L1"]:
                continue
            for branch_number, branch in enumerate(product["branches"]):
                for alt_number, alt in enumerate(branch["all_alternatives"]):
                    tree = alt["after"]["route_tree"]
                    if tree["step"] is None:
                        continue
                    location = f"/products/{row_number}/branches/{branch_number}/all_alternatives/{alt_number}/after/route_tree"
                    target = canonical(branch["requirement"]["canonical_smiles"])
                    if canonical(tree["target"]["canonical_smiles"]) != target:
                        raise ValueError("saved exact route root identity mismatch")
                    try:
                        node = _strict_node(tree, body_pin, location)
                    except (ValueError, FileNotFoundError) as exc:
                        exclusions.append(
                            {
                                "index": product["index"],
                                "branch": branch_number,
                                "alternative": alt_number,
                                "reason": str(exc),
                            }
                        )
                        continue

                    def shape(node):
                        return [
                            node.identity,
                            [shape(c) for c in node.step.reactants] if node.step else None,
                        ]

                    key = (target, content_sha(shape(node)))
                    origins[key].append(
                        {
                            "index": product["index"],
                            "family": product["family"],
                            "branch": branch_number,
                            "alternative": alt_number,
                            "receipt": {
                                "location": body_pin["path"] + "#" + location,
                                "sha256": body_pin["sha256"],
                            },
                        }
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    routes[target].append(
                        ComputationalRoute(
                            "v5:" + key[1],
                            RouteBasis.FORWARD_APPLIED,
                            node,
                            Receipt(body_pin["path"] + "#" + location, body_pin["sha256"]),
                        )
                    )
    return routes, list(origins.values()), exclusions


def _planner_shape(node):
    if node.get("is_chemical") is not True:
        raise ValueError("planner root/child is not a molecule node")
    reactions = node.get("children", [])
    if not reactions:
        return [canonical(node["smiles"]), None]
    if len(reactions) != 1 or reactions[0].get("is_reaction") is not True:
        raise ValueError("planner path contains alternative reaction branches")
    return [
        canonical(node["smiles"]),
        [_planner_shape(child) for child in reactions[0]["children"]],
    ]


def _normalized_shape(node):
    return [
        canonical(node["identity"]),
        [_normalized_shape(c) for c in node["step"]["reactants"]] if node["step"] else None,
    ]


def _location(value):
    return value.get("location", value.get("path"))


def _same_pin(first, second):
    return _location(first) == _location(second) and first["sha256"] == second["sha256"]


def _original_leaves(original, normalized, original_receipt):
    """Authenticate the entire saved mapped tree, retaining every original leaf occurrence."""
    leaves = []
    steps = 0

    def walk(raw, node, location, ancestors):
        nonlocal steps
        if (
            raw.get("is_chemical") is not True
            or raw.get("type") != "mol"
            or raw.get("hide") is not False
            or type(raw.get("in_stock")) is not bool
        ):
            raise ValueError("original path requires molecule nodes")
        identity = canonical(raw["smiles"])
        if identity != canonical(node["identity"]) or identity in ancestors:
            raise ValueError("changed identity or cyclic original route")
        if node.get("listings"):
            raise ValueError("normalized path may not inject listing observations")
        children = raw.get("children", [])
        if not children:
            if node["step"] is not None:
                raise ValueError("original terminal was expanded")
            leaves.append(
                {"pointer": location, "identity": identity, "historical_in_stock": raw["in_stock"]}
            )
            return
        if len(children) != 1 or children[0].get("is_reaction") is not True:
            raise ValueError("original path contains alternative reaction branches")
        reaction = children[0]
        if reaction.get("type") != "reaction" or reaction.get("hide") is not False:
            raise ValueError("hidden or malformed original reaction")
        metadata = reaction.get("metadata", {})
        if (
            not isinstance(metadata.get("template"), str)
            or not metadata["template"]
            or not isinstance(metadata.get("template_hash"), str)
            or len(metadata["template_hash"]) != 64
        ):
            raise ValueError("missing original template metadata")
        reactants = reaction.get("children", [])
        if not reactants or node["step"] is None:
            raise ValueError("reaction step or required reactants omitted")
        step = node["step"]
        if len(step["reactants"]) != len(reactants):
            raise ValueError("normalized child count changed")
        expected = location + "/children/0"
        if (
            _location(step["receipt"]) != expected
            or step["receipt"]["sha256"] != original_receipt["sha256"]
        ):
            raise ValueError("step receipt is not the original exact node")
        if pointer(step["receipt"]) != reaction:
            raise ValueError("step receipt body mismatch")
        if (
            step["forward_products"] is not None
            or step["transform_receipt"] is not None
            or any(value is not None for value in step["axes"].values())
        ):
            raise ValueError("vendor closure cannot invent independent chemical evidence")
        mapped = reaction.get("metadata", {}).get("mapped_reaction_smiles")
        if not isinstance(mapped, str) or mapped.count(">>") != 1:
            raise ValueError("missing original mapped reaction")
        product, precursor = mapped.split(">>")
        if canonical(product) != identity or canonical(precursor) != canonical(
            ".".join(child["smiles"] for child in reactants)
        ):
            raise ValueError("mapped reaction does not bind original target and reactants")
        # Exact structure/mapping bookkeeping, not a forward or condition verifier.
        sides = []
        for smiles in (product, precursor):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("invalid mapped reaction side")
            mapped_atoms = [
                (a.GetAtomMapNum(), a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms()
            ]
            if any(number <= 0 for number, _, _ in mapped_atoms) or len(
                {a[0] for a in mapped_atoms}
            ) != len(mapped_atoms):
                raise ValueError("missing or duplicated atom maps")
            sides.append({number: (element, isotope) for number, element, isotope in mapped_atoms})
        if any(
            sides[0][number] != sides[1][number] for number in sides[0].keys() & sides[1].keys()
        ):
            raise ValueError("shared atom map changes element or isotope")
        steps += 1
        for ordinal, (child, normalized_child) in enumerate(zip(reactants, step["reactants"])):
            walk(child, normalized_child, expected + f"/children/{ordinal}", ancestors | {identity})

    walk(original, normalized, _location(original_receipt), frozenset())
    if steps == 0 or not leaves:
        raise ValueError("vendor closure requires a nonzero complete original tree")
    return leaves


def _listing_key(row):
    return tuple(row[key] for key in ("identity", "outcome", "vendor_count", "observed_at")) + (
        _location(row["receipt"]),
        row["receipt"]["sha256"],
    )


def _bound_listing(row, originals):
    source = read_pin(row["receipt"])
    embedded_pins(source)
    if _listing_key(row) in originals:
        if originals[_listing_key(row)] != row:
            raise ValueError("original listing row changed")
        assert source["observation"]["canonical_smiles"] == row["identity"]
        assert source["accessed_at_utc"].replace("Z", "+00:00") == row["observed_at"]
        assert source["admission"]["admitted"] is True and row["vendor_count"] == 1
    else:
        if (
            source.get("schema") != "forge.vendor_listing_binding.v1"
            or source.get("exact_form_verified") is not True
        ):
            raise ValueError("positive listing binding lacks exact form qualification")
        for key in ("identity", "outcome", "vendor_count", "observed_at"):
            if source[key] != row[key]:
                raise ValueError(f"listing scalar/source mismatch: {key}")
    return VendorListing(
        row["identity"],
        LookupOutcome(row["outcome"]),
        row["vendor_count"],
        time(row["observed_at"]) if row["observed_at"] else None,
        receipt(row["receipt"]),
    )


def validate_vendor_closure(row, actual, *, policy, as_of, vendor_snapshots, policy_pin):
    if not isinstance(policy, MakeabilityPolicy) or as_of is None:
        raise ValueError("vendor closure requires explicit evaluation policy and time")
    if (
        row.get("planner_solved") is not False
        or actual.get("metadata", {}).get("is_solved") is not False
    ):
        raise ValueError("historical unsolved flag must remain false")
    if (
        row.get("status") != "current_vendor_closed"
        or row.get("source_kind") != "planner_current_vendor_leaf_closure"
    ):
        raise ValueError("missing typed current closure status")
    proof = read_pin(row["completion_receipt"])
    embedded_pins(proof)
    if proof["schema"] != "forge.planner_current_vendor_closure_proof.v1":
        raise ValueError("unrecognized vendor closure proof")
    if (
        proof["original_path_receipt"] != row["receipt"]
        or proof["original_planner_solved"] is not False
    ):
        raise ValueError("proof is not bound to the original unsolved path")
    if proof["target_identity"] != canonical(row["target_identity"]) or not _same_pin(
        proof["policy"], policy_pin
    ):
        raise ValueError("proof target/policy mismatch")
    proof_time = time(proof["as_of_utc"])
    if proof_time > as_of:
        raise ValueError("future completion proof")
    snapshot = proof["vendor_snapshot"]
    if not any(_same_pin(snapshot, supplied) for supplied in vendor_snapshots):
        raise ValueError("proof snapshot is not an explicit evaluation input")
    envelope = read_pin(snapshot)
    snapshot_rows = {_listing_key(item): item for item in envelope["listings"]}
    original_rows = read_pin(pin(OUT / "baseline_vendor_listings.json"))["listings"]
    originals = {_listing_key(item): item for item in original_rows}
    actual_leaves = _original_leaves(actual, row["root"], row["receipt"])
    supplied = proof["leaf_observations"]
    if len(supplied) != len(actual_leaves):
        raise ValueError("proof omitted or duplicated an original leaf occurrence")
    for leaf, observation in zip(actual_leaves, supplied):
        if any(
            observation[key] != leaf[key] for key in ("pointer", "identity", "historical_in_stock")
        ):
            raise ValueError("proof leaf identity/location mismatch")
        listing_row = observation["listing"]
        if snapshot_rows.get(_listing_key(listing_row)) != listing_row:
            raise ValueError("proof listing not authenticated to declared snapshot")
        listing = _bound_listing(listing_row, originals)
        for when in (proof_time, as_of):
            if (
                classify_listing(leaf["identity"], listing, policy=policy, as_of=when).state
                is not ListingState.LISTED
            ):
                raise ValueError("original terminal lacks dated current exact positive listing")
    return proof


def _external_node(node, basis, *, independently_bind):
    identity = canonical(node["identity"])
    if node["step"] is None:
        return RouteNode(identity)
    step = node["step"]
    step_source = pointer(step["receipt"])
    products = step["forward_products"]
    if independently_bind and basis is RouteBasis.FORWARD_APPLIED:
        actual_products = step_source.get("products", step_source.get("forward_products"))
        actual_reactants = step_source.get("role_ordered_reactants", step_source.get("reactants"))
        assert [canonical(x) for x in actual_products] == [canonical(x) for x in products]
        assert [canonical(x) for x in actual_reactants] == [
            canonical(x["identity"]) for x in step["reactants"]
        ]
        if "transform_receipt" in step_source:
            assert receipt(step_source["transform_receipt"]) == receipt(step["transform_receipt"])
    return RouteNode(
        identity,
        RouteStep(
            tuple(
                _external_node(c, basis, independently_bind=independently_bind)
                for c in step["reactants"]
            ),
            receipt(step["receipt"]),
            tuple(canonical(x) for x in products) if products is not None else None,
            receipt(step["transform_receipt"]) if step["transform_receipt"] else None,
            EvidenceAxes(**step["axes"]),
        ),
    )


def external_routes(envelope, *, policy=None, as_of=None, vendor_snapshots=(), policy_pin=None):
    routes = defaultdict(list)
    pending = []
    for row in envelope["paths"]:
        embedded_pins(row)
        target = canonical(row["target_identity"])
        basis = RouteBasis(row["basis"])
        actual = pointer(row["receipt"])
        if basis is RouteBasis.PLANNER_VENDOR_CLOSED:
            validate_vendor_closure(
                row,
                actual,
                policy=policy,
                as_of=as_of,
                vendor_snapshots=vendor_snapshots,
                policy_pin=policy_pin,
            )
        elif basis is RouteBasis.PLANNER_SOLVED:
            if row["planner_solved"] is not True:
                pending.append(
                    {
                        "target_identity": target,
                        "route_id": row["route_id"],
                        "status": row["status"],
                        "reason": "planner_path_not_solved",
                    }
                )
                continue
            if actual.get("metadata", {}).get("is_solved") is not True:
                raise ValueError("planner solved flag not bound to original path")
            if _planner_shape(actual) != _normalized_shape(row["root"]):
                raise ValueError("normalized planner tree changed original path")
        else:
            if row["status"] != "target_in_recorded_forward_products":
                pending.append(
                    {
                        "target_identity": target,
                        "route_id": row["route_id"],
                        "status": row["status"],
                    }
                )
                continue
        node = _external_node(row["root"], basis, independently_bind=True)
        assert node.identity == target
        routes[target].append(
            ComputationalRoute(
                row["route_id"],
                basis,
                node,
                receipt(
                    row["completion_receipt"]
                    if basis is RouteBasis.PLANNER_VENDOR_CLOSED
                    else row["receipt"]
                ),
            )
        )
    return routes, pending


def add_listings(node, listings):
    """One path retains its own graph; exact-identity observations may be reused."""
    if node.step is None:
        return RouteNode(node.identity, listings=tuple(listings.get(node.identity, ())))
    step = node.step
    return RouteNode(
        node.identity,
        RouteStep(
            tuple(add_listings(c, listings) for c in step.reactants),
            step.receipt,
            step.forward_products,
            step.transform_receipt,
            step.axes,
        ),
    )
