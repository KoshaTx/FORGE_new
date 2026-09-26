"""Complete exact-leaf audit of the one repaired-head search, without new vendor calls."""

import resource
import time
from pathlib import Path

from audit_new_leaves import MAIN, OUT, leaves, pin, read, wrapper, write
from candidate_routes import BASE, context, prior, snapshot_view
from rdkit import Chem

SEARCH = OUT / "b258_search_v1"


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    start = time.process_time()
    protocol = read(SEARCH / "protocol.json")
    completion = read(SEARCH / "completion.json")
    assert completion["all_completed"] and completion["declared_target_count"] == 1
    assert completion["protocol"] == pin(SEARCH / "protocol.json")
    for ref in protocol["inputs"]:
        assert pin(MAIN / ref["path"]) == ref
    result = prior.read_pin(completion["result"])
    declared = protocol["panel"][0]
    assert result["seed"] == declared["seed"] and result["canonical_smiles"] == declared["identity"]
    assert prior.same_pin(result["protocol"], pin(SEARCH / "protocol.json"))
    assert result["verified_assets"] == protocol["asset_pins"][:-1]
    assert len(result["top_route_hypotheses"]) == 3
    ctx = context()
    previous = read(BASE / "protocol.json")
    now = wrapper.old.time(previous["as_of_utc"])
    listing_rows = snapshot_view()["listings"]
    decisions = []
    for ordinal, raw in enumerate(result["top_route_hypotheses"]):
        ref = {
            "location": completion["result"]["path"] + f"#/top_route_hypotheses/{ordinal}",
            "sha256": completion["result"]["sha256"],
        }
        normalized = prior.normalized_original(raw, ref)
        wrapper.old._original_leaves(raw, normalized, ref)
        assert normalized["identity"] == declared["identity"]
        terminals = []
        for identity in leaves(normalized):
            positive = []
            for row in listing_rows:
                if row["identity"] != identity:
                    continue
                verdict = wrapper.old.classify_listing(
                    identity, wrapper.typed_listing(row), policy=ctx.policy, as_of=now
                )
                if verdict.state is wrapper.old.ListingState.LISTED:
                    ctx.positive(
                        identity, row, pin(OUT / "candidate_compositions_v1/listing_view.json")
                    )
                    positive.append(row)
            terminals.append(
                {
                    "identity": identity,
                    "current_listed": bool(positive),
                    "current_exact_witnesses": positive,
                }
            )
        decisions.append(
            {
                "ordinal": ordinal,
                "receipt": ref,
                "historical_stock_solved": raw["metadata"]["is_solved"],
                "all_exact_current_leaves_listed": all(row["current_listed"] for row in terminals),
                "leaves": terminals,
                "normalized_original": normalized,
            }
        )
    minimal = min(
        decisions,
        key=lambda row: (sum(not leaf["current_listed"] for leaf in row["leaves"]), row["ordinal"]),
    )
    missing = [leaf["identity"] for leaf in minimal["leaves"] if not leaf["current_listed"]]
    assert missing == ["C=CCOC"]
    bulk = (
        MAIN
        / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/computational_makeability_v1/routes/pubchem_bulk_qualification_v1/target_inventory_v1.json"
    )
    old_inventory = {row["identity"] for row in read(bulk)["identities"]}
    body = {
        "schema": "forge.stream_d.B258_exact_leaf_audit.v1",
        "inputs": [
            pin(Path(__file__)),
            pin(SEARCH / "protocol.json"),
            pin(SEARCH / "completion.json"),
            completion["result"],
            pin(OUT / "candidate_compositions_v1/listing_view.json"),
            pin(bulk),
        ],
        "target": declared["identity"],
        "request": 258,
        "as_of_utc": previous["as_of_utc"],
        "retained_tree_count": 3,
        "closed_trees": sum(row["all_exact_current_leaves_listed"] for row in decisions),
        "decisions": decisions,
        "minimum_unknown_identity_set": missing,
        "new_minimum_query": [
            {
                "identity": identity,
                "inchikey": Chem.MolToInchiKey(Chem.MolFromSmiles(identity)),
                "in_prior_6322_census": identity in old_inventory,
            }
            for identity in missing
        ],
        "new_vendor_queries": 0,
        "new_measured_product_successes": 0,
        "CPU_seconds": time.process_time() - start,
    }
    write(SEARCH / "tree_audit_v1.json", body)
    write(
        OUT / "next_two_listing_proposal.json",
        {
            "schema": "forge.stream_d.minimum_two_lookup_proposal.v1",
            "inputs": [
                pin(SEARCH / "tree_audit_v1.json"),
                pin(OUT / "leaf_search_v1/next_listing_proposal.json"),
            ],
            "status": "proposal_only_not_executed",
            "targets": body["new_minimum_query"]
            + [
                {
                    "identity": "OCCCCCCl",
                    "inchikey": "DCBJCKDOZLTTDW-UHFFFAOYSA-N",
                    "in_prior_6322_census": False,
                }
            ],
            "conditional_effects": {
                "C=CCOC": "Restore B258 makeability if exact listing admitted and intact current-head tree accepted; no molecular change",
                "OCCCCCCl": "Potentially close disulfide747 by exact saved-tree graft; new gain",
            },
            "maximum_HTTP_requests": 8,
            "maximum_response_bytes_each": 1048576,
            "request_timeout_seconds": 35,
            "maximum_wall_seconds": 300,
            "automatic_retries": 0,
            "identity_and_listing_gates": "Existing exact returned SMILES/CID match, charge/isotope/salt/tautomer preservation, positive Chemical Vendors category; listing is not stock",
            "clock_rule": "True observed timestamps; separately qualified common as_of >=lastobservation for both paired arms; unchanged30day listing expiry",
            "failure_rule": "Stop429/503; preserve errors/unknowns and censored multipleCID candidate sets",
            "independent_raw_listing_and_tree_admission_required": True,
            "new_HTTP_calls": 0,
        },
    )
    print(pin(SEARCH / "tree_audit_v1.json"))


if __name__ == "__main__":
    run()
