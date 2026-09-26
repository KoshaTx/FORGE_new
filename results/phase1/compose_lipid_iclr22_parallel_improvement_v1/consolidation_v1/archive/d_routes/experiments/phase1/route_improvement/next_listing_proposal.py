"""Minimum new exact listing query set from the completed fixed leaf-search panel."""

from pathlib import Path

from audit_new_leaves import SEARCH, pin, read, write


def main():
    audit = read(SEARCH / "tree_audit_v1.json")
    route = next(r for r in audit["trees"] if r["target_index"] == 2 and r["route_ordinal"] == 2)
    missing = [r["identity"] for r in route["leaves"] if not r["current_listed"]]
    assert missing == ["OCCCCCCl"]
    identity = next(r for r in audit["unknown_leaves"] if r["identity"] == missing[0])
    assert not identity["in_prior_6322_census"] and identity["prior_bulk_outcome"] is None
    key = identity["inchikey"]
    write(
        SEARCH / "next_listing_proposal.json",
        {
            "schema": "forge.stream_d.minimum_exact_listing_proposal.v1",
            "inputs": [pin(Path(__file__)), pin(SEARCH / "tree_audit_v1.json")],
            "status": "proposal_only_not_executed",
            "identity": identity["identity"],
            "transmitted_identifier": key,
            "property_request": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            + key
            + "/property/SMILES,InChIKey/JSON",
            "category_request_template": "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{exact_matched_CID}/JSON",
            "current_missing_listing_state": "unknown",
            "prior_6322_census_member": False,
            "known_remaining_leaves": [
                r["identity"] for r in route["leaves"] if r["current_listed"]
            ],
            "exact_original_route": route["receipt"],
            "conditional_whole_product_gain_indices": [747],
            "measured_new_products": None,
            "maximum_HTTP_requests": 4,
            "maximum_category_candidates": 3,
            "maximum_response_bytes_each": 1048576,
            "maximum_wall_seconds": 300,
            "request_timeout_seconds": 35,
            "automatic_retries": 0,
            "failure_policy": "Stop on429/503; retain every error and unknown. More than3 CID candidates is censored, not subset-selected.",
            "identity_policy": "Existing exact whole constitutional graph gate; preserve salts, isotope, formal charge and tautomer. InChIKey discovery requires returned exact SMILES/CID match plus exact Chemical Vendors category witness. A listing is not stock.",
            "clock_policy": "Preserve true observation timestamps; any recount must use a new common as_of no earlier than the observation and reclassify old observations under unchanged30day policy.",
            "admission_requirement": "Parent-reviewed bounded lookup, independent raw listing and exact saved-tree graft admission, then full1408 real evaluator. No immediate metric promotion.",
            "new_HTTP_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
