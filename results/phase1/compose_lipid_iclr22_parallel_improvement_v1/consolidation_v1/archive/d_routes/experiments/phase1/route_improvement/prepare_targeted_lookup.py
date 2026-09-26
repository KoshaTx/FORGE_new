"""Freeze the two public identifiers and exclude prior routing-workflow query targets."""

from pathlib import Path

from rdkit import Chem

from audit_new_leaves import BULK, MAIN, OUT, PAR, pin, read, write


def main():
    proposal = read(OUT / "next_two_listing_proposal.json")
    targets = {r["identity"]: r["inchikey"] for r in proposal["targets"]}
    assert len(targets) == 2
    for identity, key in targets.items():
        mol = Chem.MolFromSmiles(identity)
        assert Chem.MolToSmiles(mol) == identity and Chem.MolToInchiKey(mol) == key
    old = PAR / "routes/next_bottlenecks_v3/new_leaf_listing_diagnostic_v1"
    old_inventory = read(BULK / "target_inventory_v1.json")
    prior_targets = {r["identity"] for r in old_inventory["identities"]}
    previous_protocol = read(old / "lookup_run_v1/protocol.json")
    prior_targets.update(previous_protocol["target_keys"])
    assert not set(targets) & prior_targets
    # Search already downloaded exact-target observation bindings, excluding compressed bulk maps.
    cache_root = (
        MAIN
        / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/computational_makeability_v1/vendors"
    )
    scanned, matches = [], []
    for path in sorted(cache_root.glob("**/observations/*.json")):
        row = read(path)
        scanned.append(pin(path))
        identities = {
            row.get("identity"),
            row.get("canonical_smiles"),
            row.get("observation", {}).get("canonical_smiles"),
        }
        if identities & set(targets):
            matches.append({"source": pin(path), "identities": sorted(identities & set(targets))})
    assert not matches, "Prior exact observation exists; do not duplicate lookup"
    old_proposal = read(old / "lookup_proposal_v2.json")
    write(
        OUT / "prior_targeted_lookup_check.json",
        {
            "passed": True,
            "producer": pin(Path(__file__)),
            "inputs": [
                pin(BULK / "target_inventory_v1.json"),
                pin(old / "lookup_run_v1/protocol.json"),
                pin(OUT / "next_two_listing_proposal.json"),
            ],
            "earlier_union_identities": len(old_inventory["identities"]),
            "earlier_targeted_protocol_targets": previous_protocol["target_keys"],
            "cached_observation_sources": scanned,
            "duplicate_targets": matches,
            "new_HTTP_calls": 0,
        },
    )
    write(
        OUT / "lookup_proposal_v3.json",
        {
            "status": "frozen_two_identity_lookup",
            "inputs": [
                pin(OUT / "next_two_listing_proposal.json"),
                pin(OUT / "prior_targeted_lookup_check.json"),
                pin(old / "lookup_proposal_v2.json"),
            ],
            "transmitted_identifiers": targets,
            "property_requests": [
                {
                    "identity": identity,
                    "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                    + key
                    + "/property/SMILES,InChIKey/JSON",
                }
                for identity, key in targets.items()
            ],
            "identity_policy": old_proposal["identity_policy"],
            "as_of_policy": old_proposal["as_of_policy"],
            "maximum_HTTP_requests": 8,
            "maximum_wall_seconds": 180,
            "automatic_retries": 0,
            "priority": ["recover_B258", "disulfide747"],
            "root_authorization": "Explicit root authorization for onlyC=CCOC andOCCCCCCl under existinguserapproval;8requests and180wallseconds are insideauthorized16requestcap.",
        },
    )


if __name__ == "__main__":
    main()
