"""Read exact saved graph forms and list every current terminal witness; no new paths."""

import json
from pathlib import Path

import residual_saved_followup_v2 as previous
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from residual_debt import DEST, MAIN, pin, read, write


def run():
    protocol = read(DEST / "followup_protocol_v2.json")
    source_ref = protocol["inputs"]["old_target_result"]
    source = read(MAIN / source_ref["path"])
    assert pin(MAIN / source_ref["path"]) == source_ref
    listing_rows = []
    input_refs = [
        source_ref,
        protocol["inputs"]["policy"],
        pin(DEST / "independent_review_v2.json"),
        pin(Path(__file__)),
        pin(Path(rdBase.__file__)),
    ]
    for key, ref in protocol["inputs"].items():
        if key.startswith("listing_envelope_"):
            path = MAIN / ref["path"]
            assert pin(path) == ref
            input_refs.append(ref)
            for index, row in enumerate(read(path)["listings"]):
                listing_rows.append(
                    (
                        row,
                        {"location": ref["path"] + f"#/listings/{index}", "sha256": ref["sha256"]},
                    )
                )
    policy = previous.MakeabilityPolicy.from_mapping(
        read(MAIN / protocol["inputs"]["policy"]["path"])
    )
    as_of = previous.datetime.fromisoformat(protocol["clock"])
    records = []

    def record(node, pointer, route):
        children = node.get("children", [])
        if children:
            for index, child in enumerate(children):
                record(child, pointer + f"/children/{index}", route)
            return
        assert node["type"] == "mol"
        raw = node["smiles"]
        mol = Chem.MolFromSmiles(raw)
        assert mol is not None
        canonical = previous.canonical(raw)
        witnesses = []
        for row, ref in listing_rows:
            if row["identity"] != canonical:
                continue
            receipt = row["receipt"]
            typed = previous.VendorListing(
                row["identity"],
                previous.LookupOutcome(row["outcome"]),
                row["vendor_count"],
                previous.datetime.fromisoformat(row["observed_at"]) if row["observed_at"] else None,
                previous.Receipt(receipt.get("path", receipt.get("location")), receipt["sha256"]),
            )
            witnesses.append(
                {
                    "snapshot_row": ref,
                    "observation": row,
                    "current_state": previous.classify_listing(
                        canonical, typed, policy=policy, as_of=as_of
                    ).state.value,
                }
            )
        graph = {
            "canonical_smiles": canonical,
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "atoms": [
                {
                    "index": atom.GetIdx(),
                    "atomic_number": atom.GetAtomicNum(),
                    "formal_charge": atom.GetFormalCharge(),
                    "isotope": atom.GetIsotope(),
                    "degree": atom.GetDegree(),
                    "implicit_hydrogens": atom.GetNumImplicitHs(),
                    "explicit_hydrogens": atom.GetNumExplicitHs(),
                    "radical_electrons": atom.GetNumRadicalElectrons(),
                }
                for atom in mol.GetAtoms()
            ],
            "bonds": [
                {
                    "begin": bond.GetBeginAtomIdx(),
                    "end": bond.GetEndAtomIdx(),
                    "order": bond.GetBondTypeAsDouble(),
                }
                for bond in mol.GetBonds()
            ],
        }
        records.append(
            {
                "tree_ordinal": route,
                "source_node": {
                    "location": source_ref["path"] + pointer,
                    "sha256": source_ref["sha256"],
                },
                "source_raw_smiles": raw,
                "parsed_graph": graph,
                "positive_current_exact_evidence": any(
                    r["current_state"] == "listed" for r in witnesses
                ),
                "all_current_listing_observations": witnesses,
            }
        )

    for i, tree in enumerate(source["top_route_hypotheses"]):
        record(tree, f"#/top_route_hypotheses/{i}", i)
    sulfur = next(row for row in records if row["parsed_graph"]["canonical_smiles"] == "S")
    assert sulfur["parsed_graph"]["molecular_formula"] == "H2S" and sulfur["parsed_graph"][
        "atoms"
    ] == [
        {
            "index": 0,
            "atomic_number": 16,
            "formal_charge": 0,
            "isotope": 0,
            "degree": 0,
            "implicit_hydrogens": 2,
            "explicit_hydrogens": 0,
            "radical_electrons": 0,
        }
    ]
    result = {
        "schema": "forge.exact_saved_leaf_graph_forms.v1",
        "inputs": input_refs,
        "rdkit_version": rdBase.rdkitVersion,
        "as_of_utc": protocol["clock"],
        "leaf_records": records,
        "interpretation": "Saved SMILES S parses under pinned RDKit as one neutral sulfur atom with two implicit hydrogens, formula H2S. It does not encode elemental sulfur, S8, a salt or a supplier product form. Do not substitute by name or composition. All other exact graph/charge/isotope/tautomer gates remain unchanged.",
        "future_query_candidates_only": ["CC(O)=S", "COC(C)=S", "S"],
        "candidate_sets_alternative_not_additive": True,
        "conditional_ceiling_for_each": 6,
        "extra_requirements_for_saved_donor_options": "Exact listing plus independent full saved-tree and every full-root graft admission before paired real evaluation; not already-qualified closure.",
        "no_new_queries_paths_searches": True,
    }
    write(DEST / "exact_graph_forms.json", result)
    (DEST / "exact_graph_forms.md").write_text(
        "Saved `S` means the exact graph parsed as **H2S**: one neutral sulfur atom, two implicit hydrogens, no isotope, no bond and no radical. It is not elemental sulfur or S8; a salt or different vendor form cannot substitute.\n\n`exact_graph_forms.json` preserves the exact raw-node pointers and every matching dated listing observation. In saved tree 1, `CC(=O)Cl` already has positive current evidence; exact `S` remains unassessed. Saved tree 0 requires exact `COC(C)=S`, also unassessed. Neither option creates an observed gain. Each shares the same conditional six-product ceiling and additionally requires independent whole-tree/graft admission. No queries or new paths were made.\n"
    )
    print(
        json.dumps(
            {
                "result": pin(DEST / "exact_graph_forms.json"),
                "sulfur": sulfur["parsed_graph"],
                "records": [
                    {
                        "tree": r["tree_ordinal"],
                        "identity": r["parsed_graph"]["canonical_smiles"],
                        "positive": r["positive_current_exact_evidence"],
                        "states": [
                            w["current_state"] for w in r["all_current_listing_observations"]
                        ],
                    }
                    for r in records
                ],
            }
        )
    )


if __name__ == "__main__":
    run()
