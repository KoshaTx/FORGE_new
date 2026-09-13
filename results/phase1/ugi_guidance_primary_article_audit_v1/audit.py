"""Authenticate the supplied primary article and adjudicate one exact L2 oxidation."""

import argparse
import copy
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import write_json
from forge.corpus.source_evidence_adjudication import _l2_records, _verify_source_assets
from forge.synthesis.engine.qualified_forward import (
    load_qualified_forward_reaction,
    unique_forward_products,
)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONFIG = ROOT / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
RESULT = OUT / "result.json"


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    require(molecule is not None, "Invalid molecular identity")
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def exact_pair(compiled, definition, reactant, product):
    """Guard the exact registry pair before using its locally applicable SMARTS."""
    require(
        canonical(reactant) == canonical(definition["exact_reactant_smiles"])
        and canonical(product) == canonical(definition["exact_product_smiles"]),
        "Outside the single qualified substrate/product pair",
    )
    outcomes = unique_forward_products(
        compiled, (reactant,), max_products=32, isomeric_smiles=False
    )
    require(outcomes == (canonical(product),), "Forward product is not exact and unique")
    return outcomes


def describe():
    config = read(CONFIG)
    review = read(OUT / "review.json")
    receipt = read(OUT / "receipt.json")
    article = review["source_assets"]["jo000463n.pdf"]
    require(
        article == {key: receipt["artifact"][key] for key in ("path", "sha256")},
        "User receipt and primary article disagree",
    )
    require(
        review["source_assets"]["jo000463n_si_001.pdf"] == config["inputs"]["oppolzer_primary_si"],
        "Supplement differs from the frozen source identity",
    )
    source_config = {
        "source_assets": review["source_assets"],
        "policy": {"source_review_status": "visually_reviewed"},
        "expected": {"source_reviews": 2},
    }
    assets = _verify_source_assets(source_config, review["source_reviews"], ROOT)
    registry_path = resolve_pin(
        config["inputs"]["reaction_registry"], ROOT, label="frozen C18 registry"
    )
    variant_path = resolve_pin(
        config["inputs"]["oxidation_variant"], ROOT, label="frozen C18 oxidation variant"
    )
    steps = [s for s in config["steps"] if s["variant_input"] == "oxidation_variant"]
    require(len(steps) == 1, "Expected one frozen oxidation step")
    step = steps[0]
    reaction = review["reported_reaction"]
    for role in ("reactant", "product"):
        require(
            reaction[role]["canonical_smiles"] == step[f"{role}_smiles"]
            and reaction[role]["role"] == step[f"{role}_role"],
            f"Reviewed {role} identity or role disagrees with the frozen pair",
        )
        mol = Chem.MolFromSmiles(step[f"{role}_smiles"])
        require(
            Chem.MolToInchiKey(mol) == reaction[role]["inchi_key"],
            f"Reviewed {role} InChIKey does not match",
        )
    require(reaction["conditions"] == step["conditions"], "Conditions conflict with frozen step")
    require(
        reaction["isolated_yield_percent"] == step["isolated_yield_percent"],
        "Reported isolated yield conflicts with frozen step",
    )
    chain = reaction["reference_chain"]
    require(
        all(
            chain[key] is True
            for key in (
                "scheme_links_7_to_8",
                "table_links_exact_c18_member_to_method_a",
                "complete_general_oxidation_procedure_present",
                "product_specific_characterization_present",
            )
        )
        and chain["required_preparation_link_unavailable"] is False,
        "Incomplete source execution/procedure chain",
    )
    definitions = [
        d for d in read(registry_path)["reactions"] if d["reaction_id"] == step["reaction_id"]
    ]
    require(len(definitions) == 1, "Expected exactly one registered oxidation")
    definition = definitions[0]
    compiled = load_qualified_forward_reaction(
        registry_path, variant_path, reaction_id=step["reaction_id"]
    )
    outcomes = exact_pair(compiled, definition, step["reactant_smiles"], step["product_smiles"])
    substrate = Chem.MolFromSmiles(step["reactant_smiles"])
    sites = substrate.GetSubstructMatches(compiled.reaction.GetReactantTemplate(0), uniquify=True)
    require(len(sites) == 1, "Reactive site is not unique")
    raw_products = compiled.reaction.RunReactants((substrate,), maxProducts=32)
    require(len(raw_products) == 1 and len(raw_products[0]) == 1, "Ambiguous raw product")
    product = raw_products[0][0]
    Chem.SanitizeMol(product)
    mapping = {a.GetIdx(): a.GetIntProp("react_atom_idx") for a in product.GetAtoms()}
    require(sorted(mapping.values()) == list(range(substrate.GetNumAtoms())), "Lost atom identity")
    require(
        all(
            a.GetAtomicNum() == substrate.GetAtomWithIdx(mapping[a.GetIdx()]).GetAtomicNum()
            for a in product.GetAtoms()
        ),
        "Element identity changed",
    )
    before = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble()
        for b in substrate.GetBonds()
    }
    after = {
        tuple(
            sorted((mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()]))
        ): b.GetBondTypeAsDouble()
        for b in product.GetBonds()
    }
    require(before.keys() == after.keys(), "Heavy-atom connectivity changed")
    changes = [key for key in before if before[key] != after[key]]
    require(
        changes == [tuple(sorted(sites[0]))] and before[changes[0]] == 1 and after[changes[0]] == 2,
        "Bond change differs from the one source-resolved alcohol oxidation",
    )
    explicit_before = Counter(a.GetSymbol() for a in Chem.AddHs(substrate).GetAtoms())
    explicit_after = Counter(a.GetSymbol() for a in Chem.AddHs(product).GetAtoms())
    require(
        explicit_before == explicit_after + Counter({"H": 2}),
        "Element balance differs from oxidation with net substrate loss of two hydrogens",
    )
    records = _l2_records(review["source_reviews"], assets, {})
    require(len(records) == 1 and records[0]["disposition"] == "admit_exact", "L2 admission failed")
    require(
        records[0]["target_canonical_smiles"] == canonical(step["product_smiles"])
        and json.loads(records[0]["component_smiles_json"]) == [canonical(step["reactant_smiles"])],
        "Repository adjudicator did not emit the checked pair",
    )
    trials = config["verification_policy"]["randomized_smiles_trials"]
    rng = random.Random(0)
    for _ in range(trials):
        order = list(range(substrate.GetNumAtoms()))
        rng.shuffle(order)
        alternate = Chem.MolToSmiles(Chem.RenumberAtoms(substrate, order), canonical=False)
        exact_pair(compiled, definition, alternate, step["product_smiles"])
        reordered_review = copy.deepcopy(review["source_reviews"])
        reordered_review["supplement"]["l2_route_families"][0]["steps"][0][
            "reactant_smiles"
        ] = alternate
        require(
            _l2_records(reordered_review, assets, {}) == records, "Atom-order adjudication drift"
        )
    for reactant, target in (
        (config["molecules"]["octadec_9_yn_1_ol"]["canonical_smiles"], step["product_smiles"]),
        (step["reactant_smiles"], step["reactant_smiles"]),
    ):
        try:
            exact_pair(compiled, definition, reactant, target)
        except ValueError as exc:
            require("Outside the single qualified" in str(exc), "Unexpected negative-check failure")
        else:
            raise ValueError("Unqualified exact pair was accepted")
    return {
        "source_assets": assets,
        "evidence_records": records,
        "scope": "One reported 7d-to-8d L2 step; source_route_reported refers only to this step.",
        "mechanical_checks": {
            "unique_reacting_site_count": len(sites),
            "reactant_atom_indices_zero_based": list(sites[0]),
            "reactant_role_order": list(compiled.role_names),
            "exact_unique_products": list(outcomes),
            "heavy_atoms_conserved": substrate.GetNumHeavyAtoms(),
            "substrate_formula": rdMolDescriptors.CalcMolFormula(substrate),
            "product_formula": rdMolDescriptors.CalcMolFormula(product),
            "substrate_to_product_net_hydrogen_loss": 2,
            "mass_balance_scope": "Organic substrate/product atom accounting; oxidant products unmodeled.",
            "only_bond_change": "C1-O single to double; carbon skeleton and terminal alkyne retained.",
            "atom_order_trials_passed": trials,
            "seed": 0,
            "out_of_scope_substrate_and_wrong_target_rejected": True,
        },
        "article_evidence_gap_closed": True,
        "single_l2_evidence_disposition": "admit_exact",
        "isolated_yield_percent": reaction["isolated_yield_percent"],
        "historical_config_revalidated_in_full": False,
        "complete_c18_route_requalified": False,
        "new_terminal_observations": 0,
        "guidance_readiness_established": False,
        "remaining_requirements": [
            "Qualifying current US stearolic-acid availability or a separately qualified route to it.",
            "Separate cumulative source, complete route and synthesis-value qualification.",
        ],
        "calls": {"training": 0, "molecular_generation": 0, "guidance": 0, "remote_compute": 0},
        "scientific_scope_limits": review["scope_limits"],
        "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Authenticate inputs and replay read-only"
    )
    args = parser.parse_args()
    existing = read(RESULT) if RESULT.exists() else None
    if args.verify:
        require(existing is not None, "Missing saved article audit")
        for record in existing["inputs"].values():
            resolve_pin(
                {key: record[key] for key in ("path", "sha256")}, ROOT, label=record["path"]
            )
    else:
        require(existing is None, "Refusing to overwrite saved audit; use --verify")
    scientific = describe()
    if args.verify:
        require(
            scientific == existing["scientific"], "Saved scientific payload differs from replay"
        )
        print("verified: source assets and exact L2 scientific payload")
        return
    paths = {
        CONFIG,
        Path(__file__),
        OUT / "review.json",
        OUT / "receipt.json",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "tests/test_m0_05_source_evidence_adjudication.py",
        ROOT / "tests/test_qualified_forward.py",
        ROOT / "tests/test_pinned_sources_are_tracked.py",
    }
    for spec in read(OUT / "review.json")["source_assets"].values():
        paths.add(ROOT / spec["path"])
    for key in ("reaction_registry", "oxidation_variant"):
        paths.add(ROOT / read(CONFIG)["inputs"][key]["path"])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".venv"):
                if path.suffix == ".py" and path.is_file():
                    paths.add(path)
    result = {
        "schema_version": "forge.primary_article_chemistry_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "single_l2_article_evidence_admitted_full_guidance_qualification_pending",
        "inputs": {str(path.relative_to(ROOT)): pin_record(path, ROOT) for path in sorted(paths)},
        "scientific": scientific,
    }
    write_json(RESULT, result)
    print("admitted: one exact L2 oxidation; full route and guidance qualification pending")


if __name__ == "__main__":
    main()
