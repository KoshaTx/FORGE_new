"""Check source drawing formulas and balance, without implementing STAAR chemistry."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent


def pin(path):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inventory(molecule):
    values = Counter()
    for atom in Chem.AddHs(molecule).GetAtoms():
        values[atom.GetSymbol()] += 1
        values["formal_charge"] += atom.GetFormalCharge()
    return values


def main():
    source = json.loads((OUTPUT / "control_transcriptions.json").read_text())
    assets = [
        "control_transcriptions.json",
        "article.html",
        "si.pdf",
        "fig1.jpg",
        "fig2.jpg",
        "fig3.jpg",
        "acquisition.json",
        "figures-acquisition.json",
        "check_transcriptions.py",
    ]
    rows = []
    for control in source["controls"]:
        components = {k: Chem.MolFromSmiles(v) for k, v in control["components"].items()}
        product = Chem.MolFromSmiles(control["expected_product"])
        assert product is not None and all(m is not None for m in components.values())
        assert len(Chem.GetMolFrags(product)) == 1
        left = Counter()
        for mol in components.values():
            assert len(Chem.GetMolFrags(mol)) == 1
            left.update(inventory(mol))
        right = inventory(product)
        assert left == right
        # Compute the ion directly rather than introducing an approximate proton mass.
        ion = Chem.CombineMols(product, Chem.MolFromSmiles("[H+]"))
        mz = rdMolDescriptors.CalcExactMolWt(ion)
        assert round(mz, 2) == control["source_theoretical_mz"]
        rows.append(
            {
                "label": control["label"],
                "canonical_product": Chem.MolToSmiles(product, isomericSmiles=False),
                "neutral_formula": rdMolDescriptors.CalcMolFormula(product),
                "computed_protonated_mz": mz,
                "source_theoretical_mz": control["source_theoretical_mz"],
                "full_element_hydrogen_charge_balance": dict(sorted(right.items())),
                "computed_formulas_and_inventory_pass": True,
                "reaction_program_qualified": False,
            }
        )
    result = {
        "schema_version": "forge.source_transcription_check.v1",
        "status": "formulas_and_net_balance_checked_program_qualification_pending",
        "inputs": {name: pin(OUTPUT / name) for name in assets},
        "rdkit_version": rdBase.rdkitVersion,
        "random_sampling_used": False,
        "controls": rows,
        "training_rows_admitted": 0,
        "limitations": [
            "Formula/mass and additive inventory checks do not establish bond connectivity or reaction-site selectivity.",
            "The reported theoretical ion masses support the independent drawings; observed low-resolution masses are not exact-mass validation.",
            "No forward/inverse reaction, corpus replay, experimental bank assignment or precursor holdout is qualified here.",
        ],
    }
    (OUTPUT / "transcription_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(result["status"], [(r["label"], r["neutral_formula"]) for r in rows])


if __name__ == "__main__":
    main()
