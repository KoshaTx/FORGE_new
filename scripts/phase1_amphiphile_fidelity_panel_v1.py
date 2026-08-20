#!/usr/bin/env python3
"""Does the flow learn the amphiphile, or only the alphabet?

The constraint-only control established that formal Ugi validity is not the learning problem: a
denoiser emitting nothing but role-specific empirical marginals reaches higher admission than
the trained flow under identical constraints. So admission is out as evidence, and the question
becomes what the trained flow supplies instead.

Molecular weight, cLogP and nearest-neighbour Tanimoto pointed the right way but are too generic
to carry a claim about ionizable lipids. A model could match all three and still get the
chemistry that matters wrong. This panel asks the question in the structural language the field
uses: an ionizable lipid is an amphiphile whose ionizable head, hydrophobic regions, branching,
unsaturation and linkage chemistry have to be organized together.

The marginal null already reproduces one-variable role frequencies by construction. Any advantage
the trained flow shows here therefore has to come from **dependence** between those variables,
which is precisely the claim: the null knows the alphabet, the flow learns the grammar.

Four populations under identical precursor-architecture programs. The reference is exact rather
than approximate: every Stage 2 program was drawn from a real heldout-fold lipid, so each
program has one real molecule and one molecule from each arm.

Three levels of comparison, fixed before the numbers were seen:

  marginals    per-descriptor medians, the weakest level, reported for completeness
  dependence   the descriptor correlation matrix against the held-out one, in Frobenius norm.
               This is the level the claim lives at, because the null matches marginals already.
  separability a grouped two-sample classifier, held-out real against each arm, cross-validated
               with grouping by aldehyde component family so repeated component combinations
               cannot leak between folds. AUC near 0.5 means indistinguishable on this panel.

What this does not do: it does not model ionization, packing, particle morphology or endosomal
escape, and no descriptor here is a substitute for apparent LNP pKa. It measures molecular
structure only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
SEED = 20260816

SMARTS = {
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine": "[NX3;H0;!$(NC=O);!$(N=*)]",
    "ring_nitrogen": "[nX2,NX3;R]",
    "ester": "[CX3](=[OX1])[OX2][CX4]",
    "ether": "[CX4][OX2][CX4]",
    "ketone": "[CX4][CX3](=[OX1])[CX4]",
    "alkene": "[CX3]=[CX3]",
    "branch_carbon": "[CX4;$(C(-[#6])(-[#6])-[#6])]",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_carbon_chain(mol: Chem.Mol) -> int:
    """Longest simple path through carbon, by breadth-first eccentricity on the carbon subgraph."""
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if not carbons:
        return 0
    index = {a: i for i, a in enumerate(carbons)}
    neighbours: list[list[int]] = [[] for _ in carbons]
    for atom_index in carbons:
        for neighbour in mol.GetAtomWithIdx(atom_index).GetNeighbors():
            if neighbour.GetAtomicNum() == 6:
                neighbours[index[atom_index]].append(index[neighbour.GetIdx()])
    best = 0
    for start in range(len(carbons)):
        seen = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for other in neighbours[node]:
                if other not in seen:
                    seen[other] = seen[node] + 1
                    queue.append(other)
        best = max(best, max(seen.values()))
    return best + 1


def nitrogen_spacing(mol: Chem.Mol) -> tuple[float, float]:
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) < 2:
        return 0.0, 0.0
    distances = Chem.GetDistanceMatrix(mol)
    pairs = [distances[i][j] for k, i in enumerate(nitrogens) for j in nitrogens[k + 1:]]
    return float(min(pairs)), float(max(pairs))


def describe(product: str, parts: dict[str, str], patterns: dict) -> dict[str, float] | None:
    whole = Chem.MolFromSmiles(product)
    components = {role: Chem.MolFromSmiles(parts.get(role, "")) for role in ROLES}
    if whole is None or any(m is None for m in components.values()):
        return None
    head = components["amine_head"]
    out: dict[str, float] = {}

    # --- ionizable head architecture
    out["head_heavy_atoms"] = head.GetNumHeavyAtoms()
    out["head_nitrogens"] = sum(1 for a in head.GetAtoms() if a.GetAtomicNum() == 7)
    for name in ("primary_amine", "secondary_amine", "tertiary_amine", "ring_nitrogen"):
        out[f"head_{name}"] = len(head.GetSubstructMatches(patterns[name]))
    out["head_rings"] = rdMolDescriptors.CalcNumRings(head)
    out["head_hbd"] = rdMolDescriptors.CalcNumHBD(head)
    out["head_hba"] = rdMolDescriptors.CalcNumHBA(head)
    minimum, maximum = nitrogen_spacing(head)
    out["head_min_n_spacing"] = minimum
    out["head_max_n_spacing"] = maximum

    # --- hydrophobic regions, one block per tail-bearing role
    carbons = {}
    for role in TAIL_ROLES:
        mol = components[role]
        tag = "ald" if role.startswith("oxoester") else "iso"
        carbons[tag] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        out[f"{tag}_carbons"] = carbons[tag]
        out[f"{tag}_longest_chain"] = longest_carbon_chain(mol)
        out[f"{tag}_branches"] = len(mol.GetSubstructMatches(patterns["branch_carbon"]))
        out[f"{tag}_alkenes"] = len(mol.GetSubstructMatches(patterns["alkene"]))
        out[f"{tag}_esters"] = len(mol.GetSubstructMatches(patterns["ester"]))
        out[f"{tag}_ethers"] = len(mol.GetSubstructMatches(patterns["ether"]))
        out[f"{tag}_ketones"] = len(mol.GetSubstructMatches(patterns["ketone"]))
        out[f"{tag}_heavy_atoms"] = mol.GetNumHeavyAtoms()

    # --- whole-amphiphile organization, where head and tails have to agree
    total_tail_carbons = carbons["ald"] + carbons["iso"]
    out["total_tail_carbons"] = total_tail_carbons
    out["tail_asymmetry"] = abs(carbons["ald"] - carbons["iso"])
    out["tail_balance"] = (carbons["ald"] / total_tail_carbons) if total_tail_carbons else 0.0
    out["head_to_tail_ratio"] = (out["head_heavy_atoms"] / total_tail_carbons
                                 if total_tail_carbons else 0.0)
    out["degradable_linkages"] = out["ald_esters"] + out["iso_esters"]
    out["molecular_weight"] = Descriptors.MolWt(whole)
    out["clogp"] = Crippen.MolLogP(whole)
    out["tpsa"] = Descriptors.TPSA(whole)
    out["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(whole)
    out["hbd"] = rdMolDescriptors.CalcNumHBD(whole)
    out["hba"] = rdMolDescriptors.CalcNumHBA(whole)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, required=True)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "PATH"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    patterns = {name: Chem.MolFromSmarts(sma) for name, sma in SMARTS.items()}

    draw = json.loads(args.draw.read_text())
    family_of = {row["product_id"]: row["aldehyde_family_id"] for row in draw["samples"]}
    wanted = set(family_of)

    # --- the matched real reference: the heldout products the programs were drawn from
    from forge.design.training.ugi_training_cache import load_ugi_training_cache
    corpus, records_by_fold = load_ugi_training_cache(
        REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt")
    assignments = {a["product_id"]: a for a in corpus.assignments_by_fold["heldout"]}
    reference_rows = []
    with rdBase.BlockLogs():
        for product_id in sorted(wanted):
            assignment = assignments.get(product_id)
            if assignment is None:
                continue
            values = describe(assignment["canonical_product_smiles"],
                              {role: assignment[f"{role}_smiles"] for role in ROLES}, patterns)
            if values:
                reference_rows.append((product_id, values))
    print(f"matched real reference: {len(reference_rows)} heldout lipids, "
          f"the exact molecules the programs were drawn from")

    populations: dict[str, list[tuple[str, dict]]] = {"heldout_real": reference_rows}
    with rdBase.BlockLogs():
        for label, path in args.arm:
            rows = []
            for row in json.loads(Path(path).read_text())["samples"]:
                if not row.get("valid"):
                    continue
                values = describe(row.get("smiles") or "",
                                  row.get("component_smiles_by_role") or {}, patterns)
                if values:
                    rows.append((row["product_id"], values))
            populations[label] = rows
            print(f"{label}: {len(rows)} admitted and describable")

    columns = sorted(reference_rows[0][1])
    matrices = {label: np.array([[values[c] for c in columns] for _, values in rows], dtype=float)
                for label, rows in populations.items()}

    # --- level 1: marginals
    marginals = {label: {c: float(np.median(matrix[:, i]))
                         for i, c in enumerate(columns)}
                 for label, matrix in matrices.items()}

    # --- level 2: dependence. The null matches marginals by construction, so this is the level
    # the claim lives at.
    def correlation(matrix: np.ndarray) -> np.ndarray:
        keep = matrix.std(axis=0) > 1e-9
        safe = matrix[:, keep]
        return np.corrcoef(safe, rowvar=False), keep

    reference_corr, reference_keep = correlation(matrices["heldout_real"])
    dependence = {}
    for label, matrix in matrices.items():
        if label == "heldout_real":
            continue
        _, keep = correlation(matrix)
        shared = reference_keep & keep
        left = np.corrcoef(matrices["heldout_real"][:, shared], rowvar=False)
        right = np.corrcoef(matrix[:, shared], rowvar=False)
        difference = left - right
        dependence[label] = {
            "descriptors_compared": int(shared.sum()),
            "frobenius_distance": float(np.linalg.norm(difference)),
            "mean_absolute_correlation_difference": float(np.abs(difference).mean()),
            "max_absolute_correlation_difference": float(np.abs(difference).max()),
        }

    # --- level 3: grouped two-sample separability
    separability = {}
    reference_families = np.array([family_of.get(pid, "unknown")
                                   for pid, _ in populations["heldout_real"]])
    for label, rows in populations.items():
        if label == "heldout_real":
            continue
        features = np.vstack([matrices["heldout_real"], matrices[label]])
        labels = np.concatenate([np.zeros(len(matrices["heldout_real"])),
                                 np.ones(len(matrices[label]))])
        groups = np.concatenate([reference_families,
                                 np.array([family_of.get(pid, "unknown") for pid, _ in rows])])
        distinct = len(set(groups))
        splits = min(5, distinct)
        if splits < 2:
            separability[label] = {"skipped": "fewer than two component-family groups"}
            continue
        aucs = []
        for train_index, test_index in GroupKFold(n_splits=splits).split(features, labels, groups):
            if len(set(labels[train_index])) < 2 or len(set(labels[test_index])) < 2:
                continue
            model = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
            model.fit(features[train_index], labels[train_index])
            aucs.append(roc_auc_score(labels[test_index],
                                      model.predict_proba(features[test_index])[:, 1]))
        separability[label] = {
            "grouped_cv_auc_mean": float(np.mean(aucs)) if aucs else None,
            "grouped_cv_auc_folds": [float(a) for a in aucs],
            "groups": distinct,
            "grouping": "aldehyde component family, so repeated component combinations cannot "
                        "leak between folds",
        }

    print(f"\n{'arm':18s} {'n':>6s} {'corr Frobenius':>15s} {'mean |Δcorr|':>13s} "
          f"{'grouped AUC':>12s}")
    for label in populations:
        if label == "heldout_real":
            print(f"{label:18s} {len(populations[label]):6d} {'reference':>15s}")
            continue
        d = dependence[label]
        auc = separability[label].get("grouped_cv_auc_mean")
        print(f"{label:18s} {len(populations[label]):6d} {d['frobenius_distance']:15.2f} "
              f"{d['mean_absolute_correlation_difference']:13.3f} "
              f"{(f'{auc:.3f}' if auc is not None else '—'):>12s}")

    print(f"\nselected medians ({len(columns)} descriptors computed):")
    highlight = ["head_nitrogens", "head_tertiary_amine", "head_max_n_spacing",
                 "ald_carbons", "iso_carbons", "tail_asymmetry", "head_to_tail_ratio",
                 "ald_branches", "ald_alkenes", "degradable_linkages", "clogp"]
    print(f"  {'descriptor':24s} " + " ".join(f"{label[:13]:>14s}" for label in populations))
    for name in highlight:
        print(f"  {name:24s} " + " ".join(
            f"{marginals[label][name]:14.2f}" for label in populations))

    payload = {
        "schema_version": "phase1_forge_amphiphile_fidelity_panel.v1",
        "status": "complete_role_resolved_structural_comparison",
        "question": "the marginal null reproduces one-variable role frequencies by construction, "
                    "so any advantage the trained flow shows here comes from dependence between "
                    "head, linker and tail choices",
        "inputs": {"draw": {"path": str(args.draw), "sha256": sha256_file(args.draw)},
                   "arms": {label: {"path": path, "sha256": sha256_file(Path(path))}
                            for label, path in args.arm}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "reference": "the exact heldout lipids the Stage 2 programs were drawn from",
        "descriptors": columns,
        "smarts": SMARTS,
        "population_sizes": {label: len(rows) for label, rows in populations.items()},
        "marginals_median": marginals,
        "dependence_against_heldout_real": dependence,
        "grouped_two_sample_separability": separability,
        "nonclaims": [
            "Structure only. Nothing here models ionization, packing, particle morphology or "
            "endosomal escape, and no descriptor substitutes for apparent LNP pKa.",
            "Matching these descriptors is not evidence that a generated lipid delivers mRNA. It "
            "is evidence that its structure is organized like the chemistry the corpus contains.",
            "The classifier is a secondary aggregate. Its grouping by aldehyde component family "
            "is what keeps repeated component combinations from inflating it.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
