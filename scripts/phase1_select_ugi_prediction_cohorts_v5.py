#!/usr/bin/env python3
"""Select the v5 prospective panel under configs/bio/phase1_ugi_prediction_cohorts_v5.json.

Every threshold comes from that file, which was frozen before this script ran.  Nothing
here chooses a criterion; it only applies them and records what happened.

Five cohorts, differing in what is recombined rather than in what makes the oracle abstain:

  Q  new combination, oracle issues a calibrated prediction        rank by lcb90
  H  new amine head built from measured head vocabulary            rank by mean - sd
  N  new aldehyde inside the measured tail envelope                rank by mean - sd
  X  new aldehyde beyond the envelope                              rank by mean - sd
  B  ester-linked branched aldehyde, from the branch lane          rank by its own statistic

Membership additionally requires absence from the 66,464-product train fold, which is the
standard generative-chemistry novelty denominator.  Amine heads have no synthesis route in
this framework, so an unseen head must be purchasable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULT_SCHEMA_VERSION = "phase1_ugi_prediction_cohort_selection.v5"
LEDGER_SCHEMA_VERSION = "phase1_ugi_prediction_cohort_panel_ledger.v5"

CONFIG = "configs/bio/phase1_ugi_prediction_cohorts_v5.json"
ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
LEDGER = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
POOL_N = "results/phase1/ugi_prediction_cohorts_v1/scored_N.jsonl.gz"
POOL_X = "results/phase1/ugi_prediction_cohorts_v1/scored_X.jsonl.gz"


# ---------------------------------------------------------------- infrastructure

def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def train_fold(repo: Path) -> set[str]:
    rows = read_csv_gz(repo / ASSIGN)
    fold = {r["canonical_product_smiles"] for r in rows if r["primary_product_fold"] == "train"}
    if not fold:
        raise SystemExit("train fold is empty; refusing to run a novelty filter against nothing")
    return fold


def full_corpus(repo: Path) -> set[str]:
    return {r["canonical_product_smiles"] for r in read_csv_gz(repo / ASSIGN)}


def descriptors(smiles: str) -> dict[str, float] | None:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return {
        "molecular_weight": Descriptors.MolWt(molecule),
        "clogp": Crippen.MolLogP(molecule),
        "tpsa": Descriptors.TPSA(molecule),
        "heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
        "rotatable_bonds": float(Descriptors.NumRotatableBonds(molecule)),
        "formal_charge": float(Chem.GetFormalCharge(molecule)),
    }


def inside_envelope(values: dict[str, float], screen: dict[str, Any]) -> bool:
    tol = float(screen.get("comparison_tolerance", 0.0))
    for key in ("molecular_weight", "clogp", "tpsa"):
        low, high = screen[key]
        if not (low - tol <= values[key] <= high + tol):
            return False
    return True


def route_dossier(builder: Any, index: dict[str, Any], row: dict[str, str]) -> list[dict[str, Any]]:
    return [
        builder.component_dossier(index, "amine_head", row["canonical_amine"]),
        builder.component_dossier(index, "oxoester_aldehyde_body_tail", row["canonical_aldehyde"]),
        builder.component_dossier(index, "isocyanide_tail", row["canonical_isocyanide"]),
    ]


def route_ok(dossier: list[dict[str, Any]], max_steps: int) -> bool:
    return all(d["resolved"] for d in dossier) and sum(d.get("steps", 0) for d in dossier) <= max_steps


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- head recombination

def measured_heads(repo: Path) -> list[str]:
    return sorted({r["A_smiles"] for r in read_csv_gz(repo / MEASURED)})


def head_recombination_filter(repo: Path, rule: dict[str, Any]) -> Callable[[str], dict[str, Any] | None]:
    """Frozen test for whether an unseen head is a recombination of measured head vocabulary."""

    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    heads = measured_heads(repo)
    with rdBase.BlockLogs():
        fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in heads]
    allowed_scaffolds = set(rule["measured_head_scaffolds"])
    mw_low, mw_high = rule["measured_head_mw_range"]
    threshold = float(rule["criteria_all_required"]["nearest_neighbour_similarity"])
    cache: dict[str, dict[str, Any] | None] = {}

    def check(smiles: str) -> dict[str, Any] | None:
        if smiles in cache:
            return cache[smiles]
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                cache[smiles] = None
                return None
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "(acyclic)"
            weight = Descriptors.MolWt(molecule)
            basic = sum(
                1 for a in molecule.GetAtoms()
                if a.GetSymbol() == "N" and not a.GetIsAromatic()
                and a.GetTotalNumHs() + a.GetDegree() == 3
            )
            elements_ok = all(a.GetSymbol() in ("C", "N", "O") for a in molecule.GetAtoms())
            similarity = max(
                DataStructs.TanimotoSimilarity(generator.GetFingerprint(molecule), f)
                for f in fingerprints
            )
        ok = (scaffold in allowed_scaffolds and elements_ok
              and mw_low <= weight <= mw_high and 1 <= basic <= 4
              and similarity >= threshold)
        result = {
            "murcko_scaffold": scaffold, "molecular_weight": round(weight, 2),
            "basic_nitrogen_count": basic,
            "nearest_measured_head_similarity": round(similarity, 4),
        } if ok else None
        cache[smiles] = result
        return result

    return check


# ---------------------------------------------------------------- diversity

def greedy_diverse(candidates: list[dict[str, Any]], ceiling: float, limit: int,
                   fingerprints: dict[str, Any], extra: Callable[[dict[str, Any], list[dict[str, Any]]], bool] | None = None
                   ) -> list[dict[str, Any]]:
    """Walk a frozen ranking, keeping a candidate only if it stays under the ceiling
    against everything already kept."""

    from rdkit import DataStructs

    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(kept) >= limit:
            break
        product = candidate["canonical_product"]
        if any(DataStructs.TanimotoSimilarity(fingerprints[product], fingerprints[k["canonical_product"]]) > ceiling
               for k in kept):
            continue
        if extra is not None and not extra(candidate, kept):
            continue
        kept.append(candidate)
    return kept


def fingerprint_index(products: list[str]) -> dict[str, Any]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    index: dict[str, Any] = {}
    with rdBase.BlockLogs():
        for smiles in products:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                index[smiles] = generator.GetFingerprint(molecule)
    return index


# ---------------------------------------------------------------- pools

def build_q_pool(repo: Path, config: dict[str, Any], builder: Any, index: dict[str, Any],
                 train: set[str], corpus: set[str]) -> list[dict[str, Any]]:
    screen = config["shared_requirements"]["physicochemical_screen"]
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    # Q is the qualified role-holdout tier, not every oracle_scored row. No generated
    # design has zero unseen components, so "all identities measured" describes nothing.
    # The pipeline's own authority tier is the partition: role-holdout carries an unseen
    # isocyanide only, which is the strongest evidence class available, and restricting to
    # it also makes Q disjoint from H, N, X and B rather than overlapping them.
    tier = config["cohorts"]["Q_high"]["authority_tier"]
    pool: dict[str, dict[str, Any]] = {}
    for row in read_csv_gz(repo / LEDGER):
        if row["terminal_chemical_admitted"] != "True" or row["oracle_scored"] != "True":
            continue
        if row.get("authority_tier") != tier:
            continue
        product = row["canonical_product"]
        if product in pool or product in train or not product:
            continue
        values = descriptors(product)
        if values is None or not inside_envelope(values, screen):
            continue
        lcb90 = as_float(row["lcb90"])
        mean, sd = as_float(row["oracle_mean"]), as_float(row["oracle_sd"])
        if lcb90 is None or mean is None or sd is None:
            continue
        dossier = route_dossier(builder, index, row)
        if not route_ok(dossier, max_steps):
            continue
        pool[product] = {
            "canonical_product": product, "row": row, "descriptors": values,
            "lcb90": lcb90, "conservative": mean - sd, "oracle_mean": mean, "oracle_sd": sd,
            "route_dossier": dossier,
            "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
            "absent_from_full_corpus": product not in corpus,
        }
    return sorted(pool.values(), key=lambda c: (-c["lcb90"], c["canonical_product"]))


def build_h_pool(repo: Path, config: dict[str, Any], builder: Any, index: dict[str, Any],
                 train: set[str], corpus: set[str]) -> list[dict[str, Any]]:
    screen = config["shared_requirements"]["physicochemical_screen"]
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    check = head_recombination_filter(repo, config["head_recombination_rule"])
    pool: dict[str, dict[str, Any]] = {}
    for row in read_csv_gz(repo / LEDGER):
        if row["terminal_chemical_admitted"] != "True" or row["unseen_roles"] != "amine":
            continue
        product = row["canonical_product"]
        if product in pool or product in train or not product:
            continue
        head = check(row["canonical_amine"])
        if head is None:
            continue
        values = descriptors(product)
        if values is None or not inside_envelope(values, screen):
            continue
        mean, sd = as_float(row["oracle_mean"]), as_float(row["oracle_sd"])
        if mean is None or sd is None:
            continue
        dossier = route_dossier(builder, index, row)
        if not route_ok(dossier, max_steps):
            continue
        pool[product] = {
            "canonical_product": product, "row": row, "descriptors": values,
            "lcb90": None, "conservative": mean - sd, "oracle_mean": mean, "oracle_sd": sd,
            "route_dossier": dossier,
            "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
            "head_recombination": head,
            "absent_from_full_corpus": product not in corpus,
        }
    return sorted(pool.values(), key=lambda c: (-c["conservative"], c["canonical_product"]))


def build_nx_pool(repo: Path, path: str, builder: Any, index: dict[str, Any],
                  train: set[str], corpus: set[str], max_steps: int) -> list[dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for record in (json.loads(line) for line in gzip.open(repo / path, "rt")):
        product = record["canonical_product"]
        if product in pool or product in train:
            continue
        dossier = route_dossier(builder, index, record)
        if not route_ok(dossier, max_steps):
            continue
        pool[product] = {
            "canonical_product": product, "row": record,
            "descriptors": {
                "molecular_weight": record["molecular_weight"], "clogp": record["clogp"],
                "tpsa": record["tpsa"],
            },
            "lcb90": None, "conservative": record["conservative_score"],
            "oracle_mean": record["raw_mean"], "oracle_sd": record["raw_sd"],
            "aldehyde_similarity": record["aldehyde_similarity"],
            "route_dossier": dossier,
            "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
            "absent_from_full_corpus": product not in corpus,
        }
    return sorted(pool.values(), key=lambda c: (-c["conservative"], c["canonical_product"]))


def build_b_reference_pool(repo: Path, config: dict[str, Any], builder: Any, index: dict[str, Any],
                           train: set[str], corpus: set[str]) -> list[dict[str, Any]]:
    """In-domain reference on the measured branched acyl motif.

    These carry the exact measured AGILE branched aldehyde, so they are deliberately not
    novel against the AGILE block set. Their job is to disambiguate a branched result:
    without them, a positive or negative outcome in B cannot be attributed to branching
    rather than to extrapolation.
    """

    screen = config["shared_requirements"]["physicochemical_screen"]
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    reference_aldehyde = "CCCCCC(C)CCC(=O)OCCCCCC=O"
    pool: dict[str, dict[str, Any]] = {}
    for source in (BRANCH, LEDGER):
        for row in read_csv_gz(repo / source):
            if row.get("terminal_chemical_admitted") != "True" or row["oracle_scored"] != "True":
                continue
            if row["canonical_aldehyde"] != reference_aldehyde:
                continue
            product = row["canonical_product"]
            if product in pool or product in train or not product:
                continue
            values = descriptors(product)
            if values is None or not inside_envelope(values, screen):
                continue
            lcb90 = as_float(row["lcb90"])
            mean, sd = as_float(row["oracle_mean"]), as_float(row["oracle_sd"])
            if lcb90 is None or mean is None or sd is None:
                continue
            dossier = route_dossier(builder, index, row)
            if not route_ok(dossier, max_steps):
                continue
            pool[product] = {
                "canonical_product": product, "row": row, "descriptors": values,
                "lcb90": lcb90, "conservative": mean - sd,
                "oracle_mean": mean, "oracle_sd": sd,
                "route_dossier": dossier,
                "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
                "reference_aldehyde": reference_aldehyde,
                "absent_from_full_corpus": product not in corpus,
            }
    return sorted(pool.values(), key=lambda c: (-c["lcb90"], c["canonical_product"]))


def build_b_pool(repo: Path, config: dict[str, Any], builder: Any, index: dict[str, Any],
                 train: set[str], corpus: set[str]) -> list[dict[str, Any]]:
    """Branch-conditioned lane. Branching is tested on the aldehyde, using the lane's own
    per-component branch-point count rather than a whole-product substructure match."""

    screen = config["shared_requirements"]["physicochemical_screen"]
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    pool: dict[str, dict[str, Any]] = {}
    for row in read_csv_gz(repo / BRANCH):
        if row.get("terminal_chemical_admitted") != "True":
            continue
        branch_points = as_float(row.get("aldehyde_carbon_branch_points", ""))
        if not branch_points:
            continue
        product = row["canonical_product"]
        if product in pool or product in train or not product:
            continue
        values = descriptors(product)
        if values is None or not inside_envelope(values, screen):
            continue
        mean, sd = as_float(row["oracle_mean"]), as_float(row["oracle_sd"])
        if mean is None or sd is None:
            continue
        lcb90 = as_float(row["lcb90"]) if row["oracle_scored"] == "True" else None
        dossier = route_dossier(builder, index, row)
        if not route_ok(dossier, max_steps):
            continue
        pool[product] = {
            "canonical_product": product, "row": row, "descriptors": values,
            "lcb90": lcb90, "conservative": mean - sd, "oracle_mean": mean, "oracle_sd": sd,
            "route_dossier": dossier,
            "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
            "aldehyde_carbon_branch_points": branch_points,
            "oracle_qualified": row["oracle_scored"] == "True",
            "absent_from_full_corpus": product not in corpus,
        }
    # Rank by lcb90 where qualified, otherwise by the conservative statistic. The two are
    # never merged into one number; qualified candidates simply sort first within their group.
    return sorted(pool.values(),
                  key=lambda c: (not c["oracle_qualified"],
                                 -(c["lcb90"] if c["lcb90"] is not None else c["conservative"]),
                                 c["canonical_product"]))


# ---------------------------------------------------------------- Q_low matching

def match_controls(q_high: list[dict[str, Any]], pool: list[dict[str, Any]],
                   config: dict[str, Any], fingerprints: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Nearest-neighbour matching without replacement, in standardized covariate space,
    then the lowest lcb90 among the matched set. Predicted potency never enters the
    distance; only the frozen low-rank eligibility criterion uses it."""

    spec = config["cohorts"]["Q_low"]
    quota = int(spec["n"])
    ceiling = float(config["diversity_rules"]["Q_low"]["maximum_pairwise_within_cohort"])
    numeric = ["molecular_weight", "clogp", "tpsa", "heavy_atom_count",
               "rotatable_bonds", "formal_charge"]

    scores = sorted(c["lcb90"] for c in pool)
    quartile = scores[len(scores) // 4]
    eligible = [c for c in pool if c["lcb90"] <= quartile and c not in q_high]
    if not eligible:
        return [], {"status": "no_eligible_controls", "quartile_boundary": quartile}

    def vector(candidate: dict[str, Any]) -> list[float]:
        values = [candidate["descriptors"][k] for k in numeric]
        values.append(float(candidate["synthetic_steps"]))
        return values

    everything = eligible + q_high
    columns = list(zip(*(vector(c) for c in everything)))
    sds = [st.pstdev(col) or 1.0 for col in columns]
    means = [st.fmean(col) for col in columns]

    def standardized(candidate: dict[str, Any]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(vector(candidate), means, sds)]

    high_vectors = [(c, standardized(c)) for c in q_high]
    matched: list[dict[str, Any]] = []
    used: set[str] = set()
    for target, target_vector in high_vectors:
        best, best_distance = None, None
        for candidate in eligible:
            if candidate["canonical_product"] in used:
                continue
            vector_c = standardized(candidate)
            distance = sum((a - b) ** 2 for a, b in zip(vector_c, target_vector)) ** 0.5
            if best_distance is None or distance < best_distance or (
                    distance == best_distance and candidate["canonical_product"] < best["canonical_product"]):
                best, best_distance = candidate, distance
        if best is not None:
            best = {**best, "matched_to": target["canonical_product"],
                    "match_distance": round(best_distance, 4)}
            matched.append(best)
            used.add(best["canonical_product"])

    ranked = sorted(matched, key=lambda c: (c["lcb90"], c["canonical_product"]))
    selected = greedy_diverse(ranked, ceiling, quota, fingerprints)
    audit = {
        "low_rank_quartile_boundary": round(quartile, 4),
        "eligible_bottom_quartile": len(eligible),
        "matched_pairs": len(matched),
        "match_distance_range": [min(c["match_distance"] for c in matched),
                                 max(c["match_distance"] for c in matched)] if matched else None,
        "selected": len(selected),
        "method": "nearest-neighbour without replacement in standardized covariate space; "
                  "lowest lcb90 among matched, subject to the within-cohort diversity ceiling",
    }
    return selected, audit


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase1/ugi_prediction_cohort_panel_v5"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from forge.product.ugi_bounded_hybrid_route_cascade import (
        atomic_write, canonical_json_bytes, jsonl_gzip_bytes, sha256_file, sha256_payload,
    )
    from importlib import import_module

    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(repo)
    config = json.loads((repo / CONFIG).read_text())
    train = train_fold(repo)
    corpus = full_corpus(repo)
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    rules = config["diversity_rules"]

    pools = {
        "Q": build_q_pool(repo, config, builder, index, train, corpus),
        "H": build_h_pool(repo, config, builder, index, train, corpus),
        "N": build_nx_pool(repo, POOL_N, builder, index, train, corpus, max_steps),
        "X": build_nx_pool(repo, POOL_X, builder, index, train, corpus, max_steps),
        "B": build_b_pool(repo, config, builder, index, train, corpus),
        "Bref": build_b_reference_pool(repo, config, builder, index, train, corpus),
    }
    every_product = [c["canonical_product"] for pool in pools.values() for c in pool]
    fingerprints = fingerprint_index(sorted(set(every_product)))

    selected: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {"pool_sizes": {k: len(v) for k, v in pools.items()}}

    # Q_high: highest lcb90 under the cohort diversity ceiling.
    selected["Q_high"] = greedy_diverse(
        pools["Q"], float(rules["Q_high"]["maximum_pairwise"]),
        int(config["cohorts"]["Q_high"]["n"]), fingerprints)

    # Q_low: matched controls.
    selected["Q_low"], audit["Q_low_matching"] = match_controls(
        selected["Q_high"], pools["Q"], config, fingerprints)

    # H: at most 4 candidates may share one head.
    head_cap = 4

    def head_limit(candidate: dict[str, Any], kept: list[dict[str, Any]]) -> bool:
        head = candidate["row"]["canonical_amine"]
        return sum(1 for k in kept if k["row"]["canonical_amine"] == head) < head_cap

    selected["H"] = greedy_diverse(
        pools["H"], float(rules["H_head_recombination"]["maximum_pairwise_within_cohort"]),
        int(config["cohorts"]["H_head_recombination"]["n"]), fingerprints, extra=head_limit)

    # N: strongest, then the lower edge of the aldehyde-similarity distribution.
    n_spec = config["cohorts"]["N_transfer"]["composition"]
    n_ceiling = float(rules["N_transfer"]["maximum_pairwise_within_cohort"])
    strongest = greedy_diverse(pools["N"], n_ceiling, int(n_spec["strongest"]), fingerprints)
    similarities = sorted(c["aldehyde_similarity"] for c in pools["N"])
    edge_boundary = similarities[len(similarities) // 4]
    edge_pool = [c for c in pools["N"]
                 if c["aldehyde_similarity"] <= edge_boundary and c not in strongest]
    lower_edge = greedy_diverse(edge_pool, n_ceiling, int(n_spec["lower_edge"]), fingerprints)
    selected["N"] = strongest + lower_edge
    audit["N_lower_edge_similarity_boundary"] = round(edge_boundary, 4)

    # X: near-envelope and remote, split on the median aldehyde similarity of the X pool.
    x_spec = config["cohorts"]["X_extrapolative"]["composition"]
    x_ceiling = float(rules["X_extrapolative"]["maximum_pairwise_within_cohort"])
    x_similarities = sorted(c["aldehyde_similarity"] for c in pools["X"])
    x_median = st.median(x_similarities) if x_similarities else 0.0
    near = greedy_diverse([c for c in pools["X"] if c["aldehyde_similarity"] >= x_median],
                          x_ceiling, int(x_spec["near_envelope"]), fingerprints)
    remote = greedy_diverse([c for c in pools["X"] if c["aldehyde_similarity"] < x_median],
                            x_ceiling, int(x_spec["remote"]), fingerprints)
    selected["X"] = near + remote
    audit["X_near_remote_boundary"] = round(x_median, 4)

    # B: distinct amine head required; component context is the diversity axis.
    def distinct_head(candidate: dict[str, Any], kept: list[dict[str, Any]]) -> bool:
        return all(k["row"]["canonical_amine"] != candidate["row"]["canonical_amine"] for k in kept)

    selected["B"] = greedy_diverse(pools["B"], 1.0, int(config["cohorts"]["B_branched"]["n"]),
                                   fingerprints, extra=distinct_head)

    # In-domain reference pair on the same acyl motif. Heads must differ from each other
    # and from the exploratory four, so the six B candidates span six component contexts.
    def reference_head(candidate: dict[str, Any], kept: list[dict[str, Any]]) -> bool:
        used = {k["row"]["canonical_amine"] for k in kept} | {
            c["row"]["canonical_amine"] for c in selected["B"]}
        return candidate["row"]["canonical_amine"] not in used

    selected["Bref"] = greedy_diverse(
        pools["Bref"], 1.0, int(config["cohorts"]["B_qualified_reference"]["n"]),
        fingerprints, extra=reference_head)

    # ------------------------------------------------------------ emit
    prefix = {"Q_high": "Q", "Q_low": "L", "H": "H", "N": "N", "X": "X",
              "B": "B", "Bref": "R"}
    panel: list[dict[str, Any]] = []
    for cohort, members in selected.items():
        for order, candidate in enumerate(members, start=1):
            row = candidate["row"]
            panel.append({
                "schema_version": LEDGER_SCHEMA_VERSION,
                "candidate_id": f"{prefix[cohort]}{order:02d}",
                "cohort": cohort.split("_")[0],
                "subgroup": cohort,
                "selection_order_within_subgroup": order,
                "canonical_product": candidate["canonical_product"],
                "components": {
                    "amine_head": row["canonical_amine"],
                    "oxoester_aldehyde_body_tail": row["canonical_aldehyde"],
                    "isocyanide_tail": row["canonical_isocyanide"],
                },
                "scores": {
                    "lcb90": candidate["lcb90"],
                    "conservative_mean_minus_sd": round(candidate["conservative"], 4),
                    "oracle_mean": round(candidate["oracle_mean"], 4),
                    "oracle_sd": round(candidate["oracle_sd"], 4),
                    "ordering_statistic": ("lcb90" if cohort.startswith("Q")
                                           or (cohort == "B" and candidate["lcb90"] is not None)
                                           else "conservative_mean_minus_sd"),
                },
                "novelty": {
                    "absent_from_train_fold": True,
                    "absent_from_full_corpus": candidate["absent_from_full_corpus"],
                },
                "descriptors": {k: round(v, 3) for k, v in candidate["descriptors"].items()},
                "aldehyde_similarity_to_measured": candidate.get("aldehyde_similarity"),
                "head_recombination": candidate.get("head_recombination"),
                "authority_tier": row.get("authority_tier"),
                "route_dossier": candidate["route_dossier"],
                "route_complete": True,
                "synthetic_steps": candidate["synthetic_steps"],
                "matched_to": candidate.get("matched_to"),
                "match_distance": candidate.get("match_distance"),
            })

    # Cohorts must partition the panel. An overlap would double-count a molecule and
    # silently shrink the panel, so fail rather than emit one.
    products = [p["canonical_product"] for p in panel]
    if len(set(products)) != len(products):
        clashes = sorted({p for p in products if products.count(p) > 1})
        raise SystemExit(
            f"cohorts are not disjoint: {len(clashes)} product(s) selected more than once\n"
            + "\n".join(f"  {p}" for p in clashes))

    panel_path = out / "prediction_cohort_panel.jsonl.gz"
    atomic_write(panel_path, jsonl_gzip_bytes(panel))

    filled = {k: len(v) for k, v in selected.items()}
    quota = config["quota"]
    shortfall = {k: quota.get(
        {"Q_high": "Q_high", "Q_low": "Q_low", "H": "H_head_recombination",
         "N": "N_transfer", "X": "X_extrapolative", "B": "B_branched",
         "Bref": "B_qualified_reference"}[k], 0) - v
        for k, v in filled.items()}

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selection_complete",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "config": {"path": CONFIG, "sha256": sha256_file(repo / CONFIG)},
        "inputs": {
            name: {"path": path, "sha256": sha256_file(repo / path)}
            for name, path in (("corpus_fold_assignments", ASSIGN), ("rescoring_ledger", LEDGER),
                               ("branch_lane", BRANCH), ("measured_products", MEASURED),
                               ("scored_N_pool", POOL_N), ("scored_X_pool", POOL_X))
        },
        "train_fold_size": len(train),
        "summary": {
            "panel_size": len(panel),
            "by_subgroup": filled,
            "shortfall_against_quota": {k: v for k, v in shortfall.items() if v},
            "novelty": {
                "absent_from_train_fold": sum(1 for p in panel if p["novelty"]["absent_from_train_fold"]),
                "absent_from_full_corpus": sum(1 for p in panel if p["novelty"]["absent_from_full_corpus"]),
            },
            "distinct_components": {
                role: len({p["components"][role] for p in panel})
                for role in ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
            },
            "maximum_synthetic_steps": max(p["synthetic_steps"] for p in panel) if panel else None,
            "score_range_by_subgroup": {
                k: [round(min(c["conservative"] for c in v), 3),
                    round(max(c["conservative"] for c in v), 3)]
                for k, v in selected.items() if v
            },
            "lcb90_range_by_subgroup": {
                k: [round(min(c["lcb90"] for c in v if c["lcb90"] is not None), 3),
                    round(max(c["lcb90"] for c in v if c["lcb90"] is not None), 3)]
                for k, v in selected.items() if any(c["lcb90"] is not None for c in v)
            },
        },
        "selection_audit": audit,
        "determinism": config["determinism"],
        "artifacts": {panel_path.name: {"sha256": sha256_file(panel_path), "rows": len(panel)}},
        "nonclaims": config["nonclaims"],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(out / "result.json", canonical_json_bytes(result))
    print(json.dumps(content["summary"], indent=2))


if __name__ == "__main__":
    main()
