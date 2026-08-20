#!/usr/bin/env python3
"""Measure the novelty of generated Ugi designs against every reference set that matters.

Four reference sets answer four different questions, and sliding between them is the
main way novelty claims go wrong:

  train fold (66,464)      the standard generative-chemistry denominator; absence means
                           the model never saw this molecule during training
  full corpus (112,386)    train plus calibration plus heldout; a stricter bar, and the
                           one used as a panel membership requirement
  measured products (1,100) absence means nobody has activity data on this lipid
  AGILE library (12,276)   absence means the prior art's own pipeline never proposed it

Exact-match novelty is reported because the field expects it, but it is a binary on a
string.  Binary folded Morgan fingerprints record which local environments are present,
not how many, so alkyl homologs of different chain length can collapse to Tanimoto 1.0.
Ionizable-lipid potency is chain-length sensitive, so this pass also reports the
nearest-neighbour distribution and counts how many perfect-similarity pairs are not the
same molecule at all.

Nothing here selects, ranks or filters a candidate.  It only measures.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULT_SCHEMA_VERSION = "phase1_ugi_novelty_audit.v1"

CORPUS = "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_products.csv.gz"
ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
LEDGER = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
AGILE_LIB = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
PANEL_V4 = "results/phase1/ugi_prediction_cohort_panel_v4/prediction_cohort_panel.jsonl.gz"

SAMPLE_SEED = 20260806
NN_SAMPLE = 1500
SCAFFOLD_SAMPLE = 5000


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def reference_sets(repo: Path) -> dict[str, set[str]]:
    """The four denominators, each meaning something different."""

    assign = read_csv_gz(repo / ASSIGN)
    train = {r["canonical_product_smiles"] for r in assign if r["primary_product_fold"] == "train"}
    full = {r["canonical_product_smiles"] for r in assign}
    measured = read_csv_gz(repo / MEASURED)
    agile = {r["canonical_product_smiles"] for r in read_csv_gz(repo / AGILE_LIB)}
    # The curated measured table stores the stereo-free structure under model_smiles.
    # It is not pre-canonicalised to the same form the generator emits, so canonicalise
    # rather than string-compare; an empty reference set would silently report novelty 1.0.
    from rdkit import Chem, rdBase

    measured_products: set[str] = set()
    with rdBase.BlockLogs():
        for row in measured:
            molecule = Chem.MolFromSmiles(row["model_smiles"])
            if molecule is None:
                continue
            Chem.RemoveStereochemistry(molecule)
            measured_products.add(Chem.MolToSmiles(molecule))
    if not measured_products:
        raise SystemExit("measured product reference set is empty; refusing to report novelty against it")
    return {
        "train_fold": train,
        "full_corpus": full,
        "measured_products": measured_products,
        "agile_virtual_library": agile,
    }


def component_registries(repo: Path) -> dict[str, dict[str, set[str]]]:
    """Building blocks of the corpus, and of the measured set."""

    assign = read_csv_gz(repo / ASSIGN)
    corpus = {
        "amine_head": {r["amine_head_smiles"] for r in assign},
        "aldehyde": {r["oxoester_aldehyde_body_tail_smiles"] for r in assign},
        "isocyanide": {r["isocyanide_tail_smiles"] for r in assign},
    }
    measured_rows = read_csv_gz(repo / MEASURED)
    measured = {
        "amine_head": {r["A_smiles"] for r in measured_rows},
        "aldehyde": {r["B_smiles"] for r in measured_rows},
        "isocyanide": {r["C_smiles"] for r in measured_rows},
    }
    return {"corpus": corpus, "measured": measured}


def novelty(generated: set[str], reference: set[str]) -> dict[str, Any]:
    # An empty reference set reports novelty 1.0, which reads as a triumph and means
    # nothing. Fail loudly instead.
    if not reference:
        raise ValueError("empty reference set: novelty against it would be vacuous")
    absent = len(generated - reference)
    return {
        "reference_size": len(reference),
        "generated_distinct": len(generated),
        "absent": absent,
        "novelty": round(absent / len(generated), 4) if generated else None,
    }


def funnel(rows: list[dict[str, str]], reference: set[str]) -> list[dict[str, Any]]:
    """Where in our own pipeline novelty is actually lost."""

    stages = [
        ("all_generated", lambda r: True),
        ("valid_molecule", lambda r: r["raw_molecule_valid"] == "True"),
        ("exact_ugi_forward_verified", lambda r: r["exact_l1"] == "True"),
        ("chemically_admitted", lambda r: r["terminal_chemical_admitted"] == "True"),
        ("oracle_qualified", lambda r: r["terminal_chemical_admitted"] == "True"
                                       and r["oracle_scored"] == "True"),
    ]
    out = []
    for name, predicate in stages:
        products = {r["canonical_product"] for r in rows if predicate(r)}
        products.discard("")
        record = novelty(products, reference)
        record["stage"] = name
        out.append(record)
    return out


def component_novelty(rows: list[dict[str, str]], registries: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    """Distinct-component novelty, and the draw-weighted version that reconciles it
    with product novelty."""

    column = {"amine_head": "canonical_amine", "aldehyde": "canonical_aldehyde",
              "isocyanide": "canonical_isocyanide"}
    valid = [r for r in rows if r["raw_molecule_valid"] == "True"]
    out: dict[str, Any] = {"draws": len(valid), "by_role": {}}
    for role, col in column.items():
        distinct = {r[col] for r in valid}
        corpus_blocks = registries["corpus"][role]
        measured_blocks = registries["measured"][role]
        reuse = sum(1 for r in valid if r[col] in corpus_blocks)
        counts = Counter(r[col] for r in valid)
        top20 = sum(v for _, v in counts.most_common(20))
        out["by_role"][role] = {
            "distinct_generated": len(distinct),
            "measured_registry_size": len(measured_blocks),
            "corpus_registry_size": len(corpus_blocks),
            "distinct_novel_vs_measured": len(distinct - measured_blocks),
            "distinct_novelty_vs_measured": round(len(distinct - measured_blocks) / len(distinct), 4),
            "distinct_novel_vs_corpus_registry": len(distinct - corpus_blocks),
            "distinct_novelty_vs_corpus_registry": round(len(distinct - corpus_blocks) / len(distinct), 4),
            "share_of_draws_reusing_a_corpus_block": round(reuse / len(valid), 4),
            "top20_share_of_draws": round(top20 / len(valid), 4),
        }
    all_three = sum(
        1 for r in valid
        if r["canonical_amine"] in registries["corpus"]["amine_head"]
        and r["canonical_aldehyde"] in registries["corpus"]["aldehyde"]
        and r["canonical_isocyanide"] in registries["corpus"]["isocyanide"]
    )
    out["draws_with_all_three_components_from_corpus_registry"] = all_three
    out["share_of_draws_with_all_three"] = round(all_three / len(valid), 4)
    out["reconciliation_note"] = (
        "Distinct-component novelty exceeds product novelty because novel components are "
        "each used rarely while registry components are used constantly. A product can only "
        "enter the corpus if all three of its components are registry members and that exact "
        "triple was enumerated."
    )
    return out


def distance_novelty(generated: list[str], reference: list[str], seed: int) -> dict[str, Any]:
    """Exact match is binary. This is the continuous version, plus the count of
    perfect-similarity pairs that are not the same molecule."""

    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rng = random.Random(seed)
    reference_set = set(reference)

    with rdBase.BlockLogs():
        reference_fps = []
        for smiles in reference:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                reference_fps.append(generator.GetFingerprint(molecule))

        sample = rng.sample(generated, min(NN_SAMPLE, len(generated)))
        nearest: list[float] = []
        twins_not_identical = 0
        for smiles in sample:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            best = max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(molecule),
                                                          reference_fps))
            nearest.append(best)
            if best >= 1.0 and smiles not in reference_set:
                twins_not_identical += 1

        scaffold_sample = rng.sample(generated, min(SCAFFOLD_SAMPLE, len(generated)))
        reference_scaffolds = set()
        for smiles in rng.sample(reference, min(20000, len(reference))):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                reference_scaffolds.add(
                    MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "(acyclic)")
        generated_scaffolds = []
        for smiles in scaffold_sample:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                generated_scaffolds.append(
                    MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "(acyclic)")

    nearest.sort()

    def pct(fraction: float) -> float:
        return round(nearest[min(int(fraction * len(nearest)), len(nearest) - 1)], 4)

    novel_scaffolds = sum(1 for s in generated_scaffolds if s not in reference_scaffolds)
    return {
        "sampled": len(nearest),
        "sample_seed": seed,
        "nearest_neighbour_tanimoto": {
            "min": round(nearest[0], 4), "p10": pct(0.10), "p25": pct(0.25),
            "median": pct(0.50), "p75": pct(0.75), "max": round(nearest[-1], 4),
        },
        "fraction_below": {
            str(t): round(sum(1 for x in nearest if x < t) / len(nearest), 4)
            for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        },
        "perfect_similarity_but_different_molecule": twins_not_identical,
        "perfect_similarity_share": round(
            sum(1 for x in nearest if x >= 1.0) / len(nearest), 4),
        "fingerprint_limitation": (
            "Binary folded Morgan fingerprints encode presence of local environments, not "
            "their counts, so alkyl homologs of different chain length can score 1.0. Chain "
            "length is a real potency determinant for ionizable lipids, so a perfect score "
            "here is a metric artifact rather than chemical equivalence."
        ),
        "murcko_scaffold_novelty": {
            "sampled": len(generated_scaffolds),
            "novel": novel_scaffolds,
            "novelty": round(novel_scaffolds / len(generated_scaffolds), 4) if generated_scaffolds else None,
            "distinct_reference_scaffolds": len(reference_scaffolds),
            "distinct_generated_scaffolds": len(set(generated_scaffolds)),
            "interpretation": (
                "Product scaffold novelty is close to uninformative for this chemistry. A "
                "lipid's Murcko scaffold is essentially its head ring, so the whole reference "
                "set collapses to a small number of scaffolds."
            ),
        },
    }


def canonicalisation_check(generated: list[str], reference: list[str], seed: int) -> dict[str, Any]:
    """Guard against the failure mode where two pipelines canonicalise differently and
    novelty comes out spuriously high."""

    from rdkit import Chem, rdBase

    rng = random.Random(seed)

    def canon(smiles: str) -> str | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        Chem.RemoveStereochemistry(molecule)
        return Chem.MolToSmiles(molecule)

    with rdBase.BlockLogs():
        gen_sample = rng.sample(generated, min(5000, len(generated)))
        ref_sample = rng.sample(reference, min(5000, len(reference)))
        gen_stable = sum(1 for s in gen_sample if canon(s) == s)
        ref_stable = sum(1 for s in ref_sample if canon(s) == s)
        gen_stereo = sum(1 for s in gen_sample if "@" in s or "/" in s or "\\" in s)
        ref_stereo = sum(1 for s in ref_sample if "@" in s or "/" in s or "\\" in s)
        recanon_generated = {canon(s) for s in generated} - {None}
        recanon_reference = {canon(s) for s in reference} - {None}

    absent = len(recanon_generated - recanon_reference)
    return {
        "generated_already_canonical": f"{gen_stable}/{len(gen_sample)}",
        "reference_already_canonical": f"{ref_stable}/{len(ref_sample)}",
        "generated_with_stereo_markers": gen_stereo,
        "reference_with_stereo_markers": ref_stereo,
        "novelty_after_forcing_both_through_one_canonicaliser": (
            round(absent / len(recanon_generated), 4) if recanon_generated else None),
        "verdict": (
            "Exact-match novelty is not a canonicalisation or stereochemistry artifact when "
            "this value matches the raw string comparison."
        ),
    }


def agile_block_reach(repo: Path, panel_paths: dict[str, str]) -> dict[str, Any]:
    """AGILE fixed a curated component library and enumerated its whole cross-product,
    so 'absent from their library' means exactly 'uses a block they did not have'.

    Both sides must be stereo-free before comparison. Our pipeline is constitutional and
    emits no stereochemistry; the AGILE ledger retains it. Comparing raw strings makes an
    oleyl isocyanide look like a new building block and inflates the reach statistic.
    """

    from rdkit import Chem, rdBase

    def canon(smiles: str) -> str | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        Chem.RemoveStereochemistry(molecule)
        return Chem.MolToSmiles(molecule)

    rows = read_csv_gz(repo / AGILE_LIB)
    blocks: dict[str, set[str]] = {
        "amine_head": set(), "oxoester_aldehyde_body_tail": set(), "isocyanide_tail": set()}
    products: set[str] = set()
    with rdBase.BlockLogs():
        for row in rows:
            components = json.loads(row["candidate_routes_json"])[0]["components"]
            for role in blocks:
                blocks[role].add(canon(components[role]))
            products.add(canon(row["canonical_product_smiles"]))
        for role in blocks:
            blocks[role].discard(None)
        products.discard(None)

    out: dict[str, Any] = {
        "agile_block_set": {role: len(v) for role, v in blocks.items()},
        "agile_library_size": len(products),
        "library_is_full_cross_product": len(products) == (
            len(blocks["amine_head"]) * len(blocks["oxoester_aldehyde_body_tail"])
            * len(blocks["isocyanide_tail"])),
        "interpretation": (
            "The library is the complete cross-product of a curated block set, so product-level "
            "overlap carries no per-molecule chemical judgement. Absence means the design uses at "
            "least one building block outside the curated set."
        ),
        "stereo_note": (
            "Comparison is stereo-free on both sides. Raw string comparison inflates the reach "
            "statistic because our constitutional pipeline emits no stereochemistry."
        ),
        "panels": {},
    }
    for tag, path in panel_paths.items():
        full = repo / path
        if not full.is_file():
            continue
        panel = [json.loads(line) for line in gzip.open(full, "rt")]
        by_cohort: dict[str, list[int]] = {}
        inside = 0
        roles_beyond: Counter[str] = Counter()
        with rdBase.BlockLogs():
            for record in panel:
                if canon(record["canonical_product"]) in products:
                    inside += 1
                beyond = sorted(role for role, smiles in record["components"].items()
                                if canon(smiles) not in blocks[role])
                for role in beyond:
                    roles_beyond[role] += 1
                bucket = by_cohort.setdefault(record.get("cohort", "?"), [0, 0])
                bucket[0] += 1
                if beyond:
                    bucket[1] += 1
        reach = sum(v[1] for v in by_cohort.values())
        out["panels"][tag] = {
            "n": len(panel),
            "products_inside_agile_library": inside,
            "uses_a_block_agile_never_had": reach,
            "reach_fraction": round(reach / len(panel), 4) if panel else None,
            "by_cohort": {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_cohort.items())},
            "role_that_goes_beyond": dict(roles_beyond.most_common()),
        }
    return out


def panel_novelty(repo: Path, references: dict[str, set[str]]) -> dict[str, Any]:
    path = repo / PANEL_V4
    if not path.is_file():
        return {"status": "absent"}
    panel = [json.loads(line) for line in gzip.open(path, "rt")]
    by_cohort: dict[str, dict[str, int]] = {}
    for record in panel:
        cohort = record.get("cohort", "?")
        bucket = by_cohort.setdefault(cohort, {"n": 0, "absent_from_full_corpus": 0,
                                               "absent_from_train_fold": 0})
        bucket["n"] += 1
        product = record["canonical_product"]
        if product not in references["full_corpus"]:
            bucket["absent_from_full_corpus"] += 1
        if product not in references["train_fold"]:
            bucket["absent_from_train_fold"] += 1
    total = len(panel)
    absent_full = sum(v["absent_from_full_corpus"] for v in by_cohort.values())
    return {
        "panel": "v4, superseded by v5 selection",
        "n": total,
        "by_cohort": dict(sorted(by_cohort.items())),
        "absent_from_full_corpus": absent_full,
        "novelty": round(absent_full / total, 4) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase1/ugi_novelty_audit_v1"))
    parser.add_argument("--skip-distance", action="store_true",
                        help="omit the nearest-neighbour pass, which builds 112k fingerprints")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from forge.design.flow.ugi_bounded_hybrid_route_cascade import (
        atomic_write, canonical_json_bytes, sha256_file, sha256_payload,
    )

    references = reference_sets(repo)
    registries = component_registries(repo)
    rows = read_csv_gz(repo / LEDGER)
    generated = {r["canonical_product"] for r in rows if r["raw_molecule_valid"] == "True"}
    generated.discard("")

    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "purpose": "Measure novelty of generated designs against every reference set that matters. Selects nothing.",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "inputs": {
            name: {"path": path, "sha256": sha256_file(repo / path)}
            for name, path in (
                ("training_corpus", CORPUS), ("corpus_fold_assignments", ASSIGN),
                ("rescoring_ledger", LEDGER), ("measured_products", MEASURED),
                ("agile_virtual_library", AGILE_LIB),
            )
        },
        "corpus_fold_structure": {
            "train": len(references["train_fold"]),
            "full_corpus": len(references["full_corpus"]),
            "why_folds_exist": (
                "The corpus is fold-split at the component-family level, not at the product "
                "level. Random product-level splitting of a combinatorial enumeration leaks "
                "badly, because a held-out product shares two of three components with training "
                "products. Family-level folds make the held-out set genuinely held out, and "
                "supply the held-out likelihood used for generator checkpoint selection."
            ),
            "novelty_denominator": (
                "Novelty is defined against the train fold. Measuring against all folds counts "
                "calibration and heldout products as training data, which understates novelty."
            ),
        },
        "generator_novelty": {
            name: novelty(generated, reference) for name, reference in references.items()
        },
        "novelty_funnel_vs_train_fold": funnel(rows, references["train_fold"]),
        "novelty_funnel_vs_full_corpus": funnel(rows, references["full_corpus"]),
        "component_novelty": component_novelty(rows, registries),
        "canonicalisation_check": canonicalisation_check(
            sorted(generated), sorted(references["full_corpus"]), SAMPLE_SEED),
        "panel_v4_novelty": panel_novelty(repo, references),
        "agile_curated_block_reach": agile_block_reach(repo, {
            "v4": PANEL_V4,
            "v5": "results/phase1/ugi_prediction_cohort_panel_v5/prediction_cohort_panel.jsonl.gz",
        }),
        "nonclaims": [
            "Novelty is not evidence of activity.",
            "Exact-match novelty saturates near 1.0 for competent distribution learners and is a memorisation alarm, not a measure of how far the model reached.",
            "Absence from a corpus we enumerated ourselves is weak in both directions: present may only mean the enumeration was thorough, absent may only mean it had a filter.",
            "Scaffold novelty is close to uninformative for lipids.",
        ],
    }

    if not args.skip_distance:
        content["distance_novelty_vs_train_fold"] = distance_novelty(
            sorted(generated), sorted(references["train_fold"]), SAMPLE_SEED)

    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(out / "result.json", canonical_json_bytes(result))

    print(json.dumps({
        "generator_novelty": {k: v["novelty"] for k, v in content["generator_novelty"].items()},
        "funnel_vs_train_fold": {r["stage"]: r["novelty"]
                                 for r in content["novelty_funnel_vs_train_fold"]},
        "panel_v4": content["panel_v4_novelty"].get("novelty"),
    }, indent=2))


if __name__ == "__main__":
    main()
