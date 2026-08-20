#!/usr/bin/env python3
"""What is actually in the aldehyde population, and does the proposal arm explain its shape?

Two corrections drive this audit.

First, "non-ester" is a negative bucket, not a chemical class. Probing the 1,029 distinct
aldehyde components that lack the measured ester motif shows 503 carry neither an ether nor a
ketone at all: they are plain chain aldehydes. Calling the whole bucket one subclass, and
suggesting that measuring a few members would anchor it, treats an ether-linked tail and a
plain fatty aldehyde as the same biology. They are not.

Second, the production pool is two matched arms and only one is the base generator. The
support-enriched arm was deliberately reweighted toward morphology likely to reach predictive
support, and every measured aldehyde is ester-containing, so that arm can raise ester
prevalence by design. Any statement about what the generator learned must come from the
broad-prior arm; the combined pool describes the candidate funnel, not the model.

Categories below come from probing the observed chemistry, not from a guessed taxonomy. The
linkage labels are multi-label because a component may carry more than one motif; an exclusive
summary is also produced under a precedence fixed here before any count is taken.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402

LEDGER = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
CORPUS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"

# Recorded here so the artifact carries its own definitions.
SMARTS = {
    "ester": "[CX3](=[OX1])[OX2][CX4]",
    "ether": "[CX4][OX2][CX4]",
    "ketone": "[CX4][CX3](=[OX1])[CX4]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "primary_amine": "[NX3;H2]",
    "secondary_amine": "[NX3;H1]",
}
# Fixed before counting. Exclusive labels are assigned by first match in this order.
PRECEDENCE = ("ester_linked", "ketone_and_ether", "ketone_containing", "ether_linked",
              "unlinked_chain_aldehyde")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_patterns() -> dict[str, Chem.Mol]:
    return {name: Chem.MolFromSmarts(sma) for name, sma in SMARTS.items()}


def label_aldehyde(smiles: str, pat: dict[str, Chem.Mol], cache: dict[str, dict]) -> dict:
    """Multi-label motifs plus one exclusive chemotype under the fixed precedence."""
    if smiles in cache:
        return cache[smiles]
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out = {"parsed": False, "exclusive": "unparsed", "motifs": []}
        cache[smiles] = out
        return out
    has = {name: mol.HasSubstructMatch(p) for name, p in pat.items()}
    motifs = [n for n in ("ester", "ether", "ketone", "alkene", "alkyne") if has[n]]
    if has["ester"]:
        exclusive = "ester_linked"
    elif has["ketone"] and has["ether"]:
        exclusive = "ketone_and_ether"
    elif has["ketone"]:
        exclusive = "ketone_containing"
    elif has["ether"]:
        exclusive = "ether_linked"
    else:
        exclusive = "unlinked_chain_aldehyde"
    out = {"parsed": True, "exclusive": exclusive, "motifs": motifs}
    cache[smiles] = out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=4, help="structures per category to print")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_aldehyde_chemotype_audit_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()
    pat = compile_patterns()
    cache: dict[str, dict] = {}

    with gzip.open(LEDGER, "rt", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]
    products: dict[str, dict[str, str]] = {}
    for row in rows:
        products.setdefault(row["canonical_product"], row)
    print(f"primary population: {len(products)} distinct admitted products "
          f"from {len(rows)} admitted rows")

    with rdBase.BlockLogs():
        def secondary_head(smiles: str) -> bool:
            mol = Chem.MolFromSmiles(smiles)
            return (mol is not None
                    and not mol.HasSubstructMatch(pat["primary_amine"])
                    and mol.HasSubstructMatch(pat["secondary_amine"]))

        # ---- component level
        components = sorted({r["canonical_aldehyde"] for r in products.values()})
        comp_label = {c: label_aldehyde(c, pat, cache) for c in components}

        # ---- product level, stratified by proposal arm
        by_arm: dict[str, Counter] = defaultdict(Counter)
        supported: Counter = Counter()
        arm_head: dict[str, Counter] = defaultdict(Counter)
        for row in products.values():
            chemotype = comp_label[row["canonical_aldehyde"]]["exclusive"]
            arm = row["arm_id"]
            by_arm[arm][chemotype] += 1
            by_arm["combined"][chemotype] += 1
            if row["oracle_scored"] == "True":
                supported[chemotype] += 1
            head = "secondary" if secondary_head(row["canonical_amine"]) else "primary_or_other"
            arm_head[arm][head] += 1
            arm_head["combined"][head] += 1

        # ---- corpus, same labels, both units
        with gzip.open(CORPUS, "rt", newline="") as fh:
            crows = list(csv.DictReader(fh))
        corpus_components = Counter()
        corpus_products = Counter()
        for row in crows:
            lab = label_aldehyde(row["oxoester_aldehyde_body_tail_smiles"], pat, cache)["exclusive"]
            corpus_products[lab] += 1
        for smiles in {r["oxoester_aldehyde_body_tail_smiles"] for r in crows}:
            corpus_components[label_aldehyde(smiles, pat, cache)["exclusive"]] += 1

    comp_counts = Counter(v["exclusive"] for v in comp_label.values())

    # ---- exact reconciliation, or refuse to report
    assert sum(comp_counts.values()) == len(components), "component labels do not partition"
    assert sum(by_arm["combined"].values()) == len(products), "product labels do not partition"

    print(f"\n{'chemotype':26s} {'corpus comp':>12s} {'gen comp':>9s} "
          f"{'broad':>8s} {'enriched':>9s} {'combined':>9s} {'supported':>10s}")
    table = []
    for name in PRECEDENCE:
        rec = {
            "chemotype": name,
            "corpus_components": corpus_components[name],
            "corpus_products": corpus_products[name],
            "generated_components": comp_counts[name],
            "broad_prior_products": by_arm["broad_prior"][name],
            "support_enriched_products": by_arm["support_enriched"][name],
            "combined_products": by_arm["combined"][name],
            "prediction_supported_products": supported[name],
        }
        table.append(rec)
        print(f"{name:26s} {rec['corpus_components']:12d} {rec['generated_components']:9d} "
              f"{rec['broad_prior_products']:8d} {rec['support_enriched_products']:9d} "
              f"{rec['combined_products']:9d} {rec['prediction_supported_products']:10d}")

    b, e = by_arm["broad_prior"], by_arm["support_enriched"]
    bt, et = sum(b.values()), sum(e.values())
    print(f"\nester share of products   broad {b['ester_linked']/bt:.3f}   "
          f"enriched {e['ester_linked']/et:.3f}   "
          f"corpus {corpus_products['ester_linked']/len(crows):.3f}")
    print(f"secondary-head share      broad {arm_head['broad_prior']['secondary']/bt:.3f}   "
          f"enriched {arm_head['support_enriched']['secondary']/et:.3f}")

    print(f"\nmulti-label motif prevalence among generated components ({len(components)}):")
    multi = Counter()
    for v in comp_label.values():
        for motif in v["motifs"]:
            multi[motif] += 1
    for motif, n in multi.most_common():
        print(f"  {motif:10s} {n:5d}  {n/len(components):.3f}")

    rng = random.Random(20260815)
    samples = {}
    for name in PRECEDENCE:
        members = sorted(c for c, v in comp_label.items() if v["exclusive"] == name)
        samples[name] = rng.sample(members, min(args.sample, len(members)))
    print("\nrandom structures per chemotype, for manual inspection:")
    for name, members in samples.items():
        print(f"  {name}")
        for smiles in members:
            print(f"     {smiles}")

    payload = {
        "schema_version": "phase1_forge_aldehyde_chemotype_audit.v1",
        "status": "complete_chemotype_and_arm_stratified_audit",
        "inputs": {"terminal_ledger": {"path": str(LEDGER.relative_to(REPO)),
                                       "sha256": sha256_file(LEDGER)},
                   "structural_corpus": {"path": str(CORPUS.relative_to(REPO)),
                                         "sha256": sha256_file(CORPUS)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "smarts": SMARTS,
        "exclusive_precedence": list(PRECEDENCE),
        "precedence_fixed_before_counting": True,
        "reconciliation": {
            "distinct_products": len(products),
            "distinct_aldehyde_components": len(components),
            "product_labels_sum": sum(by_arm["combined"].values()),
            "component_labels_sum": sum(comp_counts.values()),
            "exact": True,
        },
        "table": table,
        "arm_shares": {
            "broad_prior_products": bt,
            "support_enriched_products": et,
            "ester_share_broad": b["ester_linked"] / bt,
            "ester_share_enriched": e["ester_linked"] / et,
            "ester_share_corpus_products": corpus_products["ester_linked"] / len(crows),
            "secondary_head_share_broad": arm_head["broad_prior"]["secondary"] / bt,
            "secondary_head_share_enriched": arm_head["support_enriched"]["secondary"] / et,
        },
        "multi_label_motifs": dict(multi),
        "inspection_samples": samples,
        "nonclaims": [
            "The bucket lacking the measured ester motif is not one chemical class: it spans plain chain aldehydes, ketone-containing and ether-linked chemotypes. Measuring one chemotype anchors that chemotype, not the bucket.",
            "Combined-pool composition mixes the base generator with a proposal deliberately tilted toward chemistry likely to reach predictive support. Only the broad-prior arm speaks to what the generator learned.",
            "Zero prediction-supported products in a chemotype is a measured conditional on this frozen run, not a property of the support predicate, which contains no motif test.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
