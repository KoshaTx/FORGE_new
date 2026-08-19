"""Broad prior versus reallocated proposal: does support-aware tilting concentrate the search?

Proposition 4.2 guarantees no architecture receives zero mass. It does not guarantee that a
finite sample stays diverse. This measures that directly at the matched 16,384-draw budget.
"""
import csv, gzip, json, math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LED = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
ASG = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
LIB = REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"

def effective_count(counter):
    """Exponential of Shannon entropy: the number of equally-common types that would give
    the same diversity. Robust to a long tail of singletons in a way a raw count is not."""
    n = sum(counter.values())
    if n == 0:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counter.values() if c)
    return math.exp(h)

def simpson_effective(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return 1.0 / sum((c / n) ** 2 for c in counter.values())

with gzip.open(ASG, "rt", newline="") as fh:
    rows = list(csv.DictReader(fh))
corpus_all = {r["canonical_product_smiles"] for r in rows}
train_only = {r["canonical_product_smiles"] for r in rows if r["primary_product_fold"] == "train"}

with gzip.open(LIB, "rt", newline="") as fh:
    lib = {r["canonical_product_smiles"] for r in csv.DictReader(fh)}

with gzip.open(LED, "rt", newline="") as fh:
    led = list(csv.DictReader(fh))

arms = defaultdict(list)
for r in led:
    if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]:
        arms[r["arm_id"]].append(r)

report = {}
for arm, rs in sorted(arms.items()):
    prods = [r["canonical_product"] for r in rs]
    uniq = set(prods)
    heads = Counter(r["canonical_amine"] for r in rs)
    alds = Counter(r["canonical_aldehyde"] for r in rs)
    isos = Counter(r["canonical_isocyanide"] for r in rs)
    progs = Counter(r["program_sha256"] for r in rs)
    tiers = Counter(r["authority_tier"] for r in rs if r["authority_tier"])
    report[arm] = {
        "admitted_rows": len(rs),
        "distinct_products": len(uniq),
        "unique_fraction": len(uniq) / len(rs),
        "novelty": {
            "absent_from_full_corpus_112386": sum(1 for s in uniq if s not in corpus_all) / len(uniq),
            "absent_from_train_fold_66464": sum(1 for s in uniq if s not in train_only) / len(uniq),
            "absent_from_enumerated_library_12276": sum(1 for s in uniq if s not in lib) / len(uniq),
        },
        "distinct_components": {"amine": len(heads), "aldehyde": len(alds), "isocyanide": len(isos)},
        "effective_components_shannon": {
            "amine": round(effective_count(heads), 2),
            "aldehyde": round(effective_count(alds), 2),
            "isocyanide": round(effective_count(isos), 2),
        },
        "effective_components_simpson": {
            "amine": round(simpson_effective(heads), 2),
            "aldehyde": round(simpson_effective(alds), 2),
            "isocyanide": round(simpson_effective(isos), 2),
        },
        "distinct_morphology_programs": len(progs),
        "effective_morphology_programs_shannon": round(effective_count(progs), 1),
        "authority_tier_mix": dict(tiers),
    }

print(json.dumps(report, indent=1))

b, s = report["broad_prior"], report["support_enriched"]
print("\n=== ratio, support_enriched / broad_prior ===")
print(f"  distinct products               {s['distinct_products']/b['distinct_products']:.3f}")
print(f"  unique fraction                 {s['unique_fraction']/b['unique_fraction']:.3f}")
for k in ("amine", "aldehyde", "isocyanide"):
    print(f"  effective {k:11s} (Shannon) {s['effective_components_shannon'][k]/b['effective_components_shannon'][k]:.3f}")
print(f"  effective morphology programs   {s['effective_morphology_programs_shannon']/b['effective_morphology_programs_shannon']:.3f}")
for k in ("absent_from_full_corpus_112386", "absent_from_train_fold_66464", "absent_from_enumerated_library_12276"):
    print(f"  novelty {k:38s} {s['novelty'][k]:.4f} vs {b['novelty'][k]:.4f}")
