"""Can post-hoc decomposition recover the factorization the model generated?

Half of the origin-channel ablation is answerable without training anything. Take the
admitted FORGE products, discard the generated origin map, and run the same qualified Ugi
reverse decomposition that was used to decompose the enumerated library. Then ask two
questions the paper's contribution 1 rests on:

  1. is the post-hoc decomposition unique, ambiguous, or absent?
  2. when it succeeds, does it recover the SAME components the model generated?
"""
import csv, gzip, json, random, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.data.r1_prime_audit import (  # noqa: E402
    compile_reactions, load_reaction_definitions, decompose_structure,
)

CFG = json.loads((REPO / "configs/route/m0_09_agile_virtual_ugi3_capability.json").read_text())
scope = CFG["scope"]
defs = load_reaction_definitions(
    (REPO / "data/vendor/qualified_reactions_v1.json",),
    expected_count=1,
    role_policy_overrides=scope["role_policy_overrides"],
)
reaction = compile_reactions(defs)[0]
MAX_REV, MAX_FWD = scope["max_reverse_outcomes_per_product"], scope["max_forward_outcomes_per_candidate"]

led = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
with gzip.open(led, "rt", newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]

# one row per distinct product, carrying the components the generator emitted
by_product = {}
for r in rows:
    by_product.setdefault(r["canonical_product"], r)
products = sorted(by_product)
print(f"distinct admitted products: {len(products)}")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = random.Random(20260815)
sample = rng.sample(products, min(N, len(products)))
print(f"sampling {len(sample)} with seed 20260815\n")

status = Counter()
match = Counter()
t0 = time.time()
for i, smi in enumerate(sample, 1):
    row = by_product[smi]
    generated = (row["canonical_amine"], row["canonical_aldehyde"], row["canonical_isocyanide"])
    audit_row = {
        "r0_structure_id": f"FORGE-{i:06d}",
        "canonical_isomeric_smiles": smi,
        "observed_source_ids": "forge_generated",
        "study_split_groups_json": "{}",
    }
    try:
        cands = decompose_structure(audit_row, "source_study", (reaction,), MAX_REV, MAX_FWD)
    except Exception as exc:  # noqa: BLE001
        status["error:" + type(exc).__name__] += 1
        continue
    n = len(cands)
    status["zero" if n == 0 else ("unique" if n == 1 else "ambiguous")] += 1
    if n:
        tuples = {tuple(c.reactant_smiles) for c in cands}
        match["generated_among_candidates" if generated in tuples else "generated_NOT_among_candidates"] += 1
        if n == 1:
            match["unique_and_matches_generated" if generated in tuples
                  else "unique_but_DIFFERENT_from_generated"] += 1
    if i % 250 == 0:
        print(f"  {i}/{len(sample)}  {time.time()-t0:.0f}s  {dict(status)}")

n = sum(status.values())
print(f"\n=== post-hoc decomposition of {n} generated products ({time.time()-t0:.0f}s) ===")
for k, v in status.most_common():
    print(f"  {k:34s} {v:6d}  {v/n:.4f}")
print("\n=== agreement with the generated factorization ===")
tot = sum(v for k, v in match.items() if k.startswith("generated_"))
for k, v in match.most_common():
    print(f"  {k:38s} {v:6d}" + (f"  {v/tot:.4f}" if k.startswith("generated_") else ""))
