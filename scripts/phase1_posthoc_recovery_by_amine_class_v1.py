import csv, gzip, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO/"src"))
from rdkit import Chem, rdBase
from forge.data.r1_prime_audit import compile_reactions, load_reaction_definitions, decompose_structure
CFG=json.loads((REPO/"configs/route/m0_09_agile_virtual_ugi3_capability.json").read_text()); sc=CFG["scope"]
rx=compile_reactions(load_reaction_definitions((REPO/"data/vendor/qualified_reactions_v1.json",),
    expected_count=1, role_policy_overrides=sc["role_policy_overrides"]))[0]
MR,MF=sc["max_reverse_outcomes_per_product"],sc["max_forward_outcomes_per_candidate"]
with gzip.open(REPO/"results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz","rt",newline="") as fh:
    rows=[r for r in csv.DictReader(fh) if r["terminal_chemical_admitted"]=="True" and r["canonical_product"]]
by={}
for r in rows: by.setdefault(r["canonical_product"], r)

PRIM=Chem.MolFromSmarts("[NX3;H2]")      # primary amine N
SEC =Chem.MolFromSmarts("[NX3;H1]")      # secondary amine N
def head_class(smi):
    with rdBase.BlockLogs(): m=Chem.MolFromSmiles(smi)
    if m is None: return "unparsed"
    if m.HasSubstructMatch(PRIM): return "has_primary_NH2"
    if m.HasSubstructMatch(SEC):  return "secondary_NH_only"
    return "no_NH"

res=defaultdict(Counter)
for i,(smi,row) in enumerate(sorted(by.items()),1):
    ar={"r0_structure_id":f"F{i}","canonical_isomeric_smiles":smi,
        "observed_source_ids":"forge","study_split_groups_json":"{}"}
    try: n=len(decompose_structure(ar,"source_study",(rx,),MR,MF))
    except Exception: continue
    res["recovered" if n else "unrecoverable"][head_class(row["canonical_amine"])]+=1

print(f"{'amine head class':22s} {'recovered':>12s} {'unrecoverable':>14s} {'% unrecoverable':>16s}")
allc=set()
for g in res: allc|=set(res[g])
for c in sorted(allc):
    a,b=res["recovered"][c],res["unrecoverable"][c]
    print(f"{c:22s} {a:12d} {b:14d} {b/(a+b) if a+b else 0:16.4f}")
ta,tb=sum(res["recovered"].values()),sum(res["unrecoverable"].values())
print(f"{'TOTAL':22s} {ta:12d} {tb:14d} {tb/(ta+tb):16.4f}")
