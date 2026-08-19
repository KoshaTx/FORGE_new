"""Why does the frozen inverse fail on secondary-amine-head products?

Three candidate explanations, distinguished by where in decompose_structure the failure occurs:
  (a) the reverse SMARTS finds no cut at all      -> possible non-identifiability
  (b) it finds a cut, and the ROLE POLICY rejects -> policy choice, not identifiability
  (c) it finds a cut that fails forward check     -> inverse/forward inconsistency

FORGE already holds a witness tuple for every one of these products that forward-reconstructs
exactly (that is what admission means), so (a) alone would be surprising.
"""
import csv, gzip, json, sys
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO/"src"))
from rdkit import Chem, rdBase
from forge.data.r1_prime_audit import compile_reactions, load_reaction_definitions, decompose_structure

CFG=json.loads((REPO/"configs/route/m0_09_agile_virtual_ugi3_capability.json").read_text()); sc=CFG["scope"]
defs=load_reaction_definitions((REPO/"data/vendor/qualified_reactions_v1.json",), expected_count=1,
                               role_policy_overrides=sc["role_policy_overrides"])
rx=compile_reactions(defs)[0]
MR,MF=sc["max_reverse_outcomes_per_product"],sc["max_forward_outcomes_per_candidate"]

print("declared reactant roles:")
for i,role in enumerate(defs[0].reactant_roles):
    print(f"  [{i}] {role.name}")
    for attr in ("required_handles","forbidden_patterns","handles","forbidden"):
        v=getattr(role, attr, None)
        if v: print(f"        {attr}: {v}")

SEC=Chem.MolFromSmarts("[NX3;H1]"); PRIM=Chem.MolFromSmarts("[NX3;H2]")
def sec_head(s):
    with rdBase.BlockLogs(): m=Chem.MolFromSmiles(s)
    return m is not None and not m.HasSubstructMatch(PRIM) and m.HasSubstructMatch(SEC)

with gzip.open(REPO/"results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz","rt",newline="") as fh:
    rows=[r for r in csv.DictReader(fh) if r["terminal_chemical_admitted"]=="True" and r["canonical_product"]]
by={}
for r in rows: by.setdefault(r["canonical_product"], r)
fails=[(s,r) for s,r in by.items() if sec_head(r["canonical_amine"])]
print(f"\nsecondary-amine-head distinct products: {len(fails)}")

rej=Counter(); zero=0; n=0
for i,(smi,row) in enumerate(fails[:600],1):
    ar={"r0_structure_id":f"S{i}","canonical_isomeric_smiles":smi,
        "observed_source_ids":"forge","study_split_groups_json":"{}"}
    local=Counter()
    try: c=decompose_structure(ar,"source_study",(rx,),MR,MF,local)
    except Exception as e: rej["EXC:"+type(e).__name__]+=1; continue
    n+=1
    if not c: zero+=1
    rej.update(local)
print(f"\nprobed {n} secondary-head products, {zero} returned zero candidates")
print("rejection reasons recorded inside decompose_structure:")
for k,v in rej.most_common(12):
    print(f"  {k:46s} {v}")
