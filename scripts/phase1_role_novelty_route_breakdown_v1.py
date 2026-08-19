#!/usr/bin/env python3
"""Where does off-registry chemistry sit, and does it survive route assessment?

The manuscript reports that 74.4% of admitted designs carry at least one component outside the
424-component training registry. "At least one" hides which role, and a reader can reasonably
suspect the answer is the tails: tails are a homologous series where novelty is cheap, whereas the
amine head is the dominant activity determinant and the role F.4 gives no disconnection rule, so an
unseen head has to be purchasable outright or the design fails routing. If that suspicion is right
the actionable output is catalogue heads with recombined tails, and the headline number oversells.

This resolves it by splitting off-registry status per role at both ends of the funnel:

    admitted designs           -> which role carries the novelty
    reaching route assessment  -> whether novel components still resolve

The route standard here is the one the accessibility claim uses, component_dossier(...)["resolved"],
meaning purchasable or reducible to purchasable material under a forward-verified disconnection. It
is NOT the strict adjudicated-closure standard from the bounded hybrid cascade; under that stricter
standard off-registry components are almost all "not_in_graded_development_ledger", which measures
ledger coverage rather than chemistry and would answer a different question.

Gates and population are replicated from phase1_open_world_actionability_funnel_v1 so the assessed
denominator matches the 1,549 the manuscript cites. This computes a breakdown; it selects nothing.
"""

from __future__ import annotations

import argparse, csv, gzip, hashlib, json, platform, sys, time
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def read_csv_gz(path: Path):
    with gzip.open(path, "rt", newline="") as h:
        yield from csv.DictReader(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=REPO / "results/phase1/forge_role_novelty_route_breakdown_v1/result.json")
    args = ap.parse_args()
    started = time.time()

    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen, Descriptors

    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(REPO)
    panel_mod = import_module("phase1_select_ugi_prospective_panel_v6")
    train = panel_mod.train_fold(REPO)
    config = json.loads((REPO / CONFIG).read_text())
    screen = config["shared_requirements"]["physicochemical_screen"]
    tol = float(screen.get("comparison_tolerance", 0.0))
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])

    registry = {r: set() for r in ROLES}
    for row in read_csv_gz(REPO / ASSIGN):
        registry["amine_head"].add(row["amine_head_smiles"])
        registry["oxoester_aldehyde_body_tail"].add(row["oxoester_aldehyde_body_tail_smiles"])
        registry["isocyanide_tail"].add(row["isocyanide_tail_smiles"])

    # Two admitted populations, because the manuscript cites both and they must not drift:
    # the main rescoring ledger alone, which is where the abstract's figure comes from, and the
    # union of both ledgers, which is the funnel population behind the 1,549 assessed.
    adm = {pop: {r: Counter() for r in ROLES} for pop in ("main", "union")}
    adm_n = Counter()
    adm_any = Counter()
    assessed = {r: Counter() for r in ROLES}     # role -> (off, resolved) counts at route assessment
    assessed_n = 0
    complete_by_off = Counter()                  # n off-registry components -> [assessed, complete]
    seen: set[str] = set()

    with rdBase.BlockLogs():
        for source in (MAIN, BRANCH):
            path = REPO / source
            if not path.exists():
                continue
            for row in read_csv_gz(path):
                if row.get("terminal_chemical_admitted") != "True":
                    continue
                product, aldehyde = row.get("canonical_product"), row.get("canonical_aldehyde")
                if not product or not aldehyde or product in seen:
                    continue
                seen.add(product)
                parts = {"amine_head": row["canonical_amine"],
                         "oxoester_aldehyde_body_tail": aldehyde,
                         "isocyanide_tail": row["canonical_isocyanide"]}
                off = {r: parts[r] not in registry[r] for r in ROLES}

                pops = ("main", "union") if source == MAIN else ("union",)
                for pop in pops:
                    adm_n[pop] += 1
                    for r in ROLES:
                        adm[pop][r]["off" if off[r] else "on"] += 1
                    if any(off.values()):
                        adm_any[pop] += 1

                # --- replicate the funnel's gates up to route assessment ---
                if row.get("oracle_scored") != "True" or product in train:
                    continue
                molecule = Chem.MolFromSmiles(product)
                if molecule is None:
                    continue
                if not all((
                    screen["molecular_weight"][0] - tol <= Descriptors.MolWt(molecule)
                    <= screen["molecular_weight"][1] + tol,
                    screen["clogp"][0] - tol <= Crippen.MolLogP(molecule)
                    <= screen["clogp"][1] + tol,
                    screen["tpsa"][0] - tol <= Descriptors.TPSA(molecule)
                    <= screen["tpsa"][1] + tol,
                )):
                    continue

                assessed_n += 1
                dossier = {d["role"]: d for d in
                           (builder.component_dossier(index, r, parts[r]) for r in ROLES)}
                for r in ROLES:
                    key = ("off" if off[r] else "on", bool(dossier[r]["resolved"]))
                    assessed[r][key] += 1
                n_off = sum(off.values())
                complete_by_off[(n_off, "assessed")] += 1
                if all(dossier[r]["resolved"] for r in ROLES):
                    steps = sum(dossier[r].get("steps", 0) for r in ROLES)
                    if steps <= max_steps:
                        complete_by_off[(n_off, "complete")] += 1

    def rate(c: Counter, tag: str) -> dict:
        res, unres = c[(tag, True)], c[(tag, False)]
        n = res + unres
        return {"n": n, "resolved": res, "rate": round(res / n, 4) if n else None}

    out = {
        "schema_version": "phase1_forge_role_novelty_route_breakdown.v1",
        "status": "complete",
        "question": "which role carries the off-registry novelty, and does it survive route assessment",
        "route_standard": "component_dossier resolved: purchasable, or reducible to purchasable "
                          "material under a forward-verified disconnection within the step budget. "
                          "This is the standard behind the pooled accessibility rate, not the "
                          "adjudicated-closure standard of the bounded hybrid cascade.",
        "registry": {"definition": "distinct components of the training corpus",
                     "sizes": {r: len(registry[r]) for r in ROLES},
                     "total": sum(len(v) for v in registry.values())},
        "admitted": {
            pop: {
                "definition": ("distinct admitted products of the main rescoring ledger; the "
                               "population behind the manuscript's headline off-registry figure"
                               if pop == "main" else
                               "distinct admitted products across the union of both rescoring "
                               "ledgers; the funnel population behind the 1,549 assessed"),
                "n_distinct_products": adm_n[pop],
                "any_role_off_registry": {"count": adm_any[pop],
                                          "fraction": round(adm_any[pop] / adm_n[pop], 4)},
                "per_role_off_registry": {
                    r: {"count": adm[pop][r]["off"],
                        "fraction": round(adm[pop][r]["off"] / adm_n[pop], 4)}
                    for r in ROLES},
            } for pop in ("main", "union")
        },
        "reaching_route_assessment": {
            "n": assessed_n,
            "per_role": {r: {"on_registry": rate(assessed[r], "on"),
                             "off_registry": rate(assessed[r], "off")} for r in ROLES},
            "by_number_of_off_registry_components": {
                str(k): {"assessed": complete_by_off[(k, "assessed")],
                         "route_complete": complete_by_off[(k, "complete")],
                         "rate": round(complete_by_off[(k, "complete")]
                                       / complete_by_off[(k, "assessed")], 4)
                         if complete_by_off[(k, "assessed")] else None}
                for k in range(4) if complete_by_off[(k, "assessed")]},
        },
        "inputs": {n: {"path": p, "sha256": sha256_file(REPO / p)}
                   for n, p in (("main", MAIN), ("branch", BRANCH), ("assignments", ASSIGN))
                   if (REPO / p).exists()},
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "nonclaims": [
            "Resolution is computational and is not evidence that any preparation will succeed.",
            "A component that does not resolve is not thereby unsynthesizable; it is unresolved "
            "against this evidence base within this step budget.",
            "Rates conditional on reaching route assessment; designs removed by earlier gates have "
            "unmeasured route status.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True))

    for pop in ("main", "union"):
        print(f"[{pop}] admitted {adm_n[pop]}   any role off-registry {adm_any[pop]} "
              f"({100*adm_any[pop]/adm_n[pop]:.1f}%)")
        for r in ROLES:
            print(f"   {r:<28s} {adm[pop][r]['off']:6d} "
                  f"({100*adm[pop][r]['off']/adm_n[pop]:5.1f}%)")
    print(f"\nreaching route assessment: {assessed_n}")
    for r in ROLES:
        on, off = rate(assessed[r], "on"), rate(assessed[r], "off")
        print(f"  {r:<28s} on {on['resolved']:4d}/{on['n']:<4d} "
              f"{100*on['rate']:5.1f}%    off {off['resolved']:4d}/{off['n']:<4d} "
              f"{100*off['rate']:5.1f}%" if off["rate"] is not None else "")
    print()
    for k in range(4):
        a = complete_by_off[(k, "assessed")]
        if a:
            print(f"  {k} off-registry components: {a:5d} assessed, "
                  f"{complete_by_off[(k,'complete')]:5d} route-complete "
                  f"({100*complete_by_off[(k,'complete')]/a:5.1f}%)")
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
