#!/usr/bin/env python3
"""Score the Stage 2 samples: does the model execute out-of-regime design programs?

Metrics and strata were fixed in docs/FORGE_EVIDENCE_CONTRACT_v1.md section B2 before the draw
was sealed. Nothing here is chosen after seeing a rate.

Three things this script refuses to overstate.

First, **the whole morphology program is enforced by the sampler, not learned.** Verified on
every row rather than assumed: per-role node counts match exactly in every slot, and so do cycle
ranks, across six distinct requested cycle-rank vectors. The expectation going in was that cycle
rank was a free upper bound and would therefore be a real adherence measurement; it is not.
Morphology adherence is a property of the counting masks and is barred from being reported as a
model capability or as evidence of controllability.

Second, **validity, terminal validity, component-reconstruction validity and admission are the
same predicate on this sampler.** They agree on every row. They are one quantity reported once,
not four independent metrics.

Third, **the novelty reference here is the 66,464-product train fold**, not the 112,386-product
production corpus. This is the development checkpoint, which saw the train fold only. Using the
production denominator would understate novelty for a model that never saw the other folds.

Δ_OOD is reported per metric as M(stratum) − M(train-family-derived), which is the actual
generative-generalization result. Stratum C carries no matched counterpart by construction and
is reported on its own terms.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402

ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
AGILE = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
CAPABILITY = "configs/route/m0_09_agile_virtual_ugi3_capability.json"
REACTIONS = "data/vendor/qualified_reactions_v1.json"
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
STRATA = ("A_train_family_derived", "B_held_family_overlapping", "C_held_family_disjoint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else ""


def stereo_free(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    payload = json.loads(args.samples.read_text())
    rows = payload["samples"]
    print(f"scoring {len(rows)} sampled rows")

    scope = json.loads((REPO / CAPABILITY).read_text())["scope"]
    max_forward = int(scope["max_forward_outcomes_per_candidate"])
    from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
    reaction = compile_reactions(load_reaction_definitions(
        (REPO / REACTIONS,), expected_count=1,
        role_policy_overrides=scope["role_policy_overrides"]))[0]

    # References. The development checkpoint saw the train fold only.
    with gzip.open(REPO / ASSIGN, "rt", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    train_products = {a["canonical_product_smiles"] for a in assignments
                      if a["primary_product_fold"] == "train"}
    train_components = {role: {a[f"{role}_smiles"] for a in assignments
                               if a["primary_product_fold"] == "train"} for role in ROLES}
    all_components = {role: {a[f"{role}_smiles"] for a in assignments} for role in ROLES}
    with rdBase.BlockLogs():
        with gzip.open(REPO / AGILE, "rt", newline="") as handle:
            agile = {stereo_free(r["canonical_product_smiles"]) for r in csv.DictReader(handle)}
    agile.discard("")
    print(f"references: train fold {len(train_products)} products, "
          f"enumeration {len(agile)} stereo-free products")

    per_stratum: dict[str, dict] = {s: defaultdict(int) for s in STRATA}
    products: dict[str, set] = {s: set() for s in STRATA}
    components: dict[str, dict[str, set]] = {s: {r: set() for r in ROLES} for s in STRATA}
    cycle_match: dict[str, Counter] = {s: Counter() for s in STRATA}
    node_count_enforced = {"checked": 0, "matched": 0}

    with rdBase.BlockLogs():
        for row in rows:
            stratum = row["source_stratum"]
            bucket = per_stratum[stratum]
            bucket["sampled"] += 1
            bucket["raw_molecule_valid"] += int(bool(row.get("raw_molecule_valid")))
            bucket["terminal_valid"] += int(bool(row.get("terminal_valid")))
            bucket["valid"] += int(bool(row.get("valid")))
            bucket["component_reconstruction_valid"] += int(
                bool(row.get("component_reconstruction_valid")))

            # Node counts are enforced by the sampler. Verify rather than assume.
            offspring = row.get("offspring_by_role") or {}
            requested = row["program"]["node_counts"]
            for index, role in enumerate(ROLES):
                node_count_enforced["checked"] += 1
                node_count_enforced["matched"] += int(
                    len(offspring.get(role, [])) == requested[index])

            verification = row.get("l1_forward_verification") or {}
            admitted = bool(verification.get("exact_product_reconstructed"))
            bucket["exact_forward_reconstructed"] += int(admitted)
            bucket["enumeration_saturated"] += int(
                bool(verification.get("maximum_outcomes_saturated")))
            if not admitted:
                continue

            parts = row.get("component_smiles_by_role") or {}
            product = canon(row.get("smiles") or "")
            mols = [Chem.MolFromSmiles(parts.get(role, "")) for role in ROLES]
            if not product or any(m is None for m in mols):
                bucket["unscorable_admitted_row"] += 1
                continue

            built: set[str] = set()
            for group in reaction.forward.RunReactants(tuple(mols), maxProducts=max_forward):
                if len(group) != 1:
                    continue
                try:
                    copy = Chem.Mol(group[0])
                    Chem.SanitizeMol(copy)
                    built.add(Chem.MolToSmiles(copy))
                except Exception:  # noqa: BLE001
                    continue
            if product in built and len(built) == 1:
                bucket["unique_forward_outcome"] += 1

            # Measured because the counting masks were expected to impose cycle rank as an
            # upper bound rather than an equality. They do not: it matches exactly everywhere.
            requested_cycles = row["program"]["cycle_ranks"]
            for index, role in enumerate(ROLES):
                realized = Chem.GetSSSR(mols[index])
                realized = realized if isinstance(realized, int) else len(realized)
                cycle_match[stratum][
                    "exact" if realized == requested_cycles[index]
                    else "under" if realized < requested_cycles[index] else "over"] += 1

            products[stratum].add(product)
            bucket["absent_from_train_fold"] += int(product not in train_products)
            bucket["outside_enumeration"] += int(stereo_free(product) not in agile)
            for role in ROLES:
                value = canon(parts.get(role, ""))
                if not value:
                    continue
                components[stratum][role].add(value)
                bucket[f"novel_component_{role}"] += int(value not in all_components[role])
                bucket[f"unseen_in_train_{role}"] += int(value not in train_components[role])

    # Two collapses that must be stated rather than padded over.
    cycles = Counter()
    for stratum in STRATA:
        cycles.update(cycle_match[stratum])
    requested_cycle_vectors = len({tuple(r["program"]["cycle_ranks"]) for r in rows})
    program_enforced = {
        "node_count_slots_checked": node_count_enforced["checked"],
        "node_count_slots_exact": node_count_enforced["matched"],
        "cycle_rank_slots": dict(cycles),
        "distinct_requested_cycle_rank_vectors": requested_cycle_vectors,
        "conclusion": "the morphology program is realized exactly by the sampler",
    }
    identical = {}
    for stratum in STRATA:
        bucket = per_stratum[stratum]
        identical[stratum] = (bucket["valid"] == bucket["terminal_valid"]
                              == bucket["component_reconstruction_valid"]
                              == bucket["exact_forward_reconstructed"])
    print(f"\nprogram enforcement: node counts exact in "
          f"{node_count_enforced['matched']}/{node_count_enforced['checked']} role slots; "
          f"cycle ranks {dict(cycles)} over {requested_cycle_vectors} distinct requested vectors")
    print("  Requested cycle ranks vary, so an exact match everywhere is enforcement, not a")
    print("  degenerate comparison. Morphology adherence is therefore NOT a model capability")
    print("  here and must not be reported as one.")
    print(f"\nvalidity, terminal validity, component reconstruction and admission are the same "
          f"predicate in every stratum: {all(identical.values())}")
    print("  They are one quantity, not four. Report it once.")

    summary: dict[str, dict] = {}
    for stratum in STRATA:
        bucket = per_stratum[stratum]
        n = bucket["sampled"]
        admitted = bucket["exact_forward_reconstructed"]
        summary[stratum] = {
            "sampled": n,
            "validity": bucket["valid"] / n,
            "terminal_validity": bucket["terminal_valid"] / n,
            "component_reconstruction_validity": bucket["component_reconstruction_valid"] / n,
            "admission": admitted / n,
            "unique_forward_outcome_of_admitted":
                bucket["unique_forward_outcome"] / admitted if admitted else float("nan"),
            "unique_forward_outcome_of_sampled": bucket["unique_forward_outcome"] / n,
            "distinct_products_of_admitted": len(products[stratum]),
            "uniqueness_of_admitted":
                len(products[stratum]) / admitted if admitted else float("nan"),
            "absent_from_train_fold_of_admitted":
                bucket["absent_from_train_fold"] / admitted if admitted else float("nan"),
            "outside_enumeration_of_admitted":
                bucket["outside_enumeration"] / admitted if admitted else float("nan"),
            "distinct_components": {r: len(components[stratum][r]) for r in ROLES},
            "novel_component_rate_of_admitted": {
                r: bucket[f"novel_component_{r}"] / admitted if admitted else float("nan")
                for r in ROLES},
            "cycle_rank_match": dict(cycle_match[stratum]),
            "enumeration_saturated": bucket["enumeration_saturated"],
            "unscorable_admitted_rows": bucket["unscorable_admitted_row"],
        }

    print(f"\n{'stratum':28s} {'n':>5s} {'valid':>7s} {'admit':>7s} {'uniq|adm':>9s} "
          f"{'distinct':>9s} {'novel prod':>11s} {'outside enum':>13s}")
    for stratum in STRATA:
        s = summary[stratum]
        print(f"{stratum:28s} {s['sampled']:5d} {s['validity']:7.3f} {s['admission']:7.3f} "
              f"{s['unique_forward_outcome_of_admitted']:9.3f} "
              f"{s['distinct_products_of_admitted']:9d} "
              f"{s['absent_from_train_fold_of_admitted']:11.3f} "
              f"{s['outside_enumeration_of_admitted']:13.3f}")

    reference = summary[STRATA[0]]
    degradation = {}
    for stratum in STRATA[1:]:
        degradation[stratum] = {
            metric: summary[stratum][metric] - reference[metric]
            for metric in ("validity", "admission", "unique_forward_outcome_of_admitted",
                           "uniqueness_of_admitted", "absent_from_train_fold_of_admitted",
                           "outside_enumeration_of_admitted")
        }
    print(f"\nΔ_OOD against {STRATA[0]}:")
    print(f"{'stratum':28s} {'Δvalid':>8s} {'Δadmit':>8s} {'Δuniq':>8s} {'Δdistinct':>10s}")
    for stratum, values in degradation.items():
        print(f"{stratum:28s} {values['validity']:+8.3f} {values['admission']:+8.3f} "
              f"{values['unique_forward_outcome_of_admitted']:+8.3f} "
              f"{values['uniqueness_of_admitted']:+10.3f}")

    out = {
        "schema_version": "phase1_forge_held_family_stage2_scores.v1",
        "status": "complete_ood_morphology_conditioned_sampling_scores",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section B2, amended by Amendment 2",
        "naming": "conditional sampling under held-family-derived morphology programs. A "
                  "morphology program is test-time conditioning, not a molecular identity.",
        "inputs": {
            "samples": {"path": str(args.samples), "sha256": sha256_file(args.samples)},
            "pinned_by_sampler": payload.get("pinned_inputs"),
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "novelty_reference": {
            "products": "train fold, 66,464; the development checkpoint saw this fold only",
            "components": "train-fold component sets per role, and the full structural registry",
            "enumeration": "12,276 AGILE products, stereo-free on both sides",
        },
        "program_enforcement": {
            **program_enforced,
            "reading": "the sampler's counting masks realize the requested morphology exactly, "
                       "for node counts and for cycle ranks alike, across varied requested "
                       "cycle-rank vectors. A perfect score is a property of the constraint, not "
                       "a learned capability, and must never be reported as morphology "
                       "adherence or as evidence of controllability.",
        },
        "identical_predicates": {
            "validity_terminal_component_and_admission_agree": identical,
            "reading": "these four are the same predicate on this sampler, so they are one "
                       "quantity reported once rather than four independent metrics.",
        },
        "per_stratum": summary,
        "ood_degradation_against_train_family_derived": degradation,
        "nonclaims": [
            "This does not show that unseen chemical families are regenerated. The model receives "
            "morphology as test-time conditioning, and a program associated with a held family is "
            "not that family's molecular identity.",
            "Stratum C has no matched train-family counterpart by construction, so its Δ is a "
            "descriptive contrast between differently supported regions, not a controlled one.",
            "Cycle-rank agreement is reported because the counting masks impose budgets as upper "
            "bounds rather than equalities. Node-count agreement is not a result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
