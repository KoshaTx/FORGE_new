#!/usr/bin/env python3
"""Is the target the only forward product of its own components, or merely one of them?

Chemical admission verifies two things: the target IS produced by the qualified forward
transform applied to the generated components, and the enumeration did not saturate its cap.
It does not verify that the target is produced ALONE. Those are different properties, and the
project has been using one word for both.

A 200-design sample of ketone-containing aldehydes put non-uniqueness at 17%. That sample
cannot say whether ketones are unusual, because it had no comparison. This is the full census
over every admitted design, plus the identical statistic on the 12,276-member enumerated AGILE
library, so that non-uniqueness can be attributed either to generated chemistry or to Ugi-3
lipid chemistry in general. Forward enumeration costs seconds at this scale, so sampling here
would be a choice rather than a constraint.

Where non-uniqueness occurs, the mechanism is recorded rather than guessed: the count of
distinct matches of each role's own reactant template in its own component. Two reactive amine
sites in a head, for instance, produce two regiochemical outcomes under the same reaction.

Prespecified in docs/FORGE_EVIDENCE_CONTRACT_v1.md section C, committed before this ran. The
enumeration cap is inherited from the frozen capability config and is not tuned. Disclosed
there: a 300-row head-of-ledger timing probe returned 249 of 300 unique before the rule was
written.
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
sys.path.insert(0, str(REPO / "scripts"))

from rdkit import Chem, rdBase  # noqa: E402

MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
AGILE = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
CAPABILITY = "configs/route/m0_09_agile_virtual_ugi3_capability.json"
REACTIONS = "data/vendor/qualified_reactions_v1.json"

ROLE_COLUMNS = ("canonical_amine", "canonical_aldehyde", "canonical_isocyanide")
ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX4]")
ETHER = Chem.MolFromSmarts("[CX4][OX2][CX4]")
KETONE = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[CX4]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_gz(path: Path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def chemotype(smiles: str, cache: dict[str, str]) -> str:
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            cache[smiles] = "unparsed"
        elif mol.HasSubstructMatch(ESTER):
            cache[smiles] = "ester_linked"
        elif mol.HasSubstructMatch(KETONE) and mol.HasSubstructMatch(ETHER):
            cache[smiles] = "ketone_and_ether"
        elif mol.HasSubstructMatch(KETONE):
            cache[smiles] = "ketone_containing"
        elif mol.HasSubstructMatch(ETHER):
            cache[smiles] = "ether_linked"
        else:
            cache[smiles] = "unlinked_chain_aldehyde"
    return cache[smiles]


def canon(smiles: str, cache: dict[str, str]) -> str:
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        cache[smiles] = Chem.MolToSmiles(mol) if mol is not None else ""
    return cache[smiles]


def stereo_free(smiles: str, cache: dict[str, str]) -> str:
    """Enumeration membership only, and both sides must be stripped.

    3,652 of the 12,276 enumerated products carry stereochemistry; our pipeline is
    constitutional and emits none. A raw string comparison therefore records every
    stereo-bearing library member as a design the library never contained, which inflates
    novelty. The frozen panel builder already strips both sides, and this matches it.
    """
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            cache[smiles] = ""
        else:
            Chem.RemoveStereochemistry(mol)
            cache[smiles] = Chem.MolToSmiles(mol)
    return cache[smiles]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_forward_outcome_census_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    scope = json.loads((REPO / CAPABILITY).read_text())["scope"]
    max_forward = int(scope["max_forward_outcomes_per_candidate"])
    from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
    reaction = compile_reactions(load_reaction_definitions(
        (REPO / REACTIONS,), expected_count=1,
        role_policy_overrides=scope["role_policy_overrides"]))[0]
    templates = [reaction.forward.GetReactantTemplate(i)
                 for i in range(reaction.forward.GetNumReactantTemplates())]
    print(f"forward transform with {len(templates)} reactant templates, "
          f"enumeration cap {max_forward}")

    agile: set[str] = set()
    canon_cache: dict[str, str] = {}
    free_cache: dict[str, str] = {}
    with rdBase.BlockLogs():
        for row in read_csv_gz(REPO / AGILE):
            value = stereo_free(row["canonical_product_smiles"], free_cache)
            if value:
                agile.add(value)
    print(f"enumerated AGILE library: {len(agile)} distinct stereo-free products")

    panel = [json.loads(line) for line in gzip.open(REPO / PANEL, "rt")]
    panel_by_product = {record["canonical_product"]: record for record in panel}

    def outcomes(amine: str, aldehyde: str, isocyanide: str, target: str) -> dict:
        mols = [Chem.MolFromSmiles(s) for s in (amine, aldehyde, isocyanide)]
        if any(m is None for m in mols):
            return {"status": "unparseable", "n_outcomes": 0, "sites": []}
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
        sites = [len(mol.GetSubstructMatches(template, uniquify=True))
                 for mol, template in zip(mols, templates, strict=True)]
        if target not in built:
            status = "target_not_reproduced"
        elif len(built) == 1:
            status = "unique"
        else:
            status = "non_unique"
        return {"status": status, "n_outcomes": len(built), "sites": sites,
                "saturated": len(built) >= max_forward,
                "competing": sorted(built - {target})}

    # ---------------------------------------------------------------- generated census
    census: dict[str, dict] = {}
    chem_cache: dict[str, str] = {}
    with rdBase.BlockLogs():
        for source in (MAIN, BRANCH):
            path = REPO / source
            if not path.exists():
                print(f"  (skipping absent source {source})")
                continue
            for row in read_csv_gz(path):
                if row.get("terminal_chemical_admitted") != "True":
                    continue
                product = row.get("canonical_product")
                if not product or product in census:
                    continue
                target = canon(product, canon_cache)
                result = outcomes(row["canonical_amine"], row["canonical_aldehyde"],
                                  row["canonical_isocyanide"], target)
                result["prediction_supported"] = row.get("oracle_scored") == "True"
                result["inside_enumeration"] = stereo_free(product, free_cache) in agile
                result["chemotype"] = chemotype(row["canonical_aldehyde"], chem_cache)
                result["in_frozen_40"] = product in panel_by_product
                census[product] = result

    total = len(census)
    print(f"\ngenerated census: {total} distinct admitted products")

    def summarize(rows: list[dict]) -> dict:
        status = Counter(r["status"] for r in rows)
        n = len(rows)
        return {"designs": n,
                "unique": status["unique"],
                "non_unique": status["non_unique"],
                "target_not_reproduced": status["target_not_reproduced"],
                "unique_rate": status["unique"] / n if n else float("nan")}

    everything = list(census.values())
    strata: dict[str, dict] = {"all_admitted": summarize(everything)}
    for label, predicate in (
        ("prediction_supported", lambda r: r["prediction_supported"]),
        ("not_prediction_supported", lambda r: not r["prediction_supported"]),
        ("inside_enumeration", lambda r: r["inside_enumeration"]),
        ("outside_enumeration", lambda r: not r["inside_enumeration"]),
        ("prediction_supported_and_outside", lambda r: r["prediction_supported"]
            and not r["inside_enumeration"]),
        ("frozen_40", lambda r: r["in_frozen_40"]),
    ):
        strata[label] = summarize([r for r in everything if predicate(r)])

    by_chemotype = {name: summarize([r for r in everything if r["chemotype"] == name])
                    for name in sorted({r["chemotype"] for r in everything})}

    print(f"\n{'stratum':34s} {'designs':>8s} {'unique':>8s} {'non-uniq':>9s} {'rate':>7s}")
    for label, rec in strata.items():
        print(f"{label:34s} {rec['designs']:8d} {rec['unique']:8d} "
              f"{rec['non_unique']:9d} {rec['unique_rate']:7.3f}")
    print(f"\n{'aldehyde chemotype':34s} {'designs':>8s} {'unique':>8s} {'non-uniq':>9s} {'rate':>7s}")
    for label, rec in by_chemotype.items():
        print(f"{label:34s} {rec['designs']:8d} {rec['unique']:8d} "
              f"{rec['non_unique']:9d} {rec['unique_rate']:7.3f}")

    # ------------------------------------------------------- closed-world comparison
    agile_rows: list[dict] = []
    with rdBase.BlockLogs():
        for row in read_csv_gz(REPO / AGILE):
            routes = json.loads(row["candidate_routes_json"])
            if not routes:
                continue
            components = routes[0]["components"]
            target = canon(row["canonical_product_smiles"], canon_cache)
            result = outcomes(components["amine_head"],
                              components["oxoester_aldehyde_body_tail"],
                              components["isocyanide_tail"], target)
            result["ledger_unique_forward_products"] = routes[0].get("unique_forward_products")
            agile_rows.append(result)
    enumerated = summarize(agile_rows)
    agreement = sum(int((r["n_outcomes"] == r["ledger_unique_forward_products"]))
                    for r in agile_rows if r["ledger_unique_forward_products"] is not None)
    print(f"\nenumerated AGILE library, identical statistic:")
    print(f"{'enumerated_library':34s} {enumerated['designs']:8d} {enumerated['unique']:8d} "
          f"{enumerated['non_unique']:9d} {enumerated['unique_rate']:7.3f}")
    print(f"  recomputed outcome count agrees with the stored ledger field for "
          f"{agreement}/{len(agile_rows)} rows")

    # ----------------------------------------------------------- mechanism of ambiguity
    mechanism = Counter()
    for record in everything:
        if record["status"] != "non_unique":
            continue
        sites = record.get("sites") or [0, 0, 0]
        mechanism[f"amine_sites={sites[0]} aldehyde={sites[1]} isocyanide={sites[2]}"] += 1
    print("\nmechanism of non-uniqueness, distinct reactant-template matches per component:")
    for label, count in mechanism.most_common(8):
        print(f"  {label:52s} {count:6d}")
    saturated = sum(int(r.get("saturated", False)) for r in everything)
    print(f"  designs hitting the {max_forward}-outcome enumeration cap: {saturated}")

    # Every non-unique case above carries more than one reactive amine site, so the raw
    # inside-versus-outside gap could be nothing but a difference in head composition. The
    # honest comparison conditions on the mechanism: among designs whose head already offers
    # several reactive nitrogens, does membership of the enumerated library still matter?
    def conditional(rows: list[dict]) -> dict:
        multi = [r for r in rows if (r.get("sites") or [0])[0] > 1]
        single = [r for r in rows if (r.get("sites") or [0])[0] == 1]
        return {
            "designs": len(rows),
            "single_site_heads": len(single),
            "multi_site_heads": len(multi),
            "share_with_multi_site_head": len(multi) / len(rows) if rows else float("nan"),
            "non_unique_given_single_site_head":
                sum(r["status"] == "non_unique" for r in single) / len(single) if single else 0.0,
            "non_unique_given_multi_site_head":
                sum(r["status"] == "non_unique" for r in multi) / len(multi) if multi else float("nan"),
        }

    head_conditioned = {
        "inside_enumeration": conditional([r for r in everything if r["inside_enumeration"]]),
        "outside_enumeration": conditional([r for r in everything if not r["inside_enumeration"]]),
        "prediction_supported": conditional([r for r in everything if r["prediction_supported"]]),
        "enumerated_library": conditional(agile_rows),
    }
    print("\nconditioning on the mechanism, since every non-unique case has a multi-site head:")
    print(f"  {'population':26s} {'multi-site head':>16s} {'P(non-uniq|multi)':>19s}")
    for label, rec in head_conditioned.items():
        print(f"  {label:26s} {rec['share_with_multi_site_head']:16.3f} "
              f"{rec['non_unique_given_multi_site_head']:19.3f}")

    # ------------------------------------------------------------------- frozen panel
    flagged = []
    for product, record in census.items():
        if record["in_frozen_40"] and record["status"] != "unique":
            entry = panel_by_product[product]
            flagged.append({"candidate_id": entry["candidate_id"], "status": record["status"],
                            "n_outcomes": record["n_outcomes"], "sites": record["sites"],
                            "arm": entry.get("arm"), "canonical_product": product,
                            "amine_head": entry["components"]["amine_head"],
                            "competing_products": record["competing"],
                            "required_action": "exclude from the selected 12, or carry a named "
                                               "competing-outcome risk in its dossier"})
    covered = sum(1 for p in panel_by_product if p in census)
    print(f"\nfrozen 40: {covered}/40 members found in the census; "
          f"{len(flagged)} without a unique forward outcome")
    for entry in sorted(flagged, key=lambda e: e["candidate_id"]):
        print(f"  {entry['candidate_id']}  {entry['status']}  "
              f"{entry['n_outcomes']} outcomes  sites={entry['sites']}")

    payload = {
        "schema_version": "phase1_forge_forward_outcome_census.v1",
        "status": "complete_full_census_no_sampling",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section C, committed before this ran",
        "inputs": {
            "main_ledger": {"path": MAIN, "sha256": sha256_file(REPO / MAIN)},
            "branch_ledger": {"path": BRANCH, "sha256": sha256_file(REPO / BRANCH)},
            "panel": {"path": PANEL, "sha256": sha256_file(REPO / PANEL)},
            "agile_enumeration": {"path": AGILE, "sha256": sha256_file(REPO / AGILE)},
            "capability_scope": {"path": CAPABILITY, "sha256": sha256_file(REPO / CAPABILITY)},
            "qualified_reactions": {"path": REACTIONS, "sha256": sha256_file(REPO / REACTIONS)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "unit": "distinct product graphs across the union of both rescoring sources",
        "enumeration_cap": max_forward,
        "definitions": {
            "unique": "the target is the only distinct product enumerated from its own components",
            "non_unique": "the target is enumerated, together with at least one other product",
            "target_not_reproduced": "the target does not appear among enumerated outcomes",
        },
        "strata": strata,
        "by_aldehyde_chemotype": by_chemotype,
        "enumerated_library_baseline": enumerated,
        "enumerated_library_ledger_agreement": {
            "rows": len(agile_rows), "agreeing": agreement,
        },
        "mechanism_of_non_uniqueness": dict(mechanism.most_common(20)),
        "head_conditioned_comparison": head_conditioned,
        "designs_hitting_enumeration_cap": saturated,
        "frozen_40": {
            "members_found_in_census": covered,
            "without_unique_outcome": flagged,
        },
        "nonclaims": [
            "A unique forward outcome under the qualified transform is a statement about the "
            "enumerated reaction, not about competing reactivity under experimental conditions.",
            "Non-uniqueness is a property of the component set, not a defect of the generator: "
            "the enumerated library is measured here under the identical procedure precisely so "
            "that the two can be compared rather than conflated.",
            "A design whose target is one of several enumerated outcomes remains chemically "
            "admissible. It is not unambiguous, which is a weaker and different property.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
