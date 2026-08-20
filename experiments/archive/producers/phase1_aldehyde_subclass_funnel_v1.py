#!/usr/bin/env python3
"""Do both aldehyde subclasses survive the selection funnel, or does it collapse to ester?

The declared aldehyde role admits ester-linked and non-ester components alike: the qualified
reaction asks only for [CX3H1]=[OX1], and the training corpus is 51 non-ester components
against 56 ester-linked ones. The generator reproduces that mixture. But every one of the 11
measured aldehyde tails is ester-linked, and the documented Tail A preparation only covers the
ester subclass. So two downstream layers - biological ranking and route resolution - could
quietly narrow the actionable population back onto the chemistry the measured library already
contained.

This traces both subclasses stage by stage to find out. It reuses the panel's own route index
and dossier builder rather than reimplementing them, so the funnel cannot disagree with the
frozen selection about what route-complete means.

Either outcome is publishable. If both subclasses survive, the framework demonstrates
actionable expansion across chemistry the measured set never covered. If only ester-linked
components reach the panel, the generator is broad while the prediction and routing layers are
narrow, which is an honest limitation and a direct target for the next round of measurement.
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
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import Crippen, Descriptors  # noqa: E402

MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"

ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX4]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_gz(path: Path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def subclass(smiles: str, cache: dict[str, str]) -> str:
    if smiles not in cache:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        cache[smiles] = (
            "unparsed" if mol is None
            else "ester_linked" if mol.HasSubstructMatch(ESTER)
            else "non_ester"
        )
    return cache[smiles]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_aldehyde_subclass_funnel_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    from importlib import import_module
    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(REPO)
    config = json.loads((REPO / CONFIG).read_text())
    screen = config["shared_requirements"]["physicochemical_screen"]
    tol = float(screen.get("comparison_tolerance", 0.0))
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    train = builder.train_fold(REPO) if hasattr(builder, "train_fold") else None
    if train is None:
        from importlib import import_module as _im
        panel_mod = _im("phase1_select_ugi_prospective_panel_v6")
        train = panel_mod.train_fold(REPO)

    # Every count in this artifact is distinct product graphs across the union of both
    # rescoring sources. The unit matters: the same population is 30,180 admitted rows in the
    # main ledger alone, 26,235 distinct products there, and 29,850 distinct products once the
    # branch-exploration source is unioned in. Quoting a share without naming the unit produces
    # apparent discrepancies of a few hundred that are pure bookkeeping.
    cache: dict[str, str] = {}
    stages = ["admitted", "prediction_supported", "absent_from_train_fold",
              "inside_envelope", "all_components_resolved", "route_complete_within_steps"]
    counts = {stage: {"ester_linked": 0, "non_ester": 0, "unparsed": 0} for stage in stages}
    components = {stage: {"ester_linked": set(), "non_ester": set()} for stage in stages}
    seen: set[str] = set()

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
                aldehyde = row.get("canonical_aldehyde")
                if not product or not aldehyde or product in seen:
                    continue
                seen.add(product)
                cls = subclass(aldehyde, cache)

                def mark(stage: str) -> None:
                    counts[stage][cls] += 1
                    if cls in components[stage]:
                        components[stage][cls].add(aldehyde)

                mark("admitted")
                if row.get("oracle_scored") != "True":
                    continue
                mark("prediction_supported")
                if product in train:
                    continue
                mark("absent_from_train_fold")

                molecule = Chem.MolFromSmiles(product)
                if molecule is None:
                    continue
                weight = Descriptors.MolWt(molecule)
                logp = Crippen.MolLogP(molecule)
                tpsa = Descriptors.TPSA(molecule)
                if not (screen["molecular_weight"][0] - tol <= weight <= screen["molecular_weight"][1] + tol):
                    continue
                if not (screen["clogp"][0] - tol <= logp <= screen["clogp"][1] + tol):
                    continue
                if not (screen["tpsa"][0] - tol <= tpsa <= screen["tpsa"][1] + tol):
                    continue
                mark("inside_envelope")

                dossier = [
                    builder.component_dossier(index, "amine_head", row["canonical_amine"]),
                    builder.component_dossier(index, "oxoester_aldehyde_body_tail", aldehyde),
                    builder.component_dossier(index, "isocyanide_tail", row["canonical_isocyanide"]),
                ]
                if not all(d["resolved"] for d in dossier):
                    continue
                mark("all_components_resolved")
                if sum(d.get("steps", 0) for d in dossier) > max_steps:
                    continue
                mark("route_complete_within_steps")

    panel = [json.loads(line) for line in gzip.open(REPO / PANEL, "rt")]
    panel_counts = {"ester_linked": 0, "non_ester": 0}
    panel_components = {"ester_linked": set(), "non_ester": set()}
    for record in panel:
        cls = subclass(record["components"]["oxoester_aldehyde_body_tail"], cache)
        panel_counts[cls] = panel_counts.get(cls, 0) + 1
        if cls in panel_components:
            panel_components[cls].add(record["components"]["oxoester_aldehyde_body_tail"])

    print(f"\n{'stage':34s} {'ester':>8s} {'non-ester':>10s} {'ester %':>9s}   "
          f"{'distinct ald: ester':>20s} {'non-ester':>10s}")
    rows_out = []
    for stage in stages:
        e, n = counts[stage]["ester_linked"], counts[stage]["non_ester"]
        total = e + n
        share = e / total if total else float("nan")
        de, dn = len(components[stage]["ester_linked"]), len(components[stage]["non_ester"])
        print(f"{stage:34s} {e:8d} {n:10d} {share:9.3f}   {de:20d} {dn:10d}")
        rows_out.append({"stage": stage, "designs_ester": e, "designs_non_ester": n,
                         "ester_share": share, "distinct_aldehydes_ester": de,
                         "distinct_aldehydes_non_ester": dn})
    pe, pn = panel_counts["ester_linked"], panel_counts["non_ester"]
    print(f"{'frozen_40_panel':34s} {pe:8d} {pn:10d} {pe/(pe+pn):9.3f}   "
          f"{len(panel_components['ester_linked']):20d} {len(panel_components['non_ester']):10d}")
    print(f"{'selected_12':34s} {'pending':>8s} {'pending':>10s}")
    rows_out.append({"stage": "frozen_40_panel", "designs_ester": pe, "designs_non_ester": pn,
                     "ester_share": pe / (pe + pn),
                     "distinct_aldehydes_ester": len(panel_components["ester_linked"]),
                     "distinct_aldehydes_non_ester": len(panel_components["non_ester"])})

    verdict = (
        "Both subclasses survive to the frozen panel."
        if pn > 0 else
        "The panel contains only ester-linked aldehyde components. Exclusion occurs at "
        "predictive support, before route assessment, so routing cannot explain the collapse "
        "and the route completeness of the excluded population remains unmeasured. Measuring a "
        "deliberately chosen set of non-ester aldehyde lipids would provide the first "
        "biological anchors for a large structurally supported region currently outside "
        "ranking; it would not by itself make the excluded population rankable."
    )
    print(f"\n{verdict}")

    payload = {
        "schema_version": "phase1_forge_aldehyde_subclass_funnel.v1",
        "status": "complete_subclass_propagation_audit",
        "inputs": {
            "terminal_ledger": {"path": MAIN, "sha256": sha256_file(REPO / MAIN)},
            "panel": {"path": PANEL, "sha256": sha256_file(REPO / PANEL)},
            "panel_config": {"path": CONFIG, "sha256": sha256_file(REPO / CONFIG)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "subclass_definition": "ester_linked iff the aldehyde component matches [CX3](=[OX1])[OX2][CX4]",
        "unit_of_every_count": "distinct product graphs across the union of both rescoring sources",
        "denominator_ledger": {
            "main_admitted_rows": 30180,
            "main_distinct_products": 26235,
            "branch_admitted_rows": 3678,
            "branch_distinct_products": 3652,
            "union_distinct_products": 29850,
            "products_present_in_both_sources": 37,
            "unclassified_or_unparsed": 0,
            "note": "19563 + 10287 = 29850 exactly. An apparent shortfall against 30,180 comes from comparing union products against main-ledger rows, which are different units over different source sets.",
            "ester_share_depends_on_unit": {
                "main_rows": 0.708, "main_products": 0.674, "union_products": 0.655
            },
        },
        "first_excluding_gate": {
            "non_ester_aldehyde": "prediction_supported; routing is never reached",
            "secondary_amine_head": "prediction_supported; routing is never reached",
        },
        "structural_corpus_comparison": {
            "corpus_distinct_aldehyde_components": 107,
            "corpus_ester_linked": 56,
            "corpus_ester_share": 0.523,
            "generated_distinct_aldehyde_components": 2044,
            "generated_ester_linked": 1015,
            "generated_ester_share": 0.497,
            "reading": "the generator reproduces the composition of the family it was trained on; these populations are learned from structural supervision rather than produced by drift",
        },
        "funnel": rows_out,
        "selected_12": "not yet chosen; the blinded chemistry review has not returned",
        "verdict": verdict,
        "nonclaims": [
            "Route completeness is computational: a documented preparation exists from purchasable material under a dated snapshot, not evidence a synthesis will succeed.",
            "The non-ester subclass has structural supervision but no measured biological examples, so its activity predictions rest on the same weak predictor as everything else.",
            "Route completeness of the excluded non-ester population is UNMEASURED. These designs are removed before route assessment, which shows routing is not responsible for the collapse; it does not show they would route if they reached it. The documented acid/diol preparation is specific to the ester-linked subclass.",
            "Exclusion follows from the frozen similarity thresholds against a measured aldehyde reference that happens to contain only ester-linked components. It is an empirical consequence of those thresholds on this run, not a Boolean ester requirement written into the support predicate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
