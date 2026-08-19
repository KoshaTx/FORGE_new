#!/usr/bin/env python3
"""How much of the frozen panel depends on the ranking statistic?

The v6 config asserts, pre-registration, that ranking by the ensemble mean instead of by lcb90
"changes 9 of 34 selected candidates". This checks that claim by running the frozen selector twice
against the same eligible pool and the same diversity rule, changing only the sort key.

Reusing the frozen selector matters. An independent reimplementation would silently differ on the
eligibility gates, the greedy diversity order, the novelty tie-break or the head cap, and any of
those would contaminate the comparison. Here the only difference between the two runs is the scalar
the pool is sorted by, so the delta is attributable to the statistic and nothing else.

The lcb90 run is also a regression test: it must reproduce the forty frozen candidates exactly. If it
does not, the comparison is not trustworthy and the script fails rather than reporting a number.

This selects nothing and locks nothing. It produces a sensitivity artifact and, for the candidates
that only the mean-ranked variant reaches, a dossier of the kind used for the frozen panel.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SELECTOR = REPO / "scripts/phase1_select_ugi_prospective_panel_v6.py"
CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"
FROZEN = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"


def load_selector():
    spec = importlib.util.spec_from_file_location("panel_v6", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["panel_v6"] = module
    spec.loader.exec_module(module)
    return module


def frozen_panel() -> list[dict[str, Any]]:
    with gzip.open(REPO / FROZEN, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def high_arm(sel, pool: list[dict[str, Any]], config: dict[str, Any],
             key: str) -> list[dict[str, Any]]:
    """Re-sort the pool by `key` descending, then run the frozen selection rule unchanged."""
    ordered = sorted(pool, key=lambda c: (-c[key], c["canonical_product"]))
    high, _low = sel.select(ordered, config)
    return high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/phase1/forge_rank_statistic_sensitivity_v1/result.json")
    args = parser.parse_args()

    sel = load_selector()
    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(REPO)
    config = json.loads((REPO / CONFIG).read_text())

    pool = sel.eligible_pool(REPO, config, builder, index)

    # Regression: the frozen sort key must reproduce the frozen forty exactly.
    lcb_high = high_arm(sel, pool, config, "lcb90")
    lcb_products = {c["canonical_product"] for c in lcb_high}
    frozen = frozen_panel()
    frozen_high = {r["canonical_product"] for r in frozen if r["arm"] == "HIGH"}
    reproduced = lcb_products == frozen_high
    if not reproduced:
        raise SystemExit(
            f"frozen HIGH arm not reproduced: {len(lcb_products & frozen_high)} of "
            f"{len(frozen_high)} shared; the comparison would not be trustworthy")

    mean_high = high_arm(sel, pool, config, "oracle_mean")
    mean_products = {c["canonical_product"] for c in mean_high}
    only_mean = [c for c in mean_high if c["canonical_product"] not in lcb_products]

    candidates = []
    for order, candidate in enumerate(sorted(only_mean, key=lambda c: -c["oracle_mean"]), start=1):
        row = candidate["row"]
        dossier = [builder.component_dossier(index, role, row[column]) for role, column in
                   (("amine_head", "canonical_amine"),
                    ("oxoester_aldehyde_body_tail", "canonical_aldehyde"),
                    ("isocyanide_tail", "canonical_isocyanide"))]
        candidates.append({
            "id": f"N{order:02d}",
            "product": candidate["canonical_product"],
            "oracle_mean": candidate["oracle_mean"],
            "lcb90": candidate["lcb90"],
            "tier": candidate["authority_tier"],
            "agile_novel": candidate["agile_novel"],
            "steps": sum(d.get("steps", 0) for d in dossier),
            "resolved": all(d["resolved"] for d in dossier),
            "dossier": dossier,
        })

    out = {
        "schema_version": "phase1_forge_rank_statistic_sensitivity.v1",
        "status": "sensitivity_and_backup_shortlist_complete",
        "purpose": "quantify how much of the frozen panel depends on the ranking statistic, and "
                   "provide route-resolved backup candidates",
        "eligible_pool": len(pool),
        "frozen_panel_reproduced_exactly": reproduced,
        "high_arm_changed": len(only_mean),
        "config_predicted_change": 9,
        "candidates": candidates,
        "nonclaims": [
            "This selects nothing and locks no panel.",
            "Supplier counts are a 3 August 2026 snapshot, not live availability.",
            "Route completeness is computational and is not evidence a synthesis will succeed.",
            "These candidates carry no biological measurement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=False))
    print(f"eligible pool {len(pool)}, frozen HIGH arm reproduced: {reproduced}")
    print(f"ranking by the ensemble mean changes {len(only_mean)} of {len(frozen_high)} "
          f"(config predicted 9)")
    print(f"wrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
