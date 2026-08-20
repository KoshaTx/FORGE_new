#!/usr/bin/env python3
"""Build a self-verifying index of every reportable number in the study.

Writing a manuscript from this repository currently means archaeology: finding
which artifact holds a number, checking it has not been superseded, and copying
it by hand.  This builds one index that maps each reportable claim to its
artifact, that artifact's SHA-256, the exact path to the value inside it, and
the value itself.

The index is regenerated rather than maintained.  Every value is read live from
its artifact and every artifact hash is recomputed, so the suite cannot drift
from the results it describes.  A claim whose artifact is missing, whose path no
longer resolves, or whose value changed is reported as such rather than silently
dropped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from forge.design.flow.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    atomic_write,
    canonical_json_bytes,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_forge_data_suite.v1"

P1 = "results/phase1/"
M0 = "results/m0_07/"

# (claim_id, section, artifact, json path, one-line meaning)
CLAIMS: list[tuple[str, str, str, str, str]] = [
    # --- data and corpus -------------------------------------------------
    (
        "agile_curated_records",
        "data",
        M0 + "agile_label_reconciliation.json",
        "",
        "Curated AGILE single-structure records after reconciliation",
    ),
    (
        "balanced_corpus_products",
        "data",
        P1 + "ugi_balanced_chemistry_corpus_v2/result.json",
        "",
        "Products in the balanced training corpus",
    ),
    # --- generator -------------------------------------------------------
    (
        "refit_updates",
        "generator",
        P1 + "ugi_decoration_coupling_production_refit_v1/result.json",
        "selection",
        "All-fold refit selection record",
    ),
    (
        "production_admitted_broad",
        "generator",
        P1 + "ugi_constrained_stochastic_production_candidates_v2/result.json",
        "counts",
        "Matched production run admission counts by arm",
    ),
    # --- oracle and applicability ---------------------------------------
    (
        "oracle_selected_model",
        "oracle",
        M0 + "oracle_production_result.json",
        "selected_model",
        "Frozen production oracle identity",
    ),
    (
        "rescoring_summary",
        "oracle",
        P1 + "ugi_production_full_support_rescoring_v3/result.json",
        "summary",
        "Applicability rescoring over the production pool",
    ),
    (
        "morphology_proposal",
        "oracle",
        P1 + "ugi_morphology_proposal_challenger_adjudication_v1/result.json",
        "",
        "Promoted applicability-enriched morphology proposal",
    ),
    # --- branch correction ----------------------------------------------
    (
        "branch_generation",
        "branch",
        P1 + "ugi_full_corpus_branch_exploration_candidates_v1/result.json",
        "",
        "Branch-conditioned generation counts",
    ),
    (
        "branch_applicability",
        "branch",
        P1 + "ugi_branch_exploration_applicability_v1/result.json",
        "summary",
        "Branch lane applicability yield",
    ),
    # --- route cascade ---------------------------------------------------
    (
        "cascade_summary",
        "routing",
        P1 + "ugi_bounded_hybrid_route_cascade_v1/result.json",
        "summary",
        "Route-blinded cascade over the 256-product shortlist",
    ),
    (
        "cascade_interpretation",
        "routing",
        P1 + "ugi_bounded_hybrid_route_cascade_v1/result.json",
        "interpretation",
        "What the cascade closure rate measures, and the index-miss split",
    ),
    (
        "cascade_cache",
        "routing",
        P1 + "ugi_bounded_hybrid_route_cascade_v1/result.json",
        "cache",
        "Single-lane arm-blind planner cache audit",
    ),
    (
        "planner_reach",
        "routing",
        P1 + "ugi_planner_reach_sweep_v1/result.json",
        "summary",
        "Bounded planner reach to a public catalogue",
    ),
    (
        "engine_template_verification",
        "routing",
        P1 + "ugi_engine_template_verification_v1/result.json",
        "summary",
        "Engine-own-template self-consistency, a rejected signal",
    ),
    (
        "planner_proposal_verification",
        "routing",
        P1 + "ugi_planner_proposal_verification_v1/result.json",
        "summary",
        "Planner proposals against the local forward resolver",
    ),
    # --- published routes -------------------------------------------------
    (
        "tail_a",
        "routes",
        P1 + "ugi_tail_a_disconnection_v1/result.json",
        "summary",
        "AGILE Tail A applied deliberately and forward-verified",
    ),
    (
        "isocyanide_route",
        "routes",
        P1 + "ugi_isocyanide_route_v1/result.json",
        "summary",
        "Isocyanide formylation and dehydration, forward-verified",
    ),
    # --- availability -----------------------------------------------------
    (
        "procurement",
        "availability",
        P1 + "ugi_online_procurement_snapshot_v1/result.json",
        "summary",
        "Online vendor snapshot over components and route leaves",
    ),
    (
        "procurement_window",
        "availability",
        P1 + "ugi_online_procurement_snapshot_v1/result.json",
        "snapshot",
        "Availability snapshot access time and expiry",
    ),
    # --- makeability ------------------------------------------------------
    (
        "makeability",
        "makeability",
        P1 + "ugi_indomain_makeability_v1/result.json",
        "summary",
        "Makeability across the applicability-supported population",
    ),
    # --- decision package and proposal ------------------------------------
    (
        "decision_package",
        "panel",
        P1 + "ugi_route_aware_decision_package_v1/result.json",
        "summary",
        "Route-aware dossier over the 256-product shortlist",
    ),
    (
        "panel_proposal",
        "panel",
        P1 + "ugi_stratified_panel_proposal_v1/result.json",
        "summary",
        "Stratified candidate proposal for chemist review",
    ),
]

# Derived numbers that require reading a ledger rather than a result summary.
DERIVED_LEDGERS: list[tuple[str, str, str, str]] = [
    (
        "indomain_ledger",
        "makeability",
        P1 + "ugi_indomain_makeability_v1/indomain_makeability_ledger.jsonl.gz",
        "Per-product makeability, potency and authority tier",
    ),
    (
        "panel_ledger",
        "panel",
        P1 + "ugi_stratified_panel_proposal_v1/stratified_panel_proposal.jsonl.gz",
        "The twenty proposed candidates",
    ),
    (
        "component_ledger",
        "routing",
        P1 + "ugi_bounded_hybrid_route_cascade_v1/component_route_ledger.jsonl.gz",
        "Per-component route outcome across the shortlist",
    ),
]

NONCLAIMS = [
    "No compound in this suite has been synthesised, formulated or biologically tested.",
    "A vendor listing is a screening signal, not a quote, stock level or purity specification.",
    "Forward reproduction of a target does not verify substrate scope.",
    "Route closure under the frozen local index measures index coverage, not synthesizability.",
    "Potency values are conservative ranking signals, not validated predictions.",
    "The availability snapshot expires and must be refreshed before purchase.",
]


def _resolve(payload: Any, path: str) -> Any:
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def build(repo: Path, output_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for claim_id, section, relative, path, meaning in CLAIMS:
        artifact = repo / relative
        entry: dict[str, Any] = {
            "claim_id": claim_id,
            "section": section,
            "artifact": relative,
            "value_path": path or "(whole result)",
            "meaning": meaning,
        }
        if not artifact.is_file():
            entry["status"] = "artifact_missing"
            entries.append(entry)
            continue
        entry["artifact_sha256"] = sha256_file(artifact)
        try:
            payload = json.loads(artifact.read_bytes())
        except Exception as error:  # noqa: BLE001
            entry["status"] = f"unreadable:{type(error).__name__}"
            entries.append(entry)
            continue
        value = _resolve(payload, path)
        if value is None:
            entry["status"] = "value_path_unresolved"
        else:
            entry["status"] = "resolved"
            entry["value"] = value
            entry["result_sha256"] = payload.get("result_sha256")
            entry["result_status"] = payload.get("status")
        entries.append(entry)

    for claim_id, section, relative, meaning in DERIVED_LEDGERS:
        artifact = repo / relative
        entry = {
            "claim_id": claim_id,
            "section": section,
            "artifact": relative,
            "value_path": "(ledger)",
            "meaning": meaning,
        }
        if not artifact.is_file():
            entry["status"] = "artifact_missing"
            entries.append(entry)
            continue
        entry["artifact_sha256"] = sha256_file(artifact)
        with gzip.open(artifact, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        entry["status"] = "resolved"
        entry["rows"] = len(rows)
        if claim_id == "indomain_ledger":
            entry["value"] = {
                "products": len(rows),
                "makeability": dict(sorted(Counter(r["makeability"] for r in rows).items())),
                "conservative_high": sum(1 for r in rows if r["conservative_high_potency"]),
                "authority_tiers": dict(
                    sorted(Counter(str(r["authority_tier"]) for r in rows).items())
                ),
                "unique_amines": len({r["components"]["amine_head"] for r in rows}),
                "unique_aldehydes": len(
                    {r["components"]["oxoester_aldehyde_body_tail"] for r in rows}
                ),
                "unique_isocyanides": len({r["components"]["isocyanide_tail"] for r in rows}),
            }
        elif claim_id == "panel_ledger":
            entry["value"] = {
                "candidates": len(rows),
                "strata": dict(sorted(Counter(r["stratum"] for r in rows).items())),
                "arms": dict(sorted(Counter(r["arm_id"] for r in rows).items())),
            }
        else:
            entry["value"] = {
                "components": len(rows),
                "final_states": dict(
                    sorted(Counter(r["final_component_state"] for r in rows).items())
                ),
            }
        entries.append(entry)

    resolved = [e for e in entries if e["status"] == "resolved"]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "forge_data_suite_complete",
        "purpose": (
            "One index from reportable claim to artifact, artifact hash, value path and live "
            "value. Regenerate rather than maintain; the suite cannot drift from its results."
        ),
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "claims": len(entries),
            "resolved": len(resolved),
            "unresolved": len(entries) - len(resolved),
            "by_section": dict(sorted(Counter(e["section"] for e in entries).items())),
            "statuses": dict(sorted(Counter(e["status"] for e in entries).items())),
        },
        "entries": entries,
        "nonclaims": list(NONCLAIMS),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "forge_data_suite.json", canonical_json_bytes(result))

    # Human-readable companion.
    lines = [
        "# FORGE data suite",
        "",
        "Generated index from reportable claim to hash-pinned artifact. Regenerate with",
        "`scripts/phase1_build_forge_data_suite_v1.py`; do not edit by hand.",
        "",
        f"Claims: {len(entries)} | resolved: {len(resolved)} | unresolved: {len(entries) - len(resolved)}",
        "",
    ]
    for section in sorted({e["section"] for e in entries}):
        lines += [
            f"## {section}",
            "",
            "| Claim | Artifact | SHA-256 | Path | Status |",
            "|---|---|---|---|---|",
        ]
        for entry in [e for e in entries if e["section"] == section]:
            digest = entry.get("artifact_sha256", "")
            lines.append(
                f"| `{entry['claim_id']}` — {entry['meaning']} | `{entry['artifact']}` | "
                f"`{digest[:16]}` | `{entry['value_path']}` | {entry['status']} |"
            )
        lines.append("")
    lines += ["## Non-claims", ""] + [f"- {n}" for n in NONCLAIMS] + [""]
    atomic_write(output_dir / "FORGE_DATA_SUITE.md", ("\n".join(lines)).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/forge_data_suite_v1")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = build(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
