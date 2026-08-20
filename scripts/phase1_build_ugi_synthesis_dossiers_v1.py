#!/usr/bin/env python3
"""Write a chemist-readable synthesis dossier for each proposed candidate.

For every candidate: the product, its three exact Ugi components, and for each
component either the vendor count if it can simply be bought, or the verified
route and the named starting materials with their vendor counts.

Routes come from the repository's qualified transform library and were verified
by running those transforms forward and requiring exact reconstruction of the
target.  Availability is a dated commercial snapshot.  Neither establishes
substrate scope, and nothing here has been synthesised.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from forge.design.flow.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_synthesis_dossier.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_synthesis_dossier_ledger.v1"
P1 = "results/phase1/"

ROLE_LABEL = {
    "amine_head": "amine head",
    "oxoester_aldehyde_body_tail": "oxo-ester aldehyde tail",
    "isocyanide_tail": "isocyanide tail",
}


def _index(repo: Path) -> dict[str, Any]:
    procurement: dict[str, dict[str, Any]] = {}
    for name in (
        P1 + "ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz",
        P1 + "ugi_indomain_makeability_v1/indomain_procurement_ledger.jsonl.gz",
    ):
        path = repo / name
        if path.is_file():
            for row in read_jsonl_gzip(path, label=name):
                procurement.setdefault(str(row["canonical_smiles"]), row)
    tail_a = {
        str(r["canonical_smiles"]): r
        for r in read_jsonl_gzip(
            repo / (P1 + "ugi_tail_a_disconnection_v1/tail_a_disconnection_ledger.jsonl.gz"),
            label="tail A",
        )
    }
    iso = {
        str(r["canonical_smiles"]): r
        for r in read_jsonl_gzip(
            repo / (P1 + "ugi_isocyanide_route_v1/isocyanide_route_ledger.jsonl.gz"),
            label="isocyanide",
        )
    }
    return {"procurement": procurement, "tail_a": tail_a, "isocyanide": iso}


def _vendors(index: dict[str, Any], smiles: str) -> int | None:
    return (index["procurement"].get(smiles) or {}).get("vendor_count")


def component_plan(index: dict[str, Any], role: str, smiles: str) -> dict[str, Any]:
    """Buy it, or make it by a verified route from named materials."""

    vendors = _vendors(index, smiles)
    if vendors:
        return {
            "role": role,
            "canonical_smiles": smiles,
            "action": "purchase",
            "vendor_count": vendors,
            "route": None,
            "starting_materials": [],
        }
    if role == "oxoester_aldehyde_body_tail":
        record = index["tail_a"].get(smiles)
        if record and record.get("tail_a_verified"):
            return {
                "role": role,
                "canonical_smiles": smiles,
                "action": "synthesise",
                "vendor_count": vendors,
                "route": {
                    "name": "AGILE Tail A",
                    "steps": [
                        "esterify the carboxylic acid with the alpha,omega-diol",
                        "oxidise the remaining primary alcohol to the aldehyde",
                    ],
                    "transforms": [
                        "ugi3_upstream_esterification_exact_source_v1",
                        "ugi3_upstream_primary_alcohol_oxidation_exact_source_v1",
                    ],
                    "forward_verified": True,
                },
                "starting_materials": [
                    {
                        "role": "carboxylic acid",
                        "canonical_smiles": record.get("acid"),
                        "vendor_count": record.get("acid_vendor_count"),
                    },
                    {
                        "role": "alpha,omega-diol",
                        "canonical_smiles": record.get("diol"),
                        "vendor_count": record.get("diol_vendor_count"),
                    },
                ],
            }
    if role == "isocyanide_tail":
        record = index["isocyanide"].get(smiles)
        if record and record.get("route_verified"):
            return {
                "role": role,
                "canonical_smiles": smiles,
                "action": "synthesise",
                "vendor_count": vendors,
                "route": {
                    "name": "isocyanide from primary amine",
                    "steps": [
                        "formylate the primary amine to the N-substituted formamide",
                        "dehydrate the formamide to the isocyanide",
                    ],
                    "transforms": [
                        "ugi3_upstream_amine_formylation_exact_source_v1",
                        "ugi3_upstream_formamide_dehydration_exact_source_v1",
                    ],
                    "forward_verified": True,
                },
                "starting_materials": [
                    {
                        "role": "primary amine",
                        "canonical_smiles": record.get("precursor_amine"),
                        "vendor_count": record.get("amine_vendor_count"),
                    }
                ],
            }
    return {
        "role": role,
        "canonical_smiles": smiles,
        "action": "unresolved",
        "vendor_count": vendors,
        "route": None,
        "starting_materials": [],
    }


def build(repo: Path, output_dir: Path) -> dict[str, Any]:
    index = _index(repo)
    proposal_path = repo / (
        P1 + "ugi_stratified_panel_proposal_v1/stratified_panel_proposal.jsonl.gz"
    )
    candidates = read_jsonl_gzip(proposal_path, label="panel proposal")
    snapshot = load_json(
        repo / (P1 + "ugi_online_procurement_snapshot_v1/result.json"), label="procurement"
    ).get("snapshot")

    dossiers: list[dict[str, Any]] = []
    for row in candidates:
        plans = [
            component_plan(index, role, row["components"][role])
            for role in ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
        ]
        shopping: dict[str, list[dict[str, Any]]] = {"purchase": [], "synthesise": []}
        for plan in plans:
            if plan["action"] == "purchase":
                shopping["purchase"].append(
                    {"what": plan["canonical_smiles"], "vendors": plan["vendor_count"]}
                )
            else:
                for material in plan["starting_materials"]:
                    shopping["purchase"].append(
                        {"what": material["canonical_smiles"], "vendors": material["vendor_count"]}
                    )
                shopping["synthesise"].append(
                    {"what": plan["canonical_smiles"], "via": (plan["route"] or {}).get("name")}
                )
        dossiers.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "proposal_index": row["proposal_index"],
                "stratum": row["stratum"],
                "canonical_product": row["canonical_product"],
                "generation_arm": row["arm_id"],
                "authority_tier": row["authority_tier"],
                "conservative_high_potency": row["conservative_high_potency"],
                "oracle_mean": row.get("oracle_mean"),
                "lcb90": row.get("lcb90"),
                "components": plans,
                "shopping_list": shopping,
                "steps_to_make": sum(len((p["route"] or {}).get("steps", [])) for p in plans),
                "nonclaims": [
                    "Not synthesised; no yield, purity or isolation is implied.",
                    "Forward verification does not establish substrate scope.",
                    "Vendor counts are a dated screening signal, not quotes.",
                    "Potency is a conservative ranking signal, not a validated prediction.",
                ],
            }
        )

    dossiers.sort(key=lambda d: d["proposal_index"])
    ledger = output_dir / "synthesis_dossiers.jsonl.gz"
    atomic_write(ledger, jsonl_gzip_bytes(dossiers))

    lines = [
        "# FORGE synthesis dossiers",
        "",
        f"{len(dossiers)} proposed candidates. For each: the three exact Ugi components, and for",
        "each component either a purchase or a forward-verified route with named starting",
        "materials. Vendor counts come from a commercial snapshot accessed",
        f"{(snapshot or {}).get('accessed_utc','?')} and expiring {(snapshot or {}).get('expires_utc','?')}.",
        "",
        "Nothing here has been synthesised. Forward verification confirms the transform",
        "reproduces the target; it does not establish substrate scope.",
        "",
    ]
    for d in dossiers:
        lines += [
            (
                f"## {d['proposal_index']}. {d['stratum']} — LCB90 {d['lcb90']:.2f}"
                if d["lcb90"] is not None
                else f"## {d['proposal_index']}. {d['stratum']}"
            ),
            "",
            f"**Product** `{d['canonical_product']}`",
            "",
            f"Arm {d['generation_arm']} | authority {d['authority_tier']} | "
            f"{'conservative-high' if d['conservative_high_potency'] else 'not conservative-high'} "
            f"| {d['steps_to_make']} synthetic steps",
            "",
        ]
        for plan in d["components"]:
            label = ROLE_LABEL[plan["role"]]
            if plan["action"] == "purchase":
                lines.append(
                    f"- **{label}** — purchase `{plan['canonical_smiles']}` ({plan['vendor_count']} vendors)"
                )
            elif plan["action"] == "synthesise":
                lines.append(
                    f"- **{label}** — make `{plan['canonical_smiles']}` via {plan['route']['name']}"
                )
                for step in plan["route"]["steps"]:
                    lines.append(f"    - {step}")
                for material in plan["starting_materials"]:
                    lines.append(
                        f"    - from `{material['canonical_smiles']}` "
                        f"({material['role']}, {material['vendor_count']} vendors)"
                    )
            else:
                lines.append(f"- **{label}** — `{plan['canonical_smiles']}` UNRESOLVED")
        lines.append("")
    atomic_write(output_dir / "FORGE_SYNTHESIS_DOSSIERS.md", ("\n".join(lines)).encode())

    unresolved = sum(1 for d in dossiers for p in d["components"] if p["action"] == "unresolved")
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "synthesis_dossiers_complete",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "procurement_snapshot": snapshot,
        "summary": {
            "candidates": len(dossiers),
            "components": 3 * len(dossiers),
            "components_purchased": sum(
                1 for d in dossiers for p in d["components"] if p["action"] == "purchase"
            ),
            "components_synthesised": sum(
                1 for d in dossiers for p in d["components"] if p["action"] == "synthesise"
            ),
            "components_unresolved": unresolved,
            "distinct_materials_to_buy": len(
                {m["what"] for d in dossiers for m in d["shopping_list"]["purchase"]}
            ),
            "max_steps_for_any_candidate": max(d["steps_to_make"] for d in dossiers),
        },
        "artifacts": {
            "synthesis_dossiers.jsonl.gz": {
                "path": ledger.name,
                "sha256": sha256_file(ledger),
                "rows": len(dossiers),
            },
            "FORGE_SYNTHESIS_DOSSIERS.md": {
                "path": "FORGE_SYNTHESIS_DOSSIERS.md",
                "sha256": sha256_file(output_dir / "FORGE_SYNTHESIS_DOSSIERS.md"),
            },
        },
        "nonclaims": [
            "No compound here has been synthesised, formulated or tested.",
            "Forward verification does not establish substrate scope.",
            "Vendor counts are a dated screening signal and must be refreshed before purchase.",
            "This is a proposal for chemist review, not a locked panel.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/ugi_synthesis_dossiers_v1")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = build(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
