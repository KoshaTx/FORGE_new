#!/usr/bin/env python3
"""Step 4: check online whether proposed reactants are actually obtainable.

The repository has been using a 60-item hand-curated terminal-material ledger as
the authority on availability.  That ledger is not a catalog and was never meant
to be one; treating "absent from the ledger" as "unavailable" measures our
bookkeeping rather than the world.

This queries PubChem, which aggregates supplier registrations, for every
component and every route leaf, and records a dated snapshot:

    canonical SMILES -> InChIKey -> PubChem CID -> Chemical Vendors count

Authority boundary, deliberately narrow:

  * A vendor listing means at least one supplier has registered this exact
    structure.  It is a screening signal.
  * It is NOT a quote, NOT a stock level, NOT a price, NOT a lead time, and NOT
    a purity specification.  A real purchase still needs a quote.
  * Zero vendors means no supplier has registered this structure with PubChem.
    It is evidence the compound must be synthesised, not proof it cannot be
    bought anywhere.

The snapshot carries an accessed timestamp and a declared expiry so it can serve
as time-bounded L3 evidence rather than an undated assertion.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from experiments.phase1.synthesis_guidance.route_cascade import (  # noqa: E402
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_online_procurement_lookup.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_online_procurement_ledger.v1"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest"
REQUEST_INTERVAL_SECONDS = 0.25  # PubChem asks for <= 5 requests/second
SNAPSHOT_VALID_DAYS = 30

AUTHORITY = {
    "vendor_listing_is_a_screening_signal": True,
    "vendor_listing_is_a_quote": False,
    "vendor_listing_is_stock_level_or_lead_time": False,
    "vendor_listing_is_purity_specification": False,
    "zero_vendors_proves_unobtainable": False,
    "supersedes_local_hand_curated_terminal_ledger": False,
    "requires_quote_before_purchase": True,
}


def _get(url: str, *, timeout: int = 30) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "FORGE-procurement-lookup/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    except Exception as error:  # noqa: BLE001 - recorded, never dropped
        return -1, f"{type(error).__name__}: {error}"


def inchikey(smiles: str) -> str | None:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        return Chem.MolToInchiKey(molecule)
    except Exception:  # noqa: BLE001
        return None


def lookup(smiles: str) -> dict[str, Any]:
    """Resolve one structure to a dated vendor-count observation."""

    key = inchikey(smiles)
    record: dict[str, Any] = {
        "canonical_smiles": smiles,
        "inchikey": key,
        "pubchem_cid": None,
        "vendor_count": None,
        "vendor_sample": [],
        "status": "unresolved",
        "error": None,
    }
    if key is None:
        record["status"] = "invalid_structure"
        return record

    status, body = _get(f"{PUBCHEM}/pug/compound/inchikey/{key}/cids/TXT")
    time.sleep(REQUEST_INTERVAL_SECONDS)
    if status == 404:
        record["status"] = "not_in_pubchem"
        return record
    if status != 200:
        record["status"] = "lookup_error"
        record["error"] = f"cid_lookup_http_{status}"
        return record
    cids = [line.strip() for line in body.splitlines() if line.strip().isdigit()]
    if not cids:
        record["status"] = "not_in_pubchem"
        return record
    cid = cids[0]
    record["pubchem_cid"] = int(cid)

    status, body = _get(f"{PUBCHEM}/pug_view/categories/compound/{cid}/JSON")
    time.sleep(REQUEST_INTERVAL_SECONDS)
    if status != 200:
        record["status"] = "vendor_lookup_error"
        record["error"] = f"category_http_{status}"
        return record
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        record["status"] = "vendor_lookup_error"
        record["error"] = "category_json_decode_error"
        return record

    vendors: list[str] = []
    categories = (payload.get("SourceCategories") or {}).get("Categories") or []
    for category in categories:
        if str(category.get("Category", "")).strip().lower() != "chemical vendors":
            continue
        for source in category.get("Sources") or []:
            name = source.get("SourceName")
            if name:
                vendors.append(str(name))
    record["vendor_count"] = len(vendors)
    record["vendor_sample"] = sorted(set(vendors))[:12]
    record["status"] = "vendors_listed" if vendors else "no_vendors_listed"
    return record


def collect_targets(repo: Path) -> list[dict[str, Any]]:
    """Every shortlist component plus every planner route leaf, deduplicated."""

    targets: dict[str, dict[str, Any]] = {}
    component_path = repo / (
        "results/phase1/ugi_bounded_hybrid_route_cascade_v1/component_route_ledger.jsonl.gz"
    )
    for row in read_jsonl_gzip(component_path, label="component route ledger"):
        smiles = str(row["canonical_smiles"])
        entry = targets.setdefault(
            smiles, {"canonical_smiles": smiles, "kinds": set(), "roles": set(), "unlocks": 0}
        )
        entry["kinds"].add("ugi_component")
        entry["roles"].add(str(row["role"]))
        entry["unlocks"] = max(entry["unlocks"], int(row["occurrence_count"]))

    reach_path = repo / "results/phase1/ugi_planner_reach_sweep_v1/planner_reach_ledger.jsonl.gz"
    if reach_path.is_file():
        for row in read_jsonl_gzip(reach_path, label="planner reach ledger"):
            for leaf in row.get("stock_leaves") or []:
                entry = targets.setdefault(
                    str(leaf),
                    {"canonical_smiles": str(leaf), "kinds": set(), "roles": set(), "unlocks": 0},
                )
                entry["kinds"].add("planner_route_leaf")
            for proposal in row.get("single_step_proposals") or []:
                for reactant in proposal.get("canonical_reactants") or []:
                    entry = targets.setdefault(
                        str(reactant),
                        {
                            "canonical_smiles": str(reactant),
                            "kinds": set(),
                            "roles": set(),
                            "unlocks": 0,
                        },
                    )
                    entry["kinds"].add("single_step_reactant")

    output = []
    for entry in targets.values():
        output.append(
            {
                "canonical_smiles": entry["canonical_smiles"],
                "kinds": sorted(entry["kinds"]),
                "roles": sorted(entry["roles"]),
                "unlocks_candidates": entry["unlocks"],
            }
        )
    output.sort(key=lambda row: (-row["unlocks_candidates"], row["canonical_smiles"]))
    return output


def run(repo: Path, output_dir: Path, limit: int | None) -> dict[str, Any]:
    targets = collect_targets(repo)
    if limit is not None:
        targets = targets[:limit]
    accessed = datetime.now(timezone.utc).replace(microsecond=0)
    expires = accessed + timedelta(days=SNAPSHOT_VALID_DAYS)

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        observation = lookup(target["canonical_smiles"])
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                **target,
                **observation,
                "accessed_utc": accessed.isoformat().replace("+00:00", "Z"),
                "authority": dict(AUTHORITY),
            }
        )
        if index % 25 == 0 or index == len(targets):
            listed = sum(1 for r in rows if r["status"] == "vendors_listed")
            print(f"procurement {index}/{len(targets)} vendors_listed={listed}", flush=True)

    rows.sort(key=lambda row: (-int(row["unlocks_candidates"]), str(row["canonical_smiles"])))
    ledger_path = output_dir / "procurement_ledger.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(rows))

    listed = [r for r in rows if r["status"] == "vendors_listed"]
    components = [r for r in rows if "ugi_component" in r["kinds"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "online_procurement_snapshot_complete",
        "source": {
            "provider": "PubChem PUG-REST / PUG-View",
            "endpoint": PUBCHEM,
            "aggregates": "supplier registrations from many chemical vendors",
        },
        "snapshot": {
            "accessed_utc": accessed.isoformat().replace("+00:00", "Z"),
            "expires_utc": expires.isoformat().replace("+00:00", "Z"),
            "valid_days": SNAPSHOT_VALID_DAYS,
            "interval_semantics": "start_inclusive_end_exclusive",
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "targets": len(rows),
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "vendors_listed": len(listed),
            "ugi_components_examined": len(components),
            "ugi_components_with_vendors": sum(
                1 for r in components if r["status"] == "vendors_listed"
            ),
            "ugi_components_with_vendors_by_role": dict(
                sorted(
                    Counter(
                        role
                        for r in components
                        if r["status"] == "vendors_listed"
                        for role in r["roles"]
                    ).items()
                )
            ),
            "candidate_occurrences_behind_purchasable_components": sum(
                int(r["unlocks_candidates"]) for r in components if r["status"] == "vendors_listed"
            ),
        },
        "artifacts": {
            "procurement_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "authority": dict(AUTHORITY),
        "nonclaims": [
            "A vendor listing is not a quote, stock level, lead time or purity specification.",
            "Zero listed vendors is evidence a compound must be synthesised, not proof it cannot be obtained.",
            "This snapshot expires and must be refreshed before any panel lock.",
            "This does not close a route, select a candidate or lock a panel.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_online_procurement_snapshot_v1"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    result = run(repo, output_dir.resolve(), args.limit)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
