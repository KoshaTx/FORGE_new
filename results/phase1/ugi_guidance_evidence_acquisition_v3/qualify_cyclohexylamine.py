"""Qualify only the fresh exact cyclohexylamine terminal, without cumulative route claims."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import write_json
from forge.synthesis.terminals.ugi3_high_leverage_head_terminals import _read_csv
from forge.synthesis.terminals.ugi3_third_wave_head_terminals import (
    Ugi3ThirdWaveHeadTerminalError,
    _validate_supplier_record,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_time(snapshot: dict, as_of: datetime) -> None:
    if not parse_date(snapshot["accessed_utc"]) <= as_of <= parse_date(snapshot["expires_utc"]):
        raise ValueError("Availability evidence is future-dated or expired")


def main() -> None:
    output = OUT / "cyclohexylamine_terminal_qualification.json"
    if output.exists():
        raise ValueError("Preserve the existing terminal qualification")
    historical_pack = ROOT / "configs/route/phase1_ugi3_third_wave_head_terminals_v1.json"
    historical_audit = ROOT / "configs/route/phase1_ugi3_third_wave_head_terminal_audit_v1.json"
    receipt_path = ROOT / "data/source_cache/ugi_guidance_evidence_20260913/batch_01/receipt.json"
    acquisition = json.loads((OUT / "result.json").read_text())
    for item in acquisition["inputs"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Acquisition input changed: {item['path']}")
    for identity in acquisition["identity_checks"]:
        if not all(identity["checks"].values()):
            raise ValueError("An acquired identity check failed")
    old = json.loads(historical_pack.read_text())
    config = json.loads(historical_audit.read_text())
    readiness_pin = config["inputs"]["readiness_ledger"]
    readiness_path = ROOT / readiness_pin["asset"]
    if sha256_file(readiness_path) != readiness_pin["expected_sha256"]:
        raise ValueError("Frozen registry readiness ledger changed")
    rows = _read_csv(readiness_path, label="frozen registry readiness")
    readiness = {row["component_id"]: row for row in rows}
    receipt = json.loads(receipt_path.read_text())
    (source,) = [r for r in receipt["records"] if r["id"] == "chemimpex_cyclohexylamine"]
    page = ROOT / source["response"]["path"]
    if source["http_status"] != 200 or sha256_file(page) != source["response"]["sha256"]:
        raise ValueError("Fresh primary supplier response did not authenticate")
    html = page.read_text()
    data = [
        json.loads(body)
        for body in re.findall(r'<script type="application/json">(.*?)</script>', html, re.S)
    ]
    (product,) = [p for p in data if isinstance(p, dict) and isinstance(p.get("product"), dict)]
    skus = acquisition["fresh_supplier_observation"]["same_day_skus"]
    by_sku = {v["sku"]: v for v in product["product"]["variants"]}
    for sku in skus:
        variant = by_sku[sku]
        if (
            variant["available"] is not True
            or product["formatted"][str(variant["id"])]["lead_time"] != -1
        ):
            raise ValueError("A declared same-day SKU is not currently available")
    if len(skus) != html.count('class="opt-availability">Ships Today</span>'):
        raise ValueError("Visible stock labels disagree with the declared same-day SKUs")

    accessed = parse_date(source["completed_at_utc"])
    lifetime = parse_date(old["snapshot"]["expires_utc"]) - parse_date(
        old["snapshot"]["accessed_utc"]
    )
    snapshot = {
        **old["snapshot"],
        "accessed_utc": accessed.isoformat(),
        "expires_utc": (accessed + lifetime).isoformat(),
        "interpretation": "Exact supplier availability only; no Ugi success or cumulative route closure claim.",
    }
    now = datetime.now(timezone.utc)
    validate_time(snapshot, now)
    record = {
        **old["records"][0],
        "source_asset": str(page.relative_to(ROOT)),
        "source_asset_sha256": sha256_file(page),
        "available_skus": skus,
    }
    component_id, key = _validate_supplier_record(
        record, readiness_by_id=readiness, supplier_page=page
    )
    checks = {"existing_supplier_validator_passed": True, "stock_skus_checked_individually": True}
    with tempfile.TemporaryDirectory(prefix="forge-terminal-negative-", dir="/private/tmp") as tmp:
        changed = Path(tmp) / "supplier.html"
        changed.write_text(html.replace("Ships Today", "Availability unassessed"))
        wrong_stock = {
            **record,
            "source_asset": str(changed),
            "source_asset_sha256": sha256_file(changed),
        }
        try:
            _validate_supplier_record(wrong_stock, readiness_by_id=readiness, supplier_page=changed)
        except Ugi3ThirdWaveHeadTerminalError:
            checks["rehashed_missing_stock_labels_rejected"] = True
        else:
            raise ValueError("Missing-stock negative control was admitted")
    wrong_graph = {
        **record,
        "canonical_smiles": next(
            r["canonical_smiles"]
            for r in rows
            if r["role"] == "amine_head" and r["canonical_smiles"] != record["canonical_smiles"]
        ),
    }
    try:
        _validate_supplier_record(wrong_graph, readiness_by_id=readiness, supplier_page=page)
    except Ugi3ThirdWaveHeadTerminalError:
        checks["wrong_registry_graph_rejected"] = True
    else:
        raise ValueError("Wrong-graph negative control was admitted")
    for name, timestamp in (("expired", accessed + lifetime * 2), ("future", accessed - lifetime)):
        try:
            validate_time(snapshot, timestamp)
        except ValueError:
            checks[f"{name}_snapshot_rejected"] = True
        else:
            raise ValueError(f"{name} snapshot was admitted")
    pack = copy.deepcopy(old)
    pack.update(snapshot=snapshot, records=[record])
    pack_path = OUT / "cyclohexylamine_terminal_evidence_20260913.json"
    write_json(pack_path, pack)
    report = {
        "schema_version": "forge.exact_terminal_evidence_qualification.v1",
        "created_at_utc": now.isoformat(),
        "status": "one_fresh_exact_terminal_qualified",
        "scope": "Single supplier terminal only; earlier cumulative product counts are not reassessed.",
        "inputs": {
            str(p.relative_to(ROOT)): pin_record(p, ROOT)
            for p in (
                Path(__file__),
                historical_pack,
                historical_audit,
                readiness_path,
                receipt_path,
                page,
                OUT / "result.json",
                ROOT / "forge/synthesis/terminals/ugi3_third_wave_head_terminals.py",
                ROOT / "forge/synthesis/terminals/ugi3_high_leverage_head_terminals.py",
            )
        },
        "artifact": pin_record(pack_path, ROOT),
        "component_id": component_id,
        "component_key": key,
        "qualified_terminal_count": 1,
        "availability_lifetime_days": lifetime.days,
        "snapshot": snapshot,
        "checks": checks,
        "supersedes_acquisition_review_for": "chemimpex_cyclohexylamine_only",
        "newly_qualified_routes": 0,
        "historical_pins_changed": False,
        "gates_relaxed": False,
        "guidance_run_ready": False,
        "generation_calls": 0,
        "purchase_or_contact_actions": 0,
    }
    write_json(output, report)
    print(report["status"])
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
