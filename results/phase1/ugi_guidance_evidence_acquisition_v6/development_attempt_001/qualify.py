"""Authenticate a fresh terminal observation without modifying historical route evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.core.hashing import pin_record
from forge.core.io import atomic_write, write_json
from forge.synthesis.evidence.ugi3_exact_c18_route import _terminal_evidence, _terminal_state

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONFIG = ROOT / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
PRIOR = OUT.with_name("ugi_guidance_evidence_acquisition_v4") / "result.json"
ARTICLE = OUT.with_name("ugi_guidance_primary_article_audit_v1") / "result.json"
ARCHIVE = ROOT / "provenance/recovery/ugi_guidance_evidence_v6/assets"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def plain(value: str) -> str:
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S | re.I)
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value)).split())


def parse_accel(page: str, shipping: str, expected: dict) -> dict:
    """Require exact identity, visible item stock, and an explicitly offered US destination."""
    text = plain(page)
    require("Catalog No.: EC003WL5" in text, "Supplier catalog identity differs")
    identities = set(re.findall(r"SMILES (\S+) Molecular Weight", text))
    require(len(identities) == 1, "Supplier graph missing or ambiguous")
    molecule = Chem.MolFromSmiles(identities.pop())
    require(molecule is not None, "Supplier graph cannot be parsed")
    target = Chem.MolFromSmiles(expected["canonical_smiles"])
    require(target is not None, "Frozen target cannot be parsed")
    canonical = Chem.MolToSmiles(molecule)
    require(canonical == Chem.MolToSmiles(target), "Supplier graph differs from frozen target")
    require(Chem.MolToInchiKey(molecule) == expected["inchi_key"], "Full InChIKey differs")
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    require(f"Molecular Formula {formula}" in text, "Supplier formula differs")
    require(f"CAS Number {expected['cas_number']} SMILES" in text, "Supplier CAS differs")
    table = re.findall(r"Packsize Purity Availability Price Quantity (.*?) Total", text)
    require(len(table) == 1, "Supplier pack table missing or ambiguous")
    pattern = r"(\d+(?:\.\d+)?(?:mg|g)) (\d+(?:\.\d+)?%) Global Stock (€[\d,.]+) (€[\d,.]+)"
    packs = re.findall(pattern, table[0])
    require(bool(packs), "No visible item-specific stock and purity")
    require(not re.sub(pattern, "", table[0]).strip(), "Conflicting or unparsed pack rows")
    require(len({p[0] for p in packs}) == len(packs), "Duplicate pack rows")
    require(len({p[1] for p in packs}) == 1, "Different grades require separate observations")
    shipping_text = plain(shipping)
    require("International Shipping Rates" in shipping_text, "Shipping context missing")
    require("Canada, United States €50" in shipping_text, "Explicit US destination missing")
    return {
        "canonical_smiles": canonical,
        "inchi_key": Chem.MolToInchiKey(molecule),
        "formula": formula,
        "item_specific_purity": packs[0][1],
        "packs": [
            {
                "pack_size": size,
                "purity": purity,
                "stock": "Global Stock",
                "displayed_previous_price": old,
                "displayed_price": price,
            }
            for size, purity, old, price in packs
        ],
        "us_delivery_offered_by_supplier": True,
        "us_warehouse_inventory_established": False,
        "item_specific_delivery_date_or_quantity_confirmed": False,
    }


def source(batch: str, name: str) -> tuple[dict, Path]:
    receipt = read(OUT / batch / "receipt.json")
    records = [r for r in receipt["records"] if r["id"] == name]
    require(len(records) == 1 and records[0]["http_status"] == 200, "Successful source missing")
    record = records[0]
    path = ROOT / record["response"]["path"]
    require(digest(path.read_bytes()) == record["response"]["sha256"], "Response hash changed")
    return record, path


def derive(assessment_at: str) -> tuple[dict, dict]:
    config = read(CONFIG)
    expected = config["molecules"]["stearolic_acid"]
    expected = {**expected, "cas_number": config["terminal_procurement"]["cas_number"]}
    record, page = source("batch_07", "stearolic_accelsci")
    _, shipping = source("batch_08", "accelsci_shipping")
    _, terms = source("batch_08", "accelsci_terms")
    _, drawing = source("batch_10", "accel_structure")
    parsed = parse_accel(page.read_text(), shipping.read_text(), expected)
    review = read(OUT / "review.json")
    require(
        review["drawing"]["status"] == "visually_verified_exact_graph", "Drawing review pending"
    )
    require(review["drawing"]["source"] == pin_record(drawing, ROOT), "Reviewed drawing differs")
    accessed = datetime.fromisoformat(record["completed_at_utc"])
    expiry = accessed + timedelta(days=config["procurement_policy"]["expiry_days"])
    assessment = datetime.fromisoformat(assessment_at)
    require(accessed <= assessment <= expiry, "Assessment outside observation lifetime")
    observation = {
        "schema_version": "forge.vendor_product_observation.v1",
        "vendor": "Accel Scientific",
        "official_product_url": record["resolved_url"],
        "item": {
            "catalog_number": "EC003WL5",
            "cas_number": expected["cas_number"],
            "canonical_smiles": parsed["canonical_smiles"],
            "inchi_key": parsed["inchi_key"],
        },
        "identity_exact": True,
        "item_specific_purity": parsed["item_specific_purity"],
        "explicit_current_stock_or_shipping_observed": True,
        "current_item_level_procurement_closed": True,
        "observed_at_utc": accessed.isoformat(),
        "expires_at_utc": expiry.isoformat(),
        "region": "US delivery offered from supplier-reported global stock",
        "availability_observations": parsed,
        "shipping_statement": "Public destination table explicitly includes United States",
        "source_assets": [pin_record(p, ROOT) for p in (page, shipping, terms, drawing)],
        "limitations": review["admitted_observation_limits"],
        "disposition": "admit_exact_terminal_observation",
        "complete_route_qualified": False,
    }
    terminal = {
        k: observation[k]
        for k in (
            "vendor",
            "identity_exact",
            "item_specific_purity",
            "observed_at_utc",
            "expires_at_utc",
            "explicit_current_stock_or_shipping_observed",
            "current_item_level_procurement_closed",
        )
    }
    terminal.update(
        molecule_id="stearolic_acid",
        canonical_smiles=parsed["canonical_smiles"],
        cas_number=expected["cas_number"],
        item_id="EC003WL5",
        product_url=record["resolved_url"],
        observation_source_input="fresh_observation",
        evidence_records=[
            {
                "evidence_id": "fresh-c18-terminal:accel-EC003WL5:20260913",
                "source_input": "fresh_observation",
                "source_locator": "item; source_assets; availability_observations; shipping_statement",
                "supports_current_availability": True,
            }
        ],
    )
    return observation, terminal


def check_terminal(terminal: dict, observation_path: Path, assessment_at: str) -> None:
    state = _terminal_state(terminal)
    _terminal_evidence(
        terminal=terminal,
        input_paths={"fresh_observation": observation_path},
        terminal_state=state,
        terminal_smiles=terminal["canonical_smiles"],
        terminal_inchi_key=read(CONFIG)["molecules"]["stearolic_acid"]["inchi_key"],
        assessment_as_of_utc=assessment_at,
        expiry_days=read(CONFIG)["procurement_policy"]["expiry_days"],
    )


def verify(result: dict, restore: bool = False) -> None:
    for entry in result["archived_sources"]:
        packed = (ROOT / entry["archive"]["path"]).read_bytes()
        require(digest(packed) == entry["archive"]["sha256"], "Archive hash differs")
        raw = gzip.decompress(packed)
        require(digest(raw) == entry["original"]["sha256"], "Archived source hash differs")
        path = ROOT / entry["original"]["path"]
        require(path.resolve().is_relative_to(OUT), "Restore destination escapes acquisition")
        if restore and not path.exists():
            atomic_write(path, raw)
    for pin in result["inputs"]:
        require(
            digest((ROOT / pin["path"]).read_bytes()) == pin["sha256"],
            f"Input changed: {pin['path']}",
        )
    observation, terminal = derive(result["assessment_at_utc"])
    require(observation == read(OUT / "terminal_observation.json"), "Observation differs on replay")
    require(terminal == result["terminal_contract"], "Terminal contract differs on replay")
    check_terminal(terminal, OUT / "terminal_observation.json", result["assessment_at_utc"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    result_path = OUT / "result.json"
    if args.verify or args.restore:
        verify(read(result_path), args.restore)
        print(
            "Verified exact sources and terminal-only qualification; availability expiry unchanged"
        )
        return
    require(not result_path.exists(), "Preserve the existing result")
    assessment = datetime.now(timezone.utc).isoformat()
    observation, terminal = derive(assessment)
    observation_path = OUT / "terminal_observation.json"
    if observation_path.exists():
        require(read(observation_path) == observation, "Preserve changed observation")
    else:
        write_json(observation_path, observation)
    check_terminal(terminal, observation_path, assessment)
    records = []
    paths = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [
        CONFIG,
        PRIOR,
        ARTICLE,
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "forge/synthesis/evidence/ugi3_exact_c18_route.py",
        ROOT / "forge/core/hashing.py",
        ROOT / "forge/core/io.py",
    ]
    for receipt_path in sorted(OUT.glob("batch_*/receipt.json")):
        receipt = read(receipt_path)
        records.extend(receipt["records"])
        for pin in receipt["inputs"].values():
            path = ROOT / pin["path"]
            require(digest(path.read_bytes()) == pin["sha256"], "Acquisition producer changed")
            paths.append(path)
        for record in receipt["records"]:
            if "response" in record:
                pin = record["response"]
                require(
                    digest((ROOT / pin["path"]).read_bytes()) == pin["sha256"], "Raw source changed"
                )
    archives = []
    for path in sorted(set(paths)):
        if path.is_relative_to(OUT) and path.suffix in {".html", ".png", ".webp"}:
            original = pin_record(path, ROOT)
            packed = gzip.compress(path.read_bytes(), mtime=0)
            destination = ARCHIVE / (original["sha256"] + ".gz")
            require(
                not destination.exists() or destination.read_bytes() == packed, "Archive changed"
            )
            if not destination.exists():
                atomic_write(destination, packed)
            archives.append({"original": original, "archive": pin_record(destination, ROOT)})
    prior = read(PRIOR)["fresh_evidence"]
    for item in prior["additional_observations"] + [prior["previous_cyclohexylamine_snapshot"]]:
        require(
            datetime.fromisoformat(item["accessed_utc"])
            <= datetime.fromisoformat(assessment)
            <= datetime.fromisoformat(item["expires_utc"]),
            "Prior terminal evidence expired",
        )
    result = {
        "schema_version": "forge.guidance_terminal_acquisition.v1",
        "assessment_at_utc": assessment,
        "status": "five_target_material_observations_qualified_combined_routes_pending",
        "inputs": [pin_record(p, ROOT) for p in sorted(set(paths))],
        "archived_sources": archives,
        "terminal_contract": terminal,
        "terminal_observation": pin_record(observation_path, ROOT),
        "http_outcomes": dict(
            Counter(str(r.get("http_status", "transport_failure")) for r in records)
        ),
        "public_requests": len(records),
        "previous_qualified_target_materials": prior[
            "total_qualified_target_material_observations"
        ],
        "new_qualified_target_materials": 1,
        "total_qualified_target_materials": prior["total_qualified_target_material_observations"]
        + 1,
        "complete_routes_reassessed": 0,
        "historical_artifacts_recovered_this_pass": 0,
        "random_sampling_used": False,
        "review": read(OUT / "review.json"),
        "policy": {
            "historical_pins_changed": False,
            "gates_relaxed": False,
            "guidance_ready": False,
            "training_calls": 0,
            "generation_calls": 0,
            "guidance_calls": 0,
            "purchases": 0,
            "supplier_messages": 0,
            "candidate_selection": False,
            "paid_remote_jobs": 0,
        },
        "remaining_work": [
            "Qualify a separately versioned combined source and exact route outputs against the preserved evidence.",
            "Recheck structured synthesis values, nontrivial dossier closure and zero-guidance equivalence before nonzero guidance.",
        ],
        "global_phase1_definition_of_done_met": False,
    }
    verify(result)
    write_json(result_path, result)
    print(
        json.dumps({k: result[k] for k in ("status", "public_requests", "http_outcomes")}, indent=2)
    )


if __name__ == "__main__":
    main()
