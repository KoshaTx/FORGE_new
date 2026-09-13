"""Qualify fresh exact supplier observations, independently of the frozen route source."""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import inchi, rdMolDescriptors

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
V3 = OUT.with_name("ugi_guidance_evidence_acquisition_v3")
INPUTS: set[Path] = set()


def read(path: Path) -> str:
    INPUTS.add(path)
    return path.read_text()


def read_json(path: Path) -> dict:
    return json.loads(read(path))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def source(receipt_path: Path, key: str, value: str) -> tuple[dict, Path]:
    receipt = read_json(receipt_path)
    matches = [r for r in receipt["records"] if r[key] == value]
    require(len(matches) == 1, "Source receipt must identify one response")
    record = matches[0]
    require(record["http_status"] == 200, "Supplier response was not successful")
    pin = record["response"]
    path = ROOT / pin["path"]
    require(sha256_file(path) == pin["sha256"], f"Source changed: {path}")
    INPUTS.add(path)
    for item in receipt.get("inputs", {}).values():
        p = ROOT / item["path"]
        require(sha256_file(p) == item["sha256"], f"Receipt input changed: {p}")
        INPUTS.add(p)
    return record, path


def plain(value: str) -> str:
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", value))).strip()


def canonical(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    require(molecule is not None, "Invalid source graph")
    require(len(Chem.GetMolFrags(molecule)) == 1, "Disconnected source graph")
    require(not any(a.GetIsotope() for a in molecule.GetAtoms()), "Isotope substitute rejected")
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def current_window(accessed: str, expires: str, assessment: str) -> None:
    dates = [
        datetime.fromisoformat(v.replace("Z", "+00:00")) for v in (accessed, expires, assessment)
    ]
    require(all(v.tzinfo is not None for v in dates), "Timezone required")
    require(dates[0] <= dates[2] <= dates[1], "Expired or future supplier observation")


def bld_observation(page: str, prices: dict, stock: dict, expected: dict, bd: str) -> dict:
    require(f'id="nowBD" value="{bd}"' in page, "Supplier product ID mismatch")
    require(expected["cas"] in page, "Supplier CAS mismatch")
    matches = re.findall(r"<td>SMILES Code:</td>\s*<td[^>]*>(.*?)</td>", page, re.S)
    require(len(matches) == 1, "Supplier graph missing or ambiguous")
    supplier_smiles = canonical(plain(matches[0]))
    require(supplier_smiles == canonical(expected["canonical_smiles"]), "Supplier graph mismatch")
    require(prices.get("code") == 200 and stock.get("code") == 200, "Supplier API error")
    grade_keys = [key for key in prices["value"] if key.startswith(bd + "_")]
    require(len(grade_keys) == 1, "Supplier grade ambiguity")
    grade = prices["value"][grade_keys[0]]
    require(all(item["pr_bd"] == bd for item in grade), "Supplier price identity mismatch")
    size_rows = {}
    for row in stock["value"]:
        require(row.get("is_login") == 0, "Expected public guest stock response")
        size = row["size"]
        if size in size_rows:
            require(row == size_rows[size], "Conflicting duplicate stock rows")
        size_rows[size] = row
    available = []
    for item in grade:
        row = size_rows.get(item["pr_size"], {})
        flag = row.get("has_stock_quantityusa", 0)
        require(type(flag) is int and flag in (0, 1), "Unexpected public US stock flag")
        if flag == 1 and item["price_dict"]["pr_usd"] > 0:
            available.append({"pack_size": item["pr_size"], "purity": item["p_purity"]})
    require(bool(available), "No priced pack has a positive US stock flag")
    return {
        "canonical_smiles": supplier_smiles,
        "catalog_number": bd,
        "available_packs": sorted(available, key=lambda row: row["pack_size"]),
        "region": "US",
        "availability_basis": "Public has_stock_quantityusa flag, displayed as in stock by supplier JS",
        "bottle_counts_known": False,
        "source_duplicate_rows_collapsed": len(stock["value"]) - len(size_rows),
    }


def fisher_observation(page: str, response: dict, expected: dict, sku: str) -> dict:
    require(sku in page, "Supplier SKU not present in listing")
    require(expected["cas"] in page, "Supplier CAS mismatch")
    molecule = Chem.MolFromSmiles(canonical(expected["canonical_smiles"]))
    require(
        rdMolDescriptors.CalcMolFormula(molecule) in plain(page).replace(" ", ""),
        "Formula mismatch",
    )
    key_matches = set(re.findall(r"HQ[A-Z]{12}-[A-Z]{10}-[A-Z]", page))
    standard = inchi.MolToInchi(molecule)
    fixed_h = inchi.MolToInchi(molecule, options="/FixedH")
    standard_key = inchi.InchiToInchiKey(standard)
    fixed_h_key = inchi.InchiToInchiKey(fixed_h)
    require(key_matches == {fixed_h_key}, "Supplier complete nonstandard InChIKey mismatch")
    require(
        canonical(Chem.MolToSmiles(inchi.MolFromInchi(fixed_h)))
        == canonical(expected["canonical_smiles"]),
        "Nonstandard InChI round trip changed graph",
    )
    require(str(expected["pubchem_cid"]) in page, "Supplier PubChem identity mismatch")
    rows = response.get("priceAndAvailability", {}).get(sku, [])
    require(response.get("message") == "success" and len(rows) == 1, "Supplier response error")
    row = rows[0]
    require(
        float(row["availableQuantity"]) >= 1 and float(row["backOrderedQuantity"]) == 0,
        "Supplier stock is not positive or is backordered",
    )
    require("In Stock" in plain(row["availability"]), "Supplier display does not confirm stock")
    require("98%" in page, "Item purity missing")
    return {
        "canonical_smiles": canonical(expected["canonical_smiles"]),
        "catalog_number": sku,
        "available_packs": [{"pack_size": "25g", "purity": "98%"}],
        "region": "US",
        "reference_zip_only_not_user_address": "10001",
        "availability_basis": "Positive public fulfillment quote for one EA, zero backorder",
        "shipped_direct_from_supplier": row["shippedDirect"],
        "stocked_item_at_fisher": row["stockedItem"],
        "supplier_inchi_key": fixed_h_key,
        "standard_inchi_key": standard_key,
        "identifier_resolution": "Full supplier key reproduced using FixedH; exact graph round trip",
    }


def must_reject(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise ValueError("Adversarial control was incorrectly admitted")


def build(assessment_at: str) -> dict:
    acquisition = read_json(V3 / "result.json")
    identities = {r["cas"]: r for r in acquisition["identity_checks"]}
    for item in identities.values():
        p = ROOT / item["source"]["path"]
        require(sha256_file(p) == item["source"]["sha256"], "PubChem source changed")
        INPUTS.add(p)
        properties = read_json(p)["PropertyTable"]["Properties"]
        exact = [r for r in properties if r["CID"] == item["pubchem_cid"]]
        require(
            len(exact) == 1
            and canonical(exact[0]["ConnectivitySMILES"]) == canonical(item["canonical_smiles"]),
            "PubChem graph identity mismatch",
        )
    old_c16 = read_json(ROOT / "configs/route/phase1_ugi3_exact_c16_route_v1.json")
    old_head = read_json(ROOT / "configs/route/phase1_ugi3_high_leverage_tail_terminals_v1.json")
    c16_by_cas = {r["cas_number"]: r for r in old_c16["terminal_procurement"]}
    for cas, old in c16_by_cas.items():
        require(
            canonical(old["canonical_smiles"]) == identities[cas]["canonical_smiles"],
            "Frozen C16 leaf identity changed",
        )
    require(
        old_head["records"][0]["canonical_smiles"] == identities["629-90-3"]["canonical_smiles"],
        "Frozen heptadecanal identity changed",
    )
    observations, checks = [], {}
    for bd, name, cas, batch in (
        ("BD305903", "heptadecanal", "629-90-3", "batch_01"),
        ("BD57579", "bromotridecane", "765-09-3", "batch_02"),
    ):
        _, page = source(OUT / batch / "receipt.json", "id", f"{name}_bld")
        _, prices = source(OUT / "batch_03/receipt.json", "id", f"{name}_bld_public_price")
        receipt, stock = source(OUT / "stock_responses/receipt.json", "item", bd)
        args = [read(page), read_json(prices), read_json(stock), identities[cas], bd]
        observed = bld_observation(*args)
        zero = copy.deepcopy(args)
        for row in zero[2]["value"]:
            row["has_stock_quantityusa"] = 0
        must_reject(bld_observation, *zero)
        wrong = copy.deepcopy(args)
        wrong[3]["canonical_smiles"] = identities["108-91-8"]["canonical_smiles"]
        must_reject(bld_observation, *wrong)
        conflict = copy.deepcopy(args)
        row = copy.deepcopy(conflict[2]["value"][1])
        row["has_stock_quantityusa"] = 0
        conflict[2]["value"].append(row)
        must_reject(bld_observation, *conflict)
        reordered = copy.deepcopy(args)
        mol = Chem.MolFromSmiles(args[3]["canonical_smiles"])
        reordered[3]["canonical_smiles"] = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )
        require(bld_observation(*reordered) == observed, "Atom order changed disposition")
        checks[bd] = {
            "positive_stock_passed": True,
            "zero_stock_rejected": True,
            "wrong_graph_rejected": True,
            "conflicting_stock_rows_rejected": True,
            "atom_order_invariant": True,
        }
        observed.update(
            supplier="BLD Pharm",
            cas=cas,
            accessed_utc=receipt["completed_at_utc"],
            source_assets=[pin_record(p, ROOT) for p in (page, prices, stock)],
        )
        observations.append(observed)
    _, page = source(
        ROOT / "data/source_cache/ugi_guidance_evidence_20260913/batch_02/receipt.json",
        "id",
        "fisher_protected_propargyl_alcohol",
    )
    receipt, stock = source(
        V3 / "supplier_variant_stock_responses/receipt.json", "part", "AC295060250"
    )
    _, drawing = source(OUT / "batch_05/receipt.json", "id", "fisher_protected_propargyl_structure")
    args = [read(page), read_json(stock), identities["6089-04-9"], "AC295060250"]
    observed = fisher_observation(*args)
    zero = copy.deepcopy(args)
    zero[1]["priceAndAvailability"]["AC295060250"][0]["availableQuantity"] = "0"
    must_reject(fisher_observation, *zero)
    wrong = copy.deepcopy(args)
    wrong[0] = wrong[0].replace(
        observed["supplier_inchi_key"], observed["supplier_inchi_key"].replace("HQ", "AA", 1)
    )
    must_reject(fisher_observation, *wrong)
    checks["AC295060250"] = {
        "positive_fulfillment_quote_passed": True,
        "zero_stock_rejected": True,
        "wrong_full_key_rejected": True,
        "fixed_h_exact_graph_round_trip": True,
    }
    observed.update(
        supplier="Thermo Scientific Chemicals via Fisher Scientific",
        cas="6089-04-9",
        accessed_utc=receipt["requested_at_utc"],
        source_assets=[pin_record(p, ROOT) for p in (page, stock, drawing)],
        drawing_review="Supplier image visually checked; matches the frozen constitutional graph",
    )
    observations.append(observed)
    for record in observations:
        old = c16_by_cas.get(record["cas"])
        if old:
            lifetime = datetime.fromisoformat(
                old["expires_at_utc"].replace("Z", "+00:00")
            ) - datetime.fromisoformat(old["observed_at_utc"].replace("Z", "+00:00"))
        else:
            lifetime = datetime.fromisoformat(
                old_head["snapshot"]["expires_utc"].replace("Z", "+00:00")
            ) - datetime.fromisoformat(old_head["snapshot"]["accessed_utc"].replace("Z", "+00:00"))
        accessed = datetime.fromisoformat(record["accessed_utc"].replace("Z", "+00:00"))
        record["expires_utc"] = (accessed + lifetime).isoformat()
        record["availability_lifetime_days"] = lifetime.days
        current_window(record["accessed_utc"], record["expires_utc"], assessment_at)
        for instant in (
            accessed - timedelta(seconds=1),
            accessed + lifetime + timedelta(seconds=1),
        ):
            must_reject(
                current_window, record["accessed_utc"], record["expires_utc"], instant.isoformat()
            )
        record["disposition"] = "admit_exact_terminal_observation"
        checks[record["catalog_number"]]["expired_and_future_evidence_rejected"] = True
    return {
        "observations": observations,
        "checks": checks,
        "qualified_additional_terminal_count": len(observations),
        "assessment_at_utc": assessment_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = OUT / "terminal_qualification.json"
    existing = json.loads(output.read_text()) if args.verify else None
    if existing:
        for pin in existing["inputs"].values():
            require(
                sha256_file(ROOT / pin["path"]) == pin["sha256"], f"Input changed: {pin['path']}"
            )
    else:
        require(not output.exists(), "Preserve existing qualification")
    with rdBase.BlockLogs():
        payload = build(
            existing["payload"]["assessment_at_utc"]
            if existing
            else datetime.now(timezone.utc).isoformat()
        )
    if existing:
        require(payload == existing["payload"], "Qualification replay changed")
        print("Verified pinned sources, exact identities, stock evidence and adversarial controls")
        return
    INPUTS.add(Path(__file__))
    INPUTS.add(OUT / "batch_02/bld_product_detail.js")
    write_json(
        output,
        {
            "schema_version": "forge.fresh_exact_supplier_terminal_qualification.v1",
            "payload": payload,
            "inputs": {str(p.relative_to(ROOT)): pin_record(p, ROOT) for p in sorted(INPUTS)},
            "scope": "Independent exact terminal observations only; not loaded into the frozen cumulative source",
            "raw_supplier_assets_retained": True,
            "historical_pins_changed": False,
            "gates_relaxed": False,
            "routes_reassessed": 0,
            "guidance_run_ready": False,
            "nonclaims": [
                "No bottle count is inferred from BLD guest stock flags.",
                "Fisher provides a supplier fulfillment quote, not warehouse stock proof.",
                "Stock evidence does not establish reaction compatibility or synthesis success.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "additional_terminal_observations_qualified": len(payload["observations"]),
                "checks": payload["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
