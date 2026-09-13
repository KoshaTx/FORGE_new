"""Pin new evidence and inventory recovery gaps without changing any admission decision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from experiments._runtime.historical import resolve_pinned_input

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "data/source_cache/ugi_guidance_evidence_20260913"
ARCHIVE = ROOT / "provenance/recovery/ugi_guidance_evidence_v3/assets"
PERMANENT = ROOT / "provenance/recovery/ugi_guidance_evidence_acquisition_v3.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def pin(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def check(item: dict) -> None:
    path = Path(item["path"])
    path = path if path.is_absolute() else ROOT / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Input changed: {path}")


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid molecular identity")
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def build() -> dict:
    inputs = {}
    records = []
    for receipt in sorted(CACHE.glob("batch_*/receipt.json")):
        inputs[str(receipt.relative_to(ROOT))] = pin(receipt)
        data = read(receipt)
        for item in data["inputs"].values():
            check(item)
            path = Path(item["path"])
            path = path if path.is_absolute() else ROOT / path
            inputs[str(path.relative_to(ROOT))] = pin(path)
        for record in data["records"]:
            if "response" in record:
                check(record["response"])
            records.append(record)
    for path in (OUT / "review.json", Path(__file__), CACHE / "indexed_source_responses.txt"):
        inputs[str(path.relative_to(ROOT))] = pin(path)

    configs = {}
    for label, name in {
        "head": "phase1_ugi3_third_wave_head_terminals_v1",
        "tail": "phase1_ugi3_high_leverage_tail_terminals_v1",
        "c16": "phase1_ugi3_exact_c16_route_v1",
        "c18": "phase1_ugi3_exact_c18_route_v1",
    }.items():
        path = ROOT / f"configs/route/{name}.json"
        configs[label] = read(path)
        inputs[str(path.relative_to(ROOT))] = pin(path)
    targets = []
    for label in ("head", "tail"):
        config = configs[label]
        for record in config["records"]:
            targets.append(
                {
                    "name": record["name"],
                    "canonical_smiles": record["canonical_smiles"],
                    "cas": record["cas"],
                    "original_expires_utc": config["snapshot"]["expires_utc"],
                }
            )
    for record in configs["c16"]["terminal_procurement"] + [configs["c18"]["terminal_procurement"]]:
        targets.append(
            {
                "name": record["molecule_id"],
                "canonical_smiles": record["canonical_smiles"],
                "cas": record["cas_number"],
                "original_expires_utc": record["expires_at_utc"],
            }
        )
    candidates = []
    for record in records:
        if record["id"].startswith("pubchem_") and record.get("http_status") == 200:
            for prop in read(ROOT / record["response"]["path"])["PropertyTable"]["Properties"]:
                candidates.append({"properties": prop, "source": record["response"]})
    identity_checks = []
    selected_cids = set()
    now = datetime.now(timezone.utc)
    for target in targets:
        wanted = canonical(target["canonical_smiles"])
        matches = [
            candidate
            for candidate in candidates
            if canonical(candidate["properties"]["ConnectivitySMILES"]) == wanted
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one fresh identity match for {target['name']}")
        match = matches[0]
        prop = match["properties"]
        molecule = Chem.MolFromSmiles(wanted)
        reordered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        checks = {
            "canonical_graph_matches": True,
            "formula_matches": rdMolDescriptors.CalcMolFormula(molecule)
            == prop["MolecularFormula"],
            "inchi_key_matches": Chem.MolToInchiKey(molecule) == prop["InChIKey"],
            "atom_order_invariant": canonical(Chem.MolToSmiles(reordered)) == wanted,
        }
        if not all(checks.values()):
            raise ValueError(f"Identity mismatch: {target['name']}")
        selected_cids.add(prop["CID"])
        identity_checks.append(
            {
                **target,
                "pubchem_cid": prop["CID"],
                "source": match["source"],
                "checks": checks,
                "old_availability_expired": now
                > datetime.fromisoformat(target["original_expires_utc"].replace("Z", "+00:00")),
                "current_l3_admitted": False,
            }
        )

    html_path = CACHE / "batch_01/chemimpex_03599.html"
    html = html_path.read_text()
    products = [
        json.loads(body)
        for body in re.findall(r'<script type="application/json">(.*?)</script>', html, re.S)
    ]
    (product,) = [p for p in products if isinstance(p, dict) and isinstance(p.get("product"), dict)]
    if product["product"]["title"] != configs["head"]["records"][0]["name"]:
        raise ValueError("Supplier product title mismatch")
    available = [
        v["sku"]
        for v in product["product"]["variants"]
        if v["available"] is True and product["formatted"][str(v["id"])]["lead_time"] == -1
    ]
    if len(available) != html.count('class="opt-availability">Ships Today</span>'):
        raise ValueError("Supplier stock labels disagree with variant data")

    previous_path = ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/remaining_inputs.json"
    inputs[str(previous_path.relative_to(ROOT))] = pin(previous_path)
    recovery = []
    for item in read(previous_path)["identities"]:
        expected = item["expected"]
        try:
            resolved = resolve_pinned_input(ROOT, expected["path"], expected["sha256"])
            recovery.append(
                {"expected": expected, "status": "recovered", "resolved": pin(resolved)}
            )
        except (ValueError, FileNotFoundError) as error:
            recovery.append({"expected": expected, "status": "unavailable", "error": str(error)})
    archived = []
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    for path in sorted(CACHE.rglob("*")):
        if not path.is_file():
            continue
        original = pin(path)
        destination = ARCHIVE / (original["sha256"] + ".gz")
        payload = gzip.compress(path.read_bytes(), mtime=0)
        if not destination.exists():
            destination.write_bytes(payload)
        archived.append({"original": original, "archive": pin(destination)})
    si = (
        ROOT / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/oppolzer_jo000463n_si_001.pdf"
    )
    check(configs["c18"]["inputs"]["oppolzer_primary_si"])
    inputs[str(si.relative_to(ROOT))] = pin(si)
    return {
        "schema_version": "forge.guidance_evidence_acquisition.v3",
        "created_at_utc": now.isoformat(),
        "status": "fresh_sources_preserved_qualification_incomplete",
        "inputs": inputs,
        "software": {"rdkit": rdBase.rdkitVersion},
        "seed": None,
        "random_sampling_used": False,
        "acquisition": {
            "requests": len(records),
            "http_200_responses": sum(r.get("http_status") == 200 for r in records),
            "http_error_responses": sum(r.get("http_status", 0) >= 400 for r in records),
            "transport_failures": sum(r["status"] == "retrieval_failed" for r in records),
            "records": records,
        },
        "identity_checks": identity_checks,
        "excluded_pubchem_candidates": [
            {"cid": x["properties"]["CID"], "reason": "graph_does_not_match_any_requested_terminal"}
            for x in candidates
            if x["properties"]["CID"] not in selected_cids
        ],
        "fresh_supplier_observation": {
            "source": pin(html_path),
            "same_day_skus": available,
            "admitted_to_guidance": False,
        },
        "review": read(OUT / "review.json"),
        "visual_checks": [pin(p) for p in sorted((OUT / "previews").glob("*.png"))],
        "archived_sources": archived,
        "historical_inventory": {
            "scope": "original 55 direct missing input identities, not complete transitive closure",
            "recovered": sum(x["status"] == "recovered" for x in recovery),
            "unavailable": sum(x["status"] == "unavailable" for x in recovery),
            "identities": recovery,
        },
        "policy": {
            "historical_pins_changed": False,
            "gates_relaxed": False,
            "guidance_run_ready": False,
            "newly_qualified_routes": 0,
            "newly_qualified_terminals": 0,
            "training_calls": 0,
            "generation_calls": 0,
            "candidate_selection": False,
            "purchase_or_contact_actions": 0,
        },
    }


def verify(report: dict) -> None:
    for item in report["inputs"].values():
        check(item)
    for item in report["archived_sources"]:
        check(item["archive"])
        restored = gzip.decompress((ROOT / item["archive"]["path"]).read_bytes())
        if hashlib.sha256(restored).hexdigest() != item["original"]["sha256"]:
            raise ValueError("Archive decompression does not reproduce the source bytes")
    for item in report["historical_inventory"]["identities"]:
        if item["status"] == "recovered":
            check(item["resolved"])
    for item in report["visual_checks"]:
        check(item)
    if any(
        report["policy"][k]
        for k in ("historical_pins_changed", "gates_relaxed", "guidance_run_ready")
    ):
        raise ValueError("Acquisition receipt cannot admit evidence or change frozen gates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(read(PERMANENT))
        print("All source, archive, recovered-input and visual-artifact pins verified")
        return
    if PERMANENT.exists() or (OUT / "result.json").exists():
        raise ValueError("Report already exists; preserve this acquisition version")
    result = build()
    verify(result)
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (OUT / "result.json").write_text(body)
    PERMANENT.write_text(body)
    print(
        json.dumps(
            {
                "status": result["status"],
                "identity_checks": len(result["identity_checks"]),
                "historical_inventory": {
                    k: result["historical_inventory"][k] for k in ("recovered", "unavailable")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
