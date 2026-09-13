"""Preserve bounded acquisition evidence and replay its descriptive checks offline."""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.core.hashing import pin_record
from forge.core.io import atomic_write, write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RESULT = OUT / "result.json"
ARCHIVES = ROOT / "provenance/recovery/ugi_guidance_evidence_v5/assets"
CONFIG = ROOT / "configs/route/phase1_ugi3_exact_c18_route_v1.json"
PRIOR = ROOT / "results/phase1/ugi_guidance_evidence_acquisition_v4/result.json"


def read(path):
    return json.loads(path.read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def text(html):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


def aaron_observation(html, expected):
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    parsed = [
        [text(cell) for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S | re.I)]
        for row in rows
    ]
    identity = [cells[1] for cells in parsed if len(cells) == 2 and cells[0] == "SMILES"]
    if len(identity) != 1:
        raise ValueError("Supplier SMILES is missing or ambiguous")
    molecule = Chem.MolFromSmiles(identity[0])
    if molecule is None or Chem.MolToSmiles(molecule) != expected["canonical_smiles"]:
        raise ValueError("Supplier graph disagrees with frozen identity")
    if Chem.MolToInchiKey(molecule) != expected["inchi_key"]:
        raise ValueError("Supplier full InChIKey disagrees with frozen identity")
    packs = [
        {"pack_size": c[0], "purity": c[1], "price": c[2], "availability": c[3]}
        for c in parsed
        if len(c) == 7 and c[3] == "Global Stock"
    ]
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule),
        "inchi_key": Chem.MolToInchiKey(molecule),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "observed_global_stock_packs": packs,
        "us_item_specific_fulfillment_established": False,
        "disposition": "abstain",
    }


def describe():
    config = read(CONFIG)
    html = (OUT / "batch_01/stearolic_aaron.html").read_text()
    expected = config["molecules"]["stearolic_acid"]
    observed = aaron_observation(html, expected)
    no_stock = aaron_observation(html.replace("Global Stock", "Inquiry"), expected)
    wrong_rejected = False
    wrong = {
        **expected,
        "canonical_smiles": config["molecules"]["octadec_9_yn_1_ol"]["canonical_smiles"],
    }
    try:
        aaron_observation(html, wrong)
    except ValueError:
        wrong_rejected = True
    records = [r for p in sorted(OUT.glob("batch_*/receipt.json")) for r in read(p)["records"]]
    files = read(OUT / "batch_02/oppolzer_figshare_metadata.json")["files"]
    return {
        "http_outcomes": dict(
            Counter(str(r.get("http_status", "transport_failure")) for r in records)
        ),
        "public_requests": len(records),
        "aaron": observed,
        "checks": {
            "exact_supplier_identity_matches": True,
            "zero_stock_does_not_create_positive_pack": not no_stock["observed_global_stock_packs"],
            "incorrect_graph_rejected": wrong_rejected,
            "observed_purity_preserved": all(
                p["purity"] == "80%" for p in observed["observed_global_stock_packs"]
            ),
        },
        "publisher_open_file_names": [f["name"] for f in files],
        "prior_qualified_target_materials": read(PRIOR)["fresh_evidence"][
            "total_qualified_target_material_observations"
        ],
        "new_terminal_admissions": 0,
        "new_complete_routes": 0,
        "main_article_recovered": False,
    }


def verify(result, restore):
    archived = {entry["original"]["path"]: entry for entry in result["archived_sources"]}
    for pin in result["inputs"]:
        path = ROOT / pin["path"]
        if pin["path"] in archived:
            entry = archived[pin["path"]]
            packed = (ROOT / entry["archive"]["path"]).read_bytes()
            if digest(packed) != entry["archive"]["sha256"]:
                raise ValueError("Compressed archive changed")
            raw = gzip.decompress(packed)
            if digest(raw) != pin["sha256"]:
                raise ValueError("Archive does not reproduce source")
            if restore and not path.exists():
                if not path.resolve().is_relative_to(OUT):
                    raise ValueError("Restore path escapes this acquisition")
                atomic_write(path, raw)
        if digest(path.read_bytes()) != pin["sha256"]:
            raise ValueError(f"Preserve changed input: {path}")
    if describe() != result["findings"]:
        raise ValueError("Descriptive evidence changed on replay")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.verify or args.restore:
        verify(read(RESULT), args.restore)
        print("Verified archived sources, identity controls and the abstaining result")
        return
    if RESULT.exists():
        raise ValueError("Preserve the existing acquisition result")
    findings = describe()
    if not all(findings["checks"].values()):
        raise ValueError("Descriptive source checks failed")
    paths = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [CONFIG, PRIOR, ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for receipt in OUT.glob("batch_*/receipt.json"):
        for pin in read(receipt)["inputs"].values():
            path = ROOT / pin["path"]
            if digest(path.read_bytes()) != pin["sha256"]:
                raise ValueError(f"Acquisition producer changed: {path}")
            paths.append(path)
    archives = []
    for path in sorted(set(paths)):
        if path.is_relative_to(OUT) and path.suffix in (".html", ".png"):
            original = pin_record(path, ROOT)
            packed = gzip.compress(path.read_bytes(), mtime=0)
            destination = ARCHIVES / (original["sha256"] + ".gz")
            if destination.exists() and destination.read_bytes() != packed:
                raise ValueError("Preserve changed archive")
            atomic_write(destination, packed)
            archives.append({"original": original, "archive": pin_record(destination, ROOT)})
    result = {
        "schema_version": "forge.guidance_source_acquisition_followup.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "new_stock_lead_preserved_no_additional_admission_guidance_blocked",
        "inputs": [pin_record(p, ROOT) for p in sorted(set(paths))],
        "archived_sources": archives,
        "findings": findings,
        "review": read(OUT / "review.json"),
        "random_sampling_used": False,
        "global_phase1_definition_of_done_met": False,
        "validation_scope": "Source authentication and descriptive checks only; repository-wide tests not rerun",
    }
    verify(result, False)
    write_json(RESULT, result)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
