"""Offline exact membership in the already downloaded public ZINC dataset, for disclosure review."""

import hashlib
import json
import resource
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3].parents[1]
OUT = ROOT / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    start = time.process_time()
    manifest_path = (
        ROOT / "configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json"
    )
    manifest = json.loads(manifest_path.read_text())
    stock = next(row for row in manifest["assets"] if row["path"].endswith("zinc_stock.hdf5"))
    path = ROOT / stock["path"]
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            md5.update(block)
    assert sha.hexdigest() == stock["sha256"]
    assert md5.hexdigest() == stock["published_md5"] and path.stat().st_size == stock["bytes"]
    proposal_path = OUT / "lookup_proposal_v3.json"
    targets = json.loads(proposal_path.read_text())["transmitted_identifiers"]
    dataset = pd.read_hdf(path, key="table")
    rows = []
    for identity, key in targets.items():
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(identity)) == key
        matches = dataset.index[dataset["inchi_key"].eq(key)].tolist()
        assert matches, "Exact full InChIKey absent from pinned public dataset"
        rows.append(
            {
                "identity": identity,
                "exact_full_inchikey": key,
                "dataset_row_indices": matches,
                "exact_membership": True,
            }
        )
    result = {
        "schema": "forge.public_stock_identity_disclosure_provenance.v1",
        "passed": True,
        "inputs": [pin(Path(__file__)), pin(manifest_path), pin(proposal_path)],
        "public_source_url": manifest["sources"]["stock"],
        "source_publication_provenance": "Existing hash-pinned public-stock runtime manifest; no live network retrieval during this check",
        "downloaded_dataset": {
            "path": stock["path"],
            "sha256": sha.hexdigest(),
            "md5": md5.hexdigest(),
            "published_md5_matched": True,
            "bytes": path.stat().st_size,
        },
        "dataset_rows": len(dataset),
        "targets": rows,
        "new_HTTP_calls": 0,
        "privacy_scope": "The two proposed query payloads are exact identities already present in this public historical dataset. No generated lipid, route, target product, research result, or biological data would be sent. This is evidence for disclosure review, not replacement approval.",
        "scientific_scope": "Historical public-dataset membership does not establish a current vendor listing, stock, procurement or any new makeability result.",
        "CPU_seconds": time.process_time() - start,
        "process_CPU_seconds": time.process_time(),
        "peak_RSS_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    with (OUT / "public_identity_provenance.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "result": pin(OUT / "public_identity_provenance.json"),
                "targets": rows,
                "CPU_seconds": result["CPU_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
