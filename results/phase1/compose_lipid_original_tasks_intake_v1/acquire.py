"""Acquire the user-supplied original-task bundle with the previously audited downloader."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "data/source_cache/compose_lipid_original_tasks_2026-09-20"
FOLDER = "1yWMDZ_4J1WNuVEtyw5Yi5gyatL3PbX0C"
DOWNLOADER = ROOT / "results/phase1/compose_lipid_supplement_intake_v1/acquire.py"


def main():
    spec = importlib.util.spec_from_file_location("original_task_download", DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CACHE = CACHE
    module.ACQ = CACHE / "_acquisition"
    module.FOLDER = FOLDER
    module.ACQ.mkdir(parents=True, exist_ok=True)
    listing = module.listing((FOLDER, ""))
    by_name = {item["path"]: item for item in listing}
    # README was fetched and read before this acquisition. Retain it and bootstrap the pins.
    for name in ("README.md", "SHA256SUMS", "MANIFEST.json"):
        module.download(by_name[name], {})
    module.main()
    receipt = {
        "schema_version": "forge.compose_lipid_original_tasks_acquisition.v1",
        "source_url": "https://drive.google.com/drive/folders/" + FOLDER,
        "seed": 0,
        "implementation": {
            str(path.relative_to(ROOT)): module.digest(path)
            for path in (Path(__file__), DOWNLOADER)
        },
        "download_receipt": {
            "path": str((module.ACQ / "download_receipt.json").relative_to(ROOT)),
            "sha256": module.digest(module.ACQ / "download_receipt.json"),
        },
        "training_admitted": False,
        "training_calls": 0,
    }
    (OUT / "acquisition.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
