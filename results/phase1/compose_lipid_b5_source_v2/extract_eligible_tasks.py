"""Authenticate only eligible B5 tasks; keep originals separate from chemistry admission."""

import gzip
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_full_preparation import FullPreparationCorpus
from forge.corpus.compose_lipid_original_binding import authenticate_reference, selected_lines
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/phase1/compose_lipid_b5_original_tasks_v1"
FAMILY = "vitamin_b5_multistep"


def main():
    if OUTPUT.exists():
        raise ComposeLipidError("Original B5 extraction output must be fresh")
    prep = ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
    reader = FullPreparationCorpus(ROOT, prep)
    print("Full preparation independently authenticated", flush=True)
    intake_path = resolve_pin(reader.reader.result["inputs"]["intake"], ROOT, label="intake")
    intake = json.loads(intake_path.read_text())
    constructions = resolve_pin(intake["inputs"]["constructions"], ROOT, label="constructions")
    bundle = constructions.parent
    export = bundle / "receipt.json"
    exported = json.loads(export.read_text())
    original_receipt = "artifacts/corpus_build_v2/family_23_vitamin_b5_enumeration_v8/receipt.json"
    task_dir = bundle / "original_generator_tasks/family_23_vitamin_b5_enumeration_v8"
    receipt = resolve_pin(
        {
            "path": str((task_dir / "receipt.json").relative_to(ROOT)),
            "sha256": exported["inputs"][original_receipt],
        },
        ROOT,
        label="B5 task receipt",
    )
    source = json.loads(receipt.read_text())
    if source["complete"] is not True or source["training_admissible"] is not False:
        raise ComposeLipidError("Unexpected original task receipt disposition")
    files = {}
    for entry in source["task_shards"]:
        name = entry["path"]
        local_pin = {
            "path": str((task_dir / "tasks" / Path(name).name).relative_to(ROOT)),
            "sha256": entry["file_sha256"],
        }
        local = resolve_pin(local_pin, ROOT, label=name)
        if name in files:
            raise ComposeLipidError("Duplicate original task shard")
        files[name] = {
            "source": name,
            "sha256": entry["file_sha256"],
            "local": local,
            "pin": local_pin,
        }
    prepared = {k: v for k, v in reader.preparation.items() if v["family"] == FAMILY}
    by_line = {v["construction_source_line"]: k for k, v in prepared.items()}
    if len(by_line) != len(prepared):
        raise ComposeLipidError("Duplicate eligible construction source line")
    selected = {}
    requests = defaultdict(lambda: defaultdict(list))
    for number, construction, payload_sha in selected_lines(constructions, set(by_line)):
        target = by_line[number]
        current = prepared[target]
        instances = sorted(
            [c["role"], c["component_id"], c["quantity"]]
            for c in construction["component_instances"]
        )
        if (
            construction["target_id"] != target
            or construction["primary_family"] != FAMILY
            or instances != current["component_instances"]
            or construction["construction_basis"] != current["construction_basis"]
        ):
            raise ComposeLipidError("Selected construction differs from protected preparation")
        record = {
            "preparation": current,
            "construction": construction,
            "construction_payload_sha256": payload_sha,
            "training_admitted": False,
        }
        selected[target] = record
        if current["construction_basis"] == "lnpdb_compatible_decomposition_replay":
            record["disposition"] = "compatible_decomposition_has_no_original_generator_task"
            continue
        ref = construction["provenance"]["task"]
        if ref["path"] not in files:
            raise ComposeLipidError("Selected original B5 task shard is missing")
        requests[ref["path"]][ref["line"]].append((target, ref))
    for original, lines in sorted(requests.items()):
        f = files[original]
        for number, row, payload_sha in selected_lines(f["local"], set(lines)):
            for target, reference in lines[number]:
                authenticate_reference(reference, f, number, payload_sha)
                c = selected[target]["construction"]
                if (
                    row["task_id"] != c["task_id"]
                    or row["family"] != FAMILY
                    or row["training_admissible"] is not False
                ):
                    raise ComposeLipidError("Original B5 task identity or disposition differs")
                selected[target].update(
                    original_task=row,
                    original_task_file=f["pin"],
                    disposition="original_task_bytes_authenticated",
                )
    for item in reader.iter_preparation_records(family=FAMILY):
        selected[item["preparation"]["target_id"]]["primary_metadata"] = item["source"].get(
            "primary_metadata", {}
        )
    if set(selected) != set(prepared) or any("disposition" not in r for r in selected.values()):
        raise ComposeLipidError("Original B5 extraction lost eligible rows")
    with tempfile.TemporaryDirectory(prefix=".b5-original-", dir=OUTPUT.parent) as tmp:
        stage = Path(tmp)
        ledger = stage / "selected-tasks.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
        ):
            for target in sorted(selected):
                stream.write((compact(selected[target]) + "\n").encode())
        report = {
            "schema_version": "forge.b5_eligible_original_task_extraction.v1",
            "seed": 0,
            "inputs": {
                "preparation": pin(ROOT, prep),
                "intake": pin(ROOT, intake_path),
                "export_receipt": pin(ROOT, export),
                "task_receipt": pin(ROOT, receipt),
                "constructions": pin(ROOT, constructions),
            },
            "task_shards": {k: v["pin"] for k, v in sorted(files.items())},
            "implementation": {
                p: pin(ROOT, ROOT / p)
                for p in (
                    str(Path(__file__).resolve().relative_to(ROOT)),
                    "forge/corpus/compose_lipid_original_binding.py",
                    "forge/corpus/compose_lipid_full_preparation.py",
                    "forge/core/hashing.py",
                )
            },
            "rows": len(selected),
            "dispositions": dict(Counter(r["disposition"] for r in selected.values())),
            "artifact": {
                "path": str((OUTPUT / ledger.name).relative_to(ROOT)),
                "sha256": sha256_file(ledger),
            },
            "selection_before_molecular_record_decoding": True,
            "component_structure_comparison_performed": False,
            "chemistry_qualified": False,
            "training_admitted": False,
        }
        dump(stage / "result.json", report)
        os.rename(stage, OUTPUT)
    print(json.dumps({"rows": report["rows"], "dispositions": report["dispositions"]}), flush=True)


if __name__ == "__main__":
    main()
