"""Authenticate exact upstream split sources without executing or modifying COMPOSE."""

import base64
import hashlib
import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "results/phase1/compose_lipid_v8_source_recovery_v1"


def main():
    request_path = (
        ROOT / "results/phase1/compose_lipid_training_readiness_v1/required-split-inputs.json"
    )
    request = json.loads(request_path.read_text())
    receipt_path = resolve_pin(request["receipt"], ROOT, label="original split receipt")
    receipt = json.loads(receipt_path.read_text())
    commit_path, tree_path = PREVIOUS / "commit.json", PREVIOUS / "commit-tree.json"
    commit, tree = (json.loads(p.read_text()) for p in (commit_path, tree_path))
    if tree["truncated"] is not False or tree["sha"] != commit["commit"]["tree"]["sha"]:
        raise ValueError("Incomplete or changed upstream commit tree")
    blobs = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    assets = []
    for source, local, response in [
        ("scripts/build_post_instruction_generator_splits_v8_1.py", PREVIOUS, None),
        ("configs/corpus/post_instruction_generator_splits_v8_1.json", PREVIOUS, None),
        ("src/compose_lipid/data/generator_splits_v2.py", HERE, "generator-splits-blob.json"),
        ("src/compose_lipid/data/training_corpus.py", HERE, "training-corpus-blob.json"),
    ]:
        path = local / "upstream" / source
        raw = path.read_bytes()
        git_digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_digest != blobs[source]["sha"] or len(raw) != blobs[source]["size"]:
            raise ValueError("Recovered source differs from recorded Git blob: " + source)
        value = pin(ROOT, path)
        required = next((r for r in request["required_files"] if r["path"] == source), None)
        if required:
            expected = receipt["producers"].get(source, receipt["inputs"].get(source))
            if expected != required["sha256"] or expected != value["sha256"]:
                raise ValueError("Recovered split source differs from original receipt: " + source)
        response_pin = None
        if response:
            response_path = HERE / response
            payload = json.loads(response_path.read_text())
            if payload["sha"] != git_digest or base64.b64decode(payload["content"]) != raw:
                raise ValueError("GitHub blob response differs from recovered bytes")
            response_pin = pin(ROOT, response_path)
        assets.append(
            {
                "upstream_path": source,
                **value,
                "git_blob_sha1": git_digest,
                "required_split_input": required is not None,
                "response": response_pin,
            }
        )
    recovered = {a["upstream_path"] for a in assets if a["required_split_input"]}
    if recovered != {r["path"] for r in request["required_files"]}:
        raise ValueError("Required split source recovery incomplete")
    dump(
        HERE / "result.json",
        {
            "schema_version": "forge.compose_lipid_split_source_recovery.v2",
            "date": "2026-09-21",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "original_request": pin(ROOT, request_path),
                "original_split_receipt": pin(ROOT, receipt_path),
                "upstream_commit": pin(ROOT, commit_path),
                "upstream_tree": pin(ROOT, tree_path),
            },
            "upstream_commit": commit["sha"],
            "assets": assets,
            "required_split_files_recovered": len(recovered),
            "missing_required_split_files": [],
            "upstream_code_executed": False,
            "upstream_code_modified": False,
            "full_universe_partition_qualified": False,
            "training_admitted": False,
            "next_work": "Independently reproduce supplied signatures and frozen assignments before extending partition coverage. Do not rerun group selection on the full population and thereby move prior holdouts.",
        },
    )


if __name__ == "__main__":
    main()
