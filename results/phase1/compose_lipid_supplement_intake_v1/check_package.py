"""Record the requested checksum command and authenticate supplied output receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "data/source_cache/compose_lipid_supplement_2026-09-19"


def pin(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "sha256": h.hexdigest()}


def main() -> None:
    command = ["shasum", "-a", "256", "-c", "SHA256SUMS"]
    with (OUT / "shasum.log").open("w") as stream:
        completed = subprocess.run(command, cwd=CACHE, stdout=stream, stderr=stream, check=False)
    files = []
    for line in (CACHE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe manifest path")
        path = CACHE / relative
        actual = pin(path)["sha256"] if path.is_file() else None
        files.append(dict(path=relative, expected_sha256=expected, actual_sha256=actual))
    mismatched = [
        row for row in files if row["actual_sha256"] not in (None, row["expected_sha256"])
    ]
    missing = [row["path"] for row in files if row["actual_sha256"] is None]
    receipts = []
    for relative in (
        "receipt.json",
        "post_instruction_component_manifest_v8_1/receipt.json",
        "post_instruction_generator_splits_v8_1/receipt.json",
    ):
        path = CACHE / relative
        receipt = json.loads(path.read_text())
        if receipt.get("complete") is not True:
            raise ValueError(f"Incomplete producer receipt: {relative}")
        checked = []
        for name, value in receipt["outputs"].items():
            artifact = path.parent / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe receipt output")
            expected = value["sha256"] if isinstance(value, dict) else value
            actual = pin(artifact)
            if actual["sha256"] != expected:
                raise ValueError(f"Changed receipt output: {artifact}")
            if isinstance(value, dict) and artifact.stat().st_size != value["bytes"]:
                raise ValueError(f"Changed receipt output size: {artifact}")
            checked.append(actual)
        receipts.append(dict(receipt=pin(path), verified_outputs=checked))
    original = ROOT / "data/source_cache/compose_lipid_drive_2026-09-18/global_candidates.jsonl.gz"
    export = json.loads((CACHE / "receipt.json").read_text())
    bound = export["inputs"][
        "artifacts/corpus_build_v2/post_instruction_global_union_v8/global_candidates.jsonl.gz"
    ]
    original_pin = pin(original)
    if original_pin["sha256"] != bound:
        raise ValueError("Supplement binds a different original corpus")
    result = dict(
        schema_version="forge.compose_lipid_supplement_package_check.v1",
        command=command,
        working_directory=str(CACHE.relative_to(ROOT)),
        exit_code=completed.returncode,
        status=(
            "verified"
            if completed.returncode == 0 and not missing and not mismatched
            else "incomplete"
        ),
        implementation=pin(Path(__file__)),
        checksum_manifest=pin(CACHE / "SHA256SUMS"),
        original_corpus=original_pin,
        receipt_output_checks=receipts,
        manifest_files=len(files),
        verified_files=len(files) - len(missing) - len(mismatched),
        missing_paths=missing,
        mismatched_files=mismatched,
        file_checks=files,
        seed=0,
        training_admitted=False,
        scope="Delivered checksums and receipt outputs; upstream generation not rerun.",
    )
    (OUT / "package_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "exit_code",
                    "manifest_files",
                    "verified_files",
                    "missing_paths",
                    "mismatched_files",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
