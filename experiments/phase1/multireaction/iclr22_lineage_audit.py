"""Locate exact missing ancestry files by name and hash without parsing chemical records."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    wall = time.monotonic()
    source = (
        args.root
        / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1/reference_lineage/reconstruction.json"
    )
    rows = json.loads(source.read_text())["exact_upstream_ancestor_requests"]
    globs = set()
    for row in rows:
        path = Path(row["upstream_relative_path"])
        globs.add(
            "**/" + str(Path(path.parent.name) / path.name)
            if path.name == "receipt.json"
            else "**/" + path.name
        )
    command = [
        "rg",
        "--files",
        "--hidden",
        "--no-ignore",
        "-g",
        "!.git/**",
        "-g",
        "!**/.venv/**",
        "-g",
        "!**/node_modules/**",
        "-g",
        "!**/__pycache__/**",
    ]
    for pattern in sorted(globs):
        command += ["-g", pattern]
    command.append(str(args.search_root.resolve()))
    process = subprocess.run(command, capture_output=True, timeout=60)
    if process.returncode not in (0, 1):
        raise RuntimeError(process.stderr.decode())
    paths = sorted(set(Path(p) for p in process.stdout.decode().splitlines()))
    candidates = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths]
    by_hash = {}
    for candidate in candidates:
        by_hash.setdefault(candidate["sha256"], []).append(candidate["path"])
    outcomes = [
        dict(row, recovered_exact_paths=by_hash.get(row["expected_sha256"], [])) for row in rows
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    producer_copy = args.output / "producer.py"
    producer_copy.write_bytes(Path(__file__).read_bytes())
    (args.output / "search.stdout").write_bytes(process.stdout)
    (args.output / "search.stderr").write_bytes(process.stderr)
    result = {
        "schema": "forge.iclr22.metadata_ancestry_search.v1",
        "hypothesis": "Exact previously missing ancestry files may survive under relocated paths or sibling projects.",
        "alternative_explanation": "Names may match while file versions differ; only expected SHA256 admits recovery.",
        "seed": 2026092681,
        "inputs": [{"path": str(source), "sha256": digest(source)}],
        "producer": {
            "path": str(producer_copy.resolve()),
            "sha256": digest(producer_copy),
        },
        "command": command,
        "returncode": process.returncode,
        "search_output_sha256": hashlib.sha256(process.stdout).hexdigest(),
        "candidate_files": candidates,
        "requested_ancestors": len(rows),
        "recovered_ancestors": sum(bool(row["recovered_exact_paths"]) for row in outcomes),
        "outcomes": outcomes,
        "chemical_records_parsed": 0,
        "protected_target_graphs_loaded": 0,
        "independent_reference_admitted": False,
        "network_calls": 0,
        "cost": {
            "parent_CPU_seconds": time.process_time() - started,
            "wall_seconds": time.monotonic() - wall,
            "rg_child_CPU_included": False,
        },
        "interpretation": "Exact recovery is prerequisite evidence only; source ancestry and adaptive exposure still require review.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ["requested_ancestors", "recovered_ancestors", "cost"]})
    )


if __name__ == "__main__":
    main()
