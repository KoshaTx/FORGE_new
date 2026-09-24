"""Inventory the named first-failure dependencies without altering evidence or tests."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from forge_provenance.resolver import HistoricalPinArchive

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASELINE = ROOT / "results/phase1/combinatorial_generation_pipeline_validation_v1/result.json"
ARCHIVE = ROOT / "provenance/frozen-code/manifest.json"
MOVES = ROOT / "docs/artifact_path_moves.json"
SIBLINGS = (
    ROOT.parent / "forge",
    ROOT / ".claude/worktrees/agent-af6433f37639e717d",
    ROOT / ".claude/worktrees/agent-af89163cf6155eabd",
    ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/replay_tree",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main() -> None:
    baseline = json.loads(BASELINE.read_text())
    failure_paths = defaultdict(list)
    unmatched = []
    pattern = re.compile(re.escape(str(ROOT)) + r"/([^\s'\"<>]+)")
    for failure in baseline["test_reports"]["full"]["failures"]:
        paths = sorted(set(pattern.findall(failure["message"])))
        if paths:
            for path in paths:
                failure_paths[path].append(failure)
        else:
            unmatched.append(failure)

    declarations = defaultdict(list)

    def walk(node: object, source: Path, location: str = "$") -> None:
        if isinstance(node, dict):
            path = node.get("path", node.get("asset"))
            expected = node.get("sha256")
            if (
                isinstance(path, str)
                and path in failure_paths
                and isinstance(expected, str)
                and len(expected) == 64
            ):
                declarations[path].append(
                    {
                        "declared_by": str(source.relative_to(ROOT)),
                        "location": location,
                        "sha256": expected,
                    }
                )
            for key, value in node.items():
                walk(value, source, f"{location}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, source, f"{location}[{index}]")

    documents = set((ROOT / "configs").rglob("*.json"))
    for path in failure_paths:
        parent = (ROOT / path).parent
        for name in ("result.json", "oracle_graph_tensor_cache_result.json"):
            candidate = parent / name
            if candidate.is_file():
                documents.add(candidate)
    scanned = []
    for path in sorted(documents):
        if path.is_symlink():
            raise ValueError(f"refuse symlinked metadata: {path}")
        document = json.loads(path.read_text())
        scanned.append(pin(path))
        walk(document, path)
        # Some frozen results key artifact pins by basename within their own directory.
        artifacts = document.get("artifacts", {}) if isinstance(document, dict) else {}
        for filename, value in artifacts.items() if isinstance(artifacts, dict) else ():
            if Path(filename).name != filename:
                continue
            if isinstance(value, dict) and isinstance(value.get("sha256"), str):
                local = str((path.parent / filename).relative_to(ROOT))
                if local in failure_paths:
                    declarations[local].append(
                        {
                            "declared_by": str(path.relative_to(ROOT)),
                            "location": f"$.artifacts.{filename}",
                            "sha256": value["sha256"],
                        }
                    )

    archive = HistoricalPinArchive.load(ARCHIVE, ROOT)
    moves = json.loads(MOVES.read_text())["moves"]
    records = []
    for path, failures in sorted(failure_paths.items()):
        expected = sorted({value["sha256"] for value in declarations[path]})
        # This inventory does not open sealed populations or inspect their structures.
        protected = "holdout" in path or Path(path).name == "seal.json"
        candidates = []
        if not protected:
            locations = [("active", ROOT / path)]
            if path in moves:
                locations.append(("relocated", ROOT / moves[path]))
            locations.extend(("sibling_or_replay", sibling / path) for sibling in SIBLINGS)
            for kind, candidate in locations:
                entry = {"kind": kind, "location": str(candidate)}
                if candidate.is_symlink():
                    entry["status"] = "symlink_not_read"
                elif candidate.is_file():
                    entry["sha256"] = digest(candidate)
                    entry["matches_declared_sha256"] = entry["sha256"] in expected
                    entry["status"] = "file"
                else:
                    entry["status"] = "absent"
                candidates.append(entry)
            for sha256 in expected:
                candidate = archive.resolve(path, sha256)
                if candidate is not None:
                    candidates.append(
                        {
                            "kind": "archive",
                            "location": str(candidate),
                            "sha256": sha256,
                            "matches_declared_sha256": True,
                            "status": "file",
                        }
                    )
        records.append(
            {
                "path": path,
                "failures": failures,
                "declarations": declarations[path],
                "candidates": candidates,
                "protected_content_not_read": protected,
                "any_exact_bytes_located": any(
                    value.get("matches_declared_sha256", False) for value in candidates
                ),
            }
        )
    result = {
        "schema_version": "forge.first_failure_dependency_inventory.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline": pin(BASELINE),
            "archive_manifest": pin(ARCHIVE),
            "relocations": pin(MOVES),
            "script": pin(Path(__file__).resolve()),
        },
        "scanned_documents": scanned,
        "counts": {
            "baseline_failures_by_kind": dict(
                Counter(value["kind"] for value in baseline["test_reports"]["full"]["failures"])
            ),
            "named_dependency_paths": len(records),
            "dependencies_with_some_exact_bytes": sum(
                value["any_exact_bytes_located"] for value in records
            ),
            "messages_without_an_absolute_dependency_path": len(unmatched),
        },
        "dependencies": records,
        "messages_requiring_manual_trace": unmatched,
        "limits": [
            "First reported dependencies only; later failures may be masked.",
            "Declarations are path matches, not proof of the hash requested by each failing test.",
            "Archived source bytes do not authenticate execution by a different live module.",
            "No candidate is installed and no expected hash is changed by this inventory.",
            "Protected holdout paths are reported from failure metadata without opening content.",
            "An absent candidate is not proof that every backup or remote store lacks the file.",
        ],
    }
    with (OUTPUT / "dependency_inventory.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
