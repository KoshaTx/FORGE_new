"""Archive all five source deltas before integrating reviewed implementations."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
SOURCE_DIRS = {"forge", "experiments", "tests", "configs", "docs"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def source_paths(root):
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    return sorted({
        path for path in paths if path and
        (path.split("/")[0] in SOURCE_DIRS or path in {"Makefile", "pyproject.toml"})
        and (root / path).is_file()
    })


def main():
    baseline_path = HERE.parent / "baseline/source_manifest.json"
    baseline = json.loads(baseline_path.read_text())
    original = {row["path"]: row["sha256"] for row in baseline["files"]}
    root_before = {path: digest(ROOT / path) for path in source_paths(ROOT)}
    streams, destinations = {}, {}
    for name, spec in baseline["workstreams"].items():
        worktree = Path(spec["worktree"])
        unchanged, modified, missing, additions = [], [], [], []
        for row in baseline["files"]:
            path = worktree / row["path"]
            if not path.is_file():
                missing.append(row["path"])
            elif digest(path) != row["sha256"]:
                modified.append(row["path"])
            else:
                unchanged.append(row["path"])
        assert not modified and not missing, (name, modified, missing)
        for relative in source_paths(worktree):
            if relative in original:
                continue
            source = worktree / relative
            source_hash = digest(source)
            existing = root_before.get(relative)
            assert existing is None or existing == source_hash, (name, relative, "root conflict")
            if relative in destinations:
                assert destinations[relative] == source_hash, (relative, "worktree conflict")
            destinations[relative] = source_hash
            archived = HERE / "archive" / name / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            with archived.open("xb") as stream:
                stream.write(source.read_bytes())
            assert digest(archived) == source_hash
            additions.append({
                "relative_path": relative, "source": pin(source), "archive": pin(archived),
                "main_before": "identical" if existing else "absent",
            })
        streams[name] = {
            "worktree": str(worktree.relative_to(ROOT)),
            "unchanged_baseline_files": len(unchanged),
            "modified_baseline_files": modified, "missing_baseline_files": missing,
            "additions": additions,
        }
    result = {
        "schema": "forge.iclr22.worktree_consolidation_inventory.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": pin(Path(__file__)), "baseline": pin(baseline_path),
        "baseline_source_digest": baseline["source_digest"],
        "main_source_before": root_before, "streams": streams,
        "preserved_worktrees": True, "other_worktrees_in_scope": False,
        "main_source_modified": False, "manuscript_modified": False,
        "unique_delta_paths": len(destinations),
    }
    with (HERE / "inventory.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "inventory": pin(HERE / "inventory.json"),
        "streams": {name: len(value["additions"]) for name, value in streams.items()},
        "unique_delta_paths": len(destinations),
    }))


if __name__ == "__main__":
    main()
