"""Authenticate relocated historical source inputs without changing their logical identities.

These helpers read source as evidence bytes. They do not authorize execution of archived code.
Data inputs remain at their declared locations; only Python source may use a reviewed move or
an exact original-path/digest archive entry. Result comparison normalizes only input locations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from forge.core.hashing import PinError, is_sha256, resolve_pin, sha256_file


def input_location_matches(repo: Path, path: Path, asset: str, digest: str) -> bool:
    """Require declared data location or an authenticated relocation of Python source."""
    if not isinstance(asset, str) or not asset or not is_sha256(digest):
        return False
    repo = repo.resolve()
    declared = Path(asset)
    if ".." in declared.parts or path.is_symlink():
        return False
    direct = (repo / declared).resolve()
    actual = path.resolve()
    allowed = actual == direct
    # Absolute inputs preserve the existing explicit-input behavior; they cannot be aliased.
    source = (
        not declared.is_absolute()
        and declared.suffix == ".py"
        and declared.parts[0]
        in {
            "src",
            "scripts",
            "forge",
            "experiments",
            "tools",
            "cli",
        }
    )
    try:
        if not allowed and source:
            moves_path = repo / "docs/artifact_path_moves.json"
            if moves_path.is_file():
                moved = json.loads(moves_path.read_text()).get("moves", {}).get(asset)
                if isinstance(moved, str):
                    allowed = actual == (repo / moved).resolve()
            if not allowed:
                manifest = repo / "provenance/frozen-code/manifest.json"
                archive = json.loads(manifest.read_text())
                if (
                    set(archive) != {"schema_version", "entries"}
                    or archive["schema_version"] != "forge.historical_pin_archive.v1"
                    or not isinstance(archive["entries"], list)
                ):
                    return False
                entries = [
                    e
                    for e in archive["entries"]
                    if e.get("original_path") == asset and e.get("sha256") == digest
                ]
                if len(entries) != 1 or set(entries[0]) != {
                    "original_path",
                    "sha256",
                    "blob_path",
                    "source",
                }:
                    return False
                entry = entries[0]
                blob_path = entry["blob_path"]
                if (
                    not isinstance(entry["source"], dict)
                    or not isinstance(blob_path, str)
                    or not blob_path
                    or Path(blob_path).is_absolute()
                    or ".." in Path(blob_path).parts
                ):
                    return False
                blob = repo / blob_path
                allowed = not blob.is_symlink() and actual == blob.resolve()
        if not allowed:
            return False
        # Source aliases must remain within the repository. Explicit absolute data inputs
        # remain supported by their original API and still require their exact digest.
        if source:
            actual.relative_to(repo)
        if declared.is_absolute() and not actual.is_relative_to(repo):
            return actual.is_file() and sha256_file(actual) == digest
        resolve_pin({"path": str(actual), "sha256": digest}, repo, label=asset)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, PinError):
        return False


def results_match_after_input_relocation(
    stored: Mapping[str, Any],
    fresh: Mapping[str, Any],
    *,
    repo: Path,
    specifications: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> bool:
    """Compare every field after authenticating and normalizing only inputs.*.path.

    Old workstation prefixes are location metadata. Their logical asset suffix, input name,
    SHA-256, current physical binding, scientific fields and output hashes must all agree.
    Source locators outside inputs.*.path are not rewritten or ignored.
    """
    if set(specifications) != set(input_paths):
        return False
    repo = repo.resolve()
    left, right = deepcopy(dict(stored)), deepcopy(dict(fresh))
    try:
        if set(left["inputs"]) != set(specifications) or set(right["inputs"]) != set(
            specifications
        ):
            return False
        for name, spec in specifications.items():
            asset, digest = spec["asset"], spec["expected_sha256"]
            path = input_paths[name]
            if not input_location_matches(repo, path, asset, digest):
                return False
            physical = {str(path), str(path.resolve())}
            if path.resolve().is_relative_to(repo):
                physical.add(path.resolve().relative_to(repo).as_posix())
            if right["inputs"][name]["path"] not in physical:
                return False
            for document in (left, right):
                record = document["inputs"][name]
                if set(record) != {"path", "sha256"} or record["sha256"] != digest:
                    return False
                location = record["path"]
                if not isinstance(location, str) or ".." in Path(location).parts:
                    return False
                logical = location == asset or (
                    not Path(asset).is_absolute()
                    and Path(location).is_absolute()
                    and location.endswith("/" + asset)
                )
                if not logical and location not in physical:
                    return False
                record["path"] = asset
        return left == right
    except (KeyError, TypeError, AttributeError):
        return False
