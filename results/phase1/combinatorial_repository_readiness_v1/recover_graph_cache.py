"""Replay the frozen graph cache, restoring only byte-identical missing artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from forge.potency.oracle.oracle_graph_cache import build_oracle_graph_cache

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/bio/m0_07_oracle_graph_cache.json"
AUTHORITY = ROOT / "results/m0_07/oracle_graph_tensor_cache_result.json"


def pin(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write(path: Path, document: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> None:
    candidate = OUTPUT / "graph_cache_candidate"
    if candidate.exists():
        raise FileExistsError(f"preserve existing replay: {candidate}")
    authority = json.loads(AUTHORITY.read_text())
    config = json.loads(CONFIG.read_text())
    if pin(CONFIG)["sha256"] != authority["configuration"]["sha256"]:
        raise ValueError("historical cache configuration changed")
    for name, specification in config["inputs"].items():
        if pin(ROOT / specification["path"])["sha256"] != specification["sha256"]:
            raise ValueError(f"historical cache input changed: {name}")
    sources = {
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("forge.") and getattr(module, "__file__", None)
    }
    sources.add(Path(__file__).resolve())
    request = {
        "schema_version": "forge.graph_cache_exact_recovery_request.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": pin(CONFIG),
            "original_result": pin(AUTHORITY),
            **{name: pin(ROOT / value["path"]) for name, value in config["inputs"].items()},
        },
        "source_files": [pin(path) for path in sorted(sources)],
        "software": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "rdkit")},
        },
        "seed": config["seed"],
        "policy": {
            "training_calls": 0,
            "generation_calls": 0,
            "remote_jobs": 0,
            "restore_only_original_sha256": True,
            "overwrite_existing_files": False,
            "replace_original_result": False,
        },
    }
    request_path = OUTPUT / "graph_cache_recovery_request.json"
    write(request_path, request)
    build_oracle_graph_cache(CONFIG, candidate, ROOT)
    comparisons = []
    for filename, expected in authority["artifacts"].items():
        if Path(filename).name != filename:
            raise ValueError("historical artifact is not a filename")
        observed = pin(candidate / filename)
        comparisons.append(
            {
                "filename": filename,
                "expected": expected,
                "candidate": observed,
                "exact": all(observed[key] == expected[key] for key in ("sha256", "bytes")),
            }
        )
    exact = all(record["exact"] for record in comparisons)
    restored = []
    # No candidate is admitted unless the complete frozen artifact set matches.
    if exact:
        for record in comparisons:
            target = AUTHORITY.parent / record["filename"]
            if target.exists() or target.is_symlink():
                if target.is_symlink() or pin(target)["sha256"] != record["expected"]["sha256"]:
                    raise ValueError(f"existing artifact conflicts with original pin: {target}")
        for record in comparisons:
            target = AUTHORITY.parent / record["filename"]
            if not target.exists():
                with (candidate / record["filename"]).open("rb") as source:
                    with target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                if pin(target)["sha256"] != record["expected"]["sha256"]:
                    raise ValueError(f"restored bytes changed: {target}")
                restored.append(pin(target))
    result = {
        "schema_version": "forge.graph_cache_exact_recovery.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "original_artifacts_restored" if exact else "replay_mismatch_nothing_restored",
        "request": pin(request_path),
        "comparisons": comparisons,
        "restored": restored,
        "original_result_unchanged": pin(AUTHORITY) == request["inputs"]["original_result"],
        "original_inputs_unchanged": all(
            pin(ROOT / value["path"]) == value for value in request["inputs"].values()
        ),
        "source_files_unchanged": all(
            pin(ROOT / value["path"]) == value for value in request["source_files"]
        ),
        "scientific_thresholds_changed": False,
        "new_model_or_biological_result": False,
    }
    write(OUTPUT / "graph_cache_recovery.json", result)
    print(result["status"])


if __name__ == "__main__":
    main()
