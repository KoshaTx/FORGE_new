#!/usr/bin/env python3
"""Run selected-v2 pooled/singleton/direct restartable equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_selected_v2_pool_singleton_equivalence import (
    build_selected_v2_pool_singleton_equivalence,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO
            / "configs/model/phase1_ugi_selected_v2_pool_singleton_equivalence_v1.json"
        ),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = json.loads(config_path.read_text())
    output_dir = REPO / config["output_dir"]
    result_path = output_dir / "result.json"
    rows_path = output_dir / "comparison_rows.json"
    if result_path.exists() or rows_path.exists():
        raise SystemExit("pool-singleton equivalence output exists; refusing overwrite")
    result, rows = build_selected_v2_pool_singleton_equivalence(REPO, config_path)
    _atomic_json(rows_path, rows)
    result["comparison_rows_artifact"] = {
        "path": str(rows_path.relative_to(REPO)),
        "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    content = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
