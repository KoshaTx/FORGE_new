#!/usr/bin/env python3
"""Audit actual starting-material closure beneath projected AGILE programs."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_precursor_leaf_closure import build_precursor_leaf_closure_audit

REPO = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _csv_gzip(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty ledger")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode(), mtime=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_precursor_leaf_closure_audit_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = json.loads(config_path.read_text())
    result, leaf_rows, head_rows = build_precursor_leaf_closure_audit(REPO, config_path)
    outputs = config["outputs"]
    leaf_path = REPO / outputs["leaf_ledger"]
    head_path = REPO / outputs["head_ledger"]
    result_path = REPO / outputs["result"]
    _atomic_write(leaf_path, _csv_gzip(leaf_rows))
    _atomic_write(head_path, _csv_gzip(head_rows))
    result["artifacts"].update(
        {
            "leaf_ledger": {
                "path": str(leaf_path.relative_to(REPO)),
                "sha256": sha256_file(leaf_path),
                "rows": len(leaf_rows),
            },
            "head_ledger": {
                "path": str(head_path.relative_to(REPO)),
                "sha256": sha256_file(head_path),
                "rows": len(head_rows),
            },
        }
    )
    _atomic_write(
        result_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "terminal_identity_census": result["terminal_identity_census"],
                "component_program_census": result["component_program_census"],
                "product_census": result["product_census"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
