#!/usr/bin/env python3
"""Run the corrected family-aware fresh-v2 route-gap triage."""

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
from forge.route.ugi3_route_gap_triage_v2 import build_route_gap_triage_v2

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


def _gzip_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty triage ledger")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(text.getvalue().encode(), mtime=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_route_gap_triage_v2.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = json.loads(config_path.read_text())
    result, rows = build_route_gap_triage_v2(REPO, config_path)
    output = config["outputs"]
    ledger_path = REPO / output["triage_ledger"]
    result_path = REPO / output["result"]
    _atomic_write(ledger_path, _gzip_csv(rows))
    result["artifacts"]["triage_ledger"] = {
        "path": str(ledger_path.relative_to(REPO)),
        "rows": len(rows),
        "sha256": sha256_file(ledger_path),
    }
    _atomic_write(result_path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
