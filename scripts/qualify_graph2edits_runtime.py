#!/usr/bin/env python3
"""Qualify and receipt the inactive isolated Graph2Edits macOS runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.engine.graph2edits_runtime_qualification import (
    build_runtime_receipt,
    verify_runtime_receipt,
)

REPO = Path(__file__).resolve().parents[1]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=REPO / "configs/route/graph2edits_proposal_backend_candidate_v1.json",
    )
    parser.add_argument(
        "--requirements-input",
        type=Path,
        default=REPO / "configs/route/graph2edits_runtime_py311_macos_arm64.in",
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=REPO / "configs/route/graph2edits_runtime_py311_macos_arm64.lock.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify the existing receipt and current environment without running inference.",
    )
    args = parser.parse_args()

    if args.verify_existing:
        receipt = json.loads(args.output.read_text())
        verify_runtime_receipt(
            receipt,
            repo_root=REPO,
            verify_artifacts=True,
            verify_environment=True,
        )
        print(receipt["payload_sha256"])
        return 0

    receipt = build_runtime_receipt(
        repo_root=REPO,
        candidate_config_path=args.candidate_config,
        requirements_input_path=args.requirements_input,
        runtime_lock_path=args.runtime_lock,
    )
    _write_atomic(
        args.output,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(receipt["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
