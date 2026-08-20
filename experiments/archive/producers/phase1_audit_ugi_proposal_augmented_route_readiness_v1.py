from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.synthesis.value.proposal_readiness import (
    build_proposal_augmented_route_readiness,
)


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi_proposal_augmented_route_readiness_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_proposal_augmented_route_readiness_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    result, ledger = build_proposal_augmented_route_readiness(repo, (repo / args.config).resolve())
    _write_atomic(output / "route_readiness_ledger.jsonl.gz", ledger)
    _write_atomic(
        output / "result.json",
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
