from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from experiments.archive.phase1.synthesis_audits.graph2edits_semantic_readjudication import (
    build_semantic_readjudication,
    finalize_semantic_readjudication,
)
from forge.synthesis.assessment.l2_forward_resolver import load_independent_l2_forward_resolver


def _stable_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--proposal-ledger",
        type=Path,
        default=Path("results/phase1/graph2edits_one_gap_diagnostic_v2/proposal_ledger.jsonl.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/graph2edits_semantic_readjudication_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = load_independent_l2_forward_resolver(
        repo / "configs/route/graph2edits_l2_forward_resolver_v1.json",
        repo_root=repo,
    )
    content, rows = build_semantic_readjudication(
        proposal_ledger_path=(repo / args.proposal_ledger).resolve(),
        transforms=resolver.transforms,
    )
    ledger_path = output_dir / "semantic_ledger.jsonl.gz"
    with gzip.GzipFile(filename="", mode="wb", fileobj=ledger_path.open("wb"), mtime=0) as out:
        out.write(b"".join(_stable_bytes(row) + b"\n" for row in rows))
    result = finalize_semantic_readjudication(
        content, ledger_sha256=_sha256_file(ledger_path), row_count=len(rows)
    )
    (output_dir / "result.json").write_bytes(_stable_bytes(result))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
