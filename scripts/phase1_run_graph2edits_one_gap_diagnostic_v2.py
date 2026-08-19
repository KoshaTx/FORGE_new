from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from forge.route.graph2edits_backend import (
    GRAPH2EDITS_BACKEND_ID,
    GRAPH2EDITS_CHECKPOINT_LICENSE,
    GRAPH2EDITS_IMPLEMENTATION_VERSION,
    GRAPH2EDITS_SOURCE_LOCATOR,
    GRAPH2EDITS_TRAINING_CORPUS_ID,
    Graph2EditsInferencePolicy,
    Graph2EditsLocalArtifacts,
    Graph2EditsProposalBackend,
    build_syntheseus_worker,
)
from forge.route.graph2edits_one_gap_diagnostic_v2 import (
    finalize_one_gap_diagnostic_v2,
    run_one_gap_diagnostic_v2,
)
from forge.route.l2_forward_resolver import load_independent_l2_forward_resolver
from forge.route.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
from forge.route.proposal_engine import ProposalBackendManifest


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--route-ledger",
        type=Path,
        default=Path(
            "results/phase1/ugi_route_aware_panel_feasibility_v1/"
            "route_assessment_ledger.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/graph2edits_one_gap_diagnostic_v2"),
    )
    parser.add_argument("--maximum-proposals", type=int, default=10)
    parser.add_argument("--repeat-count", type=int, default=2)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = json.loads(
        (
            repo / "configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json"
        ).read_text()
    )
    checkpoint_path = repo / runtime["artifacts"]["checkpoint"]["path"]
    corpus_manifest_path = (
        repo / "configs/route/graph2edits_usp50k_training_corpus_manifest_v1.json"
    )
    manifest = ProposalBackendManifest(
        backend_id=GRAPH2EDITS_BACKEND_ID,
        implementation_version=GRAPH2EDITS_IMPLEMENTATION_VERSION,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        checkpoint_license=GRAPH2EDITS_CHECKPOINT_LICENSE,
        training_corpus_id=GRAPH2EDITS_TRAINING_CORPUS_ID,
        training_corpus_snapshot_sha256=_sha256_file(corpus_manifest_path),
        source_locator=GRAPH2EDITS_SOURCE_LOCATOR,
    )
    artifacts = Graph2EditsLocalArtifacts(
        checkpoint_path=checkpoint_path,
        training_corpus_manifest_path=corpus_manifest_path,
        corpus_path_root=repo,
    )
    policy = Graph2EditsInferencePolicy()
    worker = build_syntheseus_worker(
        model_dir=checkpoint_path.parent,
        device="cpu",
        max_edit_steps=policy.max_edit_steps,
    )
    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=worker,
        policy=policy,
    )
    exact_resolver = load_independent_l2_forward_resolver(
        repo / "configs/route/graph2edits_l2_forward_resolver_v1.json",
        repo_root=repo,
    )
    resolver = SourceNeutralProposalDiscoveryResolver(exact_resolver=exact_resolver)
    content, rows = run_one_gap_diagnostic_v2(
        route_ledger_path=(repo / args.route_ledger).resolve(),
        backend=backend,
        resolver=resolver,
        maximum_proposals=args.maximum_proposals,
        repeat_count=args.repeat_count,
    )
    ledger_path = output_dir / "proposal_ledger.jsonl.gz"
    with gzip.GzipFile(filename="", mode="wb", fileobj=ledger_path.open("wb"), mtime=0) as out:
        out.write(b"".join(_stable_bytes(row) + b"\n" for row in rows))
    result = finalize_one_gap_diagnostic_v2(
        content,
        ledger_sha256=_sha256_file(ledger_path),
        row_count=len(rows),
    )
    (output_dir / "result.json").write_bytes(_stable_bytes(result))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
