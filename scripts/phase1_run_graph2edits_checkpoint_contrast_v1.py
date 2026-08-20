"""Run the frozen proposal-only Graph2Edits checkpoint contrast census."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from forge.route.engine.graph2edits_backend import (
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
from forge.route.audit.graph2edits_checkpoint_contrast import (
    finalize_checkpoint_component_proposals,
    run_checkpoint_component_proposals,
)
from forge.route.assessment.l2_forward_resolver import load_independent_l2_forward_resolver
from forge.route.assessment.proposal_discovery_status import SourceNeutralProposalDiscoveryResolver
from forge.route.engine.proposal_engine import ProposalBackendManifest

CONFIG_SCHEMA_VERSION = "forge.graph2edits_checkpoint_contrast_config.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _validated_inputs(repo: Path, config: Mapping[str, object]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("checkpoint proposal inputs are malformed")
    output: dict[str, Path] = {}
    for label, value in inputs.items():
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise RuntimeError(f"checkpoint proposal input is malformed: {label}")
        path = repo / str(value["path"])
        if _sha256_file(path) != str(value["sha256"]):
            raise RuntimeError(f"checkpoint proposal input hash changed: {label}")
        output[str(label)] = path
    required = {
        "support_audits",
        "runtime_qualification",
        "training_corpus_manifest",
        "forward_resolver",
    }
    if set(output) != required:
        raise RuntimeError("checkpoint proposal input set changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_graph2edits_checkpoint_contrast_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RuntimeError("checkpoint proposal config schema changed")
    if config.get("status") != "frozen_before_proposal_execution":
        raise RuntimeError("checkpoint proposal config is not frozen")
    paths = _validated_inputs(repo, config)
    output_dir = (repo / str(config["output_directory"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("checkpoint proposal output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = json.loads(paths["runtime_qualification"].read_text())
    checkpoint_path = repo / runtime["artifacts"]["checkpoint"]["path"]
    manifest = ProposalBackendManifest(
        backend_id=GRAPH2EDITS_BACKEND_ID,
        implementation_version=GRAPH2EDITS_IMPLEMENTATION_VERSION,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        checkpoint_license=GRAPH2EDITS_CHECKPOINT_LICENSE,
        training_corpus_id=GRAPH2EDITS_TRAINING_CORPUS_ID,
        training_corpus_snapshot_sha256=_sha256_file(paths["training_corpus_manifest"]),
        source_locator=GRAPH2EDITS_SOURCE_LOCATOR,
    )
    artifacts = Graph2EditsLocalArtifacts(
        checkpoint_path=checkpoint_path,
        training_corpus_manifest_path=paths["training_corpus_manifest"],
        corpus_path_root=repo,
    )
    policy = Graph2EditsInferencePolicy()
    inference = config["inference"]
    worker = build_syntheseus_worker(
        model_dir=checkpoint_path.parent,
        device=str(inference["device"]),
        max_edit_steps=policy.max_edit_steps,
    )
    backend = Graph2EditsProposalBackend(
        manifest=manifest,
        artifacts=artifacts,
        worker=worker,
        policy=policy,
    )
    exact_resolver = load_independent_l2_forward_resolver(paths["forward_resolver"], repo_root=repo)
    resolver = SourceNeutralProposalDiscoveryResolver(exact_resolver=exact_resolver)
    population = config["population"]
    support = json.loads(paths["support_audits"].read_text())
    content, rows = run_checkpoint_component_proposals(
        support_payload=support,
        support_path=paths["support_audits"],
        backend=backend,
        resolver=resolver,
        arm=str(population["arm"]),
        checkpoint_phase=str(population["checkpoint_phase"]),
        terminal_phase=str(population["terminal_phase"]),
        checkpoints=tuple(int(value) for value in population["checkpoints"]),
        maximum_proposals=int(inference["maximum_proposals_per_target"]),
        repeat_count=int(inference["repeat_count"]),
    )
    content["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": _sha256_file(config_path),
    }
    ledger_path = output_dir / "proposal_ledger.jsonl.gz"
    with gzip.GzipFile(filename="", mode="wb", fileobj=ledger_path.open("wb"), mtime=0) as out:
        out.write(b"".join(_stable_bytes(row) + b"\n" for row in rows))
    result = finalize_checkpoint_component_proposals(
        content,
        ledger_sha256=_sha256_file(ledger_path),
        row_count=len(rows),
    )
    (output_dir / "result.json").write_bytes(_stable_bytes(result))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
