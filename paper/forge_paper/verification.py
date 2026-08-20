"""Verify direct paper bytes and the recursive provenance closure behind its evidence roots."""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256, sha256_file
from forge.core.provenance_archive import HistoricalPinArchive
from forge_provenance.pins import load_baseline, load_moves
from forge_paper.contract import PaperContract, PaperPin


def _nested_pins(node: Any) -> Iterator[PaperPin]:
    if isinstance(node, Mapping):
        path, digest = node.get("path"), node.get("sha256")
        if isinstance(path, str) and isinstance(digest, str) and is_sha256(digest):
            yield PaperPin(path=path, sha256=digest)
        for value in node.values():
            yield from _nested_pins(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nested_pins(value)


def _load_json_bytes(payload: bytes, original_path: str) -> Any | None:
    if not (original_path.endswith(".json") or original_path.endswith(".json.gz")):
        return None
    try:
        if original_path.endswith(".json.gz"):
            payload = gzip.decompress(payload)
        return json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, EOFError):
        return None


def _resolve(
    repo: Path,
    pin: PaperPin,
    *,
    moves: Mapping[str, str],
    archive: HistoricalPinArchive,
) -> tuple[str, Path | None]:
    if pin.path.startswith(("/", "..")):
        return "foreign", None
    active = repo / moves.get(pin.path, pin.path)
    if active.is_file() and not active.is_symlink():
        if str(sha256_file(active)) == pin.sha256:
            return "active", active
    archived = archive.resolve(pin.path, pin.sha256)
    if archived is not None:
        return "archived", archived
    return ("missing" if not active.is_file() else "drift"), active if active.is_file() else None


def provenance_closure(repo: Path, contract: PaperContract) -> dict[str, Any]:
    """Walk every path/hash pin reachable from the paper's evidence artifacts.

    Archived JSON is parsed from its exact historical blob, so moving active code to the archive
    cannot truncate the dependency graph. Foreign container paths are recorded but never treated as
    local bytes. Known-unrecoverable identities remain visible and make full recomputation unready.
    """

    repo = repo.resolve()
    archive = HistoricalPinArchive.load(repo / "provenance/frozen-code/manifest.json", repo)
    moves = load_moves(repo / "docs/artifact_path_moves.json")
    baseline = load_baseline(repo / "docs/known_artifact_drift.json")
    queue = deque(contract.evidence_roots)
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    while queue:
        pin = queue.popleft()
        identity = (pin.path, pin.sha256)
        if identity in seen:
            continue
        seen.add(identity)
        status, resolved = _resolve(repo, pin, moves=moves, archive=archive)
        if status in {"missing", "drift"} and identity in baseline:
            status = "known_unrecoverable"
        rows.append({"path": pin.path, "sha256": pin.sha256, "status": status})
        if resolved is None or status not in {"active", "archived"}:
            continue
        document = _load_json_bytes(resolved.read_bytes(), pin.path)
        if document is not None:
            queue.extend(_nested_pins(document))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    unresolved = [row for row in rows if row["status"] not in {"active", "archived", "foreign"}]
    return {
        "counts": dict(sorted(counts.items())),
        "identities": len(rows),
        "rows": sorted(rows, key=lambda item: (item["path"], item["sha256"])),
        "unresolved": unresolved,
    }


def _direct_report(repo: Path, pins: tuple[PaperPin, ...]) -> dict[str, Any]:
    rows = []
    for pin in pins:
        path = repo / pin.path
        observed = str(sha256_file(path)) if path.is_file() and not path.is_symlink() else None
        rows.append(
            {
                "actual": observed,
                "expected": pin.sha256,
                "path": pin.path,
                "status": "verified" if observed == pin.sha256 else "missing" if observed is None else "drift",
            }
        )
    return {"ok": all(row["status"] == "verified" for row in rows), "rows": rows}


def _source_references(repo: Path, contract: PaperContract) -> dict[str, Any]:
    source = (repo / contract.source.path).read_text()
    graphics = set()
    import re

    for match in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        relative = match.group(1)
        graphics.add(f"paper/figures/{relative}")
    generated = {
        f"paper/{match.group(1)}.tex"
        for match in re.finditer(r"\\input\{(generated_[^}]+)\}", source)
    }
    declared_figures = {pin.path for pin in contract.figure_outputs}
    declared_generated = {pin.path for pin in contract.generated_outputs}
    return {
        "declared_but_unused_figures": sorted(declared_figures - graphics),
        "declared_but_unused_generated": sorted(declared_generated - generated),
        "uncontracted_figures": sorted(graphics - declared_figures),
        "uncontracted_generated": sorted(generated - declared_generated),
    }


def diagnose_paper(repo: Path, contract_path: Path) -> dict[str, Any]:
    contract = PaperContract.load(contract_path)
    direct = _direct_report(repo, contract.direct_pins)
    closure = provenance_closure(repo, contract)
    references = _source_references(repo, contract)
    producers = []
    for producer in contract.publication_producers:
        entrypoint = next((item for item in producer.command if item.endswith((".py", ".sh"))), None)
        missing_tools = [tool for tool in producer.tools if shutil.which(tool) is None]
        producers.append(
            {
                "entrypoint": entrypoint,
                "entrypoint_exists": entrypoint is None or (repo / entrypoint).is_file(),
                "id": producer.producer_id,
                "missing_tools": missing_tools,
                "outputs": list(producer.outputs),
            }
        )
    numerical = [
        {"exists": (repo / path).is_file(), "path": path} for path in contract.numerical_entrypoints
    ]
    experiments = [
        {
            "exists": (repo / "configs/experiments" / f"{experiment}.json").is_file(),
            "experiment_id": experiment,
        }
        for experiment in contract.registered_experiments
    ]
    reference_ok = not any(references.values())
    replay_ready = direct["ok"] and reference_ok
    full_ready = (
        replay_ready
        and not closure["unresolved"]
        and all(item["exists"] for item in numerical)
        and all(item["exists"] for item in experiments)
    )
    return {
        "artifact_replay_ready": replay_ready,
        "contract": str(contract_path.relative_to(repo)),
        "direct": direct,
        "full_recompute_ready": full_ready,
        "paper_id": contract.paper_id,
        "producers": producers,
        "provenance_closure": closure,
        "references": references,
        "registered_experiments": experiments,
        "numerical_entrypoints": numerical,
        "schema_version": "forge.paper_diagnosis.v1",
    }


def verify_paper(repo: Path, contract_path: Path, *, strict: bool = False) -> dict[str, Any]:
    diagnosis = diagnose_paper(repo, contract_path)
    ok = diagnosis["artifact_replay_ready"]
    if strict:
        ok = ok and diagnosis["full_recompute_ready"]
    unresolved = diagnosis["provenance_closure"]["unresolved"]
    return {
        "mode": "full-recompute" if strict else "artifact-replay",
        "ok": ok,
        "paper_id": diagnosis["paper_id"],
        "provenance_counts": diagnosis["provenance_closure"]["counts"],
        "schema_version": "forge.paper_verification.v1",
        # Artifact replay needs the closure summary but should not print thousands of unavailable
        # upstream identities. Strict mode is the diagnostic surface that reports every blocker.
        "unresolved": unresolved if strict else [],
        "unresolved_count": len(unresolved),
    }


def render_publication_outputs(repo: Path, contract_path: Path) -> dict[str, Any]:
    """Regenerate only manuscript tables/figures, then require their frozen hashes.

    The chemist packet is not regenerated here because its producer also writes the internal blind
    key. The checked-in sample pages are verified as frozen outputs by ``verify_paper``.
    """

    import subprocess

    contract = PaperContract.load(contract_path)
    completed = []
    for producer in contract.publication_producers:
        missing = [tool for tool in producer.tools if shutil.which(tool) is None]
        if missing:
            raise RuntimeError(f"producer {producer.producer_id} requires tools: {missing}")
        command = [sys.executable if item == "{python}" else item for item in producer.command]
        result = subprocess.run(command, cwd=repo, check=False)
        if result.returncode:
            raise RuntimeError(
                f"publication producer {producer.producer_id!r} failed with {result.returncode}"
            )
        completed.append(producer.producer_id)
    verification = verify_paper(repo, contract_path)
    if not verification["ok"]:
        raise RuntimeError("regenerated publication outputs do not match the frozen paper contract")
    return {"completed": completed, "verification": verification}


def reproduce_paper_artifacts(repo: Path, contract_path: Path) -> dict[str, Any]:
    """Independently repeat paper packaging and verify exact evidence bytes.

    This is deliberately named artifact reproduction. It does not relabel frozen model outputs as a
    training rerun; ``paper doctor --strict`` is the gate for the complete numerical input closure.
    """

    from tempfile import TemporaryDirectory

    from forge_paper.build import build_overleaf_bundle, build_pdf

    verification = verify_paper(repo, contract_path)
    if not verification["ok"]:
        raise RuntimeError("paper evidence artifacts do not match the frozen contract")
    with TemporaryDirectory(prefix="forge-paper-reproduce-") as temporary:
        root = Path(temporary)
        pdf_hashes = [
            build_pdf(repo, contract_path, root / f"paper-{index}.pdf")["sha256"]
            for index in range(2)
        ]
        bundle_hashes = [
            build_overleaf_bundle(
                repo,
                contract_path,
                root / f"bundle-{index}.zip",
                verify_compile=index == 0,
            )["sha256"]
            for index in range(2)
        ]
    if len(set(pdf_hashes)) != 1 or len(set(bundle_hashes)) != 1:
        raise RuntimeError("paper packaging is not byte-reproducible across clean builds")
    return {
        "bundle_sha256": bundle_hashes[0],
        "mode": "artifact-replay",
        "numerical_recomputation_executed": False,
        "paper_id": verification["paper_id"],
        "pdf_sha256": pdf_hashes[0],
        "schema_version": "forge.paper_reproduction_receipt.v1",
        "verification": verification,
    }


__all__ = [
    "diagnose_paper",
    "provenance_closure",
    "render_publication_outputs",
    "reproduce_paper_artifacts",
    "verify_paper",
]
