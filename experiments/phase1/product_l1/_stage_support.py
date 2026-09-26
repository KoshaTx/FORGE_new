"""Filesystem publication and restart helpers for product/L1 stage adapters."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from experiments._runtime.errors import StageError
from experiments._runtime.stage import ProducedArtifact, RunContext
from forge.core.hashing import sha256_file
from forge.core.io import write_json


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _deterministic_tar(paths: list[Path], target: Path, *, base: Path) -> None:
    """Archive checkpoint shards without filesystem timestamps or ownership."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
            name = path.relative_to(base).as_posix()
            info = tarfile.TarInfo(name=name)
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _resume_training_work(context: RunContext, work: Path) -> bool:
    """Resume an authenticated checkpoint, or restart unauthenticated scratch safely."""

    if not context.resume:
        return False
    if (work / "checkpoint_latest.pt").is_file():
        return True
    # A process can die before its first atomic checkpoint.  Nothing in that directory is a
    # resumable state, so rebuild it deterministically under the same stage fingerprint.
    if work.exists():
        shutil.rmtree(work)
    return False


def _publish_training_outputs(
    context: RunContext,
    work: Path,
    *,
    result_schema: str,
    checkpoint_schema: str,
    progress_schema: str,
) -> tuple[ProducedArtifact, ...]:
    result = json.loads((work / "result.json").read_text())
    for key, filename in (
        ("checkpoint", "checkpoint_best.pt"),
        ("checkpoint_latest", "checkpoint_latest.pt"),
    ):
        record = result.get(key)
        if not isinstance(record, dict) or record.get("sha256") != str(
            sha256_file(work / filename)
        ):
            raise StageError(f"training result does not authenticate {filename}")
        record["path"] = filename
    snapshots = sorted(work.glob("checkpoint_step_*.pt"))
    snapshot_records = []
    for path in snapshots:
        snapshot_records.append(
            {
                "member": path.name,
                "sha256": str(sha256_file(path)),
                "step": int(path.stem.rsplit("_", 1)[1]),
            }
        )
    result["checkpoint_snapshots"] = snapshot_records
    result["execution"] = {
        "backend": context.backend,
        "profile": context.profile,
        "resumed_partial_stage": context.resume,
    }
    write_json(context.output_path("result.json"), result)
    _copy(work / "checkpoint_best.pt", context.output_path("checkpoint_best.pt"))
    _copy(work / "checkpoint_latest.pt", context.output_path("checkpoint_latest.pt"))
    _copy(work / "progress.json", context.output_path("progress.json"))
    _deterministic_tar(snapshots, context.output_path("checkpoint_snapshots.tar"), base=work)
    return (
        ProducedArtifact("result", "result.json", result_schema),
        ProducedArtifact("checkpoint", "checkpoint_best.pt", checkpoint_schema),
        ProducedArtifact("checkpoint_latest", "checkpoint_latest.pt", checkpoint_schema),
        ProducedArtifact("progress", "progress.json", progress_schema),
        ProducedArtifact(
            "checkpoint_snapshots",
            "checkpoint_snapshots.tar",
            "forge.checkpoint_archive.v1",
            rows=len(snapshot_records),
        ),
    )
