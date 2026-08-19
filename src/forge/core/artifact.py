"""Writing a result artifact -- the receipt a stage leaves behind.

Every stage in this pipeline records not just its answer but the exact bytes it read to get there,
so a number in the manuscript can be traced back and re-checked. That bookkeeping is identical
everywhere and is currently retyped in 221 of 286 modules, with the copies having drifted: the one
in the corpus builder omits the symlink and containment checks that others perform, so how well a
pin was validated depended on which module you were in.

This is that operation, once. A stage declares what it read and what it produced; the run verifies
every input against its recorded digest, hashes every output, and writes the receipt.

Two properties are deliberate:

**Nothing is written until everything succeeds.** Outputs are buffered in memory and committed only
on a clean exit, because the repository's engineering contract requires "calculate first, then write
complete artifacts. Do not leave a plausible-looking result after an exception." A half-written
ledger that still parses is the failure this prevents.

**Receipts are reproducible by default.** `generated_utc` is opt-in, not automatic. An artifact's
own hash is pinned by whatever consumes it downstream, so stamping a timestamp into one means it
hashes differently on every run and silently breaks its consumers' pins. Twelve existing artifacts
carry a timestamp; new ones should not unless the time is genuinely part of the result.

This writes new artifacts in a consistent shape. It does not rewrite existing ones -- their bytes
are pinned and may not move.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from forge.core.hashing import PinError, resolve_pin, sha256_bytes
from forge.core.io import (
    atomic_write,
    csv_bytes,
    gzip_bytes,
    jsonl_bytes,
    pretty_json_bytes,
    read_csv,
    read_json,
    stable_json_bytes,
)


class ArtifactError(RuntimeError):
    """A stage's declared inputs or outputs violate the artifact contract."""


@dataclass
class _PendingWrite:
    path: Path
    payload: bytes
    columns: Sequence[str] | None = None


@dataclass
class ArtifactRun:
    """One stage's execution: verified inputs, buffered outputs, and the receipt.

    Use through `artifact_run`, which commits on a clean exit and discards everything otherwise.
    """

    schema_version: str
    repo: Path
    output_dir: Path
    inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    task: str | None = None
    seed: int | None = None
    record_time: bool = False

    summary: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"
    extra: dict[str, Any] = field(default_factory=dict)

    _resolved: dict[str, Path] = field(default_factory=dict, init=False)
    _input_records: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _pending: list[_PendingWrite] = field(default_factory=list, init=False)
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    # ----------------------------------------------------------------- inputs

    def input(self, label: str) -> Path:
        """Return the path for a declared input, verifying its digest on first use.

        Verification is strict and shared: the record must name exactly a path and a digest, the
        file must sit inside the repository, must not be a symlink, and must hash to what was
        declared. Fails closed on all of them.
        """
        if label in self._resolved:
            return self._resolved[label]
        if label not in self.inputs:
            known = ", ".join(sorted(self.inputs)) or "none"
            raise ArtifactError(f"undeclared input {label!r}; declared inputs are: {known}")
        try:
            path = resolve_pin(self.inputs[label], self.repo, label=label)
        except PinError as error:
            raise ArtifactError(str(error)) from error
        self._resolved[label] = path
        self._input_records[label] = {
            "path": str(path.relative_to(self.repo.resolve())),
            "sha256": self.inputs[label]["sha256"],
            "bytes": path.stat().st_size,
        }
        return path

    def read_csv(self, label: str) -> list[dict[str, str]]:
        """Verify a declared input and read it as CSV, gzip handled by extension."""
        return read_csv(self.input(label))

    def read_json(self, label: str) -> Any:
        """Verify a declared input and read it as JSON, gzip handled by extension."""
        return read_json(self.input(label))

    # ----------------------------------------------------------------- outputs

    def write_csv(
        self, name: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
    ) -> None:
        """Buffer a CSV output. Gzipped when `name` ends in `.gz`. Columns land in the receipt."""
        payload = csv_bytes(rows, fieldnames)
        self._buffer(name, gzip_bytes(payload) if name.endswith(".gz") else payload, fieldnames)

    def write_jsonl(self, name: str, records: Iterable[Any]) -> None:
        payload = jsonl_bytes(records)
        self._buffer(name, gzip_bytes(payload) if name.endswith(".gz") else payload)

    def write_json(self, name: str, value: Any, *, pretty: bool = True) -> None:
        self._buffer(name, pretty_json_bytes(value) if pretty else stable_json_bytes(value))

    def write_bytes(self, name: str, payload: bytes) -> None:
        self._buffer(name, payload)

    def _buffer(self, name: str, payload: bytes, columns: Sequence[str] | None = None) -> None:
        if name in self._artifacts:
            raise ArtifactError(f"output {name!r} written twice")
        self._pending.append(_PendingWrite(self.output_dir / name, payload, columns))
        record: dict[str, Any] = {"sha256": str(sha256_bytes(payload)), "bytes": len(payload)}
        if columns is not None:
            record["columns"] = list(columns)
        self._artifacts[name] = record

    # ----------------------------------------------------------------- receipt

    def receipt(self) -> dict[str, Any]:
        """The document describing this run. Keys are sorted on write, so order is not load-bearing."""
        document: dict[str, Any] = {"schema_version": self.schema_version, "status": self.status}
        if self.task is not None:
            document["task"] = self.task
        if self._input_records:
            document["inputs"] = dict(sorted(self._input_records.items()))
        if self._artifacts:
            document["artifacts"] = dict(sorted(self._artifacts.items()))
        if self.seed is not None:
            document["randomness"] = {"seed": self.seed}
        if self.summary:
            document["summary"] = self.summary
        if self.record_time:
            document["generated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            )
        document.update(self.extra)
        return document

    def commit(self) -> Path:
        """Write every buffered output and the receipt. Called on a clean exit; not for direct use."""
        for pending in self._pending:
            atomic_write(pending.path, pending.payload)
        target = self.output_dir / "result.json"
        atomic_write(target, pretty_json_bytes(self.receipt()))
        return target


class artifact_run:  # noqa: N801 -- used as a context manager, reads as a verb at call sites
    """Run one stage, committing its outputs and receipt only if it completes.

        with artifact_run("phase1_example_result.v1", repo, output_dir, config=config) as run:
            rows = run.read_csv("products")          # pin verified, then read
            run.write_csv("selected.csv.gz", out, fields)
            run.summary = {"selected": len(out)}

    On an exception nothing reaches disk, so a failed stage leaves no partial result behind.
    """

    def __init__(
        self,
        schema_version: str,
        repo: Path,
        output_dir: Path,
        *,
        config: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Mapping[str, Any]] | None = None,
        task: str | None = None,
        seed: int | None = None,
        record_time: bool = False,
    ) -> None:
        declared = inputs if inputs is not None else (config or {}).get("inputs", {})
        self.run = ArtifactRun(
            schema_version=schema_version,
            repo=repo,
            output_dir=output_dir,
            inputs=declared,
            task=task,
            seed=seed,
            record_time=record_time,
        )

    def __enter__(self) -> ArtifactRun:
        return self.run

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Returns None, never True: a stage's failure must always propagate. Swallowing it here
        # would turn a failed run into a silently missing result, which is precisely the outcome
        # the fail-closed discipline exists to prevent.
        if exc_type is None:
            self.run.commit()


__all__ = ["ArtifactError", "ArtifactRun", "artifact_run"]
