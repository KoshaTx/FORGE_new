"""Build 30 portable, restartable input shards for v8.1 program projection.

The frozen tasks reference thousands of local source containers.  Shipping all
of them to each remote worker would stage roughly 1.5 GiB and repeatedly scan
the same files.  This builder verifies every original content hash once, joins
only the requested records through a disk-backed SQLite index, and writes one
compact self-contained input bundle per approved one-CPU worker.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from compose_lipid.data.assets import sha256_file
from compose_lipid.data.post_instruction_program_io_v8_1 import (
    manifest_nested_references,
    reference_key,
    rows,
)
from compose_lipid.data.source_inventory import publish_json
from compose_lipid.data.training_corpus import digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corpus/post_instruction_program_projection_inputs_v8_1.json"


def _output_sha(receipt: dict, name: str) -> str:
    value = receipt["outputs"][name]
    return value["sha256"] if isinstance(value, dict) else value


def _reference_json(reference: dict) -> str:
    return json.dumps(reference, sort_keys=True, separators=(",", ":"))


def _record_json(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _add_reference(connection: sqlite3.Connection, reference: dict) -> None:
    path, line = reference_key(reference)
    encoded = _reference_json(reference)
    existing = connection.execute(
        "SELECT reference_json FROM refs WHERE path = ? AND line_key = ?",
        (path, -1 if line is None else line),
    ).fetchone()
    if existing is not None:
        prior = json.loads(existing[0])
        if prior["sha256"] != reference["sha256"]:
            raise ValueError("inconsistent source-file binding: " + path)
        for key in ("payload_sha256", "record_sha256", "task_id"):
            if (
                prior.get(key) not in (None, reference.get(key))
                and reference.get(key) is not None
            ):
                raise ValueError("inconsistent bound-record identity")
        return
    connection.execute(
        "INSERT INTO refs(path, line_key, reference_json) VALUES (?, ?, ?)",
        (path, -1 if line is None else line, encoded),
    )


def _load_requested_records(
    connection: sqlite3.Connection, source_bindings: dict[str, str]
) -> None:
    paths = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT path FROM refs WHERE record_json IS NULL ORDER BY path"
        )
    ]
    for position, relative in enumerate(paths, 1):
        specifications = connection.execute(
            "SELECT line_key, reference_json FROM refs "
            "WHERE path = ? AND record_json IS NULL ORDER BY line_key",
            (relative,),
        ).fetchall()
        references = {line: json.loads(encoded) for line, encoded in specifications}
        expected_hashes = {reference["sha256"] for reference in references.values()}
        if len(expected_hashes) != 1:
            raise ValueError("source path has multiple expected hashes: " + relative)
        expected_hash = expected_hashes.pop()
        path = ROOT / relative
        if sha256_file(path) != expected_hash:
            raise ValueError("changed bound source file: " + relative)
        source_bindings[relative] = expected_hash
        if -1 in references:
            if len(references) != 1:
                raise ValueError("whole-file and line references cannot be mixed")
            records = {-1: json.loads(path.read_bytes())}
        else:
            wanted = set(references)
            records = {}
            for line_number, record in enumerate(rows(path), 1):
                if line_number in wanted:
                    records[line_number] = record
                    wanted.remove(line_number)
                    if not wanted:
                        break
            if wanted:
                raise ValueError(
                    f"missing bound lines in {relative}: {sorted(wanted)[:10]}"
                )
        for line_key, record in records.items():
            reference = references[line_key]
            expected = reference.get("payload_sha256") or reference.get("record_sha256")
            if expected is not None and digest(record) != expected:
                raise ValueError("bound record payload changed")
            task_id = reference.get("task_id")
            if task_id is not None and record.get("task_id") != task_id:
                raise ValueError("bound task record identity changed")
            connection.execute(
                "UPDATE refs SET record_json = ? WHERE path = ? AND line_key = ?",
                (_record_json(record), relative, line_key),
            )
        connection.commit()
        if position % 250 == 0 or position == len(paths):
            print(
                json.dumps(
                    {"resolved_source_files": position, "source_files": len(paths)}
                ),
                flush=True,
            )


def _write_bundle(
    connection: sqlite3.Connection, pending: Path, shard: int
) -> tuple[Path, int, int]:
    path = pending / f"inputs_{shard:03d}.jsonl.gz"
    record_count = 0
    task_count = 0
    with (
        path.open("xb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as stream,
    ):
        reference_rows = connection.execute(
            "SELECT DISTINCT r.path, r.line_key, r.reference_json, r.record_json "
            "FROM refs r JOIN task_refs tr "
            "ON tr.path = r.path AND tr.line_key = r.line_key "
            "JOIN tasks t ON t.task_id = tr.task_id "
            "WHERE t.shard = ? ORDER BY r.path, r.line_key",
            (shard,),
        )
        for _, _, reference_json, record_json in reference_rows:
            if record_json is None:
                raise ValueError("portable shard contains an unresolved reference")
            stream.write(
                json.dumps(
                    {
                        "kind": "record",
                        "reference": json.loads(reference_json),
                        "record": json.loads(record_json),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            record_count += 1
        task_rows = connection.execute(
            "SELECT task_json, manifest_path, manifest_line "
            "FROM tasks WHERE shard = ? ORDER BY ordinal",
            (shard,),
        )
        for task_json, manifest_path, manifest_line in task_rows:
            stream.write(
                json.dumps(
                    {
                        "kind": "task",
                        "task": json.loads(task_json),
                        "manifest_reference": {
                            "path": manifest_path,
                            "line": manifest_line,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            task_count += 1
    return path, record_count, task_count


def build() -> dict:
    config = json.loads(CONFIG.read_text())
    if any(
        config[key]
        for key in ("biology_used", "beae_outcomes_used", "gpu_used", "modal_used")
    ):
        raise ValueError("portable-input build must remain local and outcome-blind")
    if config["training_admissible"]:
        raise ValueError("portable inputs cannot promote training readiness")
    if config["worker_shards"] != 30 or config["cpus_per_worker"] != 1:
        raise ValueError("approved 30 x 1-CPU execution shape changed")

    bindings = {str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG)}
    for key in ("plan", "adapter_probe"):
        path = ROOT / config[f"{key}_receipt"]
        expected = config[f"{key}_receipt_sha256"]
        if sha256_file(path) != expected:
            raise ValueError("changed bound receipt: " + key)
        receipt = json.loads(path.read_text())
        if receipt.get("complete") is not True:
            raise ValueError("incomplete bound receipt: " + key)
        bindings[str(path.relative_to(ROOT))] = expected
    plan_path = ROOT / config["plan_receipt"]
    plan = json.loads(plan_path.read_text())
    plan_dir = plan_path.parent

    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".portable_v8_1_", dir=out))
    database = pending / "join.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = FILE;
        CREATE TABLE tasks(
            task_id TEXT PRIMARY KEY,
            shard INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            task_json TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_line INTEGER NOT NULL
        );
        CREATE TABLE refs(
            path TEXT NOT NULL,
            line_key INTEGER NOT NULL,
            reference_json TEXT NOT NULL,
            record_json TEXT,
            PRIMARY KEY(path, line_key)
        );
        CREATE TABLE task_refs(
            task_id TEXT NOT NULL,
            path TEXT NOT NULL,
            line_key INTEGER NOT NULL,
            PRIMARY KEY(task_id, path, line_key)
        );
        """)
    mode_counts = Counter()
    task_ids = []
    try:
        for shard in range(config["worker_shards"]):
            name = f"tasks_{shard:03d}.jsonl.gz"
            task_path = plan_dir / name
            expected = _output_sha(plan, name)
            if sha256_file(task_path) != expected:
                raise ValueError("changed plan task shard: " + name)
            bindings[str(task_path.relative_to(ROOT))] = expected
            for ordinal, task in enumerate(rows(task_path)):
                reference = task["supervision_reference"]
                _add_reference(connection, reference)
                path, line = reference_key(reference)
                connection.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        task["task_id"],
                        shard,
                        ordinal,
                        _record_json(task),
                        path,
                        line,
                    ),
                )
                connection.execute(
                    "INSERT INTO task_refs VALUES (?, ?, ?)",
                    (task["task_id"], path, line),
                )
                task_ids.append(task["task_id"])
                mode_counts[task["resolution_mode"]] += 1
        connection.commit()
        if len(task_ids) != config["expected_training_targets"]:
            raise ValueError("portable-input task count changed")
        if dict(mode_counts) != config["expected_resolution_modes"]:
            raise ValueError("portable-input resolution-mode counts changed")

        source_bindings = {}
        _load_requested_records(connection, source_bindings)
        manifest_rows = connection.execute(
            "SELECT task_id, manifest_path, manifest_line FROM tasks ORDER BY task_id"
        )
        for task_id, path, line in manifest_rows:
            encoded = connection.execute(
                "SELECT record_json FROM refs WHERE path = ? AND line_key = ?",
                (path, line),
            ).fetchone()[0]
            manifest = json.loads(encoded)
            for reference in manifest_nested_references(manifest):
                _add_reference(connection, reference)
                nested_path, nested_line = reference_key(reference)
                connection.execute(
                    "INSERT OR IGNORE INTO task_refs VALUES (?, ?, ?)",
                    (task_id, nested_path, -1 if nested_line is None else nested_line),
                )
        connection.commit()
        _load_requested_records(connection, source_bindings)
        bindings.update(source_bindings)

        outputs = {}
        record_counts = Counter()
        task_counts = Counter()
        for shard in range(config["worker_shards"]):
            path, record_count, task_count = _write_bundle(connection, pending, shard)
            record_counts[shard] = record_count
            task_counts[shard] = task_count
            final = out / path.name
            os.replace(path, final)
            outputs[final.name] = {
                "rows": task_count,
                "embedded_records": record_count,
                "bytes": final.stat().st_size,
                "sha256": sha256_file(final),
            }

        summary = {
            "schema": "post_instruction_program_projection_inputs_v8_1_summary",
            "tasks": sum(task_counts.values()),
            "embedded_record_instances": sum(record_counts.values()),
            "unique_source_records": connection.execute(
                "SELECT COUNT(*) FROM refs"
            ).fetchone()[0],
            "unique_source_files": len(source_bindings),
            "source_bytes": sum(
                (ROOT / path).stat().st_size for path in source_bindings
            ),
            "portable_bytes": sum(spec["bytes"] for spec in outputs.values()),
            "worker_shards": config["worker_shards"],
            "cpus_per_worker": config["cpus_per_worker"],
            "memory_mib_per_worker": config["memory_mib_per_worker"],
            "minimum_shard_tasks": min(task_counts.values()),
            "maximum_shard_tasks": max(task_counts.values()),
            "resolution_modes": dict(sorted(mode_counts.items())),
            "biology_used": False,
            "beae_outcomes_used": False,
            "gpu_used": False,
            "modal_used": False,
            "training_ready": False,
            "training_admissible": False,
        }
        publish_json(out / "summary.json", summary)
        outputs["summary.json"] = sha256_file(out / "summary.json")
        producer = Path(__file__)
        io_producer = (
            ROOT / "src/compose_lipid/data/post_instruction_program_io_v8_1.py"
        )
        publish_json(
            out / "receipt.json",
            {
                "schema": "post_instruction_program_projection_inputs_v8_1_receipt",
                "complete": True,
                "inputs": dict(sorted(bindings.items())),
                "producers": {
                    str(producer.relative_to(ROOT)): sha256_file(producer),
                    str(io_producer.relative_to(ROOT)): sha256_file(io_producer),
                },
                "outputs": dict(sorted(outputs.items())),
                "task_ids_digest": digest(sorted(task_ids)),
                "training_ready": False,
                "training_admissible": False,
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        connection.close()
        shutil.rmtree(pending, ignore_errors=True)


if __name__ == "__main__":
    build()
