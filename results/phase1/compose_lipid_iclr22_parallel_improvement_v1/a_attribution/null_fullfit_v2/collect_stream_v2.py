"""Hash-verified, restartable streaming collection; no submission or compute retry API."""

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "null_monitor_collection_helpers", HERE / "monitor_progress_v1.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def artifact(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Malformed artifact pin")
    path = Path(value["path"])
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != value["path"]:
        raise ValueError("Unsafe artifact path")
    if not isinstance(value["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", value["sha256"]):
        raise ValueError("Malformed artifact hash")
    return path


def catalog(prepared, observation):
    prepared = prepared.resolve()
    observation = observation.resolve()
    observation.relative_to(prepared)
    saved = helper.submission(prepared)
    if (observation / "observation.json").exists():
        record = helper.read(observation / "observation.json")
        for value in record.get("files", {}).values():
            path = prepared / value["path"]
            path.resolve().relative_to(observation)
            if helper.digest(path) != value["sha256"]:
                raise ValueError("Observed file changed")
    else:
        raise ValueError("A durable monitor observation is required")
    terminal = (
        observation / "failure.json"
        if (observation / "failure.json").exists()
        else observation / "result.json"
    )
    result = helper.read(terminal)
    if result["request"] != saved["request"]:
        raise ValueError("Terminal request differs")
    if terminal.name == "failure.json" and (observation / "result.json").exists():
        if helper.read(observation / "result.json").get("complete") is True:
            raise ValueError("Conflicting terminal success and failure snapshots")
    success = result.get("complete") is True and result.get("passed") is True
    artifacts = {}

    def add(value):
        artifact(value)
        old = artifacts.get(value["path"])
        if old is not None and old != value:
            raise ValueError("Conflicting artifact hashes")
        artifacts[value["path"]] = value

    add({"path": terminal.name, "sha256": helper.digest(terminal)})
    if success:
        completion = record.get("completion")
        expected = {
            "path": "training/" + saved["request_id"] + "/result.json",
            "sha256": helper.digest(terminal),
        }
        if (
            not isinstance(completion, dict)
            or completion.get("complete") is not True
            or completion.get("result") != expected
        ):
            raise ValueError(
                "Complete function-return pin required before successful-fit collection"
            )
        if record.get("validation_errors"):
            raise ValueError("Observation has unresolved validation errors")
        for key in ("fit", "staging", "inventory"):
            add(result[key])
        if result["fit"]["path"] != "fit/result.json":
            raise ValueError("Unexpected fit key path")
    else:
        if result.get("complete") is not False:
            raise ValueError("Terminal state is neither authenticated success nor failure")
        for key, value in result.get("artifacts", {}).items():
            if key != value.get("path"):
                raise ValueError("Outer artifact key/path differ")
            add(value)
    fit_path = observation / "fit/result.json"
    if fit_path.exists():
        if (
            "fit/result.json" not in artifacts
            or helper.digest(fit_path) != artifacts["fit/result.json"]["sha256"]
        ):
            raise ValueError("Nested fit result lacks exact outer binding")
        fit = helper.read(fit_path)
        if success and (
            fit.get("passed") is not True
            or fit.get("complete") is not True
            or fit.get("protocol_sha256") != saved["request"]["protocol"]["sha256"]
        ):
            raise ValueError("Fit result incomplete or protocol changed")
        for key, value in fit.get("artifacts", {}).items():
            if key != value.get("path"):
                raise ValueError("Nested artifact key/path differ")
            artifact(value)
            add({"path": "fit/" + value["path"], "sha256": value["sha256"]})
    elif success:
        raise ValueError("Nested successful fit receipt missing")
    return {
        "request_id": saved["request_id"],
        "submitted": helper.pin(prepared / "submitted.json", prepared),
        "outer_terminal": helper.pin(terminal, prepared),
        "observation": helper.pin(observation / "observation.json", prepared),
        "terminal_success": success,
        "function_return_authenticated": success,
        "artifacts": artifacts,
    }


def target_path(output, value):
    path = output / artifact(value)
    if path.is_symlink():
        raise ValueError("Collected artifacts cannot be symlinks")
    path.resolve().relative_to(output.resolve())
    return path


async def collect(
    prepared, observation, volume, *, timeout=120, record_name="collection_stream_v2.json"
):
    prepared = Path(prepared).resolve()
    observation = Path(observation).resolve()
    if Path(record_name).name != record_name:
        raise ValueError("Collection record must be a local filename")
    description = catalog(prepared, observation)
    identity = {
        k: description[k]
        for k in ("request_id", "submitted", "outer_terminal", "terminal_success", "artifacts")
    }
    identity["outer_terminal"] = {"sha256": description["outer_terminal"]["sha256"]}
    catalog_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    record = prepared / record_name
    output = prepared / "collected"
    output.mkdir(exist_ok=True)
    with (prepared / ".stream_collection_v2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            manifest = helper.read(record)
            if manifest["catalog_sha256"] != catalog_hash or manifest[
                "producer_sha256"
            ] != helper.digest(Path(__file__)):
                raise ValueError(
                    "Preserve other catalog/producer record; choose a new record filename"
                )
            if manifest["complete"]:
                try:
                    for name, value in description["artifacts"].items():
                        target = target_path(output, value)
                        if (
                            helper.digest(target) != value["sha256"]
                            or name not in manifest["verified"]
                        ):
                            raise ValueError("Previously complete collection changed: " + name)
                except Exception as error:
                    directory = prepared / "collection_validation_errors_v1"
                    directory.mkdir(exist_ok=True)
                    helper.write(
                        directory / f"{time.time_ns()}.json",
                        {
                            "error": repr(error),
                            "prior_record": helper.pin(record, prepared),
                            "new_GPU_jobs": 0,
                            "scientific_completion_admitted": False,
                        },
                    )
                    raise
                return record, manifest
        else:
            manifest = {
                "schema": "forge.null_fullfit_stream_collection.v2",
                "catalog_sha256": catalog_hash,
                "producer_sha256": helper.digest(Path(__file__)),
                "catalog": description,
                "verified": {},
                "attempts": [],
                "complete": False,
                "new_GPU_jobs": 0,
                "scientific_completion_admitted": False,
                "per_file_deadline_seconds": 120,
            }
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt = {
            "id": stamp,
            "observation": helper.pin(observation / "observation.json", prepared),
            "finished": False,
            "network_downloads": 0,
            "deduplicated_files": 0,
            "failure": None,
        }
        manifest["attempts"].append(attempt)
        helper.write(record, manifest)
        known = {}
        try:
            for value in description["artifacts"].values():
                target = target_path(output, value)
                if target.exists():
                    if helper.digest(target) != value["sha256"]:
                        raise ValueError("Changed existing artifact: " + value["path"])
                    known[value["sha256"]] = target
            for name, value in sorted(description["artifacts"].items()):
                target = target_path(output, value)
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + "." + stamp + ".stream.partial")
                if not target.exists():
                    if value["sha256"] in known:
                        with (
                            known[value["sha256"]].open("rb") as source,
                            partial.open("xb") as stream,
                        ):
                            shutil.copyfileobj(source, stream, 1024 * 1024)
                            stream.flush()
                            os.fsync(stream.fileno())
                        attempt["deduplicated_files"] += 1
                    else:
                        attempt["active_file"] = name
                        helper.write(record, manifest)
                        attempt["network_downloads"] += 1
                        with partial.open("x+b") as stream:
                            await asyncio.wait_for(
                                volume.read_file_into_fileobj.aio(
                                    "training/" + description["request_id"] + "/" + value["path"],
                                    stream,
                                ),
                                timeout=timeout,
                            )
                            stream.flush()
                            os.fsync(stream.fileno())
                    if helper.digest(partial) != value["sha256"]:
                        raise ValueError("Downloaded hash mismatch: " + name)
                    partial.replace(target)
                known[value["sha256"]] = target
                manifest["verified"][name] = {
                    "remote": value,
                    "local": helper.pin(target, prepared),
                    "size_bytes": target.stat().st_size,
                }
                attempt.pop("active_file", None)
                helper.write(record, manifest)
            manifest["complete"] = True
            attempt["finished"] = True
            attempt["wall_seconds"] = time.monotonic() - started
            helper.write(record, manifest)
            return record, manifest
        except BaseException as error:
            attempt["failure"] = {"type": type(error).__name__, "error": repr(error)}
            if "partial" in locals() and partial.exists():
                attempt["failure"]["partial"] = helper.pin(partial, prepared)
                attempt["failure"]["partial_bytes"] = partial.stat().st_size
            attempt["wall_seconds"] = time.monotonic() - started
            attempt["finished"] = True
            manifest["complete"] = False
            helper.write(record, manifest)
            raise


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--record", default="collection_stream_v2.json")
    args = parser.parse_args()
    catalog(args.prepared, args.observation)
    from experiments._runtime.modal_app import experiment_volume

    path, result = await collect(
        args.prepared, args.observation, experiment_volume, record_name=args.record
    )
    print(
        json.dumps(
            {
                "collection": str(path),
                "sha256": helper.digest(path),
                "complete": result["complete"],
                "artifacts": len(result["verified"]),
                "terminal_success": result["catalog"]["terminal_success"],
                "new_GPU_jobs": 0,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
