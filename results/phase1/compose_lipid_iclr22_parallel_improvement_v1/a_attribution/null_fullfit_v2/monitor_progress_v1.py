"""Observe committed full-fit progress once; never submit or restart computation."""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = (
    "function-started.json",
    "staging.json",
    "inventory.json",
    "inventory.log",
    "failure.json",
    "result.json",
    "fit/result.json",
    "fit/child_receipt.json",
    "fit/training_run.log",
    "fit/training_run/training/latest.json",
    "fit/training_run/events.jsonl",
    "fit/training_run/draws.jsonl",
    "fit/training_run/measurement.json",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pin(path, root):
    resolved = Path(path).resolve()
    location = (
        resolved.relative_to(root.resolve())
        if resolved.is_relative_to(root.resolve())
        else resolved
    )
    return {"path": str(location), "sha256": digest(path)}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def submission(prepared):
    submitted = read(prepared / "submitted.json")
    request = read(prepared / "request.json")
    identity = "compose-null-fullfit-" + digest(prepared / "request.json")[:16]
    if submitted["request"] != request or submitted["request_id"] != identity:
        raise ValueError("Submitted request/identity differs from prepared bytes")
    if not all(
        isinstance(submitted.get(k), str) and submitted[k]
        for k in ("application_id", "function_call_id")
    ):
        raise ValueError("Missing persisted application/function-call IDs")
    if submitted.get("detached") is not True or submitted.get("automatic_retry") is not False:
        raise ValueError("Expected detached, non-retrying submission")
    return submitted


def sequence(path, key, start):
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError("Incomplete committed JSONL line")
    rows = [json.loads(line) for line in text.splitlines()]
    if [r[key] for r in rows] != list(range(start, start + len(rows))):
        raise ValueError("Committed sequence is discontinuous")
    if len(rows) > 2794:
        raise ValueError("Committed sequence exceeds fixed update budget")
    return rows


async def observe(prepared, volume, *, not_found=(FileNotFoundError,), timeout=45, completion=None):
    prepared = Path(prepared).resolve()
    submitted = submission(prepared)
    base = "training/" + submitted["request_id"]
    destination = (
        prepared
        / "monitor_observations_v1"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "forge.null_fullfit_readonly_observation.v1",
        "request_id": submitted["request_id"],
        "producer": pin(Path(__file__), prepared),
        "submitted": pin(prepared / "submitted.json", prepared),
        "files": {},
        "missing": [],
        "errors": {},
        "validation_errors": {},
        "committed_observed_update_events": None,
        "committed_observed_draws": None,
        "planned_updates": 2794,
        "observed_update_fraction": None,
        "checkpoint_pointer": None,
        "completion": None,
        "completion_status": "not_requested",
        "complete_observation": False,
        "quality_evidence": False,
        "scientific_completion_admitted": False,
        "new_GPU_jobs": 0,
        "note": "Separate committed reads are monitoring evidence, not an atomic training audit. Missing or unreadable events mean unknown progress, never zero updates.",
    }
    record = destination / "observation.json"
    write(record, result)
    semaphore = asyncio.Semaphore(4)

    async def fetch(name):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        async with semaphore:
            try:
                with partial.open("xb") as stream:
                    await asyncio.wait_for(
                        volume.read_file_into_fileobj.aio(base + "/" + name, stream), timeout
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                partial.replace(path)
                result["files"][name] = pin(path, prepared)
            except not_found:
                if partial.exists() and partial.stat().st_size == 0:
                    partial.unlink()
                result["missing"].append(name)
            except Exception as error:
                result["errors"][name] = {
                    "type": type(error).__name__,
                    "error": repr(error),
                    "partial": pin(partial, prepared) if partial.exists() else None,
                }
            finally:
                write(record, result)

    await asyncio.gather(*(fetch(name) for name in NAMES))
    for name in ("function-started.json", "result.json", "failure.json"):
        if name not in result["files"]:
            continue
        try:
            if read(destination / name)["request"] != submitted["request"]:
                raise ValueError("Remote request binding differs")
        except Exception as error:
            result["validation_errors"][name] = repr(error)
    for name, key, start, target in (
        ("fit/training_run/events.jsonl", "completed_steps", 1, "committed_observed_update_events"),
        ("fit/training_run/draws.jsonl", "step", 0, "committed_observed_draws"),
    ):
        if name not in result["files"]:
            continue
        try:
            rows = sequence(destination / name, key, start)
            if key == "step" and any(r["examples"] != 3168 for r in rows):
                raise ValueError("Draw exposure differs")
            result[target] = len(rows)
        except Exception as error:
            result["validation_errors"][name] = repr(error)
    if result["committed_observed_update_events"] is not None:
        result["observed_update_fraction"] = result["committed_observed_update_events"] / 2794
    latest = "fit/training_run/training/latest.json"
    if latest in result["files"]:
        try:
            result["checkpoint_pointer"] = read(destination / latest)
        except Exception as error:
            result["validation_errors"][latest] = repr(error)
    outer = (
        read(destination / "result.json")
        if "result.json" in result["files"] and "result.json" not in result["validation_errors"]
        else None
    )
    if outer and outer.get("complete") is True and "fit/result.json" in result["files"]:
        value = outer.get("fit", {})
        if value.get("path") != "fit/result.json" or value.get("sha256") != digest(
            destination / "fit/result.json"
        ):
            result["validation_errors"]["fit/result.json"] = "Outer fit artifact binding differs"
    if completion is not None:
        try:
            value = await asyncio.wait_for(completion(), timeout)
            result["completion"] = value
            result["completion_status"] = "returned"
            if value.get("complete") is True:
                expected = (
                    {"path": base + "/result.json", "sha256": digest(destination / "result.json")}
                    if outer
                    else None
                )
                if expected is None or value.get("result") != expected:
                    result["validation_errors"][
                        "function_return"
                    ] = "Function-returned result binding differs or outer result unavailable"
        except TimeoutError:
            result["completion_status"] = "not_yet_returned_or_deadline"
            result["completion_error"] = "TimeoutError: return status unknown"
        except Exception as error:
            result["completion_status"] = "error"
            result["completion_error"] = repr(error)
    result["missing"].sort()
    result["complete_observation"] = True
    if any(
        name in result["validation_errors"]
        for name in ("function-started.json", "result.json", "failure.json")
    ):
        result["committed_observed_update_events"] = None
        result["observed_update_fraction"] = None
        result["committed_observed_draws"] = None
    write(record, result)
    return record, result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE)
    args = parser.parse_args()
    saved = submission(args.prepared.resolve())
    import modal

    from experiments._runtime.modal_app import experiment_volume

    call = modal.FunctionCall.from_id(saved["function_call_id"])

    async def returned():
        return await call.get.aio(timeout=0)

    path, result = await observe(
        args.prepared,
        experiment_volume,
        not_found=(FileNotFoundError, modal.exception.NotFoundError),
        completion=returned,
    )
    print(
        json.dumps(
            {
                "observation": str(path),
                "updates": result["committed_observed_update_events"],
                "network_errors": result["errors"],
                "validation_errors": result["validation_errors"],
                "completion_status": result["completion_status"],
                "completion_error": result.get("completion_error"),
            }
        )
    )
    if result["errors"] or result["validation_errors"] or result["completion_status"] == "error":
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
