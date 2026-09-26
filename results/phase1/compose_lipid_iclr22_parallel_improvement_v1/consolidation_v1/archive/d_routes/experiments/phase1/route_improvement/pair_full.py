"""Detached paired real-evaluator census; preserves every original request and old result."""

import argparse
import copy
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import candidate_routes as candidate
from audit_new_leaves import MAIN, OUT, PREP, ROOT, pin, read, write

import forge

sys.path.insert(0, str(PREP))
import paired_runner as previous  # noqa: E402

BASE = candidate.BASE
DEST = OUT / "full_pair_v1"
FIELDS = ["exact_L1", "combined_primary", "L2_ready", "L3_direct_only", "strict_secondary"]


def freeze():
    admission = read(candidate.CAND / "root_admission.json")
    envelope = read(candidate.CAND / "admitted_paths.json")
    assert admission["passed"] and admission["reviewer"] == "root"
    assert admission["validator"] == pin(Path(candidate.__file__))
    assert admission["paths_sha256"] == candidate.prior.content_sha(envelope["paths"])
    assert read(candidate.CAND / "qualification.json")["passed"]
    DEST.mkdir(exist_ok=False)
    refs = {}
    for arm in ["control", "treatment"]:
        path = DEST / arm
        path.mkdir()
        protocol = copy.deepcopy(read(BASE / "protocol.json"))
        protocol["implementation"].extend([pin(Path(__file__)), pin(Path(candidate.__file__))])
        if arm == "treatment":
            protocol["route_envelopes"].append(pin(candidate.CAND / "admitted_paths.json"))
            protocol["proof_vendor_snapshots"].append(pin(candidate.CAND / "listing_view.json"))
        protocol["stream_d_pair"] = {
            "arm": arm,
            "baseline_result": pin(BASE / "result.json"),
            "baseline_products": pin(BASE / "products.json"),
            "admission": pin(candidate.CAND / "root_admission.json"),
            "qualification": pin(candidate.CAND / "qualification.json"),
            "CPU_cap_seconds": 160,
            "wall_cap_seconds": 600,
            "seed": 0,
            "population": 1408,
            "new_vendor_observations": 0,
            "expected_primary_gain_indices": (
                [] if arm == "control" else [1281, 1290, 1305, 1306, 1313]
            ),
        }
        write(path / "protocol.json", protocol)
        refs[arm] = pin(path / "protocol.json")
    write(
        DEST / "protocol.json",
        {
            "schema": "forge.stream_d.full_pair.v1",
            "arms": refs,
            "driver": pin(Path(__file__)),
            "validator": pin(Path(candidate.__file__)),
            "independent_admission": pin(candidate.CAND / "root_admission.json"),
            "maximum_workers": 2,
            "aggregate_CPU_cap_seconds": 325,
            "no_network_model_TEST": True,
            "no_retry": True,
        },
    )


def execute(arm):
    resource.setrlimit(resource.RLIMIT_CPU, (160, 160))
    started = time.process_time()
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    directory = DEST / arm
    protocol = read(directory / "protocol.json")
    for ref in protocol["implementation"]:
        assert pin(MAIN / ref["path"]) == ref
    write(
        directory / "started.json",
        {
            "protocol": pin(directory / "protocol.json"),
            "pid": os.getpid(),
            "time_epoch": time.time(),
        },
    )
    try:
        import run_v10 as evaluator

        bridge = candidate.Bridge(evaluator.external_routes)
        evaluator.external_routes = bridge.external_routes
        evaluator.configure_documentary = previous.warm_documentary
        evaluator.run(directory, protocol)
        old_rows = read(BASE / "products.json")["products"]
        rows = read(directory / "products.json")["products"]
        assert len(rows) == len(old_rows) == 1408
        changes = {
            key: {
                "gains": [
                    b["index"] for a, b in zip(old_rows, rows, strict=True) if not a[key] and b[key]
                ],
                "losses": [
                    b["index"] for a, b in zip(old_rows, rows, strict=True) if a[key] and not b[key]
                ],
            }
            for key in FIELDS
        }
        assert all(
            a["index"] == b["index"] and a["family"] == b["family"]
            for a, b in zip(old_rows, rows, strict=True)
        )
        for key in FIELDS:
            assert changes[key] == {
                "gains": (
                    protocol["stream_d_pair"]["expected_primary_gain_indices"]
                    if key == "combined_primary"
                    else []
                ),
                "losses": [],
            }, (key, changes[key])
        if arm == "control":
            assert rows == old_rows
            assert (
                read(BASE / "components.json")["components"]
                == read(directory / "components.json")["components"]
            )
        summary = {
            "passed": True,
            "protocol": pin(directory / "protocol.json"),
            "result": pin(directory / "result.json"),
            "full_denominator": 1408,
            "complete_branch_denominator": sum(len(r["branches"]) for r in rows),
            "transitions": changes,
            "totals": {key: sum(r[key] for r in rows) for key in FIELDS},
            "by_family": [
                {
                    "family": family,
                    "requests": 64,
                    "before": {
                        key: sum(r[key] for r in old_rows if r["family"] == family)
                        for key in FIELDS
                    },
                    "after": {
                        key: sum(r[key] for r in rows if r["family"] == family) for key in FIELDS
                    },
                    "transitions": {
                        key: {
                            side: [i for i in changes[key][side] if rows[i]["family"] == family]
                            for side in ["gains", "losses"]
                        }
                        for key in FIELDS
                    },
                }
                for family in sorted({r["family"] for r in rows})
            ],
            "process_CPU_seconds": time.process_time(),
            "execute_CPU_seconds": time.process_time() - started,
            "imported_forge": str(Path(forge.__file__).resolve()),
            "new_HTTP_calls": 0,
        }
        write(directory / "validation.json", summary)
        write(
            directory / "execution.json",
            {
                "complete": True,
                "result": pin(directory / "result.json"),
                "validation": pin(directory / "validation.json"),
                "process_CPU_seconds": time.process_time(),
            },
        )
    except Exception as error:
        write(
            directory / "failure.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "message": str(error),
                "process_CPU_seconds": time.process_time(),
            },
        )
        raise


def supervise():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

    def one(arm):
        path = DEST / arm
        command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "execute", "--arm", arm]
        with (path / "stdout.log").open("xb") as stdout, (path / "stderr.log").open("xb") as stderr:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            write(
                path / "launch_receipt.json",
                {"pid": proc.pid, "command": command, "protocol": pin(path / "protocol.json")},
            )
            try:
                code, timeout = proc.wait(timeout=600), False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code, timeout = proc.wait(), True
        record = {
            "arm": arm,
            "exit_code": code,
            "timeout": timeout,
            "complete": code == 0 and (path / "execution.json").exists(),
        }
        write(path / "exit.json", record)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, ["control", "treatment"]))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        DEST / "completion.json",
        {
            "protocol": pin(DEST / "protocol.json"),
            "exits": records,
            "all_completed": all(row["complete"] for row in records),
            "children_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "supervisor_CPU_seconds": time.process_time(),
            "no_retry": True,
        },
    )


def launch():
    write(
        DEST / "launch_intent.json",
        {"protocol": pin(DEST / "protocol.json"), "time_epoch": time.time()},
    )
    command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "supervise"]
    with (DEST / "stdout.log").open("xb") as stdout, (DEST / "stderr.log").open("xb") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write(
        DEST / "launch_receipt.json",
        {"pid": proc.pid, "protocol": pin(DEST / "protocol.json"), "detached": True},
    )
    print(proc.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "supervise", "execute"])
    parser.add_argument("--arm", choices=["control", "treatment"])
    args = parser.parse_args()
    execute(args.arm) if args.action == "execute" else globals()[args.action]()
