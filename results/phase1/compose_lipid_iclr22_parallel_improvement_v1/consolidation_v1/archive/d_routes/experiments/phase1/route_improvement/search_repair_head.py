"""One bounded, previously unattempted exact head search to test the B258 routing loss."""

import argparse
import copy
import os
import resource
import signal
import subprocess
import time
from pathlib import Path

import search_new_leaves as engine
from audit_new_leaves import MAIN, OUT, ROOT, pin, read, write

SEARCH = OUT / "b258_search_v1"
BASELINE_SEARCH = OUT / "leaf_search_v1"
engine.SEARCH = SEARCH
IDENTITY = "COCC1CCN(CCN)C1"


def freeze():
    start = time.process_time()
    protocol = copy.deepcopy(read(BASELINE_SEARCH / "protocol.json"))
    inputs = read(OUT / "audit_protocol_v1.json")["inputs"]
    attempted, sources = set(), []
    for key, ref in inputs.items():
        if key.startswith("prior_search_protocol_") or key == "root_search_v2":
            value = engine.read(MAIN / ref["path"])
            assert pin(MAIN / ref["path"]) == ref
            sources.append(ref)
            attempted.update(
                row.get("identity", row.get("canonical_smiles")) for row in value.get("panel", [])
            )
    attempted.update(row["identity"] for row in protocol["panel"])
    assert IDENTITY not in attempted
    assessment = OUT / "joint_b_d_v1/assessment/products.json"
    request = read(assessment)["products"][258]
    missing = [b for b in request["branches"] if not b["makeable"]]
    assert (
        len(missing) == 1
        and missing[0]["identity"] == IDENTITY
        and missing[0]["state"] == "unknown"
    )
    protocol["schema"] = "forge.stream_d.B258_exact_head_search.v1"
    protocol["inputs"].extend(
        [pin(Path(__file__)), pin(Path(engine.__file__)), pin(assessment)] + sources
    )
    protocol["panel"] = [
        {
            "index": 0,
            "family_priority": "aldehyde_ugi4",
            "identity": IDENTITY,
            "seed": 2026092684,
            "conditional_gain_indices": [258],
        }
    ]
    protocol["maximum_workers"] = (
        3  # Qualified target environment; supervisor below runs exactly one.
    )
    protocol["actual_maximum_concurrent_targets"] = 1
    protocol["maximum_CPU_allocation_seconds"] = 160
    protocol["parent_authorization"] = (
        "Root explicitly authorized one previously untried exact258 component route recovery within remaining1800CPU; no network, chemistry gate changes, or paid jobs."
    )
    protocol["hypothesis"] = (
        "B repair258 loses makeability because the changed exact head has no cached route; a bounded whole-head search may recover evidence without changing the repaired molecular identity."
    )
    protocol["preparation_CPU_seconds"] = time.process_time() - start
    SEARCH.mkdir(exist_ok=False)
    write(SEARCH / "protocol.json", protocol)


def preflight():
    protocol = engine.validate()
    records = []
    for correct in [True, False]:
        env = engine.environment(protocol)
        if not correct:
            env["PYTHONHASHSEED"] = str(protocol["pythonhashseed"] + 1)
        child = subprocess.run(
            [str(MAIN / protocol["runtime_executable"]), str(Path(__file__)), "validate_runtime"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        records.append(
            {
                "correct_environment": correct,
                "exit_code": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
            }
        )
    assert (
        records[0]["exit_code"] == 0
        and records[1]["exit_code"] != 0
        and "Set PYTHONHASHSEED" in records[1]["stderr"]
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        SEARCH / "preflight.json",
        {
            "passed": True,
            "protocol": pin(SEARCH / "protocol.json"),
            "cases": records,
            "child_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "new_searches": 0,
        },
    )


def supervise():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 160))
    protocol = engine.validate()
    command = [str(MAIN / protocol["runtime_executable"]), str(Path(__file__)), "target"]
    with (
        (SEARCH / "target.stdout").open("xb") as stdout,
        (SEARCH / "target.stderr").open("xb") as stderr,
    ):
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=engine.environment(protocol),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        write(SEARCH / "target_launch.json", {"pid": proc.pid, "command": command})
        try:
            code, timeout = proc.wait(timeout=240), False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code, timeout = proc.wait(), True
    path = SEARCH / "target_0/result.json"
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        SEARCH / "completion.json",
        {
            "protocol": pin(SEARCH / "protocol.json"),
            "declared_target_count": 1,
            "exit_code": code,
            "timeout": timeout,
            "all_completed": code == 0 and path.exists() and read(path)["status"] == "completed",
            "result": pin(path) if path.exists() else None,
            "children_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "supervisor_CPU_seconds": time.process_time(),
            "automatic_retry": False,
        },
    )


def launch():
    assert read(SEARCH / "preflight.json")["passed"]
    write(
        SEARCH / "launch_intent.json",
        {"protocol": pin(SEARCH / "protocol.json"), "time_epoch": time.time()},
    )
    command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "supervise"]
    with (SEARCH / "stdout.log").open("xb") as stdout, (SEARCH / "stderr.log").open("xb") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=engine.environment(read(SEARCH / "protocol.json")),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write(
        SEARCH / "launch_receipt.json",
        {"pid": proc.pid, "protocol": pin(SEARCH / "protocol.json"), "detached": True},
    )
    print(proc.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["freeze", "preflight", "validate_runtime", "launch", "supervise", "target"],
    )
    action = parser.parse_args().action
    if action == "validate_runtime":
        engine.validate_runtime()
    elif action == "target":
        engine.target(0)
    else:
        globals()[action]()
