"""One bounded offline search for each frozen weak-family shared leaf target."""

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parents[1]
OUT = MAIN / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
SEARCH = OUT / "leaf_search_v1"
WORK = MAIN / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1"
ROUTES = WORK / "computational_makeability_v1/routes"
PILOT = WORK / "planner/remote_engine_reuse_v1/discovery"
FAMILIES = ["reductive_amination", "ketone_ugi4", "disulfide_michael", "aryl_reductive_amination"]


def read(path):
    return json.loads(path.read_text())


def pin(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while b := stream.read(1024 * 1024):
            h.update(b)
    return {"path": str(path.relative_to(MAIN)), "sha256": h.hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def environment(protocol):
    return dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONHASHSEED=str(protocol["pythonhashseed"]),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        ORT_DISABLE_TELEMETRY="1",
    )


def core_module():
    sys.path.insert(0, str(ROUTES))
    import search_residual

    return search_residual


def freeze():
    start = time.process_time()
    frontier = read(OUT / "leaf_frontier_v1.json")["rows"]
    panel = []
    for index, family in enumerate(FAMILIES):
        row = next(
            r
            for r in frontier
            if r["eligible_new_search"] and r["conditional_gain_by_family"].get(family, 0) > 0
        )
        panel.append(
            {
                "index": index,
                "identity": row["identity"],
                "family_priority": family,
                "seed": 2026092680 + index,
                "conditional_gain_indices": row["single_leaf_conditional_gain_indices"],
                "source_traces": row["traces"],
                "previous_searches": row["previous_declared_searches"],
            }
        )
    assert len({p["identity"] for p in panel}) == 4 and all(
        not p["previous_searches"] for p in panel
    )
    old = read(PILOT / "protocol.json")
    assets = old["verified_assets"] + [old["runtime_config"]]
    for value in assets:
        assert pin(MAIN / value["path"]) == value
    sources = [
        Path(__file__),
        OUT / "audit_protocol_v1.json",
        OUT / "result.json",
        OUT / "leaf_frontier_v1.json",
        OUT / "source_verification_v1.json",
        PILOT / "protocol.json",
        PILOT / "run.py",
        PILOT / "prepare.py",
        PILOT / "smoke_review.json",
        PILOT / "smoke_0/result.json",
        PILOT / "smoke_1/result.json",
        ROUTES / "search_residual.py",
        ROUTES / "bridge.py",
    ]
    SEARCH.mkdir(exist_ok=False)
    write(
        SEARCH / "protocol.json",
        {
            "schema": "forge.iclr22_stream_D_new_leaf_search.v1",
            "inputs": [pin(p) for p in sources],
            "asset_pins": assets,
            "runtime_config": old["runtime_config"],
            "runtime_executable": old["runtime_executable"],
            "pythonhashseed": old["seed"],
            "search": dict(
                old["search"], time_limit_seconds=90, iteration_limit=900, maximum_transforms=8
            ),
            "panel": panel,
            "maximum_workers": 3,
            "child_CPU_seconds": 140,
            "child_wall_seconds": 240,
            "maximum_CPU_allocation_seconds": 610,
            "stream_CPU_cap_seconds": 1800,
            "seed": 2026092680,
            "baseline_primary": 616,
            "requests": 1408,
            "selection": "First eligible saved leaf for each weakfamily in frozen frontier rank: weakfamily conditionalgain, wholeproduct conditionalgain, incidence, size, exactidentity. No previously declared exacttarget is reused.",
            "parent_authorization": "Execute stream D isolated route closure; bounded CPU1800seconds <=3threads; no paid jobs, no external bulk lookup. Preserve source/listing/strict gates and full cohort.",
            "bounds": "4newleaf searches only; top3 trees each; one-shot pertarget; no automaticretry. Bounds enlarged explicitly to90sec/900iterations/depth8; improvements attributed to additional route proposals and target choice, not modellearning.",
            "original_stock_flags_are_current_listing_evidence": False,
            "network_disabled": True,
            "independent_evidence_admission_required_before_official_promotion": True,
            "source_snapshot_digest": read(OUT / "source_verification_v1.json")["source_digest"],
            "preparation_CPU_seconds": time.process_time() - start,
        },
    )
    print(
        json.dumps(
            {
                "protocol": pin(SEARCH / "protocol.json"),
                "panel": [
                    (p["family_priority"], p["identity"], p["conditional_gain_indices"])
                    for p in panel
                ],
            }
        )
    )


def validate():
    protocol = read(SEARCH / "protocol.json")
    for value in protocol["inputs"]:
        assert pin(MAIN / value["path"]) == value, value["path"]
    assert protocol["maximum_workers"] == 3 and protocol["child_CPU_seconds"] == 140
    return protocol


def validate_runtime():
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    protocol = validate()
    qualified = core_module().load_qualified_runner()
    old = qualified.validate_protocol()
    assert old["seed"] == protocol["pythonhashseed"]
    print(json.dumps({"passed": True, "searches": 0, "seed": old["seed"]}))


def preflight():
    protocol = validate()
    rows = []
    for correct in (True, False):
        env = environment(protocol)
        if not correct:
            env["PYTHONHASHSEED"] = str(protocol["pythonhashseed"] + 1)
        proc = subprocess.run(
            [str(MAIN / protocol["runtime_executable"]), str(Path(__file__)), "validate_runtime"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows.append(
            {
                "correct_environment": correct,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
    passed = (
        rows[0]["exit_code"] == 0
        and rows[1]["exit_code"] != 0
        and "Set PYTHONHASHSEED" in rows[1]["stderr"]
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        SEARCH / "preflight.json",
        {
            "passed": passed,
            "protocol": pin(SEARCH / "protocol.json"),
            "cases": rows,
            "child_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "new_searches": 0,
        },
    )
    if not passed:
        raise ValueError("Fresh runtime qualification failed")


def target(index):
    resource.setrlimit(resource.RLIMIT_CPU, (140, 140))
    protocol = validate()
    assert read(SEARCH / "preflight.json")["passed"]
    import forge

    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    import socket

    def deny(*_, **__):
        raise PermissionError("Network disabled for streamD route search")

    for name in ("connect", "connect_ex", "send", "sendall", "sendto"):
        setattr(socket.socket, name, deny)
    socket.create_connection = deny
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    core = core_module()
    core.OUT = SEARCH
    core.validate = lambda: protocol
    start = time.monotonic()
    try:
        core.target(index)
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write(
            SEARCH / f"target_{index}_usage.json",
            {
                "CPU_seconds": usage.ru_utime + usage.ru_stime,
                "wall_seconds": time.monotonic() - start,
                "protocol": pin(SEARCH / "protocol.json"),
                "Python_network_guard_installed": True,
                "onnx_telemetry_disabled": True,
                "imported_forge": str(Path(forge.__file__).resolve()),
            },
        )


def supervise():
    resource.setrlimit(resource.RLIMIT_CPU, (15, 1800))
    protocol = validate()

    def one(row):
        index = row["index"]
        cmd = [
            str(MAIN / protocol["runtime_executable"]),
            str(Path(__file__)),
            "target",
            "--index",
            str(index),
        ]
        write(
            SEARCH / f"target_{index}.launch.json",
            {"command": cmd, "protocol": pin(SEARCH / "protocol.json")},
        )
        start = time.monotonic()
        with (
            (SEARCH / f"target_{index}.stdout").open("xb") as stdout,
            (SEARCH / f"target_{index}.stderr").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=environment(protocol),
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=protocol["child_wall_seconds"])
                timeout = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                timeout = True
        path = SEARCH / f"target_{index}/result.json"
        result = {
            "index": index,
            "exit_code": code,
            "timeout": timeout,
            "wall_seconds": time.monotonic() - start,
            "completed": code == 0 and path.exists() and read(path).get("status") == "completed",
            "result": pin(path) if path.exists() else None,
        }
        write(SEARCH / f"target_{index}.exit.json", result)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        exits = list(pool.map(one, protocol["panel"]))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        SEARCH / "completion.json",
        {
            "protocol": pin(SEARCH / "protocol.json"),
            "exits": exits,
            "declared_indices": [0, 1, 2, 3],
            "all_completed": all(x["completed"] for x in exits),
            "unstarted_indices": [],
            "children_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "supervisor_CPU_seconds": time.process_time(),
            "new_measured_makeable_products": 0,
            "no_retry": True,
        },
    )


def launch():
    protocol = validate()
    assert read(SEARCH / "preflight.json")["passed"]
    write(
        SEARCH / "launch_intent.json",
        {"protocol": pin(SEARCH / "protocol.json"), "time_epoch": time.time()},
    )
    cmd = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "supervise"]
    with (
        (SEARCH / "supervisor.stdout").open("xb") as stdout,
        (SEARCH / "supervisor.stderr").open("xb") as stderr,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=environment(protocol),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write(
        SEARCH / "launch_receipt.json",
        {
            "pid": process.pid,
            "command": cmd,
            "protocol": pin(SEARCH / "protocol.json"),
            "detached": True,
        },
    )
    print(json.dumps({"pid": process.pid, "receipt": pin(SEARCH / "launch_receipt.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("freeze", "preflight", "validate_runtime", "target", "supervise", "launch"),
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "target":
        target(args.index)
    else:
        globals()[args.action]()
