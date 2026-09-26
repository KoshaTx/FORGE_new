"""Real route evaluation of the separate B+D molecular cohort, with fresh component identities."""

import argparse
import copy
import os
import resource
import signal
import subprocess
import time
from pathlib import Path

import candidate_routes as candidate
import joint_population as population
from audit_new_leaves import MAIN, OUT, ROOT, pin, read, write
from pair_full_v2 import FIELDS, previous

import forge

DEST = population.DEST / "assessment"


def freeze():
    review = read(population.DEST / "root_population_admission.json")
    assert review["passed"] and review["reviewer"] == "root"
    assert review["population"] == pin(population.DEST / "requests.json")
    assert review["assembly_bridge"] == pin(population.DEST / "assembly_receipt_bridge.json")
    assert review["producer"] == pin(Path(population.__file__))
    protocol = copy.deepcopy(read(OUT / "full_pair_v2/treatment/protocol.json"))
    protocol["population"] = pin(population.DEST / "requests.json")
    protocol["admission"] = pin(population.DEST / "root_population_admission.json")
    protocol["implementation"].extend([pin(Path(__file__)), pin(Path(population.__file__))])
    protocol.pop("stream_d_pair")
    protocol["joint_B_D"] = {
        "scope": "Separate1408 candidate cohort with19 changed molecular graphs; D route evidence held identical. Every changed component is assessed from exact identity, no copied route verdict.",
        "changed_molecules": 19,
        "original_baseline_result": pin(candidate.BASE / "result.json"),
        "D_only_result": pin(OUT / "full_pair_v2/treatment/result.json"),
        "CPU_cap_seconds": 160,
        "wall_cap_seconds": 600,
        "maximum_workers": 1,
        "seed": 0,
        "no_network_model_TEST": True,
    }
    DEST.mkdir(exist_ok=False)
    write(DEST / "protocol.json", protocol)


def execute():
    resource.setrlimit(resource.RLIMIT_CPU, (160, 160))
    started = time.process_time()
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    protocol = read(DEST / "protocol.json")
    for ref in protocol["implementation"]:
        assert pin(MAIN / ref["path"]) == ref
    write(
        DEST / "started.json",
        {"pid": os.getpid(), "protocol": pin(DEST / "protocol.json"), "time_epoch": time.time()},
    )
    try:
        import run_v10 as evaluator

        evaluator.external_routes = candidate.Bridge(evaluator.external_routes).external_routes
        evaluator.configure_documentary = previous.warm_documentary
        evaluator.run(DEST, protocol)
        rows = read(DEST / "products.json")["products"]
        old = read(candidate.BASE / "products.json")["products"]
        d_only = read(OUT / "full_pair_v2/treatment/products.json")["products"]
        source = read(population.DEST / "requests.json")
        changed = set(source["changed_indices"])
        assert len(rows) == len(old) == len(d_only) == 1408
        assert all(a == b for a, b in zip(d_only, rows, strict=True) if a["index"] not in changed)
        for source_row, row in zip(source["requests"], rows, strict=True):
            assert source_row["index"] == row["index"]
            assert [(b["branch_id"], b["identity"]) for b in source_row["requirements"]] == [
                (b["branch_id"], b["identity"]) for b in row["branches"]
            ]

        def compare(before):
            return {
                key: {
                    "gains": [
                        b["index"]
                        for a, b in zip(before, rows, strict=True)
                        if not a[key] and b[key]
                    ],
                    "losses": [
                        b["index"]
                        for a, b in zip(before, rows, strict=True)
                        if a[key] and not b[key]
                    ],
                }
                for key in FIELDS
            }

        write(
            DEST / "validation.json",
            {
                "passed": True,
                "protocol": pin(DEST / "protocol.json"),
                "result": pin(DEST / "result.json"),
                "full_denominator": 1408,
                "changed_indices": sorted(changed),
                "unchanged1389_identical_to_D_only": True,
                "changed_branch_identities_match_new_source_witness": True,
                "totals": {key: sum(row[key] for row in rows) for key in FIELDS},
                "against_original": compare(old),
                "against_D_only": compare(d_only),
                "by_family": [
                    {
                        "family": family,
                        "requests": 64,
                        "original": {
                            key: sum(row[key] for row in old if row["family"] == family)
                            for key in FIELDS
                        },
                        "D_only": {
                            key: sum(row[key] for row in d_only if row["family"] == family)
                            for key in FIELDS
                        },
                        "B_and_D": {
                            key: sum(row[key] for row in rows if row["family"] == family)
                            for key in FIELDS
                        },
                    }
                    for family in sorted({r["family"] for r in rows})
                ],
                "process_CPU_seconds": time.process_time(),
                "execute_CPU_seconds": time.process_time() - started,
                "new_HTTP_calls": 0,
            },
        )
        write(
            DEST / "execution.json",
            {
                "complete": True,
                "result": pin(DEST / "result.json"),
                "validation": pin(DEST / "validation.json"),
                "process_CPU_seconds": time.process_time(),
            },
        )
    except Exception as error:
        write(
            DEST / "failure.json",
            {
                "complete": False,
                "type": type(error).__name__,
                "message": str(error),
                "process_CPU_seconds": time.process_time(),
            },
        )
        raise


def supervise():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 165))
    command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "execute"]
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    write(DEST / "child_launch_receipt.json", {"pid": child.pid, "command": command})
    try:
        code, timed_out = child.wait(timeout=600), False
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        code, timed_out = child.wait(), True
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        DEST / "exit.json",
        {
            "exit_code": code,
            "timeout": timed_out,
            "complete": code == 0 and (DEST / "execution.json").exists(),
            "children_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "supervisor_CPU_seconds": time.process_time(),
        },
    )


def launch():
    write(
        DEST / "launch_intent.json",
        {"protocol": pin(DEST / "protocol.json"), "time_epoch": time.time()},
    )
    command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "supervise"]
    with (DEST / "stdout.log").open("xb") as stdout, (DEST / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(
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
        DEST / "launch_receipt.json",
        {
            "pid": child.pid,
            "protocol": pin(DEST / "protocol.json"),
            "command": command,
            "detached": True,
        },
    )
    print(child.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "execute", "supervise"])
    globals()[parser.parse_args().action]()
