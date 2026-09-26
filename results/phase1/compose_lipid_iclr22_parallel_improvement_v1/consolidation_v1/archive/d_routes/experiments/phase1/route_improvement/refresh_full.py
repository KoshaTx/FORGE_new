"""One-shot real-evaluator refresh after a bounded route-search intervention."""

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import forge

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parents[1]
OUT = MAIN / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
PAR = MAIN / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1"
PREP = (
    PAR
    / "routes/next_bottlenecks_v3/new_leaf_listing_diagnostic_v1/paired_evaluation_preparation_v1"
)
BASE = PREP / "paired_run_v1/treatment"
DEST = OUT / "full_refresh_v1"
sys.path.insert(0, str(PREP))
import paired_runner as previous  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(MAIN)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze():
    audit = read(OUT / "leaf_search_v1/tree_audit_v1.json")
    # A positive tree needs separately reviewed graft/route admission before a different runner.
    assert audit["closed_trees"] == 0 and audit["all_four_declared_targets_retained"]
    protocol = copy.deepcopy(read(BASE / "protocol.json"))
    protocol["implementation"].append(pin(Path(__file__)))
    protocol["stream_d_refresh"] = {
        "baseline": pin(BASE / "result.json"),
        "baseline_products": pin(BASE / "products.json"),
        "baseline_components": pin(BASE / "components.json"),
        "new_search_tree_audit": pin(OUT / "leaf_search_v1/tree_audit_v1.json"),
        "admitted_new_route_count": 0,
        "new_vendor_observations": 0,
        "scope": "Full real evaluator refresh of unchanged admitted evidence after zero new exact-listed-leaf closures. New search failures retained as diagnostics, not admitted paths.",
        "seed": 0,
        "CPU_cap_seconds": 150,
        "wall_cap_seconds": 600,
        "threads": 1,
    }
    DEST.mkdir(exist_ok=False)
    write(DEST / "protocol.json", protocol)
    print(json.dumps({"protocol": pin(DEST / "protocol.json")}))


def execute():
    resource.setrlimit(resource.RLIMIT_CPU, (150, 150))
    start = time.process_time()
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    protocol = read(DEST / "protocol.json")
    settings = protocol["stream_d_refresh"]
    for ref in protocol["implementation"] + [
        v for v in settings.values() if isinstance(v, dict) and "sha256" in v
    ]:
        assert pin(MAIN / ref["path"]) == ref
    write(
        DEST / "started.json",
        {"protocol": pin(DEST / "protocol.json"), "pid": os.getpid(), "time_epoch": time.time()},
    )
    try:
        import run_v10 as evaluator

        bridge = previous.wrapper.Bridge(evaluator.external_routes)
        evaluator.external_routes = bridge.external_routes
        evaluator.configure_documentary = previous.warm_documentary
        evaluator.run(DEST, protocol)
        before = read(BASE / "products.json")["products"]
        after = read(DEST / "products.json")["products"]
        assert len(before) == len(after) == 1408
        assert before == after, "Full product/branch verdicts changed under unchanged evidence"
        assert (
            read(BASE / "components.json")["components"]
            == read(DEST / "components.json")["components"]
        )
        fields = ["exact_L1", "combined_primary", "L2_ready", "L3_direct_only", "strict_secondary"]
        paired = []
        for family in sorted({r["family"] for r in before}):
            selected = [r for r in after if r["family"] == family]
            paired.append(
                {
                    "family": family,
                    "requests": len(selected),
                    "before": {k: sum(r[k] for r in selected) for k in fields},
                    "after": {k: sum(r[k] for r in selected) for k in fields},
                    "gains": {k: [] for k in fields},
                    "losses": {k: [] for k in fields},
                }
            )
        totals = {k: sum(r[k] for r in after) for k in fields}
        assert totals == dict(zip(fields, [1387, 616, 1189, 119, 27], strict=True))
        write(
            DEST / "validation.json",
            {
                "passed": True,
                "protocol": pin(DEST / "protocol.json"),
                "result": pin(DEST / "result.json"),
                "baseline_result": settings["baseline"],
                "all_1408_product_and_branch_objects_identical": True,
                "all_2044_component_objects_identical": True,
                "totals": totals,
                "by_family": paired,
                "network_model_TEST_calls": 0,
                "historical_clock_proofs": bridge.audit,
                "CPU_seconds": time.process_time() - start,
                "imported_forge": str(Path(forge.__file__).resolve()),
            },
        )
        write(
            DEST / "execution.json",
            {
                "complete": True,
                "result": pin(DEST / "result.json"),
                "validation": pin(DEST / "validation.json"),
                "process_CPU_seconds": time.process_time(),
                "automatic_retries": 0,
            },
        )
    except Exception as error:
        write(
            DEST / "failure.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "message": str(error),
                "CPU_seconds": time.process_time() - start,
            },
        )
        raise


def launch():
    protocol = pin(DEST / "protocol.json")
    write(DEST / "launch_intent.json", {"protocol": protocol, "time_epoch": time.time()})
    command = [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "execute"]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with (DEST / "stdout.log").open("xb") as stdout, (DEST / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write(
        DEST / "launch_receipt.json",
        {"pid": child.pid, "command": command, "protocol": protocol, "detached": True},
    )
    print(json.dumps({"pid": child.pid, "protocol": protocol}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "execute"])
    globals()[parser.parse_args().action]()
