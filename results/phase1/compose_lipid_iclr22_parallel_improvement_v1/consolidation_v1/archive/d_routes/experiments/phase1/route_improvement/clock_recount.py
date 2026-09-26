"""Three fixed full cohorts, each paired through the unchanged typed classifier at one clock."""

import argparse
import copy
import json
import os
import resource
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import new_clock_routes as routes
from audit_new_leaves import MAIN, OUT, ROOT, pin, read, write
from pair_full_v2 import FIELDS, previous

import forge

DEST = OUT / "new_clock_recount_v1"
JOINT = OUT.parent / "integration/joint_selection_v1/run_v2"
ARMS = ("original", "B", "joint")


def listing_key(value):
    receipt = value.receipt
    return (receipt.location, receipt.sha256) if receipt else None


def strip_new_component(component, route_ids, receipt_keys):
    def node(value):
        step = value.step
        return replace(
            value,
            listings=tuple(x for x in value.listings if listing_key(x) not in receipt_keys),
            step=(
                replace(step, reactants=tuple(node(child) for child in step.reactants))
                if step
                else None
            ),
        )

    return replace(
        component,
        direct=tuple(x for x in component.direct if listing_key(x) not in receipt_keys),
        routes=tuple(
            replace(path, root=node(path.root))
            for path in component.routes
            if path.route_id not in route_ids
        ),
    )


def freeze():
    assert not DEST.exists()
    admission = read(routes.DEST / "root_admission.json")
    assert admission["passed"] and admission["validator"] == pin(Path(routes.__file__))
    envelope = read(routes.DEST / "admitted_paths.json")
    assert admission["paths_sha256"] == routes.prior.content_sha(envelope["paths"])
    assert read(routes.DEST / "qualification.json")["passed"]
    review = read(OUT / "joint_binding_review_v1/result.json")
    assert review["passed"] and review["changed_requests"] == 36
    joint = read(JOINT / "result.json")
    assert pin(JOINT / "result.json") in review["inputs"]
    DEST.mkdir()
    original = read(routes.previous.BASE / "protocol.json")
    population = copy.deepcopy(routes.prior.read_pin(original["population"]))
    population["requests"] = [r["route_request"] for r in joint["selected_evidence"]]
    assert (
        len(population["requests"]) == 1408
        and sum(r["strict_complete"] for r in population["requests"]) == 27
    )
    population["inputs"]["assemblies"] = pin(JOINT / "result.json")
    population["inputs"]["joint_binding_review"] = pin(OUT / "joint_binding_review_v1/result.json")
    population["inputs"]["producer"] = pin(Path(__file__))
    population["official_cohort_replaced"] = False
    write(DEST / "joint_requests.json", population)
    write(
        DEST / "joint_control_products.json",
        {
            "products": [r["route_verdict"] for r in joint["selected_evidence"]],
            "source": pin(JOINT / "result.json"),
        },
    )
    definitions = {
        "original": (
            OUT / "full_pair_v2/treatment/protocol.json",
            original["population"],
            OUT / "full_pair_v2/treatment/products.json",
        ),
        "B": (
            OUT / "joint_b_d_v1/assessment/protocol.json",
            pin(OUT / "joint_b_d_v1/requests.json"),
            OUT / "joint_b_d_v1/assessment/products.json",
        ),
        "joint": (
            OUT / "full_pair_v2/treatment/protocol.json",
            pin(DEST / "joint_requests.json"),
            DEST / "joint_control_products.json",
        ),
    }
    refs = {}
    for arm, (old_protocol_path, population_pin, old_products) in definitions.items():
        directory = DEST / arm
        directory.mkdir()
        protocol = copy.deepcopy(read(old_protocol_path))
        protocol["as_of_utc"] = routes.AS_OF
        protocol["population"] = population_pin
        protocol["implementation"].extend([pin(Path(__file__)), pin(Path(routes.__file__))])
        protocol["vendor_envelopes"].append(pin(routes.LISTINGS))
        protocol["route_envelopes"].append(pin(routes.DEST / "admitted_paths.json"))
        protocol.pop("stream_d_pair", None)
        protocol["new_clock_pair"] = {
            "arm": arm,
            "fixed_population": True,
            "previous_protocol": pin(old_protocol_path),
            "previous_products": pin(old_products),
            "common_as_of_utc": routes.AS_OF,
            "new_routes": [r["route_id"] for r in envelope["paths"]],
            "new_listing_envelope": pin(routes.LISTINGS),
            "control": "Actual unchanged typed product classifier on identical requests at the common new clock, removing exactly declared two new paths and exactly the two new listing receipts from direct observations and every route node; assert full per-product identity to saved control.",
            "root_route_admission": pin(routes.DEST / "root_admission.json"),
            "route_qualification": pin(routes.DEST / "qualification.json"),
            "joint_binding_review": pin(OUT / "joint_binding_review_v1/result.json"),
            "CPU_cap_seconds": 160,
            "wall_cap_seconds": 600,
            "seed": 0,
        }
        write(directory / "protocol.json", protocol)
        refs[arm] = pin(directory / "protocol.json")
    write(
        DEST / "protocol.json",
        {
            "schema": "forge.stream_d.three_fixed_cohort_common_clock_recount.v1",
            "arms": refs,
            "population": 1408,
            "families": 22,
            "paired_typed_assessments_per_arm": 2816,
            "producer": pin(Path(__file__)),
            "validator": pin(Path(routes.__file__)),
            "joint_population": pin(DEST / "joint_requests.json"),
            "aggregate_CPU_cap_seconds": 490,
            "maximum_workers": 3,
            "no_model_search_HTTP_TEST": True,
            "no_reselection": True,
            "no_retry": True,
        },
    )
    print(json.dumps({"protocol": pin(DEST / "protocol.json")}))


def execute(arm):
    resource.setrlimit(resource.RLIMIT_CPU, (160, 160))
    started = time.process_time()
    directory = DEST / arm
    protocol = read(directory / "protocol.json")
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    for ref in protocol["implementation"]:
        assert pin(MAIN / ref["path"]) == ref
    write(
        directory / "started.json",
        {
            "pid": os.getpid(),
            "protocol": pin(directory / "protocol.json"),
            "time_epoch": time.time(),
        },
    )
    try:
        import run_v10 as evaluator

        ids = set(protocol["new_clock_pair"]["new_routes"])
        receipt_keys = {
            (routes.prior.location(r["receipt"]), r["receipt"]["sha256"])
            for r in read(routes.LISTINGS)["listings"]
        }
        assert len(ids) == len(receipt_keys) == 2
        requests = {
            r["index"]: r for r in routes.prior.read_pin(protocol["population"])["requests"]
        }
        old_rows = routes.prior.read_pin(protocol["new_clock_pair"]["previous_products"])[
            "products"
        ]
        originals, controls, component_cache = {}, {}, {}
        direct = evaluator.classify_product

        def paired(**args):
            result = direct(**args)
            items = []
            for branch in args["evidence"]:
                identity = branch.component.identity
                if identity in originals:
                    assert originals[identity] == branch.component
                else:
                    originals[identity] = branch.component
                    component_cache[identity] = strip_new_component(
                        branch.component, ids, receipt_keys
                    )
                items.append(replace(branch, component=component_cache[identity]))
            before = direct(**{**args, "evidence": tuple(items)})
            index = int(args["request_id"])
            assert index not in controls
            branch_rows = []
            for branch in before.branches:
                assessed = branch.assessment
                assert assessed is not None
                branch_rows.append(
                    {
                        "branch_id": branch.requirement.branch_id,
                        "identity": branch.requirement.identity,
                        "state": assessed.state.value,
                        "direct_listed": any(x.state.value == "listed" for x in assessed.direct),
                        "computational_path": any(x.path_supported for x in assessed.routes),
                        "listed_leaf_path": any(x.complete for x in assessed.routes),
                        "makeable": assessed.makeable,
                    }
                )
            request = requests[index]
            controls[index] = {
                "index": index,
                "family": request["family"],
                "exact_L1": request["exact_L1"],
                "state": before.state.value,
                "combined_primary": before.state is evaluator.ProductState.MAKEABLE,
                "L2_ready": request["exact_L1"]
                and all(b["direct_listed"] or b["computational_path"] for b in branch_rows),
                "L3_direct_only": request["exact_L1"]
                and all(b["direct_listed"] for b in branch_rows),
                "strict_secondary": request["strict_complete"],
                "branches": branch_rows,
                "reasons": list(before.reasons),
            }
            return result

        evaluator.classify_product = paired
        evaluator.external_routes = routes.Bridge(evaluator.external_routes).external_routes
        evaluator.configure_documentary = previous.warm_documentary
        evaluator.run(directory, protocol)
        after = read(directory / "products.json")["products"]
        control = [controls[i] for i in range(1408)]
        assert control == old_rows and len(after) == len(control) == 1408
        write(
            directory / "control_products.json",
            {"products": control, "protocol": pin(directory / "protocol.json")},
        )
        transitions = {
            field: {
                "gains": [
                    i
                    for i, (a, b) in enumerate(zip(control, after, strict=True))
                    if not a[field] and b[field]
                ],
                "losses": [
                    i
                    for i, (a, b) in enumerate(zip(control, after, strict=True))
                    if a[field] and not b[field]
                ],
            }
            for field in FIELDS
        }
        assert not any(change["losses"] for change in transitions.values())
        assert not transitions["exact_L1"]["gains"] and not transitions["strict_secondary"]["gains"]
        families = sorted({r["family"] for r in after})
        assert len(families) == 22
        validation = {
            "passed": True,
            "protocol": pin(directory / "protocol.json"),
            "result": pin(directory / "result.json"),
            "control_products": pin(directory / "control_products.json"),
            "full_denominator": 1408,
            "branches": sum(len(r["branches"]) for r in after),
            "control_exactly_reproduces_saved_records": True,
            "common_as_of_utc": routes.AS_OF,
            "transitions": transitions,
            "before": {k: sum(r[k] for r in control) for k in FIELDS},
            "after": {k: sum(r[k] for r in after) for k in FIELDS},
            "by_family": [
                {
                    "family": f,
                    "requests": sum(r["family"] == f for r in after),
                    "before": {k: sum(r[k] for r in control if r["family"] == f) for k in FIELDS},
                    "after": {k: sum(r[k] for r in after if r["family"] == f) for k in FIELDS},
                }
                for f in families
            ],
            "process_CPU_seconds": time.process_time(),
            "execute_CPU_seconds": time.process_time() - started,
            "new_model_search_HTTP_TEST_calls": 0,
        }
        write(directory / "validation.json", validation)
        write(
            directory / "execution.json",
            {
                "complete": True,
                "result": pin(directory / "result.json"),
                "validation": pin(directory / "validation.json"),
            },
        )
    except Exception as error:
        write(
            directory / "failure.json",
            {"complete": False, "error": repr(error), "process_CPU_seconds": time.process_time()},
        )
        raise


def supervise():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 490))

    def child(arm):
        directory = DEST / arm
        with (
            (directory / "stdout.log").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "execute", "--arm", arm],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            write(
                directory / "launch_receipt.json",
                {"pid": process.pid, "protocol": pin(directory / "protocol.json")},
            )
            try:
                code, timeout = process.wait(timeout=600), False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code, timeout = process.wait(), True
        return {
            "arm": arm,
            "exit_code": code,
            "timeout": timeout,
            "complete": code == 0
            and (directory / "execution.json").exists()
            and read(directory / "execution.json")["complete"],
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        exits = list(pool.map(child, ARMS))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        DEST / "completion.json",
        {
            "protocol": pin(DEST / "protocol.json"),
            "all_completed": all(r["complete"] for r in exits),
            "exits": exits,
            "children_CPU_seconds": usage.ru_utime + usage.ru_stime,
            "supervisor_CPU_seconds": time.process_time(),
            "no_retry": True,
        },
    )


def launch():
    protocol = read(DEST / "protocol.json")
    review = read(DEST / "root_review.json")
    assert review["passed"] and review["protocol"] == pin(DEST / "protocol.json")
    assert protocol["producer"] == pin(Path(__file__))
    assert not (DEST / "launch_intent.json").exists()
    write(
        DEST / "launch_intent.json",
        {"protocol": pin(DEST / "protocol.json"), "time_epoch": time.time()},
    )
    with (DEST / "stdout.log").open("xb") as stdout, (DEST / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            [str(MAIN / ".venv/bin/python"), str(Path(__file__)), "supervise"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    write(
        DEST / "launch_receipt.json",
        {
            "pid": process.pid,
            "protocol": pin(DEST / "protocol.json"),
            "review": pin(DEST / "root_review.json"),
            "detached": True,
        },
    )
    print(json.dumps({"pid": process.pid, "receipt": pin(DEST / "launch_receipt.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "supervise", "execute"])
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    execute(args.arm) if args.action == "execute" else globals()[args.action]()
