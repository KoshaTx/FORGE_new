"""Two real typed E product assessments per request while loading route evidence once."""

import argparse
import copy
import os
import resource
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import candidate_routes as candidate
import e_population as population
from audit_new_leaves import MAIN, OUT, ROOT, pin, read, write
from pair_full_v2 import FIELDS, previous

import forge

DEST = population.DEST / "assessment"


def without_new_routes(evidence, route_ids):
    return tuple(
        replace(
            branch,
            component=replace(
                branch.component,
                routes=tuple(
                    route for route in branch.component.routes if route.route_id not in route_ids
                ),
            ),
        )
        for branch in evidence
    )


def freeze():
    review = read(population.DEST / "root_population_admission.json")
    assert review["passed"] and review["reviewer"] == "root"
    assert review["population"] == pin(population.DEST / "requests.json")
    assert review["assembly_bridge"] == pin(population.DEST / "assembly_receipt_bridge.json")
    assert review["producer"] == pin(Path(population.__file__))
    assert review["evaluator_candidate"] == pin(Path(__file__).with_name("evaluator_e_v1.py"))
    protocol = copy.deepcopy(read(OUT / "full_pair_v2/treatment/protocol.json"))
    protocol["population"] = pin(population.DEST / "requests.json")
    protocol["admission"] = pin(population.DEST / "root_population_admission.json")
    protocol["implementation"].extend(
        [
            pin(Path(__file__)),
            pin(Path(population.__file__)),
            pin(Path(__file__).with_name("evaluator_e_v1.py")),
        ]
    )
    protocol.pop("stream_d_pair")
    protocol["E_pair"] = {
        "scope": "E-only and E+D use identical245 changed molecular identities and identical listing observations. Each request is classified twice by the unchanged actual product classifier; control removes exactly six independently admitted D routes, preserving all original evidence.",
        "changed_molecules": 245,
        "strict_original": 27,
        "strict_unchanged_admitted": 16,
        "changed_old_dossiers_unestablished": read(population.DEST / "requests.json")[
            "old_strict_positives_not_transferred"
        ],
        "original_baseline_result": pin(candidate.BASE / "result.json"),
        "D_only_result": pin(OUT / "full_pair_v2/treatment/result.json"),
        "CPU_cap_seconds": 160,
        "wall_cap_seconds": 600,
        "maximum_workers": 1,
        "seed": 0,
        "no_network_model_TEST": True,
        "control_removed_route_ids": sorted(
            row["route_id"] for row in read(candidate.CAND / "admitted_paths.json")["paths"]
        ),
    }
    DEST.mkdir(exist_ok=False)
    write(DEST / "protocol.json", protocol)


def execute():
    resource.setrlimit(resource.RLIMIT_CPU, (160, 160))
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    protocol = read(DEST / "protocol.json")
    for ref in protocol["implementation"]:
        assert pin(MAIN / ref["path"]) == ref
    write(
        DEST / "started.json",
        {"pid": os.getpid(), "protocol": pin(DEST / "protocol.json"), "time_epoch": time.time()},
    )
    try:
        import evaluator_e_v1 as evaluator

        route_ids = set(protocol["E_pair"]["control_removed_route_ids"])
        source = read(population.DEST / "requests.json")
        requests = {row["index"]: row for row in source["requests"]}
        controls = {}
        direct_product = evaluator.classify_product

        def classify_pair(**arguments):
            result = direct_product(**arguments)
            control_evidence = without_new_routes(arguments["evidence"], route_ids)
            control_arguments = {**arguments, "evidence": control_evidence}
            control = direct_product(**control_arguments)
            index = int(arguments["request_id"])
            assert index not in controls
            assert [b.requirement for b in control_evidence] == [
                b.requirement for b in arguments["evidence"]
            ]
            for before, after in zip(arguments["evidence"], control_evidence, strict=True):
                assert after.component.direct == before.component.direct
                assert after.component.identity == before.component.identity
                assert after.component.routes == tuple(
                    route for route in before.component.routes if route.route_id not in route_ids
                )
            if control_evidence == arguments["evidence"]:
                assert control == result
            branch_rows = []
            for branch in control.branches:
                item = branch.assessment
                assert item is not None
                branch_rows.append(
                    {
                        "branch_id": branch.requirement.branch_id,
                        "identity": branch.requirement.identity,
                        "state": item.state.value,
                        "direct_listed": any(v.state.value == "listed" for v in item.direct),
                        "computational_path": any(v.path_supported for v in item.routes),
                        "listed_leaf_path": any(v.complete for v in item.routes),
                        "makeable": item.makeable,
                    }
                )
            request = requests[index]
            controls[index] = {
                "index": index,
                "family": request["family"],
                "exact_L1": request["exact_L1"],
                "state": control.state.value,
                "combined_primary": control.state is evaluator.ProductState.MAKEABLE,
                "L2_ready": request["exact_L1"]
                and all(b["direct_listed"] or b["computational_path"] for b in branch_rows),
                "L3_direct_only": request["exact_L1"]
                and all(b["direct_listed"] for b in branch_rows),
                "strict_secondary": request["strict_complete"],
                "branches": branch_rows,
                "reasons": list(control.reasons),
            }
            return result

        evaluator.classify_product = classify_pair
        evaluator.external_routes = candidate.Bridge(evaluator.external_routes).external_routes
        evaluator.configure_documentary = previous.warm_documentary
        evaluator.run(DEST, protocol)
        rows = read(DEST / "products.json")["products"]
        control_rows = [controls[i] for i in range(1408)]
        assert len(controls) == len(rows) == 1408
        write(
            DEST / "control_products.json",
            {
                "protocol": pin(DEST / "protocol.json"),
                "scope": "Actual unchanged classifier on the same E products with six D routes absent",
                "products": control_rows,
            },
        )
        old = read(candidate.BASE / "products.json")["products"]
        d_only = read(OUT / "full_pair_v2/treatment/products.json")["products"]
        changed = set(source["changed_indices"])
        assert all(a == b for a, b in zip(d_only, rows, strict=True) if a["index"] not in changed)
        assert all(
            a == b for a, b in zip(old, control_rows, strict=True) if a["index"] not in changed
        )

        def compare(before, after):
            return {
                key: {
                    "gains": [
                        b["index"]
                        for a, b in zip(before, after, strict=True)
                        if not a[key] and b[key]
                    ],
                    "losses": [
                        b["index"]
                        for a, b in zip(before, after, strict=True)
                        if a[key] and not b[key]
                    ],
                }
                for key in FIELDS
            }

        summary = {
            "passed": True,
            "protocol": pin(DEST / "protocol.json"),
            "result": pin(DEST / "result.json"),
            "control_products": pin(DEST / "control_products.json"),
            "full_denominator": 1408,
            "actual_typed_product_classifier_calls": 2816,
            "changed_indices": sorted(changed),
            "unchanged1163_identical_to_matching_baseline": True,
            "E_only_totals": {key: sum(r[key] for r in control_rows) for key in FIELDS},
            "E_and_D_totals": {key: sum(r[key] for r in rows) for key in FIELDS},
            "E_only_against_original": compare(old, control_rows),
            "E_and_D_against_D_only": compare(d_only, rows),
            "D_intervention_with_E_fixed": compare(control_rows, rows),
            "by_family": [
                {
                    "family": family,
                    "requests": 64,
                    "original": {
                        key: sum(r[key] for r in old if r["family"] == family) for key in FIELDS
                    },
                    "D_only": {
                        key: sum(r[key] for r in d_only if r["family"] == family) for key in FIELDS
                    },
                    "E_only": {
                        key: sum(r[key] for r in control_rows if r["family"] == family)
                        for key in FIELDS
                    },
                    "E_and_D": {
                        key: sum(r[key] for r in rows if r["family"] == family) for key in FIELDS
                    },
                }
                for family in sorted({r["family"] for r in rows})
            ],
            "strict_secondary_interpretation": "11 formerly strict products changed identity; their new dossiers are unestablished, not experimental failures.16 unchanged strict positives retain original proofs.",
            "process_CPU_seconds": time.process_time(),
            "new_HTTP_calls": 0,
        }
        write(DEST / "validation.json", summary)
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
        DEST / "launch_receipt.json",
        {"pid": proc.pid, "protocol": pin(DEST / "protocol.json"), "detached": True},
    )
    print(proc.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "launch", "execute", "supervise"])
    globals()[parser.parse_args().action]()
