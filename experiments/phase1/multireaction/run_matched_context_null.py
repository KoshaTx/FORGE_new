"""Resumable CPU full-cohort inference/constructor runner; root admits each fixed request."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.synthesis_program_sampling import _initial_state
from results.phase1.compose_lipid_iclr22_table_completion_v1.assembly.recover_readouts_v4 import (
    score,
    summarize,
)


def local(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture = local("matched_context_evaluation_v5")
pipeline = local("matched_context_pipeline_v2")
adapter = fixture.adapter
ROOT, OUT = fixture.ROOT, fixture.OUT


def environment():
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rdkit": rdBase.rdkitVersion,
    }


def relpin(path):
    value = fixture.pin(path)
    value["path"] = str(Path(path).resolve().relative_to(ROOT))
    return value


def validate_checkpoint(state, config, config_pin, families):
    if (
        state.get("identity", {}).get("config_sha256") != config_pin["sha256"]
        or state["identity"].get("source_sha256") != config["runtime_source_sha256"]
        or state["identity"].get("inputs") != config["inputs"]
        or state["identity"].get("seed") != 2026092401
        or state.get("completed_steps") != 2794
        or state.get("examples_seen") != 8851392
        or set(state.get("family_presentations", {})) != set(families)
        or len(families) != 22
        or set(state["family_presentations"].values()) != {402336}
        or config["model"].get("semantic_conditioning") != "semantic_null"
    ):
        raise ValueError("Requires exact complete, exposure-matched2794 null checkpoint")


def prepare_inputs():
    p = fixture.read(OUT / "protocol.json")
    proof = fixture.read(OUT / "fixture_v1/result.json")
    if not proof["passed"] or proof["protocol"] != fixture.pin(OUT / "protocol.json"):
        raise ValueError("Fixture not fully qualified")
    selector_proof = fixture.read(OUT / "pipeline_qualification_v2/result.json")
    if (
        not selector_proof["passed"]
        or selector_proof["requests"] != 1408
        or selector_proof["protocol"]
        != fixture.pin(OUT / "pipeline_qualification_v2/protocol.json")
    ):
        raise ValueError("Full original selector replay not qualified")
    files = [
        Path(__file__),
        Path(pipeline.__file__),
        Path(adapter.__file__),
        Path(fixture.construction.__file__),
        Path(__file__).with_name("verify_saved_pool_controls.py"),
        OUT / "pipeline_qualification_v2/protocol.json",
        OUT / "pipeline_qualification_v2/result.json",
        Path(__file__).resolve().parents[3] / "tests/test_matched_context_execution.py",
        OUT / "execution_tests_v2.xml",
    ]
    reference = ROOT / "results/phase1/compose_lipid_quality_selection_v2"
    files += [
        reference / name
        for name in (
            "policy.json",
            "aema_domain/protocol.json",
            "reference_support.json",
            "reference_support.pkl",
        )
    ]
    evaldir = ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation"
    gate_policy = fixture.read(evaldir / "gate_policy_v3.json")
    for value in gate_policy["inputs"].values():
        relative = Path(value["path"])
        actual = (
            fixture.NULL / "source" / relative if relative.parts[0] == "forge" else ROOT / relative
        )
        if fixture.digest(actual) != value["sha256"]:
            raise ValueError("Actual executable/data gate input differs: " + str(relative))
        files.append(actual)
    quality = ROOT / "results/phase1/compose_lipid_quality_v1"
    files += [quality / name for name in ("reference.json", "policy.json", "run.py")]
    evaldir = ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation"
    files += [
        evaldir / name
        for name in ("design_selector.py", "gates.py", "gate_policy_v3.json", "replay.py")
    ]
    files += [
        ROOT / "results/phase1/compose_lipid_quality_selection_v2/run.py",
        ROOT / "results/phase1/compose_lipid_quality_confirmation_v1/select.py",
        ROOT / "results/phase1/compose_lipid_quality_decode_v2/compact.py",
        ROOT / "results/phase1/compose_lipid_mapped_preparation_v3/cache/sources.sqlite",
        ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/assembly/recover_readouts_v4.py",
    ]
    recipe = {
        "schema": "forge.matched_context_null_execution_recipe.v1",
        "preflight_protocol": fixture.pin(OUT / "protocol.json"),
        "qualification": fixture.pin(OUT / "fixture_v1/result.json"),
        "expected_training_configuration": fixture.pin(
            OUT.parent / "null_fullfit_readiness_v2/configuration.json"
        ),
        "inputs": {str(x): fixture.pin(x) for x in files},
        "seed_schedule": p["schedule"],
        "device": "cpu",
        "environment": environment(),
        "threads": 4,
        "sampling_CPU_cap_seconds": 5400,
        "pipeline_CPU_cap_seconds": 1800,
        "sampling_wall_cap_seconds": 2400,
        "pipeline_wall_cap_seconds": 1800,
        "requests": 1408,
        "draws": 5,
        "trajectories": 7040,
        "full_constructor_shards": 920,
        "checkpoint_binding": "bind --checkpoint <authenticatedfinal> --configuration <fullfitconfig> --training-admission <rootadmission> --output <freshdirectory>",
        "execution_binding": "run --request <boundrequest> --review <root exact request admission> --phase sample|pipeline [--resume]",
        "allowed_admission_key": "matched_context_evaluation_authorized",
        "automatic_retry": False,
        "resume_budget": "Aggregate per-phase CPU cap across explicit invocations; an unclosed attempt consumes its full reserved cap. No automatic retry.",
        "basis_limit": "Full selector uses unchanged original gate_policy_v3; tree-transport evidence on new identities remains separately unadmitted. Compare corresponding original rubric, not silently1324.",
        "scope": "No new fit; jointly changed null training prior/masks/conditioning, shared externally suppliedstructure/noise and decoderbudgets.",
        "quality_promotion": False,
    }
    fixture.write(OUT / "launch_inputs.json", recipe)
    print(json.dumps({"launch_inputs": fixture.pin(OUT / "launch_inputs.json")}))


def bind(args):
    recipe = fixture.read(OUT / "launch_inputs.json")
    for value in recipe["inputs"].values():
        fixture.authenticate(value)
    fixture.authenticate(recipe["preflight_protocol"])
    fixture.authenticate(recipe["qualification"])
    proof = fixture.read(args.training_admission)
    if proof.get("passed") is not True:
        raise ValueError("Independent completed-training admission missing")
    checkpoint = fixture.pin(args.checkpoint.resolve())
    config_pin = fixture.pin(args.configuration.resolve())
    expected_config = recipe["expected_training_configuration"]
    fixture.authenticate(expected_config)
    if config_pin["sha256"] != expected_config["sha256"]:
        raise ValueError("Training configuration differs from frozen paired null fit")
    config = fixture.read(args.configuration)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(state, config, config_pin, fixture.read(OUT / "protocol.json")["families"])
    # An exact root inference review subsequently binds this checkpoint, admission and request together.
    out = args.output.resolve()
    if not out.is_relative_to(OUT.parent):
        raise ValueError("Output must stay inside A ownership")
    out.mkdir(exist_ok=False)
    request = {
        "schema": "forge.matched_context_null_bound_request.v1",
        "recipe": fixture.pin(OUT / "launch_inputs.json"),
        "checkpoint": checkpoint,
        "configuration": config_pin,
        "training_admission": fixture.pin(args.training_admission.resolve()),
        "output": str(out),
        "root_review_required": True,
        "model_seed": 2026092401,
        "device": "cpu",
        "requests": 1408,
        "draws": 5,
        "trajectories": 7040,
        "automatic_retry": False,
    }
    fixture.write(out / "request.json", request)
    print(json.dumps({"request": fixture.pin(out / "request.json")}))


def commit():
    if "FORGE_COMMIT_REQUEST_FD" not in os.environ:
        return
    with os.fdopen(os.dup(int(os.environ["FORGE_COMMIT_REQUEST_FD"])), "w", buffering=1) as stream:
        stream.write("commit\n")
    with os.fdopen(os.dup(int(os.environ["FORGE_COMMIT_REPLY_FD"])), "r") as reply:
        if reply.readline() != "committed\n":
            raise RuntimeError("Durable commit not acknowledged")


def atomic_tensor(path, value):
    if path.exists():
        raise FileExistsError("Unclosed tensor artifact exists")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("Preserve partial tensor artifact for explicit recovery")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_closed_sampling(path, entry, request_sha256, root):
    receipt = fixture.read(path)
    if receipt["request_sha256"] != request_sha256 or receipt["schedule"] != entry:
        raise ValueError("Sampling resume schedule/input changed")
    if (receipt["draw"], receipt["offset"]) != (entry["draw"], entry["offset"]):
        raise ValueError("Sampling resume index changed")
    for key in ("predictions", "readouts", "states"):
        resolve_pin(receipt[key], root, label="closed sampling output")
    rows = fixture.read(root / receipt["readouts"]["path"])
    if len(rows) != 8 or [(r["draw"], r["index"]) for r in rows] != [
        (entry["draw"], i) for i in range(entry["offset"], entry["offset"] + 8)
    ]:
        raise ValueError("Sampling resume readout denominator changed")
    return receipt, rows


def open_attempt(destination, cap, request_sha256):
    used = 0.0
    for path in destination.glob("*.attempt.json"):
        prior = fixture.read(path)
        if prior["request_sha256"] != request_sha256:
            raise ValueError("Prior attempt belongs to another request")
        closed = path.with_name(path.name.replace(".attempt.json", ".closed.json"))
        used += (
            fixture.read(closed)["CPU_seconds"]
            if closed.exists()
            else prior["reserved_CPU_seconds"]
        )
    remaining = int(cap - used)
    if remaining <= 0:
        raise ValueError("Aggregate phase CPU budget exhausted; no automatic retry")
    attempt = destination / f"{time.time_ns()}.attempt.json"
    fixture.write(
        attempt,
        {
            "request_sha256": request_sha256,
            "previous_CPU_seconds": used,
            "reserved_CPU_seconds": remaining,
        },
    )
    return attempt, remaining


def run(args):
    start_cpu, start_wall = time.process_time(), time.monotonic()
    request = fixture.read(args.request)
    review = fixture.read(args.review)
    if (
        review.get("passed") is not True
        or review.get("matched_context_evaluation_authorized") is not True
        or review.get("request_sha256") != fixture.digest(args.request)
    ):
        raise ValueError("Exact separate inference admission required")
    recipe = fixture.read(fixture.authenticate(request["recipe"]))
    if recipe["environment"] != environment():
        raise ValueError("Inference environment differs from qualified CPU fixture")
    for value in recipe["inputs"].values():
        fixture.authenticate(value)
    fixture.authenticate(recipe["preflight_protocol"])
    fixture.authenticate(recipe["qualification"])
    for key in ("checkpoint", "configuration", "training_admission"):
        fixture.authenticate(request[key])
    p, payload = fixture.inputs()
    if recipe["seed_schedule"] != p["schedule"]:
        raise ValueError("Frozen draw schedule changed")
    out = Path(request["output"])
    cap = recipe[
        "sampling_CPU_cap_seconds" if args.phase == "sample" else "pipeline_CPU_cap_seconds"
    ]
    wall = recipe[
        "sampling_wall_cap_seconds" if args.phase == "sample" else "pipeline_wall_cap_seconds"
    ]
    if review.get("phase_CPU_caps", {}).get(args.phase) != cap:
        raise ValueError("Root review must explicitly bind phase CPU cap")

    def timeout(*_):
        raise TimeoutError("Fixed inference phase wall cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, wall)
    with (out / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = out / args.phase
        if destination.exists() != args.resume:
            raise ValueError("Explicit resume/fresh boundary differs")
        destination.mkdir(exist_ok=args.resume)
        if (destination / "result.json").exists():
            raise FileExistsError("Preserve completed phase")
        attempt, remaining_cpu = open_attempt(destination, cap, fixture.digest(args.request))
        resource.setrlimit(resource.RLIMIT_CPU, (remaining_cpu, remaining_cpu + 1))
        complete = False
        try:
            torch.set_num_threads(4 if args.phase == "sample" else 1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            hashes = fixture.frozen_construction.load(
                fixture.frozen_construction.MIRROR
                / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py",
                "matched_run_hashes",
            ).VerifiedHashes()
            with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
                context = fixture.construction.setup(
                    ROOT, fixture.read(Path(p["inputs"]["constructor_protocol"]["path"])), payload
                )
                if args.phase == "sample":
                    state = torch.load(
                        Path(request["checkpoint"]["path"]), map_location="cpu", weights_only=False
                    )
                    config = fixture.read(request["configuration"]["path"])
                    validate_checkpoint(state, config, request["configuration"], p["families"])
                    base = fixture.model_for(payload, state["model"], "semantic_null")
                    model = adapter.MatchedContextNull(base)
                    noise = [
                        torch.tensor(payload["noise"][key], dtype=torch.float32)
                        for key in ("node", "bond")
                    ]
                    receipts = []
                    all_rows = []
                    for entry in p["schedule"]:
                        draw, offset, seed = entry["draw"], entry["offset"], entry["seed"]
                        stem = destination / f"draw-{draw}-{offset:04d}"
                        closed = stem.with_suffix(".receipt.json")
                        if closed.exists():
                            receipt, rows = read_closed_sampling(
                                closed, entry, fixture.digest(args.request), ROOT
                            )
                        else:
                            tick = time.process_time()
                            layouts = payload["layouts"][offset : offset + 8]
                            batch = collate_generated_layouts(layouts, maximum_closures=12)
                            captured = adapter.capture(
                                model, base, batch, *noise, seed=seed, semantic_null=True
                            )
                            initial = _initial_state(
                                batch, *noise, torch.Generator().manual_seed(seed)
                            )
                            if {
                                k: adapter.tensor_digest(initial[k]) for k in adapter.FIELDS
                            } != captured["trace"]["initial_state"]:
                                raise ValueError("Seeded reference initial noise mismatch")
                            rows = score(
                                {
                                    "true_endpoint": captured["raw"],
                                    "terminal_argmax": captured["terminal"],
                                },
                                layouts,
                                payload["atoms"],
                                context["found"],
                                draw,
                                offset,
                            )
                            predictions = {
                                k: v.detach().cpu() for k, v in captured.pop("predictions").items()
                            }
                            atomic_tensor(stem.with_suffix(".pt"), predictions)
                            atomic_tensor(stem.with_suffix(".states.pt"), captured)
                            fixture.write(stem.with_suffix(".readouts.json"), rows)
                            receipt = {
                                "request_sha256": fixture.digest(args.request),
                                "schedule": entry,
                                "draw": draw,
                                "offset": offset,
                                "predictions": relpin(stem.with_suffix(".pt")),
                                "states": relpin(stem.with_suffix(".states.pt")),
                                "readouts": relpin(stem.with_suffix(".readouts.json")),
                                "CPU_seconds": time.process_time() - tick,
                            }
                            fixture.write(closed, receipt)
                            commit()
                        receipts.append(receipt)
                        all_rows.extend(rows)
                    if (
                        len(receipts) != 880
                        or len(all_rows) != 7040
                        or len({(r["draw"], r["index"]) for r in all_rows}) != 7040
                    ):
                        raise ValueError("Incomplete full five-draw denominator")
                    result = {
                        "complete": True,
                        "requests": 1408,
                        "trajectories": 7040,
                        "receipts": receipts,
                        "summaries": summarize(all_rows, p["families"]),
                        "quality_promotion": False,
                    }
                else:
                    generated = fixture.read(out / "sample/result.json")
                    if (
                        not generated["complete"]
                        or generated["request"] != fixture.pin(args.request)
                        or generated["requests"] != 1408
                        or generated["trajectories"] != 7040
                        or len(generated["receipts"]) != 880
                        or len(p["schedule"]) != 880
                    ):
                        raise ValueError("Sampling is incomplete")
                    for receipt, entry in zip(generated["receipts"], p["schedule"], strict=True):
                        closed = (
                            out
                            / "sample"
                            / f"draw-{entry['draw']}-{entry['offset']:04d}.receipt.json"
                        )
                        authenticated, _ = read_closed_sampling(
                            closed, entry, fixture.digest(args.request), ROOT
                        )
                        if authenticated != receipt:
                            raise ValueError("Sampling phase receipt changed")
                    rows, receipts = pipeline.construct_all(
                        ROOT,
                        destination,
                        payload,
                        generated["receipts"],
                        context,
                        fixture.construction,
                        relpin,
                        fixture.write,
                        commit,
                    )
                    selected = pipeline.selection(
                        ROOT, destination, payload, rows, relpin, fixture.write, commit
                    )
                    result = {
                        "complete": True,
                        "requests": 1408,
                        "trajectories": 7040,
                        "constructor_receipts": receipts,
                        "selection": selected,
                        "quality_promotion": False,
                    }
                hashes.validate()
            result.update(
                request=fixture.pin(args.request),
                CPU_seconds=time.process_time() - start_cpu,
                wall_seconds=time.monotonic() - start_wall,
                automatic_retry=False,
            )
            fixture.write(destination / "result.json", result)
            commit()
            complete = True
            print(
                json.dumps(
                    {
                        "complete": True,
                        "result": relpin(destination / "result.json"),
                        "CPU_seconds": result["CPU_seconds"],
                    }
                )
            )
        except BaseException as error:
            path = destination / f"failure-{time.time_ns()}.json"
            fixture.write(
                path,
                {
                    "complete": False,
                    "error": repr(error),
                    "request": fixture.pin(args.request),
                    "CPU_seconds": time.process_time() - start_cpu,
                    "wall_seconds": time.monotonic() - start_wall,
                    "partial_metrics_admitted": False,
                    "automatic_retry": False,
                },
            )
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            fixture.write(
                attempt.with_name(attempt.name.replace(".attempt.json", ".closed.json")),
                {"CPU_seconds": time.process_time(), "complete": complete},
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-inputs")
    bound = sub.add_parser("bind")
    for key in ("checkpoint", "configuration", "training-admission", "output"):
        bound.add_argument("--" + key, type=Path, required=True)
    launch = sub.add_parser("run")
    for key in ("request", "review"):
        launch.add_argument("--" + key, type=Path, required=True)
    launch.add_argument("--phase", choices=("sample", "pipeline"), required=True)
    launch.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-inputs":
        prepare_inputs()
    elif args.command == "bind":
        bind(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
