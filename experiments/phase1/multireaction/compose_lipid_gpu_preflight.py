"""Bounded CUDA forward/backward diagnostic; never updates real-corpus model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import write_json


def requested_gpu(request: dict) -> str:
    """Select an explicit qualified allocation without fallback or retry expansion."""
    if (
        request.get("gpu") not in {"L4", "L40S"}
        or request.get("timeout_seconds") != 600
        or request.get("retries") != 0
        or request.get("optimizer_steps") != 0
    ):
        raise ValueError("Preflight requires one explicit L4/L40S, 600-second cap and no retries")
    return request["gpu"]


def check_payload(payload: dict) -> None:
    if (
        payload.get("schema_version") != "forge.compose_lipid_gpu_preflight_input.v1"
        or payload.get("optimizer_steps") != 0
        or payload.get("training_admitted") is not False
        or payload.get("precision") != "float32"
        or payload.get("deterministic") is not True
        or not 1 <= payload.get("batch_size", 0) <= 32
        or not 1 <= payload.get("repeats", 0) <= 3
        or payload.get("model", {}).get("maximum_heavy_atoms", 0) < payload.get("maximum_atoms", 0)
    ):
        raise ValueError(
            "Preflight requires bounded deterministic float32 and zero optimizer steps"
        )


def run_preflight(payload: dict, *, device: str = "cuda") -> dict:
    """Use the training noising/loss path, including every prepared source binding.

    Uniform noise is diagnostic only. It is not a fitted training marginal. Each
    repeat resets the RNG and checks exact loss/gradient reproducibility on one device.
    """
    check_payload(payload)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from forge.corpus.qualified_program_cache import _vocabulary
    from forge.model.compose_lipid_training import compose_lipid_forward_loss
    from forge.model.synthesis_program_training import build_synthesis_program_flow, move_tensors

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (
        target.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested preflight device is unavailable")
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(payload["seed"])
    vocabulary = _vocabulary(payload["vocabulary"])
    clean = {
        name: torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]))
        for name, value in payload["batch"].items()
    }
    if clean["nodes"].shape[0] != len(payload["target_ids"]):
        raise ValueError("Preflight graph identities and tensors differ")
    if set(clean["program_states"].tolist()) != set(range(1, len(vocabulary.program_states))):
        raise ValueError("Preflight omitted a prepared source binding")
    if int(clean["node_mask"].sum(1).max()) != payload["maximum_atoms"]:
        raise ValueError("Preflight omitted the largest prepared graph")
    if int(clean["closure_mask"].sum(1).max()) != payload["maximum_closures"]:
        raise ValueError("Preflight omitted maximum closure support")
    model = (
        build_synthesis_program_flow(
            vocabulary=vocabulary,
            node_classes=payload["node_classes"],
            model_config=payload["model"],
            device=target,
        )
        .float()
        .train()
    )
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    node = torch.full((payload["node_classes"],), 1 / payload["node_classes"], device=target)
    bonds = payload["model"]["bond_classes"]
    bond = torch.full((bonds,), 1 / bonds, device=target)
    maximum = int(clean["node_mask"].sum(1).argmax())
    groups = [
        list(range(i, min(i + payload["batch_size"], len(payload["target_ids"]))))
        for i in range(0, len(payload["target_ids"]), payload["batch_size"])
    ]
    groups.append([maximum] * payload["batch_size"])
    reports = []
    for group_number, indices in enumerate(groups):
        batch = move_tensors({name: value[indices] for name, value in clean.items()}, target)
        reference = None
        timings = []
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(payload["repeats"]):
            torch.manual_seed(payload["seed"])
            generator = torch.Generator(device=target).manual_seed(payload["seed"])
            model.zero_grad(set_to_none=True)
            if target.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _ = compose_lipid_forward_loss(
                model,
                batch,
                architecture=payload["model"]["architecture"],
                node_marginal=node,
                bond_marginal=bond,
                times=torch.full((len(indices),), 0.5, device=target),
                generator=generator,
                semantic_weights=payload["semantic_weights"],
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if target.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            digest = hashlib.sha256()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    digest.update(parameter.grad.detach().cpu().numpy().tobytes())
            observed = (float(loss.detach()), float(norm), digest.hexdigest())
            if reference is not None and observed != reference:
                raise ValueError("Repeated CUDA/CPU diagnostic changed loss or gradients")
            reference = observed
        reports.append(
            {
                "batch": group_number,
                "records": len(indices),
                "maximum_size_stress": group_number == len(groups) - 1,
                "loss": reference[0],
                "gradient_norm": reference[1],
                "gradient_sha256": reference[2],
                "seconds": timings,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if target.type == "cuda" else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if target.type == "cuda" else None
                ),
            }
        )
    if any(
        not torch.equal(value.detach().cpu(), before[name])
        for name, value in model.state_dict().items()
    ):
        raise ValueError("Diagnostic modified model parameters or buffers")
    return {
        "schema_version": "forge.compose_lipid_gpu_preflight.v1",
        "passed": True,
        "seed": payload["seed"],
        "device": str(target),
        "gpu": torch.cuda.get_device_name() if target.type == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "precision": "float32",
        "deterministic": True,
        "model": payload["model"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_bindings": len(vocabulary.program_states) - 1,
        "maximum_atoms": payload["maximum_atoms"],
        "maximum_closures": payload["maximum_closures"],
        "batches": reports,
        "model_state_unchanged": True,
        "optimizer_steps": 0,
        "training_admitted": False,
        "noise_policy": "diagnostic_uniform_unfitted",
        "scope": "Forward/backward and memory diagnostic, not optimizer convergence or training admission",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cpu", "submit", "collect"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    directory = args.directory.resolve()
    directory.relative_to(repo)
    request = json.loads((directory / "request.json").read_text())
    gpu_type = requested_gpu(request)
    if args.action == "collect":
        import modal

        receipt = json.loads((directory / "submitted.json").read_text())
        result = modal.FunctionCall.from_id(receipt["function_call_id"]).get(timeout=0)
        if result["request_id"] != request["request_id"]:
            raise ValueError("Remote preflight request identity differs")
        write_json(directory / "result.json", result)
        print(json.dumps(result, indent=2))
        return
    payload_path = resolve_pin(request["payload"], repo, label="preflight payload")
    payload = json.loads(payload_path.read_text())
    check_payload(payload)
    for name, digest in request["source_files"].items():
        resolve_pin({"path": name, "sha256": digest}, repo, label="preflight source")
    if args.action == "cpu":
        result = run_preflight(payload, device="cpu")
        write_json(directory / "cpu-result.json", result)
        return
    attempt = directory / "submission-attempt.json"
    with attempt.open("x") as handle:
        json.dump({"request_id": request["request_id"], "automatic_retry": False}, handle)
    import modal

    from experiments._runtime.modal_app import image

    app = modal.App("forge-compose-lipid-gpu-preflight")

    @app.function(
        image=image, gpu=gpu_type, cpu=4, memory=16384, timeout=600, retries=0, serialized=True
    )
    def execute(payload: dict, request: dict) -> dict:
        remote = Path("/opt/forge-project")
        for name, digest in request["source_files"].items():
            if str(sha256_file(remote / name)) != digest:
                raise ValueError("Remote preflight source mismatch: " + name)
        result = run_preflight(payload)
        return dict(result, request_id=request["request_id"], inputs=request)

    try:
        with modal.enable_output(), app.run(detach=True):
            call = execute.spawn(payload, request)
            receipt = {
                "request_id": request["request_id"],
                "function_call_id": call.object_id,
                "application_id": app.app_id,
                "payload": request["payload"],
                "source_sha256": request["source_sha256"],
                "gpu": gpu_type,
                "timeout_seconds": 600,
                "retries": 0,
                "detached": True,
            }
            write_json(directory / "submitted.json", receipt)
            print(json.dumps(receipt), flush=True)
    except BaseException as error:
        write_json(
            directory / "submission-failure.json", {"error": str(error), "automatic_retry": False}
        )
        raise


if __name__ == "__main__":
    main()
