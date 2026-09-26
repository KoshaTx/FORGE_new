"""Paired, no-repair generation evaluation of two frozen all-family checkpoints."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments._runtime.source import source_fingerprint
from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import write_json
from forge.corpus.compose_lipid_source_view import pin
from forge.corpus.qualified_program_cache import _vocabulary
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import ComposeLipidLayoutPrior, collate_generated_layouts
from forge.model.compose_lipid_sampling import sample
from forge.model.synthesis_program_training import build_synthesis_program_flow, move_tensors

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FIT = HERE.with_name("compose_lipid_training_eight_fp32_v2")


def setup(payload, arm, device, seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = build_synthesis_program_flow(
        vocabulary=_vocabulary(payload["vocabulary"]),
        node_classes=len(payload["atoms"]),
        model_config=payload["config"]["model"],
        device=device,
    ).float()
    model.load_state_dict(payload["models"][arm], strict=True)
    model.eval()
    return model, *(
        torch.tensor(payload["noise"][k], device=device, dtype=torch.float32)
        for k in ("node", "bond")
    )


def generate(model, layouts, atoms, node, bond, *, steps, seed, device):
    batch = move_tensors(
        collate_generated_layouts(layouts, maximum_closures=model.maximum_closures), device
    )
    with torch.inference_mode():
        raw, predictions = sample(
            model, batch, node, bond, steps=steps, seed=seed, return_predictions=True
        )
        strict, reasons = constrained_readout(predictions, batch, layouts, atoms)
    return dict(raw=(raw, (None,) * len(layouts)), strict=(strict, reasons))


def prepare():
    if (HERE / "protocol.json").exists():
        raise FileExistsError("Evaluation protocol is immutable")
    current_config = json.loads((FIT / "training-config.json").read_text())
    completion = json.loads((FIT / "completion.json").read_text())
    assert str(sha256_file(FIT / "checkpoint.pt")) == completion["checkpoint"]["sha256"]
    old_dir = HERE.with_name("compose_lipid_generation_v1")
    old_protocol = json.loads((old_dir / "protocol.json").read_text())
    old = torch.load(
        resolve_pin(old_protocol["inputs"]["payload"], ROOT, label="qualified generation payload"),
        map_location="cpu",
        weights_only=False,
    )
    prior_path = resolve_pin(old_protocol["inputs"]["prior"], ROOT, label="layout prior")
    prior = ComposeLipidLayoutPrior(json.loads(prior_path.read_text()))
    assert len(prior._families) == 22
    mapped = json.loads(
        resolve_pin(
            current_config["inputs"]["mapped_cache"], ROOT, label="mapped cache"
        ).read_text()
    )
    assert mapped["program_vocabulary"] == old["vocabulary"]
    noise = json.loads(
        resolve_pin(current_config["inputs"]["noise_marginals"], ROOT, label="noise").read_text()
    )
    assert all(np.array_equal(old["noise"][k], noise[k]) for k in ("node", "bond"))
    pilot_dir = HERE.with_name("compose_lipid_iteration_pilot_v1")
    pilot_receipt = json.loads((pilot_dir / "completion-summary.json").read_text())
    pilot_path = resolve_pin(pilot_receipt["checkpoint"], ROOT, label="pilot checkpoint")
    pilot_config = json.loads((pilot_dir / "training-config.json").read_text())
    for key in ("model", "semantic_weights", "objective"):
        assert current_config[key] == pilot_config[key] == old["config"][key]
    for key in ("population", "verification", "measure", "mapped_cache", "noise_marginals"):
        assert current_config["inputs"][key] == pilot_config["inputs"][key]
    seed = 2026092402
    layouts = [
        prior.sample(
            f, rng=np.random.default_rng(seed + 100000 * j + i), identity=f"evaluation-{j}-{i}"
        )
        for j, f in enumerate(sorted(prior._families))
        for i in range(64)
    ]
    models = {}
    for name, path in [("current", FIT / "checkpoint.pt"), ("pilot", pilot_path)]:
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected = current_config if name == "current" else pilot_config
        assert state["identity"]["inputs"] == expected["inputs"]
        models[name] = state["model"]
    payload = dict(
        models=models,
        config=current_config,
        vocabulary=old["vocabulary"],
        atoms=old["atoms"],
        noise={k: noise[k] for k in ("node", "bond")},
        layouts=layouts,
    )
    torch.save(payload, HERE / "input.pt")
    inputs = dict(
        current_checkpoint=pin(ROOT, FIT / "checkpoint.pt"),
        pilot_checkpoint=pin(ROOT, pilot_path),
        current_config=pin(ROOT, FIT / "training-config.json"),
        pilot_config=pin(ROOT, pilot_dir / "training-config.json"),
        payload=pin(ROOT, HERE / "input.pt"),
        prior=pin(ROOT, prior_path),
        mapped=current_config["inputs"]["mapped_cache"],
        checker=pin(ROOT, HERE.with_name("compose_lipid_posttraining_v1") / "adjudicate.py"),
        measure=current_config["inputs"]["measure"],
        population=current_config["inputs"]["population"],
        prior_payload=old_protocol["inputs"]["payload"],
    )
    protocol = dict(
        schema_version="forge.compose_lipid_checkpoint_evaluation.v1",
        seed=seed,
        samples_per_family=64,
        families=sorted(prior._families),
        models=["current", "pilot"],
        decoders=["raw", "strict"],
        sample_steps=64,
        batch_size=8,
        device="cuda",
        precision="float32",
        deterministic=True,
        gpu="H100",
        timeout_seconds=1200,
        automatic_retries=0,
        source_sha256=source_fingerprint(ROOT),
        inputs=inputs,
        implementation=pin(ROOT, Path(__file__)),
        score_implementation=pin(ROOT, HERE / "score.py"),
        limits=[
            "TRAIN-derived independent layouts; no heldout-performance claim or sealed TEST access.",
            "One checkpoint per model; uncertainty across requests is not training-seed uncertainty.",
            "Pilot has 40,128 presentations per family versus 402,336; comparison does not isolate batch size.",
            "All failed attempts retained. No optional decoder, repair, completion, retry, selection or optimizer update.",
            "Diagnostic quality assessment; no post-hoc acceptance threshold or production promotion.",
        ],
    )
    write_json(HERE / "protocol.json", protocol)
    model, node, bond = setup(payload, "current", "cpu", seed)
    # Two disposable flow steps test the exact interface, not quality or GPU speed.
    sampled = generate(
        model, [layouts[0], layouts[-1]], old["atoms"], node, bond, steps=2, seed=seed, device="cpu"
    )
    assert set(sampled) == {"raw", "strict"} and all(len(v[1]) == 2 for v in sampled.values())
    write_json(
        HERE / "smoke.json",
        dict(
            passed=True,
            quality_claim=False,
            inputs=inputs,
            protocol=pin(ROOT, HERE / "protocol.json"),
            steps=2,
        ),
    )
    print(json.dumps(dict(prepared=True, requests=len(layouts), models=2, decoders=2)))


def execute(protocol, request_id):
    from experiments._runtime.modal_app import VOLUME_ROOT, experiment_volume

    root = Path("/opt/forge-project")
    assert source_fingerprint(root) == protocol["source_sha256"]
    assert str(sha256_file(root / "evaluation_driver.py")) == protocol["implementation"]["sha256"]
    assert str(sha256_file(Path("/opt/evaluation.pt"))) == protocol["inputs"]["payload"]["sha256"]
    payload = torch.load("/opt/evaluation.pt", map_location="cpu", weights_only=False)
    volume = VOLUME_ROOT.resolve()
    output = volume / "evaluations" / request_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    shards = []
    for arm in protocol["models"]:
        model, node, bond = setup(payload, arm, "cuda", protocol["seed"])
        pending = []
        for offset in range(0, len(payload["layouts"]), protocol["batch_size"]):
            layouts = payload["layouts"][offset : offset + protocol["batch_size"]]
            outputs = generate(
                model,
                layouts,
                payload["atoms"],
                node,
                bond,
                steps=protocol["sample_steps"],
                seed=protocol["seed"] + offset,
                device="cuda",
            )
            rows = []
            for decoder, (states, reasons) in outputs.items():
                for i, layout in enumerate(layouts):
                    rows.append(
                        dict(
                            index=offset + i,
                            model=arm,
                            arm=decoder,
                            family=layout.family,
                            program=layout.record.program_id,
                            reason=reasons[i],
                            state={
                                k: v[
                                    i,
                                    : (
                                        layout.record.graph.closure_count
                                        if k.startswith("closure")
                                        else layout.record.node_count
                                    ),
                                ]
                                .cpu()
                                .tolist()
                                for k, v in states.items()
                            },
                        )
                    )
            pending.extend(rows)
            if (offset + len(layouts)) % protocol["samples_per_family"]:
                continue
            path = output / f"{arm}-{offset + len(layouts):05d}.json"
            write_json(path, pending)
            pending = []
            shards.append(pin(volume, path))
            write_json(
                output / "progress.json",
                dict(
                    completed=False,
                    model=arm,
                    requests_completed=offset + len(layouts),
                    total_requests=len(payload["layouts"]),
                    elapsed_seconds=time.monotonic() - start,
                    shards=shards,
                ),
            )
            experiment_volume.commit()
        del model, node, bond
        torch.cuda.empty_cache()
    result = dict(
        completed=True,
        protocol=protocol,
        shards=shards,
        seconds=time.monotonic() - start,
        gpu_name=torch.cuda.get_device_name(),
        optimizer_updates=0,
    )
    write_json(output / "result.json", result)
    experiment_volume.commit()
    return dict(completed=True, result=pin(volume, output / "result.json"))


def submit():
    import modal

    from experiments._runtime.modal_app import VOLUME_ROOT, experiment_volume, image

    protocol = json.loads((HERE / "protocol.json").read_text())
    assert source_fingerprint(ROOT) == protocol["source_sha256"]
    for name, value in protocol["inputs"].items():
        resolve_pin(value, ROOT, label=name)
    resolve_pin(protocol["implementation"], ROOT, label="driver")
    resolve_pin(protocol["score_implementation"], ROOT, label="scorer")
    assert json.loads((HERE / "smoke.json").read_text())["passed"]
    request_id = "compose-evaluation-" + pin(ROOT, HERE / "protocol.json")["sha256"][:16]
    with (HERE / "attempt.json").open("x") as handle:
        json.dump(dict(request_id=request_id, protocol=protocol), handle)
    remote = (
        image.env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        .add_local_file(HERE / "input.pt", remote_path="/opt/evaluation.pt", copy=True)
        .add_local_file(
            Path(__file__), remote_path="/opt/forge-project/evaluation_driver.py", copy=True
        )
    )
    app = modal.App("forge-compose-checkpoint-quality")

    @app.function(
        image=remote,
        gpu="H100",
        cpu=4,
        memory=16384,
        timeout=1200,
        retries=0,
        serialized=True,
        volumes={str(VOLUME_ROOT): experiment_volume},
    )
    def run(protocol, request_id):
        from evaluation_driver import execute

        return execute(protocol, request_id)

    with modal.enable_output(), app.run(detach=True):
        write_json(
            HERE / "application.json",
            dict(application_id=app.app_id, request_id=request_id, protocol=protocol),
        )
        call = run.spawn(protocol, request_id)
        write_json(
            HERE / "submitted.json",
            dict(
                application_id=app.app_id,
                function_call_id=call.object_id,
                request_id=request_id,
                detached=True,
                automatic_retry=False,
                protocol=protocol,
            ),
        )
        print(call.object_id)


def collect():
    import modal

    from experiments._runtime.modal_app import experiment_volume

    receipt = json.loads((HERE / "submitted.json").read_text())
    call = modal.FunctionCall.from_id(receipt["function_call_id"])
    print("Call:", [n.status.name for n in call.get_call_graph()])
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        try:
            data = b"".join(
                experiment_volume.read_file(
                    "evaluations/" + receipt["request_id"] + "/progress.json"
                )
            )
            progress = json.loads(data)
            write_json(HERE / "observed-progress.json", progress)
            print(json.dumps({k: v for k, v in progress.items() if k != "shards"}))
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
        return
    except Exception as error:
        write_json(
            HERE / "failure.json",
            dict(
                error=repr(error),
                automatic_retry=False,
                submitted=pin(ROOT, HERE / "submitted.json"),
            ),
        )
        raise

    def download(value, path):
        if path.exists():
            assert str(sha256_file(path)) == value["sha256"]
            return
        data = b"".join(experiment_volume.read_file(value["path"]))
        assert hashlib.sha256(data).hexdigest() == value["sha256"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    download(result["result"], HERE / "remote-result.json")
    remote = json.loads((HERE / "remote-result.json").read_text())
    assert remote["completed"] and remote["protocol"] == receipt["protocol"]
    for value in remote["shards"]:
        download(value, HERE / "shards" / Path(value["path"]).name)
    write_json(HERE / "completion.json", result)
    print("Collected and verified all generation shards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "collect"))
    globals()[parser.parse_args().action]()
