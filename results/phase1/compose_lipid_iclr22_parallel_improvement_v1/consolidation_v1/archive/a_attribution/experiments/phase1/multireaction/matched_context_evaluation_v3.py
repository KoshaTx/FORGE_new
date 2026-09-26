"""Freeze and qualify matched-context null sampling; no count-only policy is reused."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path
from unittest.mock import patch

import torch
from rdkit import rdBase

import forge
from experiments._runtime.source import source_fingerprint
from forge.core.hashing import resolve_pin
from forge.corpus.qualified_program_cache import _vocabulary
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.synthesis_program_training import build_synthesis_program_flow
from results.phase1.compose_lipid_quality_decode_v2 import run as frozen_construction


def local_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


construction = local_module("matched_context_construction")
adapter = local_module("matched_context_null")
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
BASE = ROOT / "results/phase1"
OUT = BASE / "compose_lipid_iclr22_parallel_improvement_v1/a_attribution/matched_context_null_v3"
NULL = BASE / "compose_lipid_iclr22_table_completion_v1/null_runtime"
CONFIRM = BASE / "compose_lipid_quality_confirmation_v1"
ASSEMBLY = BASE / "compose_lipid_iclr22_table_completion_v1/assembly"
GATE = BASE / "compose_lipid_iclr22_parallel_improvement_v1/a_attribution/null_fullbatch_v2"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path):
    return {"path": str(Path(path).absolute()), "sha256": digest(path)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def authenticate(value):
    if digest(value["path"]) != value["sha256"]:
        raise ValueError("Input digest changed: " + value["path"])
    return Path(value["path"])


def ensure_source():
    if not Path(forge.__file__).resolve().is_relative_to((NULL / "source").resolve()):
        raise ValueError("Use immutable qualified null source first in PYTHONPATH")
    if (
        source_fingerprint(NULL / "source")
        != "1c77ed6e9c7b9251c9423666aadafa9967083db8ad540d369186577ec9980ae1"
    ):
        raise ValueError("Qualified model source changed")


def model_for(payload, state, mode):
    torch.manual_seed(2026092401)
    config = {**payload["config"]["model"], "semantic_conditioning": mode}
    model = build_synthesis_program_flow(
        vocabulary=_vocabulary(payload["vocabulary"]),
        node_classes=len(payload["atoms"]),
        model_config=config,
        device="cpu",
    ).float()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def freeze():
    ensure_source()
    OUT.mkdir(exist_ok=False)
    prior = read(ASSEMBLY / "readout_recovery_v4/protocol.json")
    inputs = {
        "payload": pin(ROOT / prior["inputs"]["payload"]["path"]),
        "original_checkpoint": pin(ROOT / prior["inputs"]["checkpoint"]["path"]),
        "saved_replay_protocol": pin(ASSEMBLY / "readout_recovery_v4/protocol.json"),
        "constructor_protocol": pin(CONFIRM / "construction-protocol.json"),
        "fixture_null_checkpoint": pin(
            GATE / "collected/qualification/full_batch/final/checkpoint.pt"
        ),
        "fixture_null_config": pin(NULL / "full_batch_readiness_v1/configuration.json"),
        "fixture_capacity_admission": pin(GATE / "capacity_admission.json"),
        "conditioned_selector_qualification": pin(OUT.parent / "ordering_v2/qualification.json"),
        "conditioned_current_gate_recount": pin(
            OUT.parent / "ordering_v3/independent_recount_v1.json"
        ),
        "preserved_tensor_conversion_failure": pin(
            OUT.parent / "matched_context_null_v2/failure.json"
        ),
        "preserved_three_thread_failure": pin(OUT.parent / "matched_context_null_v1/failure.json"),
        "inference_control_scope": pin(
            OUT.parent / "null_fullfit_readiness_v1/control_readiness.json"
        ),
        "producer": pin(Path(__file__)),
        "adapter": pin(Path(adapter.__file__)),
        "constructor_adapter": pin(Path(construction.__file__)),
    }
    payload = torch.load(authenticate(inputs["payload"]), map_location="cpu", weights_only=False)
    fixtures = []
    families = sorted({layout.family for layout in payload["layouts"]})
    selected_families = [families[0], "aema_aza_thiol_addition", "maleate_addition"]
    for family in dict.fromkeys(selected_families):
        offset = next(i for i, layout in enumerate(payload["layouts"]) if layout.family == family)
        row = next(r for r in prior["schedule"] if r["draw"] == 0 and r["offset"] == offset)
        fixtures.append(row)
    protocol = {
        "schema": "forge.matched_structural_context_null_inference.v1",
        "inputs": inputs,
        "immutable_null_source": str(NULL / "source"),
        "source_sha256": source_fingerprint(NULL / "source"),
        "fixture_schedule": fixtures,
        "schedule": prior["schedule"],
        "families": prior["families"],
        "requests": 1408,
        "per_family": 64,
        "draws": 5,
        "trajectories": 7040,
        "steps": 64,
        "batch_size": 8,
        "threads": 4,
        "precision": "float32",
        "deterministic": True,
        "fixture_CPU_cap_seconds": 1800,
        "fixture_device": "cpu",
        "fixture_only_weights": "22-update capacity checkpoint; no trained-null quality or convergence metric admitted",
        "full_execution_requires": "Fresh root admission binding final2794 checkpoint and this separate inference estimand; no GPU calls in preflight",
        "population": "Original TRAIN-derived1408layouts, all failures retained; no TEST or heldout claim",
        "information_boundary": "Identical supplied layout/core/masks/role-conditioned source-noise tensors/drawseeds outside forward; trained null native model plus explicit adapter zeros allnine semantics eachforward",
        "not_count_only": "No CountOnlySampler, count-only evaluation protocol or null-policy admission is passed to this adapter",
        "causal_scope": "Joint null training mask/prior/conditioning contrast; neither isolated semantic embedding ablation nor modelsvsrules",
        "primary": [
            "draw0 true endpointN1408",
            "draw0 terminal argmaxN1408",
            "full matched decoder/selectionN1408",
        ],
        "secondary": "all5 trajectory yieldN7040, no bestexact selection",
        "constructor_policy": read(CONFIRM / "construction-protocol.json")["limits"],
        "selector": "Unchanged original arm-own firstexact/support/design/context selector objectives and budgets; current authenticated identity-bound gate evidence only",
        "automatic_retry": False,
        "quality_promotion": False,
    }
    write(OUT / "protocol.json", protocol)
    print(json.dumps({"protocol": pin(OUT / "protocol.json"), "fixture_batches": len(fixtures)}))


def inputs():
    ensure_source()
    p = read(OUT / "protocol.json")
    for value in p["inputs"].values():
        authenticate(value)
    payload = torch.load(
        Path(p["inputs"]["payload"]["path"]), map_location="cpu", weights_only=False
    )
    return p, payload


def fixture():
    start = time.process_time()
    p, payload = inputs()
    out = OUT / "fixture_v1"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(p["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old = torch.load(
        Path(p["inputs"]["original_checkpoint"]["path"]), map_location="cpu", weights_only=False
    )
    adapter.equal_tensors(old["model"], payload["models"]["current"])
    trained = torch.load(
        Path(p["inputs"]["fixture_null_checkpoint"]["path"]), map_location="cpu", weights_only=False
    )
    if (
        trained["completed_steps"] != 22
        or trained["identity"]["config_sha256"] != p["inputs"]["fixture_null_config"]["sha256"]
    ):
        raise ValueError("Capacity checkpoint binding changed")
    base = model_for(payload, old["model"], "conditioned")
    native = model_for(payload, trained["model"], "semantic_null")
    null = adapter.MatchedContextNull(native)
    noise = [torch.tensor(payload["noise"][k], dtype=torch.float32) for k in ("node", "bond")]
    hashes = frozen_construction.load(
        frozen_construction.MIRROR / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py",
        "matched_context_hashes",
    ).VerifiedHashes()
    evidence = []
    with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
        cproto = read(Path(p["inputs"]["constructor_protocol"]["path"]))
        context = construction.setup(ROOT, cproto, payload)
        for entry in p["fixture_schedule"]:
            tick = time.process_time()
            offset = entry["offset"]
            seed = entry["seed"]
            batch = collate_generated_layouts(
                payload["layouts"][offset : offset + 8], maximum_closures=12
            )
            baseline = adapter.capture(base, base, batch, *noise, seed=seed)
            saved = torch.load(
                resolve_pin(entry["predictions"], ROOT, label="original predictions"),
                weights_only=False,
                map_location="cpu",
            )
            if offset == 0:
                torch.set_num_threads(3)
                three = adapter.capture(base, base, batch, *noise, seed=seed)
                torch.set_num_threads(4)
                deviations = {
                    k: {
                        "exact": torch.equal(three["predictions"][k], saved[k]),
                        "max_absolute_error": float(
                            (three["predictions"][k].detach() - saved[k].detach()).abs().max()
                        ),
                        "changed_elements": int((three["predictions"][k] != saved[k]).sum()),
                    }
                    for k in saved
                }
                write(
                    out / "three_thread_deviations.json",
                    {
                        "protocol": pin(OUT / "protocol.json"),
                        "head_deviations": deviations,
                        "reproduced_diagnostic_not_original_failed_tensor": True,
                    },
                )
            adapter.equal_tensors(baseline["predictions"], saved)
            candidate = adapter.capture(null, native, batch, *noise, seed=seed, semantic_null=True)
            repeated = adapter.capture(null, native, batch, *noise, seed=seed, semantic_null=True)
            for key in ("raw", "terminal", "predictions"):
                adapter.equal_tensors(candidate[key], repeated[key])
            if (
                baseline["trace"]["context"] != candidate["trace"]["context"]
                or baseline["trace"]["initial_state"] != candidate["trace"]["initial_state"]
            ):
                raise ValueError("Common context or seeded initial noise mismatch")
            detached_predictions = {k: v.detach().cpu() for k, v in baseline["predictions"].items()}
            rows = construction.construct(
                context, payload, baseline["predictions"], draw=0, offset=offset
            )
            saved_path = CONFIRM / "constructed" / f"base-0-{offset:04d}.json.gz"
            expected = json.loads(gzip.decompress(saved_path.read_bytes()))
            if rows != expected:
                raise ValueError("Frozen base constructor fixture differs")
            variants = [{"mode": "base", "source": pin(saved_path), "rows": len(rows)}]
            if offset in context["domain_policies"]:
                domain = construction.construct(
                    context,
                    payload,
                    detached_predictions,
                    draw=0,
                    offset=offset,
                    mode="domain",
                    base_rows=rows,
                )
                domain_path = CONFIRM / "constructed" / f"domain-0-{offset:04d}.json.gz"
                if domain != json.loads(gzip.decompress(domain_path.read_bytes())):
                    raise ValueError("Frozen domain constructor fixture differs")
                variants.append({"mode": "domain", "source": pin(domain_path), "rows": len(domain)})
            state_path = out / f"fixture-{offset:04d}.pt"
            torch.save(
                {"conditioned": baseline, "capacity_null": candidate, "outcome_scored": False},
                state_path,
            )
            evidence.append(
                {
                    "offset": offset,
                    "seed": seed,
                    "family": payload["layouts"][offset].family,
                    "conditioned_14heads_bit_exact": len(saved) == 14,
                    "null_deterministic": True,
                    "context_and_initial_noise_identical": True,
                    "trace": candidate["trace"],
                    "constructors": variants,
                    "artifact": pin(state_path),
                    "CPU_seconds": time.process_time() - tick,
                }
            )
            print(
                json.dumps(
                    {"offset": offset, "passed": True, "CPU_seconds": time.process_time() - tick}
                ),
                flush=True,
            )
        hashes.validate()
    write(
        out / "result.json",
        {
            "passed": True,
            "protocol": pin(OUT / "protocol.json"),
            "evidence": evidence,
            "CPU_seconds": time.process_time() - start,
            "new_optimizer_GPU_TEST_network_calls": 0,
            "fixture_quality_scored": False,
            "full1408_evaluation": False,
            "selector_qualification_reused": p["inputs"]["conditioned_selector_qualification"],
        },
    )
    print(
        json.dumps({"result": pin(out / "result.json"), "CPU_seconds": time.process_time() - start})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fixture"))
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    {"freeze": freeze, "fixture": fixture}[args.stage]()


if __name__ == "__main__":
    main()
