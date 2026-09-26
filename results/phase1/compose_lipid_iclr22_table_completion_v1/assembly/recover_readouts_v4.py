"""Recover terminal argmax and qualify true-endpoint replay on the unchanged 1408 requests."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
from rdkit import rdBase

from forge.core.hashing import resolve_pin
from forge.model.compose_lipid_layout import collate_generated_layouts
from forge.model.compose_lipid_sampling import sample
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from forge.model.reaction_program_flow import decode_synthesis_program_argmax
from forge.model.synthesis_program_sampling import _fixed_state_exact_tensor
from results.phase1.compose_lipid_component_decoder_v1.contracts import assess, load_all
from results.phase1.compose_lipid_eight_fp32_evaluation_v1.evaluate import setup
from results.phase1.compose_lipid_iclr22_research_v1.parallel_completion_v1.training_evaluation.forward_controls import (
    SEMANTIC_FIELDS,
    ForwardControl,
)
from results.phase1.compose_lipid_iclr22_table_completion_v1.assembly.extract_saved import (
    CONFIRM,
    ROOT,
    TRAIN,
    pin,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "readout_recovery_v4"
FIELDS = ("nodes", "parents", "parent_bonds", "closure_left", "closure_right", "closure_bonds")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def check_equal(left, right):
    if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
        raise ValueError("Replay does not reproduce every saved terminal prediction bit exactly")


def validate_predictions(predictions):
    if not set(FIELDS).issubset(predictions):
        raise ValueError("Incomplete terminal prediction fields")
    if any(
        v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in predictions.values()
    ):
        raise ValueError("Nonfinite or non-FP32 terminal prediction")


def capture(model, base, batch, node, bond, seed, mapping=None):
    captured, calls = [], []
    expected = batch["program_states"]
    if mapping is not None:
        expected = model.program_permutation[expected]

    def hook(_model, _args, kwargs):
        if not torch.equal(kwargs["program_states"], expected):
            raise ValueError("Program intervention does not match frozen inference mapping")
        for key in (*SEMANTIC_FIELDS[1:], "node_mask", "child_mask", "closure_mask"):
            if kwargs[key] is not batch[key]:
                raise ValueError(f"Structural context changed: {key}")
        calls.append(float(kwargs["t"][0]))
        if bool((kwargs["t"] == 1).all()):
            captured.append({k: kwargs[k].detach().clone() for k in FIELDS})

    handle = base.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            terminal, predictions = sample(
                model, batch, node, bond, steps=64, seed=seed, return_predictions=True
            )
    finally:
        handle.remove()
    if calls != [i / 64 for i in range(65)] or len(captured) != 1:
        raise ValueError("Read-only hook changed the original 65-forward schedule")
    validate_predictions(predictions)
    for state in (terminal, captured[0]):
        if not bool(_fixed_state_exact_tensor(state, batch)):
            raise ValueError("Fixed context changed")
    reconstructed = decode_synthesis_program_argmax(predictions, batch)
    check_equal(terminal, reconstructed)
    return captured[0], terminal, predictions


def score(states, layouts, atoms, contracts, draw, offset):
    rows = []
    for local, layout in enumerate(layouts):
        arms = {}
        for arm, state in states.items():
            size, cycles = layout.record.node_count, layout.record.graph.closure_count
            serial = {
                k: state[k][local, : cycles if k.startswith("closure") else size].tolist()
                for k in FIELDS
            }
            smiles, reason = None, None
            try:
                nodes, edges = state_graph(serial)
                smiles = graph_smiles(nodes, edges, atoms)
                if smiles is None:
                    reason = "invalid_or_disconnected_graph"
            except (ValueError, RuntimeError, KeyError) as error:
                reason = str(error)
            checked = (
                assess(contracts, layout, smiles)
                if smiles is not None
                else {"status": "invalid_or_refused", "exact": False, "reason": reason}
            )
            if type(checked.get("exact")) is not bool:
                raise ValueError("Invalid exact checker verdict")
            arms[arm] = {"state": serial, "smiles": smiles, "reason": reason, "check": checked}
        rows.append({"index": offset + local, "family": layout.family, "draw": draw, "arms": arms})
    return rows


def freeze():
    OUT.mkdir(exist_ok=False)
    original = read(ROOT / CONFIRM / "protocol.json")
    qualification = TRAIN / "cyclic_inference_qualification_v1"
    cyclic = read(ROOT / qualification / "result.json")
    if not cyclic["passed"]:
        raise ValueError("Cyclic implementation qualification did not pass")
    inputs = {
        **original["inputs"],
        "original_protocol": pin(CONFIRM / "protocol.json"),
        "original_generation": pin(CONFIRM / "generation.json"),
        "original_generator": pin(CONFIRM / "generate.py"),
        "cyclic_protocol": pin(qualification / "protocol.json"),
        "cyclic_qualification": pin(qualification / "result.json"),
    }
    payload = torch.load(
        resolve_pin(inputs["payload"], ROOT, label="payload"),
        weights_only=False,
        map_location="cpu",
    )
    checkpoint = torch.load(
        resolve_pin(inputs["checkpoint"], ROOT, label="checkpoint"),
        weights_only=False,
        map_location="cpu",
    )
    check_equal(payload["models"]["current"], checkpoint["model"])
    if len(payload["layouts"]) != 1408 or {x.family for x in payload["layouts"]} != set(
        original["families"]
    ):
        raise ValueError("Original 1408 layout population changed")
    with rdBase.BlockLogs():
        _, contract_inputs, _ = load_all()
    schedule = []
    for draw, seed in enumerate(original["flow_seeds"]):
        for shard in original["shards"]:
            offset = shard["offset"]
            receipt_path = CONFIRM / "logits" / f"draw-{draw}-{offset:04d}.json"
            receipt = read(ROOT / receipt_path)
            if (
                receipt["draw"] != draw
                or receipt["offset"] != offset
                or receipt["seed"] != seed + offset
            ):
                raise ValueError("Stored prediction shard does not match original seed schedule")
            resolve_pin(receipt["predictions"], ROOT, label="saved prediction")
            schedule.append(
                {
                    "draw": draw,
                    "offset": offset,
                    "seed": seed + offset,
                    "receipt": pin(receipt_path),
                    "predictions": receipt["predictions"],
                }
            )
    assert len(schedule) == 880 and all(
        len([x for x in schedule if x["draw"] == i]) == 176 for i in range(5)
    )
    loaded_sources = {
        str(Path(module.__file__).resolve().relative_to(ROOT))
        for name, module in sorted(sys.modules.items())
        if (name.startswith("forge.") or name.startswith("results.phase1."))
        and (getattr(module, "__file__", None) or "").endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(ROOT)
    } | {str(Path(__file__).resolve().relative_to(ROOT))}
    historical = [
        {
            "original": x,
            "current": pin(Path(x["path"])),
            "same_bytes": pin(Path(x["path"]))["sha256"] == x["sha256"],
        }
        for x in original["implementation"]
    ]
    protocol = {
        "schema": "forge.same1408.readout_recovery.v1",
        "inputs": inputs,
        "implementation": [pin(Path(x)) for x in sorted(loaded_sources)],
        "contract_inputs": contract_inputs,
        "historical_source_audit": historical,
        "schedule": schedule,
        "families": original["families"],
        "requests": 1408,
        "draws": 5,
        "trajectories": 7040,
        "batch_size": 8,
        "steps": 64,
        "threads": 4,
        "device": "cpu",
        "precision": "float32",
        "deterministic": True,
        "program_mapping_indices": cyclic["program_mapping_indices"],
        "primary": "draw0 only; one output for each original request; N1408",
        "secondary": "all5 draws as trajectory yield; N7040; no best-of5 exact selection",
        "pool_selection": "Not executed; requires separately frozen valid original selector policy",
        "arms": ["terminal_argmax_from_saved", "conditioned_true_endpoint", "cyclic_true_endpoint"],
        "controls": "Cyclic changes only inference program ID; all other structural context unchanged. Correctly conditioned matching5draw readout is comparator, not final98.5% selected pipeline.",
        "replay_gate": "First shard and every subsequent conditioned shard must match all saved terminal predictions bit exactly; otherwise raw result is not admitted.",
        "terminal_gate": "First shard reconstructed argmax must equal unchanged sampler's returned terminal state bit exactly.",
        "state_capture": "Read-only pre-forward hook at actualt=1. sample() return is terminal argmax, never called raw.",
        "scope": "Existing one-fit TRAIN-derived layouts; no heldout/independent-seed evidence; no constructors, extra proposals, optimization or quality promotion.",
        "budget": {
            "smoke_CPU_seconds": 60,
            "terminal_CPU_seconds": 893,
            "terminal_wall_seconds": 893,
            "full_replay_requires_root_admission": True,
        },
        "automatic_retry": False,
        "new_fit_GPU_TEST_network_calls": 0,
        "cost_reference": read(ROOT / CONFIRM / "generation.json"),
    }
    write(OUT / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "protocol": pin((OUT / "protocol.json").relative_to(ROOT)),
                "schedule": len(schedule),
                "historical_changed_sources": [
                    x["original"]["path"] for x in historical if not x["same_bytes"]
                ],
            }
        )
    )


def load_protocol():
    protocol = read(OUT / "protocol.json")
    for value in protocol["inputs"].values():
        resolve_pin(value, ROOT, label="frozen recovery input")
    for value in protocol["implementation"]:
        resolve_pin(value, ROOT, label="frozen recovery implementation")
    for value in protocol["contract_inputs"].values():
        resolve_pin(value, ROOT, label="qualified source contract")
    payload = torch.load(
        resolve_pin(protocol["inputs"]["payload"], ROOT, label="payload"),
        weights_only=False,
        map_location="cpu",
    )
    return protocol, payload


def smoke():
    resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
    path = OUT / "smoke_v1"
    path.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    protocol, payload = load_protocol()
    original = protocol["schedule"][0]
    model, node, bond = setup(payload, "current", "cpu", original["seed"])
    batch = collate_generated_layouts(payload["layouts"][:8], maximum_closures=12)
    endpoint, terminal, predictions = capture(model, model, batch, node, bond, original["seed"])
    saved = torch.load(
        resolve_pin(original["predictions"], ROOT, label="saved predictions"),
        weights_only=False,
        map_location="cpu",
    )
    check_equal(predictions, saved)
    check_equal(terminal, decode_synthesis_program_argmax(saved, batch))
    mapping = {int(k): v for k, v in protocol["program_mapping_indices"].items()}
    wrapped = ForwardControl(model, "conditioned")
    wrapped_endpoint, wrapped_terminal, wrapped_predictions = capture(
        wrapped, model, batch, node, bond, original["seed"]
    )
    check_equal(predictions, wrapped_predictions)
    check_equal(endpoint, wrapped_endpoint)
    check_equal(terminal, wrapped_terminal)
    cyclic = ForwardControl(model, "cyclic", mapping)
    cyclic_endpoint, cyclic_terminal, cyclic_predictions = capture(
        cyclic, model, batch, node, bond, original["seed"], mapping
    )
    with rdBase.BlockLogs():
        contracts, _, _ = load_all()
        rows = score(
            {
                "conditioned_true_endpoint": endpoint,
                "terminal_argmax_from_saved": terminal,
                "cyclic_true_endpoint": cyclic_endpoint,
                "cyclic_terminal_argmax": cyclic_terminal,
            },
            payload["layouts"][:8],
            payload["atoms"],
            contracts,
            0,
            0,
        )
    torch.save(
        {
            "endpoint": endpoint,
            "terminal": terminal,
            "predictions": predictions,
            "cyclic_endpoint": cyclic_endpoint,
            "cyclic_terminal": cyclic_terminal,
            "cyclic_predictions": cyclic_predictions,
        },
        path / "states.pt",
    )
    write(
        path / "result.json",
        {
            "passed": True,
            "protocol": pin((OUT / "protocol.json").relative_to(ROOT)),
            "original_prediction_equality": True,
            "terminal_reconstruction_equality": True,
            "conditioned_wrapper_equality": True,
            "cyclic_context_unchanged_all65calls": True,
            "draw": 0,
            "offset": 0,
            "rows": rows,
            "states": pin((path / "states.pt").relative_to(ROOT)),
            "CPU_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - start,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "CPU_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - start,
            }
        )
    )


def summarize(rows, families):
    result = {}
    for arm in rows[0]["arms"]:
        result[arm] = {}
        for label, population in (("draw0", [x for x in rows if x["draw"] == 0]), ("all5", rows)):
            result[arm][label] = {
                family: {
                    "trajectories": len(subset),
                    "valid_connected": sum(x["arms"][arm]["smiles"] is not None for x in subset),
                    "exact_L1": sum(x["arms"][arm]["check"]["exact"] for x in subset),
                }
                for family in ["ALL", *families]
                if (subset := [x for x in population if family == "ALL" or x["family"] == family])
            }
    return result


def run(mode):
    start, cpu = time.monotonic(), time.process_time()
    protocol, payload = load_protocol()
    qualified = read(resolve_pin(protocol["qualification"], ROOT, label="qualified original smoke"))
    if not qualified["passed"] or qualified["protocol"] != protocol["qualified_smoke_protocol"]:
        raise ValueError("Missing exact first-shard admission")
    cap = protocol["budget"]["terminal_CPU_seconds"]
    wall_cap = protocol["budget"]["terminal_wall_seconds"]
    if mode != "terminal":
        admission = read(OUT / f"{mode}_root_admission.json")
        if not admission["approved"] or admission["protocol"] != pin(
            (OUT / "protocol.json").relative_to(ROOT)
        ):
            raise ValueError("Full replay requires root admission bound to exact protocol")
        cap = admission["CPU_seconds_cap"]
        wall_cap = admission["wall_seconds_cap"]
        if admission["runner"] != pin(Path(__file__).relative_to(ROOT)):
            raise ValueError("Admission does not bind executable runner")
    resource.setrlimit(resource.RLIMIT_CPU, (int(cap) - 1, int(cap)))
    path = OUT / f"{mode}_v1"
    path.mkdir(exist_ok=False)
    write(
        path / "started.json",
        {
            "pid": os.getpid(),
            "protocol": pin((OUT / "protocol.json").relative_to(ROOT)),
            "mode": mode,
            "CPU_seconds_cap": cap,
            "wall_seconds_cap": wall_cap,
        },
    )
    (path / "shards").mkdir()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        with rdBase.BlockLogs():
            contracts, _, _ = load_all()
        model = None
        if mode != "terminal":
            base, node, bond = setup(payload, "current", "cpu", protocol["schedule"][0]["seed"])
            mapping = (
                {int(k): v for k, v in protocol["program_mapping_indices"].items()}
                if mode == "cyclic"
                else None
            )
            model = ForwardControl(base, "cyclic", mapping) if mapping else base
        rows, receipts = [], []
        for shard in protocol["schedule"]:
            if time.monotonic() - start >= wall_cap:
                raise TimeoutError("Frozen wall budget exceeded; no partial metrics admitted")
            offset, draw = shard["offset"], shard["draw"]
            layouts = payload["layouts"][offset : offset + 8]
            batch = collate_generated_layouts(layouts, maximum_closures=12)
            saved = torch.load(
                resolve_pin(shard["predictions"], ROOT, label="saved predictions"),
                weights_only=False,
                map_location="cpu",
            )
            validate_predictions(saved)
            if mode == "terminal":
                states = {
                    "terminal_argmax_from_saved": decode_synthesis_program_argmax(saved, batch)
                }
            else:
                endpoint, terminal, predictions = capture(
                    model, base, batch, node, bond, shard["seed"], mapping
                )
                if mode == "conditioned":
                    check_equal(predictions, saved)
                states = {f"{mode}_true_endpoint": endpoint, f"{mode}_terminal_argmax": terminal}
                pred_path = path / "shards" / f"draw-{draw}-{offset:04d}.pt"
                torch.save({"states": states, "predictions": predictions}, pred_path)
            if any(not bool(_fixed_state_exact_tensor(state, batch)) for state in states.values()):
                raise ValueError("Decoded state changed fixed context")
            with rdBase.BlockLogs():
                current = score(states, layouts, payload["atoms"], contracts, draw, offset)
            shard_path = path / "shards" / f"draw-{draw}-{offset:04d}.json"
            record = {"schedule": shard, "rows": current}
            if mode != "terminal":
                record["states_and_final_predictions"] = pin(pred_path.relative_to(ROOT))
            write(shard_path, record)
            rows.extend(current)
            receipts.append(pin(shard_path.relative_to(ROOT)))
            write(
                path / "progress.json",
                {
                    "complete": False,
                    "completed_shards": len(receipts),
                    "expected_shards": 880,
                    "CPU_seconds": time.process_time() - cpu,
                    "wall_seconds": time.monotonic() - start,
                },
            )
        if len(rows) != 7040 or len({(x["index"], x["draw"]) for x in rows}) != 7040:
            raise ValueError("Incomplete unchanged cohort denominator")
        write(
            path / "result.json",
            {
                "complete": True,
                "mode": mode,
                "protocol": pin((OUT / "protocol.json").relative_to(ROOT)),
                "smoke": protocol["qualification"],
                "summaries": summarize(rows, protocol["families"]),
                "shards": receipts,
                "trajectories": 7040,
                "requests": 1408,
                "model_flow_calls": 0 if mode == "terminal" else 880,
                "CPU_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - start,
            },
        )
    except BaseException as error:
        write(
            path / "failure.json",
            {
                "complete": False,
                "error": repr(error),
                "CPU_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "smoke", "terminal", "conditioned", "cyclic"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "smoke":
        smoke()
    else:
        run(args.action)
