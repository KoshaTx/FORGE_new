"""Matched local precursor-reuse conditioning experiment; original chemistry gates stay fixed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from experiments.phase1.multireaction.combinatorial_generation import _pin, _write, summarize
from experiments.phase1.multireaction.combinatorial_generation_verify import (
    verify as verify_baseline,
)
from experiments.phase1.multireaction.combinatorial_training_pilot import _record_losses
from forge.assembly.families import load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.library_splits import constitution_id
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.precursor_reuse import (
    PrecursorReuseError,
    PrecursorReuseFlow,
    ReuseSamplingView,
    group_tensor,
    qualify_reuse,
)
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)
from forge.model.synthesis_program_training import (
    collate_synthesis_program_training_batch,
    synthesis_program_fixed_state_exact,
)
from forge.model.tensor_checkpoint import encode_tensor_state

SCHEMA = "forge.combinatorial_reuse_pilot.v1"
ARMS = ("frozen", "self_control", "reuse", "reuse_disabled")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_config(config: dict) -> None:
    if config.get("schema_version") != SCHEMA + "_config" or config.get("policy") != {
        "remote_compute": False,
        "heldout_record_use": False,
        "candidate_selection": False,
        "gate_changes": False,
        "checkpoint_promotion": False,
    }:
        raise PrecursorReuseError("reuse experiment schema/policy changed")
    for key, low, high in (("steps", 1, 768), ("cpu_threads", 1, 8)):
        if (
            type(config["training"].get(key)) is not int
            or not low <= config["training"][key] <= high
        ):
            raise PrecursorReuseError(f"training {key} exceeds bounded pilot contract")
    training = config["training"]
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise PrecursorReuseError("seed must be a nonnegative integer")
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "minimum_flow_time",
        "maximum_flow_time",
    ):
        value = training.get(key)
        if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
            raise PrecursorReuseError(f"invalid finite positive training parameter: {key}")
    if not training["minimum_flow_time"] < training["maximum_flow_time"] < 1:
        raise PrecursorReuseError("flow times must lie strictly between zero and one")
    if (
        config.get("arms") != list(ARMS)
        or config.get("sampling_source") != "authenticated_baseline_layouts_and_noise"
    ):
        raise PrecursorReuseError("paired arms/layout source changed")
    if (
        config.get("decision_rule")
        != "improve_repeated_exact_and_preserve_all_family_metrics_vs_both_controls"
    ):
        raise PrecursorReuseError("decision rule changed")


def assess_rows(
    raw: list[dict],
    records: list,
    arm: str,
    dataset: dict,
    libraries: dict,
    train_products: set,
    train_components: set,
) -> list[dict]:
    rows = []
    for row, record in zip(raw, records, strict=True):
        policy = dataset["programs"][record.program_id]
        smiles = row["canonical_smiles"]
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        valid = molecule is not None and len(Chem.GetMolFrags(molecule)) == 1
        assembly = {"status": "invalid_graph", "programs": [], "expansions": 0}
        if valid:
            assembly = asdict(
                check_generated_program(
                    libraries[record.program_id],
                    smiles,
                    depth=record.program_depth,
                    accumulator_role=policy["accumulator_role"],
                    limits=LibraryProgramLimits(policy["maximum_steps"], **dataset["limits"]),
                )
            )
        witnesses = assembly["programs"]
        rows.append(
            {
                **row,
                "arm": arm,
                "valid_connected": valid,
                "target_scope": policy["target_scope"],
                "requested_depth": record.program_depth,
                "assembly": assembly,
                "novel_vs_train": valid and smiles not in train_products,
                "all_witnesses_have_novel_component": bool(witnesses)
                and all(
                    any(
                        constitution_id(s) not in train_components for s in w["components"].values()
                    )
                    for w in witnesses
                ),
                "failure_reason": (
                    row["constraint_abstention_reason"] or "invalid_or_disconnected_graph"
                    if not valid
                    else (None if witnesses else assembly["status"])
                ),
            }
        )
    return rows


def comparisons(rows: list[dict], families: list[str], dataset: dict) -> dict:
    per_arm = {a: summarize([r for r in rows if r["arm"] == a], families) for a in ARMS}
    metrics = (
        "valid_connected",
        "exact_program",
        "unique_valid_products",
        "effective_product_count",
        "unique_exact_products",
        "novel_valid_attempts_vs_all_cache_train",
        "unique_novel_exact_products",
        "exact_with_novel_component_in_every_witness",
    )
    comparisons = {}
    for control in ("frozen", "self_control"):
        deltas = {
            f: {m: per_arm["reuse"][f][m] - per_arm[control][f][m] for m in metrics}
            for f in families
        }
        repeated = [
            r
            for r in rows
            if r["requested_depth"] > 1
            and dataset["programs"][r["program_id"]]["accumulator_role"] is not None
        ]
        counts = {
            a: sum(
                r["assembly"]["status"] == "exact_computed_program"
                for r in repeated
                if r["arm"] == a
            )
            for a in ARMS
        }
        comparisons[control] = {
            "per_family_deltas": deltas,
            "repeated_exact_delta": counts["reuse"] - counts[control],
            "all_family_metrics_preserved": all(
                v >= 0 for d in deltas.values() for v in d.values()
            ),
            "paired_exact_changes": {
                f: {
                    "gains": sum(
                        not b and t
                        for b, t in zip(
                            [
                                r["assembly"]["status"] == "exact_computed_program"
                                for r in rows
                                if r["arm"] == control and r["program_id"] == f
                            ],
                            [
                                r["assembly"]["status"] == "exact_computed_program"
                                for r in rows
                                if r["arm"] == "reuse" and r["program_id"] == f
                            ],
                            strict=True,
                        )
                    ),
                    "losses": sum(
                        b and not t
                        for b, t in zip(
                            [
                                r["assembly"]["status"] == "exact_computed_program"
                                for r in rows
                                if r["arm"] == control and r["program_id"] == f
                            ],
                            [
                                r["assembly"]["status"] == "exact_computed_program"
                                for r in rows
                                if r["arm"] == "reuse" and r["program_id"] == f
                            ],
                            strict=True,
                        )
                    ),
                }
                for f in families
            },
        }
    depth = []
    for arm in ARMS:
        for family in families:
            for n in sorted({r["requested_depth"] for r in rows if r["program_id"] == family}):
                local = [
                    r
                    for r in rows
                    if r["arm"] == arm and r["program_id"] == family and r["requested_depth"] == n
                ]
                depth.append(
                    {"arm": arm, "family": family, "depth": n, **summarize(local, [family])[family]}
                )
    return {
        "per_arm": per_arm,
        "by_depth": depth,
        "comparisons": comparisons,
        "observed_preservation_rule_passed": all(
            c["repeated_exact_delta"] > 0 and c["all_family_metrics_preserved"]
            for c in comparisons.values()
        ),
    }


def train_pair(
    base: object,
    cache: object,
    config: dict,
    package: dict,
    plans: dict,
    selection: list,
    work: Path,
) -> tuple[dict, dict]:
    networks, reports = {}, {}
    torch.manual_seed(config["seed"])
    initial = PrecursorReuseFlow(copy.deepcopy(base))
    nodes, bonds = (
        torch.as_tensor(package[k], dtype=torch.float32) for k in ("node_marginal", "bond_marginal")
    )
    training = config["training"]
    for arm, share in (("self_control", False), ("reuse", True)):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
        )
        noise = torch.Generator(device="cpu").manual_seed(config["seed"] + 2)
        noise_digest, losses = hashlib.sha256(), []
        start = time.monotonic()
        gradient_max = 0.0
        model.train()
        for step, indices in enumerate(selection, 1):
            records = cache.records(indices)
            clean = collate_synthesis_program_training_batch(
                records,
                maximum_closures=base.maximum_closures,
                conditioning="program",
                vocabulary=cache.vocabulary,
            )
            times = torch.empty(len(indices)).uniform_(
                training["minimum_flow_time"], training["maximum_flow_time"], generator=noise
            )
            state = noise_synthesis_program_batch(clean, nodes, bonds, times, noise)
            if not synthesis_program_fixed_state_exact(state, clean):
                raise PrecursorReuseError("corruption changed fixed chemistry")
            for value in (times, *state.values()):
                noise_digest.update(value.numpy().tobytes())
            groups = group_tensor([plans[i] for i in indices], clean["nodes"].shape[1])
            inputs = {
                k: clean[k]
                for k in (
                    "node_mask",
                    "child_mask",
                    "closure_mask",
                    "program_states",
                    "role_states",
                    "core_position_states",
                    "program_depths",
                    "adapter_mask",
                )
            }
            optimizer.zero_grad(set_to_none=True)
            predictions = model(**inputs, **state, t=times, reuse_groups=groups, share=share)
            loss = torch.stack(_record_losses(predictions, clean)).mean()
            if not bool(torch.isfinite(loss)):
                raise PrecursorReuseError(f"nonfinite {arm} loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["gradient_clip_norm"]
            )
            if not bool(torch.isfinite(norm)):
                raise PrecursorReuseError(f"nonfinite {arm} gradient at step {step}")
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            losses.append(float(loss.detach()))
            if step in {1, 32, 128, 256, len(selection)}:
                print(f"{arm} {step}/{len(selection)} loss={losses[-1]:.6f}", flush=True)
        saved = {
            "schema_version": SCHEMA + "_checkpoint",
            "base_checkpoint": config["inputs"]["checkpoint"],
            "share": share,
            "model_state": encode_tensor_state(model.state_dict()),
            "model_state_sha256": _model_state_sha256(model),
            "training_resumption_supported": False,
        }
        _write(work / f"{arm}_checkpoint.json", saved)
        networks[arm] = model
        reports[arm] = {
            "steps": len(selection),
            "examples": sum(map(len, selection)),
            "losses": losses,
            "initial_state_sha256": _model_state_sha256(initial),
            "final_state_sha256": saved["model_state_sha256"],
            "noise_sha256": noise_digest.hexdigest(),
            "maximum_gradient_norm": gradient_max,
            "duration_seconds": time.monotonic() - start,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
    if reports["self_control"]["noise_sha256"] != reports["reuse"]["noise_sha256"]:
        raise PrecursorReuseError("training corruption streams differ")
    return networks, reports


def run(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise PrecursorReuseError("config/output must be local and output fresh")
    config = json.loads(config_path.read_text())
    validate_config(config)
    pins = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    verify_baseline(repo, pins["baseline_result"])
    baseline = json.loads(pins["baseline_result"].read_text())
    baseline_config = json.loads((repo / baseline["config"]["path"]).read_text())
    for key in ("cache", "checkpoint", "dataset_config", "component_partitions"):
        if baseline["inputs"][key] != config["inputs"][key]:
            raise PrecursorReuseError(f"baseline input changed: {key}")
    dataset = json.loads(pins["dataset_config"].read_text())
    registries = [dataset["inputs"][k] for k in dataset["registries"]]
    libraries = load_assembly_libraries(
        [(resolve_pin(p, repo, label="registry"), p["sha256"]) for p in registries],
        expected_families=dataset["programs"],
    )
    source_names = sorted(
        set(
            [p["path"] for p in baseline["sources"]]
            + [
                "forge/model/precursor_reuse.py",
                "experiments/phase1/multireaction/combinatorial_reuse_pilot.py",
                "experiments/phase1/multireaction/combinatorial_reuse_verify.py",
                "forge/model/phase1_flow.py",
                "forge/model/reaction_program_conditioning.py",
                "forge/model/synthesis_program_graph.py",
                "forge/model/sparse_topology_feasibility.py",
                "forge/model/tensor_checkpoint.py",
            ]
        )
    )
    source_pins = [_pin(repo / p, repo) for p in source_names]
    config_pin = _pin(config_path, repo)
    previous_threads, previous_determinism = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory(prefix=".reuse-pilot-", dir=output.parent) as temporary:
            work = Path(temporary)
            base, vocabulary, atoms, nodes, bonds, package = load_synthesis_program_checkpoint(
                pins["checkpoint"], device="cpu"
            )
            families = list(vocabulary.program_states[1:])
            layout_rows = json.loads(
                (repo / baseline["artifacts"]["layouts.json"]["path"]).read_text()
            )
            indices = [r["index"] for r in layout_rows]
            with SynthesisProgramProductionCache(pins["cache"]) as cache, rdBase.BlockLogs():
                measure = cache.training_measure({f: 1 / len(families) for f in families})
                choices = {f: cache.indices(program_id=f, fold="train") for f in families}
                rng = np.random.default_rng(config["seed"] + 1)
                selection = [
                    [
                        int(
                            rng.choice(
                                choices[f], p=measure[choices[f]] / measure[choices[f]].sum()
                            )
                        )
                        for f in families
                    ]
                    for _ in range(config["training"]["steps"])
                ]
                used = sorted(set(indices + [i for s in selection for i in s]))
                if any(cache.fold(i) != "train" for i in used):
                    raise PrecursorReuseError("nontraining reuse record requested")
                plans = {}
                for i in used:
                    r = cache.record(i)
                    plans[i] = qualify_reuse(
                        r,
                        libraries[r.program_id],
                        dataset["programs"][r.program_id],
                        dataset["limits"],
                    )
                _write(work / "reuse_plans.json", {str(i): asdict(p) for i, p in plans.items()})
                _write(work / "training_indices.json", selection)
                _write(work / "layouts.json", layout_rows)
                print(
                    f"reuse qualification: {dict(Counter(p.status for p in plans.values()))}",
                    flush=True,
                )
                records = cache.records(indices)
                train_products = {
                    cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")
                }
                train_components = {
                    r["constitution_id"]
                    for r in json.loads(pins["component_partitions"].read_text())
                    if r["fold"] == "train"
                }
                networks, training = train_pair(
                    base, cache, config, package, plans, selection, work
                )
            rows, sampler = [], {}
            sampling = baseline_config["sampling"]
            for arm in ARMS:
                network = (
                    base
                    if arm == "frozen"
                    else ReuseSamplingView(
                        networks["reuse" if arm == "reuse_disabled" else arm],
                        records,
                        [plans[i] for i in indices],
                        batch_size=sampling["batch_size"],
                        share=arm == "reuse",
                    )
                )
                with rdBase.BlockLogs():
                    raw, report = sample_synthesis_program_products(
                        network,
                        records,
                        atoms,
                        nodes,
                        bonds,
                        samples_per_program=1,
                        sample_steps=sampling["flow_steps"],
                        batch_size=sampling["batch_size"],
                        seed=sampling["flow_seed"],
                        device="cpu",
                        terminal_decode_policy="strict_valence_topology_argmax",
                    )
                    if report["fixed_state_failures"] or len(raw) != len(records):
                        raise PrecursorReuseError("sampling lost attempts or changed fixed states")
                    if arm != "frozen":
                        network.assert_consumed()
                    enriched = assess_rows(
                        raw, records, arm, dataset, libraries, train_products, train_components
                    )
                rows.extend(enriched)
                sampler[arm] = report
                print(
                    f"{arm}: valid={sum(r['valid_connected'] for r in enriched)} exact={sum(r['assembly']['status'] == 'exact_computed_program' for r in enriched)}",
                    flush=True,
                )
            original = [
                json.loads(s)
                for s in (repo / baseline["artifacts"]["attempts.jsonl"]["path"])
                .read_text()
                .splitlines()
            ]
            original = [r for r in original if r["arm"] == "trained"]
            frozen = [{**r, "arm": "trained"} for r in rows if r["arm"] == "frozen"]
            if _digest(original) != _digest(frozen):
                raise PrecursorReuseError("frozen baseline replay changed")
            with (work / "attempts.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            for pin in [config_pin, *source_pins, *config["inputs"].values(), *registries]:
                resolve_pin(pin, repo, label="final authentication")
            result = {
                "schema_version": SCHEMA,
                "status": "numerical_complete",
                "config": config_pin,
                "inputs": config["inputs"],
                "sources": source_pins,
                "registry_inputs": registries,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - start,
                "environment": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "rdkit": rdBase.rdkitVersion,
                    "device": "cpu",
                    "deterministic": True,
                    "cpu_threads": config["training"]["cpu_threads"],
                },
                "training": training,
                "training_selection_sha256": _digest(selection),
                "sampler": sampler,
                "reuse_qualification": dict(Counter(p.status for p in plans.values())),
                "generation_reuse_qualification": dict(Counter(plans[i].status for i in indices)),
                "reference_train_products": len(train_products),
                "reference_train_components": len(train_components),
                "baseline_replay_identical": True,
                **comparisons(rows, families, dataset),
                "model_promoted": False,
                "nonclaims": [
                    "Soft reuse conditioning is not guaranteed graph equality or relaxed assembly acceptance.",
                    "One seed and reused diagnostic TRAIN layouts do not establish generalization or statistical noninferiority.",
                    "Source graphs qualify correspondence; only equality classes, not target atoms/bonds/pointers, enter the new conditioning path.",
                    "Computed reconstruction does not establish chemical realism, ionizability, source-executed synthesis or L2/L3 closure.",
                    "Cache arrays include heldout bytes; only TRAIN records are materialized, with no heldout evaluation.",
                    "Continued training uses fresh optimizer states in both arms; this is not faithful optimizer resumption.",
                ],
            }
            result["artifacts"] = {
                p.name: {
                    "path": str((output / p.name).relative_to(repo)),
                    "sha256": str(sha256_file(p)),
                }
                for p in sorted(work.iterdir())
            }
            _write(work / "result.json", result)
            os.rename(work, output)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo_root, args.config, args.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "observed_preservation_rule_passed")}))


if __name__ == "__main__":
    main()
