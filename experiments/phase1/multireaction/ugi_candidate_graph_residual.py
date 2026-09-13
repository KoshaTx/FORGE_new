"""Bounded TRAIN-only complete-candidate topology scoring on a frozen backbone."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.phase1.multireaction.reaction_specialization import (
    _load_base_package,
    _set_determinism,
)
from experiments.phase1.multireaction.ugi_candidate_graph_coverage import (
    build_selection,
    load_train_assignments,
)
from experiments.phase1.multireaction.ugi_realism_diagnostic_suite import implementation_snapshot
from experiments.phase1.multireaction.ugi_realism_model_attribution import _tensor_digest
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from forge.core.hashing import pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_candidate_graph_residual import CandidateGraphResidual, candidate_class_loss
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_topology_candidates import score_candidates, topology_candidates

ROLES = ("amine_head", "oxoester_aldehyde_body_tail")


class CandidateGraphStudyError(ValueError):
    """The study violated its frozen data, computation, or evidence contract."""


def _read(path: Path) -> dict[str, Any]:
    return read_json_object(path, error=CandidateGraphStudyError)


def _validate(config: dict[str, Any]) -> None:
    expected = {
        "schema_version": "forge.ugi_candidate_graph_residual_config.v1",
        "target_program": "ugi_3cr_agile",
        "partition": {
            "component_domain": "forge.ugi_candidate_graph_component_partition.v1:2026090824:",
            "product_domain": "forge.ugi_candidate_graph_product_partition.v1:2026090824:",
            "bucket_count": 5,
            "evaluation_bucket": 0,
        },
        "flow_times": [0.2, 0.5, 0.8, 0.95],
        "draws": {
            "fit": 1024,
            "amine_disjoint": 256,
            "aldehyde_disjoint": 256,
            "repeated_component": 256,
        },
        "feature_batch_size": 64,
        "training_batch_size": 64,
        "optimizer_steps": 64,
        "candidate_chunk_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "width": 32,
        "cpu_threads": 2,
        "device": "cpu",
        "precision": "float32",
        "seeds": {
            "draws": 2026090825,
            "noise": 2026090826,
            "order": 2026090827,
            "initialization": 2026090828,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise CandidateGraphStudyError(f"frozen study field changed: {key}")
    if config.get("policy") != {
        "backbone_training": False,
        "molecular_generation": False,
        "remote_compute": False,
        "gate_changes": False,
        "final_checkpoint_only": True,
        "feature_records_train_only": True,
    }:
        raise CandidateGraphStudyError("study policy changed")


def _candidate_receipt(candidate: Any) -> dict[str, Any]:
    return {
        "role": candidate.role,
        "candidate_count": len(candidate.candidates),
        "graph_class_count": candidate.graph_class_count,
        "positive_count": int(np.count_nonzero(candidate.positive_mask)),
        "target_present": bool(np.any(candidate.positive_mask)),
        "unavailable_reason": candidate.unavailable_reason,
        "failed_candidate_index": candidate.failed_candidate_index,
        "enumerated_identity_sha256": candidate.enumerated_identity_sha256,
        "graph_node_indices": list(candidate.graph_node_indices),
        "graph_class_ids": list(candidate.graph_class_ids),
        "positive_mask": candidate.positive_mask.tolist(),
        "adjacency_sha256": _tensor_digest(
            {"adjacency": torch.as_tensor(candidate.adjacency.copy())}
        ),
        "candidates": [
            {
                "offspring": np.asarray(value.offspring).tolist(),
                "closures": [list(edge) for edge in value.closures],
            }
            for value in candidate.candidates
        ],
        "enumerated_candidates": [
            {
                "offspring": list(value.offspring),
                "closures": [list(edge) for edge in value.closures],
            }
            for value in candidate.enumerated_candidates
        ],
    }


def _coverage(candidates: dict, selection: dict) -> dict[str, Any]:
    output = {}
    for group, selected in selection.items():
        output[group] = {}
        for role in ROLES:
            items = [candidates[(row["cache_index"], role)] for row in selected["draws"]]
            output[group][role] = {
                "draws": len(items),
                "empty": sum(len(item.candidates) == 0 for item in items),
                "law_unavailable": sum(item.unavailable_reason is not None for item in items),
                "target_absent": sum(not np.any(item.positive_mask) for item in items),
                "singleton_graph_class": sum(item.graph_class_count == 1 for item in items),
                "reachable_competing_graph_classes": int(
                    sum(item.graph_class_count > 1 and np.any(item.positive_mask) for item in items)
                ),
            }
    return output


def _extract_features(model, package, cache, selection, candidates, config, output):
    features, ledger = {}, []
    for group, selected in selection.items():
        rows = selected["draws"]
        records = cache.records([row["cache_index"] for row in rows])
        cases = []
        for flow_time in config["flow_times"]:
            for start in range(0, len(rows), config["feature_batch_size"]):
                stop = min(len(rows), start + config["feature_batch_size"])
                clean = collate_synthesis_program_training_batch(
                    records[start:stop],
                    maximum_closures=package["model_config"]["maximum_closures"],
                    conditioning="program",
                    vocabulary=cache.vocabulary,
                )
                t = torch.full((stop - start,), flow_time, dtype=torch.float32)
                seed = config["seeds"]["noise"] + len(ledger)
                noisy = noise_synthesis_program_batch(
                    clean,
                    torch.as_tensor(package["node_marginal"], dtype=torch.float32),
                    torch.as_tensor(package["bond_marginal"], dtype=torch.float32),
                    t,
                    torch.Generator().manual_seed(seed),
                )
                # Native state only: clean target topology must not enter the feature input.
                with torch.no_grad():
                    predictions = _synthesis_program_predict(
                        model, clean, noisy, t, return_hidden_state=True
                    )
                for local, draw in enumerate(range(start, stop)):
                    for role_index, role in enumerate(ROLES):
                        key = (rows[draw]["cache_index"], role)
                        candidate = candidates[key]
                        one = {
                            name: value[local : local + 1] for name, value in predictions.items()
                        }
                        cases.append(
                            {
                                "cache_index": key[0],
                                "role": role,
                                "role_index": role_index,
                                "draw_index": draw,
                                "flow_time": flow_time,
                                "hidden": predictions["hidden_state"][
                                    local, list(candidate.graph_node_indices)
                                ]
                                .detach()
                                .clone(),
                                "base_scores": score_candidates(candidate, one).detach().clone(),
                            }
                        )
                ledger.append(
                    {
                        "group": group,
                        "start": start,
                        "stop": stop,
                        "flow_time": flow_time,
                        "noise_seed": seed,
                        "clean_sha256": _tensor_digest(clean),
                        "noisy_sha256": _tensor_digest(noisy),
                        "predictions_sha256": _tensor_digest(predictions),
                        "target_topology_substituted": False,
                    }
                )
                print(f"features {group} t={flow_time} {stop}/{len(rows)}", flush=True)
        features[group] = cases
        torch.save(cases, output / f"features_{group}.pt")
    write_json(output / "feature_batches.json", {"forward_calls": len(ledger), "rows": ledger})
    return features, ledger


def _candidate_deltas(head, case, candidate, chunk_size):
    n = len(candidate.candidates)
    colors = torch.tensor(candidate.node_colors, dtype=torch.float32)
    nodes = case["hidden"].shape[0]
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        count = stop - start
        yield start, stop, (
            head(
                case["hidden"].unsqueeze(0).expand(count, -1, -1),
                torch.tensor(candidate.adjacency[start:stop], dtype=torch.float32),
                colors.unsqueeze(0).expand(count, -1, -1),
                torch.ones(count, nodes, dtype=torch.bool),
                torch.full((count,), case["flow_time"], dtype=torch.float32),
                torch.full((count,), case["role_index"], dtype=torch.long),
            )
        )


def _residual_scores(head, case, candidate, chunk_size):
    if not candidate.candidates:
        return case["base_scores"].clone()
    chunks = [value for _, _, value in _candidate_deltas(head, case, candidate, chunk_size)]
    return case["base_scores"] + torch.cat(chunks).to(torch.float64)


def _backward_case(head, case, candidate, chunk_size, denominator):
    """Accumulate class-loss gradients with only one chunk's live activations.

    A detached pass gives dL/ds = softmax(all) - softmax(positives). Recompute each
    residual chunk and apply that coefficient; weights stay fixed until the batch update.
    """
    n = len(candidate.candidates)
    positive = torch.tensor(candidate.positive_mask, dtype=torch.bool)
    if not n or not positive.any() or positive.all():
        return 0.0
    with torch.no_grad():
        scores = _residual_scores(head, case, candidate, chunk_size)
        loss = candidate_class_loss(scores, torch.tensor([0, n]), positive).per_row_loss[0]
        if not torch.isfinite(loss):
            raise CandidateGraphStudyError("nonfinite candidate loss")
        coefficient = scores.softmax(0)
        coefficient[positive] -= scores[positive].softmax(0)
        coefficient /= denominator
    for start, stop, delta in _candidate_deltas(head, case, candidate, chunk_size):
        (delta.to(torch.float64) * coefficient[start:stop]).sum().backward()
    return float(loss) / denominator


def _case_metrics(scores, candidate):
    n = len(scores)
    positive = torch.tensor(candidate.positive_mask, dtype=torch.bool)
    result = {
        "candidate_count": n,
        "graph_class_count": candidate.graph_class_count,
        "target_present": bool(positive.any()),
        "informative": bool(positive.any()) and candidate.graph_class_count > 1,
        "target_probability": 0.0,
        "target_nll": None,
        "graph_argmax_correct": False,
    }
    if n and positive.any():
        log_probability = torch.logsumexp(scores[positive], 0) - torch.logsumexp(scores, 0)
        result["target_probability"] = float(log_probability.exp())
        result["target_nll"] = float(-log_probability)
        probabilities = scores.softmax(0)
        masses = torch.zeros(candidate.graph_class_count, dtype=probabilities.dtype)
        classes = torch.tensor(candidate.graph_class_ids, dtype=torch.long)
        masses.scatter_add_(0, classes, probabilities)
        result["graph_argmax_correct"] = int(masses.argmax()) == int(classes[positive][0])
    return result


def _summary(rows):
    informative = [row for row in rows if row["informative"]]
    return {
        "cases": len(rows),
        "empty": sum(row["candidate_count"] == 0 for row in rows),
        "target_absent": sum(not row["target_present"] for row in rows),
        "graph_argmax_correct_all": sum(row["graph_argmax_correct"] for row in rows),
        "mean_target_probability_all": (
            float(np.mean([r["target_probability"] for r in rows])) if rows else None
        ),
        "informative_cases": len(informative),
        "informative_mean_target_probability": (
            float(np.mean([r["target_probability"] for r in informative])) if informative else None
        ),
        "informative_mean_nll": (
            float(np.mean([r["target_nll"] for r in informative])) if informative else None
        ),
        "informative_graph_argmax_correct": sum(r["graph_argmax_correct"] for r in informative),
    }


def _fit(features, candidates, config, output):
    cases = features["fit"]
    if not cases or len(cases) % len(ROLES):
        raise CandidateGraphStudyError("fitting cases must contain complete paired roles")
    for start in range(0, len(cases), len(ROLES)):
        pair = cases[start : start + len(ROLES)]
        if any(
            case["role"] != role
            or case["role_index"] != index
            or any(case[key] != pair[0][key] for key in ("cache_index", "draw_index", "flow_time"))
            for index, (case, role) in enumerate(zip(pair, ROLES, strict=True))
        ):
            raise CandidateGraphStudyError("fitting paired exposure identity/order differs")
    exposures = len(cases) // len(ROLES)
    if exposures != config["training_batch_size"] * config["optimizer_steps"]:
        raise CandidateGraphStudyError("fitting exposure count changed")
    order = torch.randperm(
        exposures, generator=torch.Generator().manual_seed(config["seeds"]["order"])
    )
    write_json(output / "exposure_order.json", {"product_time_indices": order.tolist()})
    sample = candidates[(cases[0]["cache_index"], cases[0]["role"])]
    heads, receipts, initial = {}, {}, None
    for arm in ("degree_control", "graph"):
        head = CandidateGraphResidual(
            cases[0]["hidden"].shape[-1],
            np.asarray(sample.node_colors).shape[1],
            width=config["width"],
            initialization_seed=config["seeds"]["initialization"],
            use_adjacency=arm == "graph",
        )
        if initial is None:
            initial = copy.deepcopy(head.state_dict())
            torch.save(initial, output / "initial_head.pt")
        elif any(not torch.equal(initial[key], value) for key, value in head.state_dict().items()):
            raise CandidateGraphStudyError("matched initialization changed")
        with torch.no_grad():
            for case in cases:
                candidate = candidates[(case["cache_index"], case["role"])]
                if not torch.equal(
                    _residual_scores(head, case, candidate, config["candidate_chunk_size"]),
                    case["base_scores"],
                ):
                    raise CandidateGraphStudyError("zero residual changes candidate scores")
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        losses, counts = [], {"empty": 0, "absent": 0, "all_positive": 0, "informative": 0}
        for start in range(0, exposures, config["training_batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            batch = order[start : start + config["training_batch_size"]].tolist()
            denominator = len(batch) * len(ROLES)
            for exposure in batch:
                for role_index in range(len(ROLES)):
                    case = cases[exposure * len(ROLES) + role_index]
                    candidate = candidates[(case["cache_index"], case["role"])]
                    n = len(candidate.candidates)
                    positive = torch.tensor(candidate.positive_mask, dtype=torch.bool)
                    if not n:
                        counts["empty"] += 1
                        continue
                    if not positive.any():
                        counts["absent"] += 1
                        continue
                    if positive.all():
                        counts["all_positive"] += 1
                        continue
                    counts["informative"] += 1
                    value += _backward_case(
                        head, case, candidate, config["candidate_chunk_size"], denominator
                    )
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise CandidateGraphStudyError("nonfinite gradient norm")
            optimizer.step()
            losses.append(value)
            print(f"fit {arm} step={len(losses)} loss={value:.8f}", flush=True)
        head.eval()
        torch.save(
            {"head_state": head.state_dict(), "optimizer_state": optimizer.state_dict()},
            output / f"{arm}_final.pt",
        )
        heads[arm] = head
        receipts[arm] = {
            "updates": len(losses),
            "losses": losses,
            "exposure_counts": counts,
            "parameters": sum(p.numel() for p in head.parameters()),
            "state_sha256": _tensor_digest(head.state_dict()),
            "zero_residual_equal_all_fit_cases": True,
        }
        write_json(output / "training_receipts.json", receipts)
    return heads, receipts


def _evaluate(heads, features, candidates, config, output):
    summary = {}
    with torch.no_grad(), gzip.open(output / "evaluation_rows.jsonl.gz", "wt") as stream:
        for group, cases in features.items():
            if group == "fit":
                continue
            rows = {role: {arm: [] for arm in ("baseline", *heads)} for role in ROLES}
            for case in cases:
                candidate = candidates[(case["cache_index"], case["role"])]
                for arm in ("baseline", *heads):
                    scores = (
                        case["base_scores"]
                        if arm == "baseline"
                        else _residual_scores(
                            heads[arm], case, candidate, config["candidate_chunk_size"]
                        )
                    )
                    metrics = _case_metrics(scores, candidate)
                    row = {
                        key: case[key] for key in ("cache_index", "role", "draw_index", "flow_time")
                    }
                    row.update(group=group, arm=arm, scores=scores.tolist(), **metrics)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows[case["role"]][arm].append({**metrics, "flow_time": case["flow_time"]})
            summary[group] = {
                role: {
                    arm: {
                        "pooled": _summary(values),
                        "by_flow_time": {
                            str(t): _summary([r for r in values if r["flow_time"] == t])
                            for t in config["flow_times"]
                        },
                    }
                    for arm, values in arms.items()
                }
                for role, arms in rows.items()
            }
    checks = {}
    for role, group in zip(ROLES, ("amine_disjoint", "aldehyde_disjoint"), strict=True):
        arms = {arm: summary[group][role][arm]["pooled"] for arm in ("baseline", *heads)}
        graph = arms["graph"]
        checks[role] = {
            "probability_better_than_both": all(
                graph["informative_mean_target_probability"]
                > arms[arm]["informative_mean_target_probability"]
                for arm in ("baseline", "degree_control")
            ),
            "nll_better_than_both": all(
                graph["informative_mean_nll"] < arms[arm]["informative_mean_nll"]
                for arm in ("baseline", "degree_control")
            ),
            "disjoint_argmax_preserved": graph["graph_argmax_correct_all"]
            >= arms["baseline"]["graph_argmax_correct_all"],
            "repeated_argmax_preserved": summary["repeated_component"][role]["graph"]["pooled"][
                "graph_argmax_correct_all"
            ]
            >= summary["repeated_component"][role]["baseline"]["pooled"][
                "graph_argmax_correct_all"
            ],
        }
    return {
        "populations": summary,
        "checks": checks,
        "mechanism_gate_passed": all(all(v.values()) for v in checks.values()),
    }


def run_study(repo_root: Path, config_path: Path, output_dir: Path):
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise CandidateGraphStudyError("fresh output directory inside repository required")
    config = _read(config_path)
    _validate(config)
    config_pin = pin_record(config_path, repo)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    for name, pin in config["sources"].items():
        resolve_pin(pin, repo, label=name)
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        sources = implementation_snapshot(repo)
        write_json(output / "source_snapshot.json", sources)
        rows = [json.loads(line) for line in paths["coverage_rows"].read_text().splitlines()]
        split = build_selection(rows, load_train_assignments(paths["ugi_assignments"]), config)
        selection = split["groups"]
        with (output / "coverage_rows.jsonl").open("w") as stream:
            for row in split.pop("coverage_rows"):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "selection.json", split)
        _set_determinism(
            config["seeds"]["initialization"], config["cpu_threads"], torch.device("cpu")
        )
        base = config["base_checkpoint"]
        package, member = _load_base_package(
            archive_path=paths["base_checkpoint_archive"],
            training=_read(paths["base_training_result"]),
            arm_id=base["arm_id"],
            step=base["step"],
            expected_member_sha256=base["member_sha256"],
            expected_model_state_sha256=base["model_state_sha256"],
            expected_design_sha256=str(sha256_file(paths["base_design"])),
            expected_cache_sha256=str(sha256_file(paths["production_cache"])),
            device=torch.device("cpu"),
        )
        before = _tensor_digest(package["model_state"])
        with SynthesisProgramProductionCache(paths["production_cache"]) as cache:
            train = set(map(int, cache.indices(program_id=config["target_program"], fold="train")))
            if len(rows) != len(train) or {row["cache_index"] for row in rows} != train:
                raise CandidateGraphStudyError("coverage does not match full TRAIN indices")
            for row in rows:
                i = row["cache_index"]
                if (
                    cache.record_id(i) != row["product_id"]
                    or float(cache.arrays["source_weights"][i]) != row["source_weight"]
                ):
                    raise CandidateGraphStudyError("coverage identity/source weight mismatch")
            local = LocalChemistrySupport.from_mapping(_read(paths["role_morphology_policy"]))
            ester = UgiEsterChemotypePolicy.from_qualified_registry(
                paths["qualified_reactions"],
                training_assignments_path=paths["ugi_assignments"],
                reaction_id=config["target_program"],
                expected_sha256=str(sha256_file(paths["qualified_reactions"])),
                expected_training_assignments_sha256=str(sha256_file(paths["ugi_assignments"])),
            )
            topology = _topology_policy(paths)
            unique = {r["cache_index"]: r for group in selection.values() for r in group["draws"]}
            candidates = {}
            for index, row in sorted(unique.items()):
                if index not in train:
                    raise CandidateGraphStudyError("non-TRAIN record materialization rejected")
                record = cache.record(index)
                for role in ROLES:
                    candidates[(index, role)] = topology_candidates(
                        record,
                        role,
                        cache.atom_vocabulary,
                        topology,
                        ester,
                        local,
                        UgiAmineSemanticTarget.from_key(tuple(row["semantic_target"])),
                        core_position_classes=len(cache.vocabulary.core_position_states),
                        maximum_children=package["model_config"]["maximum_children"],
                    )
            coverage = _coverage(candidates, selection)
            write_json(output / "candidate_coverage.json", coverage)
            write_json(
                output / "candidate_sets.json",
                {f"{i}:{role}": _candidate_receipt(c) for (i, role), c in candidates.items()},
            )
            preflight = all(
                coverage[group][role]["reachable_competing_graph_classes"] > 0
                for role, evaluation in zip(
                    ROLES, ("amine_disjoint", "aldehyde_disjoint"), strict=True
                )
                for group in ("fit", evaluation)
            )
            training, evaluation, ledger = {}, {}, []
            if preflight:
                model = build_synthesis_program_flow(
                    vocabulary=cache.vocabulary,
                    node_classes=len(cache.atom_vocabulary),
                    model_config=package["model_config"],
                    device=torch.device("cpu"),
                )
                model.load_state_dict(package["model_state"], strict=True)
                model.eval().requires_grad_(False)
                features, ledger = _extract_features(
                    model, package, cache, selection, candidates, config, output
                )
                heads, training = _fit(features, candidates, config, output)
                evaluation = _evaluate(heads, features, candidates, config, output)
                if _tensor_digest(model.state_dict()) != before:
                    raise CandidateGraphStudyError("frozen backbone changed")
        if sources != implementation_snapshot(repo) or config_pin != pin_record(config_path, repo):
            raise CandidateGraphStudyError("source/config changed during study")
        for name, pin in {**config["inputs"], **config["sources"]}.items():
            resolve_pin(pin, repo, label=name)
        result = {
            "schema_version": "forge.ugi_candidate_graph_residual_result.v1",
            "status": (
                "complete_development_reconstruction_only"
                if preflight
                else "complete_support_preflight_failed_no_fit"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_pin,
            "inputs": config["inputs"],
            "sources": config["sources"],
            "source_snapshot_sha256": str(sha256_json(sources)),
            "checkpoint": {**base, "member_name": member, "state_sha256": before},
            "coverage": coverage,
            "partition_coverage": split["coverage"],
            "preflight_passed": preflight,
            "training": training,
            "evaluation": evaluation,
            "mechanism_gate_passed": evaluation.get("mechanism_gate_passed", False),
            "forward_calls": len(ledger),
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": {
                p.name: pin_record(p, repo) for p in sorted(output.iterdir()) if p.is_file()
            },
            "nonclaims": config["nonclaims"],
        }
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_no_admitted_result",
                "config": config_pin,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_study(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
