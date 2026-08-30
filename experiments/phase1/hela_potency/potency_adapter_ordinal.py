"""Paired ordinal cross-fit for the bounded Ugi HeLa potency adapter.

This v2 diagnostic corrects the two identifiable limitations of v1: real and shuffled controls now
share initialization and every stochastic training draw, and target percentiles are identified by
counterfactual q10/q50/q90 reconstruction comparisons.  It still does not run generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.hela_potency.potency_adapter import (
    BASE_CHECKPOINT_SCHEMA,
    PROGRAM_ID,
    _clean_batch,
    _deterministic_tar,
    _joined_observations,
    _load_base_package,
    _model_with_adapter,
    _morphology_features,
    _policy,
    _set_determinism,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_csv, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.potency_adapter import (
    apply_potency_adapter_state,
    optimizer_parameters,
    potency_adapter_state_dict,
    potency_parameter_report,
)
from forge.model.potency_conditioning import PotencyAdapterPolicy, PotencyCondition
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import _synthesis_program_predict
from forge.model.training_restart import atomic_torch_save
from forge.potency.adapter_failure_attribution import factorial_dataset_summary
from forge.potency.adapter_guidance import (
    clustered_auroc_difference_interval,
    clustered_bootstrap_interval,
    clustered_mean_difference_interval,
    deterministic_permutation,
    map_to_training_ecdf,
    masked_retention_kl,
    monotonicity_gate,
    morphology_residuals,
    ordinal_contrastive_loss,
    ordinal_monotonicity_rows,
    ordinal_nll_matrix,
    quartile_balanced_sample,
    roc_auc,
    signal_gate,
    spearman,
    training_ecdf_quantiles,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.ugi_hela_potency_ordinal_adapter_config.v2"
RESULT_SCHEMA = "forge.ugi_hela_potency_ordinal_adapter_crossfit_result.v2"
LEDGER_SCHEMA = "forge.ugi_hela_potency_ordinal_score.v2"
CHECKPOINT_SCHEMA = "forge.ugi_hela_potency_ordinal_adapter_checkpoint.v2"
FOLD_PROGRESS_SCHEMA = "forge.ugi_hela_potency_ordinal_fold_progress.v1"


class UgiPotencyOrdinalAdapterError(ValueError):
    """The paired ordinal adapter violated its frozen diagnostic contract."""


def _condition(policy: PotencyAdapterPolicy, quantile: float) -> PotencyCondition:
    return PotencyCondition(
        endpoint_id=policy.endpoint_id,
        target_quantile=float(quantile),
        policy_id=policy.policy_id,
    )


def _anchor_nll(
    model: Any,
    clean: Mapping[str, Any],
    noisy: Mapping[str, Any],
    t: Any,
    *,
    policy: PotencyAdapterPolicy,
    anchors: Sequence[float],
) -> Any:
    predictions = [
        _synthesis_program_predict(
            model,
            clean,
            noisy,
            t,
            potency_condition=_condition(policy, quantile),
        )
        for quantile in anchors
    ]
    return ordinal_nll_matrix(predictions, clean)


def _fit_paired_adapters(
    *,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    config: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    real_quantiles: np.ndarray,
    shuffled_quantiles: np.ndarray,
    policy: PotencyAdapterPolicy,
    device: Any,
    seed: int,
    steps: int,
    labelled_batch_size: int,
    retention_batch_size: int,
) -> tuple[Any, Any, dict[str, list[dict[str, float]]], dict[str, Any]]:
    """Fit paired controls whose only difference is the target-percentile assignment."""

    real_model = _model_with_adapter(package, cache, config, device, policy)
    initial_adapter = potency_adapter_state_dict(real_model)
    shuffled_model = _model_with_adapter(package, cache, config, device, policy)
    apply_potency_adapter_state(shuffled_model, initial_adapter)
    if _model_state_sha256(real_model) != _model_state_sha256(shuffled_model):
        raise UgiPotencyOrdinalAdapterError("paired models do not share exact initialization")
    initialization_sha256 = _model_state_sha256(real_model)
    optimizers = {
        "real": torch.optim.AdamW(
            optimizer_parameters(real_model),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        ),
        "shuffled": torch.optim.AdamW(
            optimizer_parameters(shuffled_model),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        ),
    }
    models = {"real": real_model, "shuffled": shuffled_model}
    targets = {"real": real_quantiles, "shuffled": shuffled_quantiles}
    anchors = tuple(float(value) for value in config["ordinal"]["anchors"])
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    retention_support = cache.indices(program_id=PROGRAM_ID, fold="train")
    retention_weights = cache.arrays["source_weights"][retention_support].astype(np.float64)
    retention_weights /= retention_weights.sum()
    traces: dict[str, list[dict[str, float]]] = {"real": [], "shuffled": []}
    for step in range(1, steps + 1):
        selected = quartile_balanced_sample(
            real_quantiles,
            size=labelled_batch_size,
            rng=rng,
        )
        labelled_clean = _clean_batch(
            cache,
            [int(train_rows[index]["cache_index"]) for index in selected],
            real_model,
            device,
        )
        labelled_t = torch.rand(labelled_batch_size, generator=generator, device=device).clamp(
            0.02, 0.98
        )
        labelled_noisy = noise_synthesis_program_batch(
            labelled_clean,
            node_p0,
            bond_p0,
            labelled_t,
            generator,
        )
        retention_indices = rng.choice(
            retention_support,
            size=retention_batch_size,
            replace=True,
            p=retention_weights,
        )
        retention_clean = _clean_batch(cache, retention_indices, real_model, device)
        retention_t = torch.rand(retention_batch_size, generator=generator, device=device).clamp(
            0.02, 0.98
        )
        retention_noisy = noise_synthesis_program_batch(
            retention_clean,
            node_p0,
            bond_p0,
            retention_t,
            generator,
        )
        with torch.no_grad():
            teacher = _synthesis_program_predict(
                real_model,
                retention_clean,
                retention_noisy,
                retention_t,
            )
        for arm in ("real", "shuffled"):
            model = models[arm]
            optimizer = optimizers[arm]
            nll = _anchor_nll(
                model,
                labelled_clean,
                labelled_noisy,
                labelled_t,
                policy=policy,
                anchors=anchors,
            )
            primary, ordinal = ordinal_contrastive_loss(
                nll,
                torch.as_tensor(targets[arm][selected], dtype=torch.float32, device=device),
                anchors=anchors,
                margin=float(config["ordinal"]["margin"]),
            )
            student = _synthesis_program_predict(
                model,
                retention_clean,
                retention_noisy,
                retention_t,
                potency_condition=_condition(
                    policy, float(config["condition"]["retention_quantile"])
                ),
            )
            retention = masked_retention_kl(student, teacher, retention_clean)
            loss = (
                primary
                + float(config["ordinal"]["contrast_weight"]) * ordinal
                + float(config["training"]["retention_weight"]) * retention
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters(model),
                float(config["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            if step == 1 or step == steps or step % max(1, steps // 10) == 0:
                values = torch.stack(
                    (
                        primary.detach(),
                        ordinal.detach(),
                        retention.detach(),
                        loss.detach(),
                        gradient_norm.detach(),
                    )
                ).cpu()
                traces[arm].append(
                    {
                        "step": float(step),
                        "primary_nll": float(values[0]),
                        "ordinal_contrast": float(values[1]),
                        "retention_kl": float(values[2]),
                        "total_loss": float(values[3]),
                        "gradient_norm": float(values[4]),
                    }
                )
    receipt = {
        "identical_initialization": True,
        "identical_labelled_molecule_batches": True,
        "identical_retention_batches": True,
        "identical_flow_times_and_corruptions": True,
        "initial_combined_model_state_sha256": initialization_sha256,
        "only_difference": "real versus deterministically permuted target percentile",
    }
    return real_model, shuffled_model, traces, receipt


def _evaluate_paired_adapters(
    *,
    real_model: Any,
    shuffled_model: Any,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[str],
    policy: PotencyAdapterPolicy,
    anchors: Sequence[float],
    times: Sequence[tuple[str, float]],
    batch_size: int,
    seed: int,
    device: Any,
) -> list[dict[str, Any]]:
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    output: list[dict[str, Any]] = []
    for time_index, (time_name, time_value) in enumerate(times):
        for offset in range(0, len(rows), batch_size):
            local = rows[offset : offset + batch_size]
            clean = _clean_batch(
                cache,
                [int(row["cache_index"]) for row in local],
                real_model,
                device,
            )
            t = torch.full((len(local),), time_value, dtype=torch.float32, device=device)
            generator = torch.Generator(device=device).manual_seed(
                seed + 100_000 * time_index + offset
            )
            noisy = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, generator)
            with torch.inference_mode():
                real_nll = (
                    _anchor_nll(
                        real_model,
                        clean,
                        noisy,
                        t,
                        policy=policy,
                        anchors=anchors,
                    )
                    .cpu()
                    .numpy()
                )
                shuffled_nll = (
                    _anchor_nll(
                        shuffled_model,
                        clean,
                        noisy,
                        t,
                        policy=policy,
                        anchors=anchors,
                    )
                    .cpu()
                    .numpy()
                )
            for local_index, row in enumerate(local):
                output.append(
                    {
                        "schema_version": LEDGER_SCHEMA,
                        "label": row["label"],
                        "potency": float(row["potency"]),
                        "held_quantile": float(row["held_quantile"]),
                        "cluster_id": clusters[offset + local_index],
                        "time_bin": time_name,
                        "real_q10_nll": float(real_nll[local_index, 0]),
                        "real_q50_nll": float(real_nll[local_index, 1]),
                        "real_q90_nll": float(real_nll[local_index, 2]),
                        "shuffled_q10_nll": float(shuffled_nll[local_index, 0]),
                        "shuffled_q50_nll": float(shuffled_nll[local_index, 1]),
                        "shuffled_q90_nll": float(shuffled_nll[local_index, 2]),
                    }
                )
    return output


def _monotonic_metrics(
    local: Sequence[Mapping[str, Any]],
    *,
    anchors: Sequence[float],
    bootstrap: Mapping[str, Any],
    seed: int,
    thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, bool], np.ndarray, np.ndarray]:
    targets = np.asarray([float(row["held_quantile"]) for row in local])
    clusters = [str(row["cluster_id"]) for row in local]
    real_nll = np.asarray(
        [[row[f"real_q{suffix}_nll"] for suffix in ("10", "50", "90")] for row in local]
    )
    shuffled_nll = np.asarray(
        [[row[f"shuffled_q{suffix}_nll"] for suffix in ("10", "50", "90")] for row in local]
    )
    real = ordinal_monotonicity_rows(real_nll, targets, anchors=anchors)
    shuffled = ordinal_monotonicity_rows(shuffled_nll, targets, anchors=anchors)
    outer = real["outer_quartile"]
    if not bool(outer.any()):
        raise UgiPotencyOrdinalAdapterError("held fold has no outer-quartile observations")
    outer_interval = clustered_mean_difference_interval(
        real["outer_monotonic"][outer].astype(float),
        shuffled["outer_monotonic"][outer].astype(float),
        np.asarray(clusters, dtype=object)[outer].tolist(),
        replicates=int(bootstrap["replicates"]),
        seed=seed,
    )
    anchor_interval = clustered_mean_difference_interval(
        real["anchor_correct"].astype(float),
        shuffled["anchor_correct"].astype(float),
        clusters,
        replicates=int(bootstrap["replicates"]),
        seed=seed + 10_000,
    )
    values = {
        "real_outer_monotonic_fraction": float(np.mean(real["outer_monotonic"][outer])),
        "shuffled_outer_monotonic_fraction": float(np.mean(shuffled["outer_monotonic"][outer])),
        "outer_monotonic_difference_bootstrap": outer_interval,
        "real_anchor_accuracy": float(np.mean(real["anchor_correct"])),
        "shuffled_anchor_accuracy": float(np.mean(shuffled["anchor_correct"])),
        "anchor_accuracy_difference_bootstrap": anchor_interval,
    }
    gate = monotonicity_gate(
        real_outer_fraction=values["real_outer_monotonic_fraction"],
        shuffled_outer_fraction=values["shuffled_outer_monotonic_fraction"],
        outer_difference_lower=float(outer_interval["lower_95"]),
        real_anchor_accuracy=values["real_anchor_accuracy"],
        shuffled_anchor_accuracy=values["shuffled_anchor_accuracy"],
        anchor_difference_lower=float(anchor_interval["lower_95"]),
        thresholds=thresholds,
    )
    return values, gate, real["direction_score"], shuffled["direction_score"]


def _fold_progress_signature(
    *,
    scheme: str,
    fold: int,
    local_seed: int,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    time_bins: Sequence[Mapping[str, Any]],
) -> str:
    return str(
        sha256_json(
            {
                "fold": fold,
                "local_seed": local_seed,
                "scheme": scheme,
                "test_labels": [str(row["label"]) for row in test_rows],
                "time_bins": list(time_bins),
                "train_labels": [str(row["label"]) for row in train_rows],
            }
        )
    )


def _load_fold_progress(
    path: Path,
    *,
    expected_signature: str,
    expected_test_rows: int,
    expected_evaluation_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = read_json_object(
        path,
        error=UgiPotencyOrdinalAdapterError,
        label="ordinal cross-fit fold progress",
    )
    record = value.get("fold_record")
    rows = value.get("evaluation_rows")
    if (
        value.get("schema_version") != FOLD_PROGRESS_SCHEMA
        or value.get("fold_signature") != expected_signature
        or not isinstance(record, dict)
        or not isinstance(rows, list)
        or int(record.get("test_rows", -1)) != expected_test_rows
        or len(rows) != expected_evaluation_rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise UgiPotencyOrdinalAdapterError(f"ordinal fold progress is invalid: {path}")
    return dict(record), [dict(row) for row in rows]


def _write_fold_progress(
    path: Path,
    *,
    fold_signature: str,
    fold_record: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_json(
        path,
        {
            "evaluation_rows": [dict(row) for row in evaluation_rows],
            "fold_record": dict(fold_record),
            "fold_signature": fold_signature,
            "schema_version": FOLD_PROGRESS_SCHEMA,
        },
    )


def run_potency_ordinal_crossfit(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    allocated_device: str,
    progress_dir: Path,
    progress_commit: Callable[[], None],
    resume: bool,
) -> dict[str, Any]:
    """Fit paired real/shuffled ordinal adapters and qualify direct percentile control."""

    if torch is None:
        raise UgiPotencyOrdinalAdapterError("ordinal potency adapter training requires torch")
    config = read_json_object(
        config_path,
        error=UgiPotencyOrdinalAdapterError,
        label="ordinal potency adapter config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or profile not in {"smoke", "full"}:
        raise UgiPotencyOrdinalAdapterError("unsupported ordinal adapter config or profile")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise UgiPotencyOrdinalAdapterError("ordinal adapter authorization is missing")
    if profile == "full" and authorization.get("adapter_fit_execution_authorized") is not True:
        raise UgiPotencyOrdinalAdapterError("full ordinal adapter fitting is not authorized")
    if authorization.get("nonzero_generation_authorized") is not False:
        raise UgiPotencyOrdinalAdapterError("ordinal cross-fit must not authorize generation")
    runtime = config[profile]
    if (
        runtime["device"] != allocated_device
        or runtime["precision"] != "float32"
        or runtime["deterministic_algorithms"] is not True
    ):
        raise UgiPotencyOrdinalAdapterError("ordinal adapter runtime differs from its contract")
    anchors = tuple(float(value) for value in config["ordinal"]["anchors"])
    if anchors != (0.1, 0.5, 0.9):
        raise UgiPotencyOrdinalAdapterError("v2 ordinal anchors must be q10/q50/q90")
    paired_contract = config["ordinal"].get("paired_control")
    expected_paired_contract = {
        "identical_initialization": True,
        "identical_labelled_molecule_batches": True,
        "identical_retention_batches": True,
        "identical_flow_times_and_corruptions": True,
        "only_difference": "real versus deterministically permuted target percentile",
    }
    if paired_contract != expected_paired_contract:
        raise UgiPotencyOrdinalAdapterError("paired real/shuffled contract changed")
    data_scope = config.get("data_scope")
    if not isinstance(data_scope, Mapping) or (
        int(data_scope.get("observations", -1)),
        int(data_scope.get("independent_heads", -1)),
        int(data_scope.get("independent_aldehyde_isocyanide_pairs", -1)),
        int(data_scope.get("new_measured_head_or_tail_chemotypes", -1)),
    ) != (1100, 20, 55, 0):
        raise UgiPotencyOrdinalAdapterError("measured-data scope changed")
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiPotencyOrdinalAdapterError("CUDA ordinal fitting requested but unavailable")
    _set_determinism(int(config["seed"]), int(runtime["cpu_threads"]), device)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    with SynthesisProgramProductionCache(paths["production_cache"]) as cache:
        package, member_name, member_sha256 = _load_base_package(
            paths["base_checkpoint_archive"],
            paths["base_training_result"],
            paths["base_design"],
            paths["production_cache"],
            arm_id=str(config["base"]["arm_id"]),
            checkpoint_step=int(config["base"]["checkpoint_step"]),
            device=device,
        )
        if package.get("schema_version") != BASE_CHECKPOINT_SCHEMA:
            raise UgiPotencyOrdinalAdapterError("base checkpoint schema changed")
        joined, splits = _joined_observations(
            paths["potency_observations"],
            paths["ugi_assignments"],
            paths["oracle_split_assignments"],
            cache,
        )
        observed_scope = factorial_dataset_summary(
            [
                {
                    "source_lipid_name": str(row["source_lipid_name"]),
                    "model_smiles": str(row["model_smiles"]),
                    "label_value": str(row["potency"]),
                }
                for row in joined
            ]
        )
        if (
            int(observed_scope["rows"]),
            int(observed_scope["unique_heads"]),
            int(observed_scope["unique_aldehyde_isocyanide_pairs"]),
        ) != (1100, 20, 55) or observed_scope["complete_cartesian_library"] is not True:
            raise UgiPotencyOrdinalAdapterError("observed measured-data support changed")
        fit_policy = _policy(
            config,
            [row["interval"] for row in config["validation"]["time_bins"]],
        )
        schemes = list(config["validation"]["schemes"])
        folds = list(range(5))
        if profile == "smoke":
            schemes = schemes[:1]
            folds = folds[:1]
        fold_records: list[dict[str, Any]] = []
        evaluation_rows: list[dict[str, Any]] = []
        for scheme_index, scheme in enumerate(schemes):
            for fold in folds:
                train_rows: list[dict[str, Any]] = []
                test_rows: list[dict[str, Any]] = []
                test_clusters: list[str] = []
                for row in joined:
                    split = splits[(scheme, fold, row["label"])]
                    if split[0] == "train":
                        train_rows.append(row)
                    elif split[0] == "test":
                        test_rows.append(dict(row))
                        test_clusters.append(split[1])
                if not train_rows or not test_rows:
                    raise UgiPotencyOrdinalAdapterError("component-disjoint fold is empty")
                local_seed = int(config["seed"]) + 10_000 * scheme_index + 100 * fold
                fold_signature = _fold_progress_signature(
                    scheme=scheme,
                    fold=fold,
                    local_seed=local_seed,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    time_bins=config["validation"]["time_bins"],
                )
                progress_path = progress_dir / f"{scheme}__fold_{fold}.json"
                if progress_path.exists():
                    if not resume:
                        raise UgiPotencyOrdinalAdapterError(
                            f"unexpected ordinal fold progress without resume: {progress_path}"
                        )
                    restored_record, restored_rows = _load_fold_progress(
                        progress_path,
                        expected_signature=fold_signature,
                        expected_test_rows=len(test_rows),
                        expected_evaluation_rows=(
                            len(test_rows) * len(config["validation"]["time_bins"])
                        ),
                    )
                    fold_records.append(restored_record)
                    evaluation_rows.extend(restored_rows)
                    continue
                train_values = np.asarray([float(row["potency"]) for row in train_rows])
                real_quantiles = training_ecdf_quantiles(train_values)
                permutation = deterministic_permutation(
                    len(train_rows),
                    seed=int(config["seed"]) + fold,
                    namespace=scheme,
                )
                shuffled_quantiles = real_quantiles[permutation]
                held_quantiles = map_to_training_ecdf(
                    train_values,
                    [float(row["potency"]) for row in test_rows],
                )
                for row, quantile in zip(test_rows, held_quantiles, strict=True):
                    row["held_quantile"] = float(quantile)
                real_model, shuffled_model, traces, pairing = _fit_paired_adapters(
                    package=package,
                    cache=cache,
                    config=config,
                    train_rows=train_rows,
                    real_quantiles=real_quantiles,
                    shuffled_quantiles=shuffled_quantiles,
                    policy=fit_policy,
                    device=device,
                    seed=local_seed,
                    steps=int(runtime["optimizer_steps"]),
                    labelled_batch_size=int(runtime["labelled_batch_size"]),
                    retention_batch_size=int(runtime["retention_batch_size"]),
                )
                rows = _evaluate_paired_adapters(
                    real_model=real_model,
                    shuffled_model=shuffled_model,
                    package=package,
                    cache=cache,
                    rows=test_rows,
                    clusters=test_clusters,
                    policy=fit_policy,
                    anchors=anchors,
                    times=[
                        (str(row["name"]), float(row["evaluation_time"]))
                        for row in config["validation"]["time_bins"]
                    ],
                    batch_size=int(runtime["evaluation_batch_size"]),
                    seed=local_seed,
                    device=device,
                )
                train_features = np.stack(
                    [
                        _morphology_features(
                            cache.record(int(row["cache_index"])),
                            role_count=len(cache.vocabulary.role_states),
                        )
                        for row in train_rows
                    ]
                )
                test_features = np.stack(
                    [
                        _morphology_features(
                            cache.record(int(row["cache_index"])),
                            role_count=len(cache.vocabulary.role_states),
                        )
                        for row in test_rows
                    ]
                )
                residuals = morphology_residuals(
                    train_features,
                    train_values,
                    test_features,
                    [float(row["potency"]) for row in test_rows],
                )
                residual_by_label = {
                    row["label"]: float(value)
                    for row, value in zip(test_rows, residuals, strict=True)
                }
                for row in rows:
                    row.update(
                        {
                            "scheme": scheme,
                            "fold": fold,
                            "residual_potency": residual_by_label[row["label"]],
                            "high_potency": int(float(row["held_quantile"]) >= 0.75),
                        }
                    )
                fold_record = {
                    "scheme": scheme,
                    "fold": fold,
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "real_losses": traces["real"],
                    "shuffled_losses": traces["shuffled"],
                    "paired_control": pairing,
                }
                _write_fold_progress(
                    progress_path,
                    fold_signature=fold_signature,
                    fold_record=fold_record,
                    evaluation_rows=rows,
                )
                progress_commit()
                fold_records.append(fold_record)
                evaluation_rows.extend(rows)

        bootstrap = config["validation"]["clustered_bootstrap"]
        signal_thresholds = config["validation"]["signal_gates"]
        monotonic_thresholds = config["validation"]["monotonicity_gates"]
        bin_metrics: dict[str, Any] = {}
        active_by_scheme: dict[str, set[str]] = {}
        for scheme_index, scheme in enumerate(schemes):
            active_by_scheme[scheme] = set()
            for time_index, time_bin in enumerate(config["validation"]["time_bins"]):
                name = str(time_bin["name"])
                local = [
                    row
                    for row in evaluation_rows
                    if row["scheme"] == scheme and row["time_bin"] == name
                ]
                seed = int(bootstrap["seed"]) + 100 * scheme_index + time_index
                monotonic, monotonic_gate, real_score, shuffled_score = _monotonic_metrics(
                    local,
                    anchors=anchors,
                    bootstrap=bootstrap,
                    seed=seed + 30_000,
                    thresholds=monotonic_thresholds,
                )
                truth = [int(row["high_potency"]) for row in local]
                residual = [float(row["residual_potency"]) for row in local]
                clusters = [f"{row['fold']}::{row['cluster_id']}" for row in local]
                real_auc = roc_auc(truth, real_score)
                shuffled_auc = roc_auc(truth, shuffled_score)
                real_rho = spearman(residual, real_score)
                auc_interval = clustered_bootstrap_interval(
                    real_score,
                    clusters,
                    statistic="auroc",
                    labels=truth,
                    replicates=int(bootstrap["replicates"]),
                    seed=seed,
                )
                rho_interval = clustered_bootstrap_interval(
                    real_score,
                    clusters,
                    statistic="spearman",
                    labels=residual,
                    replicates=int(bootstrap["replicates"]),
                    seed=seed + 10_000,
                )
                auc_difference = clustered_auroc_difference_interval(
                    truth,
                    real_score,
                    shuffled_score,
                    clusters,
                    replicates=int(bootstrap["replicates"]),
                    seed=seed + 20_000,
                )
                signal = signal_gate(
                    real_auroc=real_auc,
                    real_auroc_lower=float(auc_interval["lower_95"]),
                    residual_spearman=real_rho,
                    residual_spearman_lower=float(rho_interval["lower_95"]),
                    shuffled_auroc=shuffled_auc,
                    shuffled_difference_lower=float(auc_difference["lower_95"]),
                    thresholds=signal_thresholds,
                )
                combined = bool(signal["passes"] and monotonic_gate["passes"])
                if combined:
                    active_by_scheme[scheme].add(name)
                bin_metrics[f"{scheme}::{name}"] = {
                    "rows": len(local),
                    "real_direction_auroc": real_auc,
                    "shuffled_direction_auroc": shuffled_auc,
                    "direction_residual_spearman": real_rho,
                    "real_direction_auroc_bootstrap": auc_interval,
                    "direction_residual_spearman_bootstrap": rho_interval,
                    "real_minus_shuffled_direction_auroc_bootstrap": auc_difference,
                    "monotonicity": monotonic,
                    "signal_gates": signal,
                    "monotonicity_gates": monotonic_gate,
                    "passes": combined,
                }
        active_bins = sorted(set.intersection(*active_by_scheme.values()))
        eligible = {str(value) for value in config["validation"]["eligible_time_bins"]}
        active_bins = [name for name in active_bins if name in eligible]
        signal_pass = profile == "full" and bool(active_bins)

        checkpoints: dict[str, Any] = {}
        if signal_pass:
            intervals = {
                str(row["name"]): row["interval"] for row in config["validation"]["time_bins"]
            }
            final_policy = _policy(config, [intervals[name] for name in active_bins])
            values = np.asarray([float(row["potency"]) for row in joined])
            real_quantiles = training_ecdf_quantiles(values)
            shuffled_quantiles = real_quantiles[
                deterministic_permutation(len(joined), seed=int(config["seed"]), namespace="final")
            ]
            real_model, shuffled_model, traces, pairing = _fit_paired_adapters(
                package=package,
                cache=cache,
                config=config,
                train_rows=joined,
                real_quantiles=real_quantiles,
                shuffled_quantiles=shuffled_quantiles,
                policy=final_policy,
                device=device,
                seed=int(config["seed"]),
                steps=int(runtime["optimizer_steps"]),
                labelled_batch_size=int(runtime["labelled_batch_size"]),
                retention_batch_size=int(runtime["retention_batch_size"]),
            )
            for arm, model in (("potency_q90", real_model), ("shuffled_label_q90", shuffled_model)):
                checkpoint_path = output_dir / f"{arm}_ordinal_adapter.pt"
                package_out = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "trusted_local_checkpoint": True,
                    "arm": arm,
                    "seed": int(config["seed"]),
                    "base": {
                        "archive_sha256": str(sha256_file(paths["base_checkpoint_archive"])),
                        "member_name": member_name,
                        "member_sha256": member_sha256,
                        "model_state_sha256": package["model_state_sha256"],
                    },
                    "production_cache_sha256": str(sha256_file(paths["production_cache"])),
                    "model_config": {
                        **dict(package["model_config"]),
                        "potency_adapter_dim": int(config["adapter"]["bottleneck_dim"]),
                        "potency_condition_dim": int(config["adapter"]["condition_dim"]),
                    },
                    "potency_policy": final_policy.to_mapping(),
                    "ordinal_contract": dict(config["ordinal"]),
                    "paired_control": pairing,
                    "adapter_state": potency_adapter_state_dict(model),
                    "combined_model_state_sha256": _model_state_sha256(model),
                    "parameter_report": potency_parameter_report(model),
                    "losses": traces["real" if arm == "potency_q90" else "shuffled"],
                }
                atomic_torch_save(checkpoint_path, package_out)
                checkpoints[arm] = artifact_record(checkpoint_path)

    ledger_path = output_dir / "evaluation_scores.csv.gz"
    ledger_fields = [
        "schema_version",
        "scheme",
        "fold",
        "label",
        "cluster_id",
        "time_bin",
        "potency",
        "held_quantile",
        "residual_potency",
        "high_potency",
        "real_q10_nll",
        "real_q50_nll",
        "real_q90_nll",
        "shuffled_q10_nll",
        "shuffled_q50_nll",
        "shuffled_q90_nll",
    ]
    write_csv(
        ledger_path,
        sorted(
            evaluation_rows,
            key=lambda row: (
                str(row["scheme"]),
                int(row["fold"]),
                str(row["time_bin"]),
                str(row["label"]),
            ),
        ),
        ledger_fields,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "signal_gate_pass"
            if signal_pass
            else "smoke_complete"
            if profile == "smoke"
            else "signal_gate_fail"
        ),
        "profile": profile,
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in sorted(paths.items())},
        "base_checkpoint": {
            "member_name": member_name,
            "member_sha256": member_sha256,
            "model_state_sha256": package["model_state_sha256"],
        },
        "observations": 1100,
        "data_scope": {
            **dict(config["data_scope"]),
            "observed_complete_cartesian_library": observed_scope["complete_cartesian_library"],
            "observed_unique_aldehydes": observed_scope["unique_aldehydes"],
            "observed_unique_isocyanides": observed_scope["unique_isocyanides"],
        },
        "folds": fold_records,
        "time_bin_metrics": bin_metrics,
        "active_time_bins": active_bins,
        "evaluation_scores": artifact_record(ledger_path),
        "checkpoints": checkpoints,
        "gates": {
            "component_disjoint_schemes_complete": profile == "smoke" or len(fold_records) == 10,
            "paired_control_invariants_recorded": all(
                all(
                    bool(value)
                    for key, value in row["paired_control"].items()
                    if key != "only_difference" and not key.endswith("sha256")
                )
                for row in fold_records
            ),
            "at_least_one_mid_or_late_bin_passes_both_schemes": signal_pass,
            "nonzero_generation_remains_unauthorized": True,
        },
        "nonclaims": [
            "This cross-fit does not show that guided generation improves potency.",
            "No new measured head or tail chemotypes were created.",
            "No synthesis, proposal-engine, candidate-selection or generation call was run.",
        ],
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    _deterministic_tar(
        [
            result_path,
            ledger_path,
            *(output_dir / f"{arm}_ordinal_adapter.pt" for arm in sorted(checkpoints)),
        ],
        output_dir / "adapter_bundle.tar",
        base=output_dir,
    )
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "FOLD_PROGRESS_SCHEMA",
    "LEDGER_SCHEMA",
    "RESULT_SCHEMA",
    "UgiPotencyOrdinalAdapterError",
    "run_potency_ordinal_crossfit",
]
