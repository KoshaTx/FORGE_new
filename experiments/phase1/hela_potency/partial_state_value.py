"""Direct partial-state mTP prediction over a frozen FORGE Transformer.

This module is deliberately diagnostic.  It cross-fits a small graph-level scalar head, compares
it with an identically initialized shuffled-label head, and never runs guided generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.hela_potency.potency_adapter import (
    BASE_CHECKPOINT_SCHEMA,
    _clean_batch,
    _deterministic_tar,
    _joined_observations,
    _load_base_package,
    _morphology_features,
    _set_determinism,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_csv, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.partial_state_value import (
    RoleAwarePartialStateValueHead,
    direct_value_loss,
    role_aware_graph_summary,
)
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
)
from forge.model.training_restart import atomic_torch_save
from forge.potency.adapter_failure_attribution import factorial_dataset_summary
from forge.potency.adapter_guidance import (
    clustered_auroc_difference_interval,
    clustered_bootstrap_interval,
    deterministic_permutation,
    map_to_training_ecdf,
    morphology_residuals,
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


CONFIG_SCHEMA = "forge.ugi_hela_partial_state_value_config.v1"
RESULT_SCHEMA = "forge.ugi_hela_partial_state_value_crossfit_result.v1"
LEDGER_SCHEMA = "forge.ugi_hela_partial_state_value_score.v1"
CHECKPOINT_SCHEMA = "forge.ugi_hela_partial_state_value_checkpoint.v1"
FOLD_PROGRESS_SCHEMA = "forge.ugi_hela_partial_state_value_fold_progress.v1"


class UgiPartialStateValueError(ValueError):
    """The bounded direct partial-state value diagnostic violated its contract."""


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
        error=UgiPartialStateValueError,
        label="partial-state value fold progress",
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
        raise UgiPartialStateValueError(f"partial-state fold progress is invalid: {path}")
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


def _frozen_base_model(
    package: Mapping[str, Any], cache: SynthesisProgramProductionCache, device: Any
) -> Any:
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=dict(package["model_config"]),
        device=device,
    )
    model.load_state_dict(package["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if _model_state_sha256(model) != package["model_state_sha256"]:
        raise UgiPartialStateValueError("frozen base model state changed during reconstruction")
    return model


def _extract_feature_bank(
    *,
    model: Any,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    rows: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    replicas: int,
    batch_size: int,
    seed: int,
    device: Any,
) -> Any:
    """Encode deterministic corruptions once; all cross-fit heads reuse the frozen features."""

    if replicas < 1 or batch_size < 1 or not times:
        raise UgiPartialStateValueError("partial-state feature-bank geometry is empty")
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    role_count = len(cache.vocabulary.role_states)
    bank = None
    for time_index, time_value in enumerate(times):
        if not 0.0 < time_value < 1.0:
            raise UgiPartialStateValueError("partial-state evaluation times must be interior")
        for replica in range(replicas):
            for offset in range(0, len(rows), batch_size):
                local = rows[offset : offset + batch_size]
                clean = _clean_batch(
                    cache,
                    [int(row["cache_index"]) for row in local],
                    model,
                    device,
                )
                t = torch.full((len(local),), float(time_value), dtype=torch.float32, device=device)
                generator = torch.Generator(device=device).manual_seed(
                    seed + 10_000_000 * replica + 100_000 * time_index + offset
                )
                noisy = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, generator)
                with torch.inference_mode():
                    predictions = _synthesis_program_predict(
                        model,
                        clean,
                        noisy,
                        t,
                        return_hidden_state=True,
                    )
                    summary = role_aware_graph_summary(
                        predictions["hidden_state"],
                        clean["node_mask"],
                        clean["role_states"],
                        t,
                        role_count=role_count,
                    ).to(device="cpu", dtype=torch.float32)
                if bank is None:
                    bank = torch.empty(
                        (len(rows), len(times), replicas, summary.shape[1]),
                        dtype=torch.float32,
                    )
                bank[offset : offset + len(local), time_index, replica] = summary
    if bank is None or not bool(torch.isfinite(bank).all()):
        raise UgiPartialStateValueError("partial-state feature bank is incomplete or non-finite")
    return bank


def _head(
    *, hidden_dim: int, role_count: int, value_hidden_dim: int, seed: int, device: Any
) -> Any:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return RoleAwarePartialStateValueHead(
        hidden_dim=hidden_dim,
        role_count=role_count,
        value_hidden_dim=value_hidden_dim,
    ).to(device)


def _fit_paired_heads(
    *,
    feature_bank: Any,
    row_indices: np.ndarray,
    values: np.ndarray,
    node_counts: np.ndarray,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    hidden_dim: int,
    role_count: int,
    seed: int,
    namespace: str,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Fit real, shuffled, and structural-sanity heads on identical partial states."""

    value_hidden_dim = int(config["value_head"]["hidden_dim"])
    real = _head(
        hidden_dim=hidden_dim,
        role_count=role_count,
        value_hidden_dim=value_hidden_dim,
        seed=seed,
        device=device,
    )
    shuffled = _head(
        hidden_dim=hidden_dim,
        role_count=role_count,
        value_hidden_dim=value_hidden_dim,
        seed=seed,
        device=device,
    )
    structural = _head(
        hidden_dim=hidden_dim,
        role_count=role_count,
        value_hidden_dim=value_hidden_dim,
        seed=seed + 1,
        device=device,
    )
    for key, value in real.state_dict().items():
        if not torch.equal(value, shuffled.state_dict()[key]):
            raise UgiPartialStateValueError(
                "real and shuffled value heads differ at initialization"
            )

    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    node_mean = float(node_counts.mean())
    node_scale = float(node_counts.std(ddof=0))
    if scale <= 0.0 or node_scale <= 0.0:
        raise UgiPartialStateValueError("training-fold target variance is zero")
    standardized = (values - mean) / scale
    standardized_nodes = (node_counts - node_mean) / node_scale
    quantiles = training_ecdf_quantiles(values)
    permutation = deterministic_permutation(len(values), seed=seed, namespace=namespace)
    shuffled_targets = standardized[permutation]

    heads = {"real": real, "shuffled": shuffled, "structural": structural}
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        for name, head in heads.items()
    }
    rng = np.random.default_rng(seed)
    steps = int(runtime["optimizer_steps"])
    batch_size = int(runtime["labelled_batch_size"])
    traces: dict[str, list[dict[str, float]]] = {name: [] for name in heads}
    for step in range(1, steps + 1):
        local_indices = quartile_balanced_sample(quantiles, size=batch_size, rng=rng)
        time_indices = rng.integers(0, feature_bank.shape[1], size=batch_size)
        replica_indices = rng.integers(0, feature_bank.shape[2], size=batch_size)
        global_indices = row_indices[local_indices]
        features = feature_bank[
            torch.as_tensor(global_indices),
            torch.as_tensor(time_indices),
            torch.as_tensor(replica_indices),
        ].to(device)
        targets = {
            "real": standardized[local_indices],
            "shuffled": shuffled_targets[local_indices],
            "structural": standardized_nodes[local_indices],
        }
        for name, head in heads.items():
            prediction = head.forward_summary(features)
            target = torch.as_tensor(targets[name], dtype=torch.float32, device=device)
            rank_weight = 0.0 if name == "structural" else float(config["training"]["rank_weight"])
            loss, parts = direct_value_loss(
                prediction,
                target,
                rank_weight=rank_weight,
                huber_delta=float(config["training"]["huber_delta"]),
            )
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizers[name].step()
            if step == 1 or step == steps or step % max(1, steps // 10) == 0:
                traces[name].append(
                    {
                        "step": float(step),
                        "loss": float(loss.detach().cpu()),
                        "regression": float(parts["regression"].detach().cpu()),
                        "ranking": float(parts["ranking"].detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                    }
                )
    return (
        real.eval(),
        shuffled.eval(),
        structural.eval(),
        {
            "traces": traces,
            "target_normalization": {"mean": mean, "scale": scale},
            "node_count_normalization": {"mean": node_mean, "scale": node_scale},
            "paired_control": {
                "identical_initialization": True,
                "identical_partial_state_batches": True,
                "only_difference": "real versus deterministically permuted mTP label",
                "permutation_sha256": str(sha256_json(permutation.tolist())),
            },
        },
    )


def _score_fold(
    *,
    heads: tuple[Any, Any, Any],
    feature_bank: Any,
    test_rows: Sequence[Mapping[str, Any]],
    test_indices: np.ndarray,
    test_clusters: Sequence[str],
    held_quantiles: np.ndarray,
    time_names: Sequence[str],
    fit: Mapping[str, Any],
    node_counts: np.ndarray,
    scheme: str,
    fold: int,
    device: Any,
) -> list[dict[str, Any]]:
    real, shuffled, structural = heads
    normalization = fit["target_normalization"]
    node_normalization = fit["node_count_normalization"]
    output: list[dict[str, Any]] = []
    for time_index, time_name in enumerate(time_names):
        features = feature_bank[torch.as_tensor(test_indices), time_index, 0].to(device)
        with torch.inference_mode():
            real_score = real.forward_summary(features).cpu().numpy()
            shuffled_score = shuffled.forward_summary(features).cpu().numpy()
            structural_score = structural.forward_summary(features).cpu().numpy()
        real_raw = real_score * float(normalization["scale"]) + float(normalization["mean"])
        shuffled_raw = shuffled_score * float(normalization["scale"]) + float(normalization["mean"])
        node_raw = structural_score * float(node_normalization["scale"]) + float(
            node_normalization["mean"]
        )
        for index, row in enumerate(test_rows):
            output.append(
                {
                    "schema_version": LEDGER_SCHEMA,
                    "scheme": scheme,
                    "fold": fold,
                    "label": str(row["label"]),
                    "cluster_id": str(test_clusters[index]),
                    "time_bin": str(time_name),
                    "potency": float(row["potency"]),
                    "held_quantile": float(held_quantiles[index]),
                    "high_potency": int(float(held_quantiles[index]) >= 0.75),
                    "real_prediction": float(real_raw[index]),
                    "shuffled_prediction": float(shuffled_raw[index]),
                    "node_count": float(node_counts[index]),
                    "node_count_prediction": float(node_raw[index]),
                }
            )
    return output


def _bin_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    bootstrap = config["validation"]["clustered_bootstrap"]
    thresholds = config["validation"]["signal_gates"]
    metrics: dict[str, Any] = {}
    active: dict[str, set[str]] = {}
    schemes = list(config["validation"]["schemes"])
    for scheme_index, scheme in enumerate(schemes):
        active[scheme] = set()
        for time_index, time_bin in enumerate(config["validation"]["time_bins"]):
            name = str(time_bin["name"])
            local = [row for row in rows if row["scheme"] == scheme and row["time_bin"] == name]
            if not local:
                continue
            truth = [int(row["high_potency"]) for row in local]
            residual = [float(row["residual_potency"]) for row in local]
            real = [float(row["real_prediction"]) for row in local]
            shuffled = [float(row["shuffled_prediction"]) for row in local]
            clusters = [f"{row['fold']}::{row['cluster_id']}" for row in local]
            seed = int(bootstrap["seed"]) + 100 * scheme_index + time_index
            real_auc = roc_auc(truth, real)
            shuffled_auc = roc_auc(truth, shuffled)
            residual_rho = spearman(residual, real)
            node_rho = spearman(
                [float(row["node_count"]) for row in local],
                [float(row["node_count_prediction"]) for row in local],
            )
            auc_interval = clustered_bootstrap_interval(
                real,
                clusters,
                statistic="auroc",
                labels=truth,
                replicates=int(bootstrap["replicates"]),
                seed=seed,
            )
            rho_interval = clustered_bootstrap_interval(
                real,
                clusters,
                statistic="spearman",
                labels=residual,
                replicates=int(bootstrap["replicates"]),
                seed=seed + 10_000,
            )
            difference = clustered_auroc_difference_interval(
                truth,
                real,
                shuffled,
                clusters,
                replicates=int(bootstrap["replicates"]),
                seed=seed + 20_000,
            )
            gates = signal_gate(
                real_auroc=real_auc,
                real_auroc_lower=float(auc_interval["lower_95"]),
                residual_spearman=residual_rho,
                residual_spearman_lower=float(rho_interval["lower_95"]),
                shuffled_auroc=shuffled_auc,
                shuffled_difference_lower=float(difference["lower_95"]),
                thresholds=thresholds,
            )
            sanity_pass = node_rho >= float(
                config["validation"]["minimum_node_count_sanity_spearman"]
            )
            passed = bool(gates["passes"] and sanity_pass)
            if passed:
                active[scheme].add(name)
            errors = np.asarray(real) - np.asarray([float(row["potency"]) for row in local])
            metrics[f"{scheme}::{name}"] = {
                "rows": len(local),
                "real_auroc": real_auc,
                "shuffled_auroc": shuffled_auc,
                "residual_spearman": residual_rho,
                "raw_spearman": spearman([float(row["potency"]) for row in local], real),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "node_count_sanity_spearman": node_rho,
                "real_auroc_bootstrap": auc_interval,
                "residual_spearman_bootstrap": rho_interval,
                "real_minus_shuffled_auroc_bootstrap": difference,
                "signal_gates": gates,
                "node_count_sanity_pass": sanity_pass,
                "passes": passed,
            }
    return metrics, active


def run_partial_state_value_crossfit(
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
    """Cross-fit a direct scalar value head; never sample or invoke an oracle."""

    if torch is None:
        raise UgiPartialStateValueError("partial-state value fitting requires torch")
    config = read_json_object(
        config_path,
        error=UgiPartialStateValueError,
        label="partial-state value config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or profile not in {"smoke", "full"}:
        raise UgiPartialStateValueError("unsupported partial-state value config or profile")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise UgiPartialStateValueError("partial-state value authorization is missing")
    if profile == "full" and authorization.get("value_fit_execution_authorized") is not True:
        raise UgiPartialStateValueError("full partial-state value fitting is not authorized")
    if authorization.get("nonzero_generation_authorized") is not False:
        raise UgiPartialStateValueError("partial-state value fitting must not authorize generation")
    runtime = config[profile]
    if (
        runtime["device"] != allocated_device
        or runtime["precision"] != "float32"
        or runtime["deterministic_algorithms"] is not True
    ):
        raise UgiPartialStateValueError("partial-state value runtime differs from its contract")
    paired = config["training"].get("paired_control")
    if paired != {
        "identical_initialization": True,
        "identical_partial_state_batches": True,
        "only_difference": "real versus deterministically permuted mTP label",
    }:
        raise UgiPartialStateValueError("paired real/shuffled value-head contract changed")
    data_scope = config.get("data_scope")
    if not isinstance(data_scope, Mapping) or (
        int(data_scope.get("observations", -1)),
        int(data_scope.get("independent_heads", -1)),
        int(data_scope.get("independent_aldehyde_isocyanide_pairs", -1)),
        int(data_scope.get("new_measured_head_or_tail_chemotypes", -1)),
    ) != (1100, 20, 55, 0):
        raise UgiPartialStateValueError("measured-data scope changed")

    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiPartialStateValueError("CUDA partial-state fitting requested but unavailable")
    _set_determinism(int(config["seed"]), int(runtime["cpu_threads"]), device)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
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
            raise UgiPartialStateValueError("base checkpoint schema changed")
        model = _frozen_base_model(package, cache, device)
        base_state_before = _model_state_sha256(model)
        joined, splits = _joined_observations(
            paths["potency_observations"],
            paths["ugi_assignments"],
            paths["oracle_split_assignments"],
            cache,
        )
        for index, row in enumerate(joined):
            row["row_index"] = index
        scope = factorial_dataset_summary(
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
            int(scope["rows"]),
            int(scope["unique_heads"]),
            int(scope["unique_aldehyde_isocyanide_pairs"]),
        ) != (1100, 20, 55):
            raise UgiPartialStateValueError("observed measured-data support changed")
        times = [float(row["evaluation_time"]) for row in config["validation"]["time_bins"]]
        train_bank = _extract_feature_bank(
            model=model,
            package=package,
            cache=cache,
            rows=joined,
            times=times,
            replicas=int(runtime["training_corruption_replicas"]),
            batch_size=int(runtime["feature_batch_size"]),
            seed=int(config["seed"]) + 1_000_000,
            device=device,
        )
        evaluation_bank = _extract_feature_bank(
            model=model,
            package=package,
            cache=cache,
            rows=joined,
            times=times,
            replicas=1,
            batch_size=int(runtime["feature_batch_size"]),
            seed=int(config["seed"]) + 2_000_000,
            device=device,
        )
        if _model_state_sha256(model) != base_state_before:
            raise UgiPartialStateValueError("feature extraction mutated the frozen generator")
        hidden_dim = int(package["model_config"]["hidden_dim"])
        role_count = len(cache.vocabulary.role_states)
        schemes = list(config["validation"]["schemes"])
        folds = list(range(5))
        if profile == "smoke":
            schemes = schemes[:1]
            folds = folds[:1]
        fold_records: list[dict[str, Any]] = []
        evaluation_rows: list[dict[str, Any]] = []
        for scheme_index, scheme in enumerate(schemes):
            for fold in folds:
                train_rows = [
                    row for row in joined if splits[(scheme, fold, row["label"])][0] == "train"
                ]
                test_rows = [
                    row for row in joined if splits[(scheme, fold, row["label"])][0] == "test"
                ]
                if not train_rows or not test_rows:
                    raise UgiPartialStateValueError("component-disjoint fold is empty")
                test_clusters = [splits[(scheme, fold, row["label"])][1] for row in test_rows]
                train_indices = np.asarray(
                    [int(row["row_index"]) for row in train_rows], dtype=np.int64
                )
                test_indices = np.asarray(
                    [int(row["row_index"]) for row in test_rows], dtype=np.int64
                )
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
                        raise UgiPartialStateValueError(
                            f"unexpected partial-state fold progress without resume: {progress_path}"
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
                train_values = np.asarray(
                    [float(row["potency"]) for row in train_rows], dtype=np.float64
                )
                train_node_counts = np.asarray(
                    [float(cache.record(int(row["cache_index"])).node_count) for row in train_rows],
                    dtype=np.float64,
                )
                test_node_counts = np.asarray(
                    [float(cache.record(int(row["cache_index"])).node_count) for row in test_rows],
                    dtype=np.float64,
                )
                real, shuffled, structural, fit = _fit_paired_heads(
                    feature_bank=train_bank,
                    row_indices=train_indices,
                    values=train_values,
                    node_counts=train_node_counts,
                    config=config,
                    runtime=runtime,
                    hidden_dim=hidden_dim,
                    role_count=role_count,
                    seed=local_seed,
                    namespace=f"{scheme}:{fold}",
                    device=device,
                )
                held_quantiles = map_to_training_ecdf(
                    train_values, [float(row["potency"]) for row in test_rows]
                )
                rows = _score_fold(
                    heads=(real, shuffled, structural),
                    feature_bank=evaluation_bank,
                    test_rows=test_rows,
                    test_indices=test_indices,
                    test_clusters=test_clusters,
                    held_quantiles=held_quantiles,
                    time_names=[str(row["name"]) for row in config["validation"]["time_bins"]],
                    fit=fit,
                    node_counts=test_node_counts,
                    scheme=scheme,
                    fold=fold,
                    device=device,
                )
                train_morphology = np.stack(
                    [
                        _morphology_features(
                            cache.record(int(row["cache_index"])), role_count=role_count
                        )
                        for row in train_rows
                    ]
                )
                test_morphology = np.stack(
                    [
                        _morphology_features(
                            cache.record(int(row["cache_index"])), role_count=role_count
                        )
                        for row in test_rows
                    ]
                )
                residuals = morphology_residuals(
                    train_morphology,
                    train_values,
                    test_morphology,
                    [float(row["potency"]) for row in test_rows],
                )
                residual_by_label = {
                    str(row["label"]): float(value)
                    for row, value in zip(test_rows, residuals, strict=True)
                }
                for row in rows:
                    row["residual_potency"] = residual_by_label[str(row["label"])]
                fold_record = {
                    "scheme": scheme,
                    "fold": fold,
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    **fit,
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

        bin_metrics, active_by_scheme = _bin_metrics(evaluation_rows, config=config)
        eligible = {str(value) for value in config["validation"]["eligible_time_bins"]}
        active_bins = (
            sorted(set.intersection(*active_by_scheme.values())) if active_by_scheme else []
        )
        active_bins = [name for name in active_bins if name in eligible]
        signal_pass = profile == "full" and bool(active_bins)

        checkpoint_records: dict[str, Any] = {}
        if signal_pass:
            all_indices = np.arange(len(joined), dtype=np.int64)
            all_values = np.asarray([float(row["potency"]) for row in joined])
            all_node_counts = np.asarray(
                [float(cache.record(int(row["cache_index"])).node_count) for row in joined]
            )
            real, shuffled, _, fit = _fit_paired_heads(
                feature_bank=train_bank,
                row_indices=all_indices,
                values=all_values,
                node_counts=all_node_counts,
                config=config,
                runtime=runtime,
                hidden_dim=hidden_dim,
                role_count=role_count,
                seed=int(config["seed"]),
                namespace="final",
                device=device,
            )
            for arm, head in (("real_mtp", real), ("shuffled_mtp", shuffled)):
                checkpoint_path = output_dir / f"{arm}_partial_state_value.pt"
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "trusted_local_checkpoint": True,
                    "arm": arm,
                    "base": {
                        "archive_sha256": str(sha256_file(paths["base_checkpoint_archive"])),
                        "member_name": member_name,
                        "member_sha256": member_sha256,
                        "model_state_sha256": package["model_state_sha256"],
                    },
                    "head_config": dict(config["value_head"]),
                    "role_count": role_count,
                    "hidden_dim": hidden_dim,
                    "active_time_bins": active_bins,
                    "time_bins": list(config["validation"]["time_bins"]),
                    "target_normalization": fit["target_normalization"],
                    "head_state": {
                        key: value.detach().cpu().clone()
                        for key, value in sorted(head.state_dict().items())
                    },
                    "paired_control": fit["paired_control"],
                    "guided_generation_authorized": False,
                }
                atomic_torch_save(checkpoint_path, checkpoint)
                checkpoint_paths.append(checkpoint_path)
                checkpoint_records[arm] = artifact_record(checkpoint_path)

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
        "real_prediction",
        "shuffled_prediction",
        "node_count",
        "node_count_prediction",
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
    bundle_path = output_dir / "value_head_bundle.tar"
    _deterministic_tar(checkpoint_paths, bundle_path, base=output_dir)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "signal_gate_pass"
            if signal_pass
            else "smoke_complete" if profile == "smoke" else "signal_gate_fail"
        ),
        "profile": profile,
        "observations": len(joined),
        "folds": fold_records,
        "bin_metrics": bin_metrics,
        "active_time_bins": active_bins,
        "checkpoints": checkpoint_records,
        "base_generator_frozen": True,
        "base_model_state_sha256": base_state_before,
        "feature_bank": {
            "training_corruption_replicas": int(runtime["training_corruption_replicas"]),
            "evaluation_corruption_replicas": 1,
            "times": times,
            "summary_dim": int(train_bank.shape[-1]),
        },
        "paired_real_shuffled_controls": True,
        "direct_scalar_target": "training-fold standardized HeLa mTP",
        "guided_generation": False,
        "oracle_calls": 0,
        "synthesis_calls": 0,
        "inputs": {name: pin_record(path, repo) for name, path in sorted(paths.items())},
        "artifacts": {
            "evaluation_scores": artifact_record(ledger_path),
            "value_head_bundle": artifact_record(bundle_path),
        },
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "FOLD_PROGRESS_SCHEMA",
    "LEDGER_SCHEMA",
    "RESULT_SCHEMA",
    "UgiPartialStateValueError",
    "run_partial_state_value_crossfit",
]
