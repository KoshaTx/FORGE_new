"""Cross-fitted, frozen-backbone Ugi potency-adapter diagnostic.

This module fits only zero-initialized Transformer residual adapters.  It does not run guided
generation, invoke the completed-molecule oracle, or authorize any nonzero biological tilt.
"""

from __future__ import annotations

import hashlib
import io
import random
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_csv, read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.potency_adapter import (
    configure_potency_adapter,
    freeze_base_parameters,
    initialize_potency_adapter_from_base,
    optimizer_parameters,
    potency_adapter_state_dict,
    potency_parameter_report,
)
from forge.model.potency_conditioning import (
    PotencyAdapterPolicy,
    PotencyCondition,
    PotencyConditionBatch,
)
from forge.model.reaction_program_flow import (
    derive_role_morphology_states,
    noise_synthesis_program_batch,
    synthesis_program_flow_loss,
)
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
)
from forge.model.training_restart import atomic_torch_save
from forge.potency.adapter_guidance import (
    clustered_auroc_difference_interval,
    clustered_bootstrap_interval,
    deterministic_permutation,
    map_to_training_ecdf,
    masked_retention_kl,
    morphology_residuals,
    normalized_agile_label,
    per_record_masked_nll,
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


CONFIG_SCHEMA = "forge.ugi_hela_potency_adapter_config.v1"
RESULT_SCHEMA = "forge.ugi_hela_potency_adapter_crossfit_result.v1"
CHECKPOINT_SCHEMA = "forge.ugi_hela_potency_adapter_checkpoint.v1"
BASE_CHECKPOINT_SCHEMA = "forge.synthesis_program_production_checkpoint.v1"
PROGRAM_ID = "ugi_3cr_agile"


class UgiPotencyAdapterError(ValueError):
    """The bounded Ugi potency-adapter diagnostic violated its frozen contract."""


def _deterministic_tar(paths: Sequence[Path], target: Path, *, base: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
            info = tarfile.TarInfo(path.relative_to(base).as_posix())
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _set_determinism(seed: int, cpu_threads: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def _load_base_package(
    archive_path: Path,
    training_result_path: Path,
    design_path: Path,
    cache_path: Path,
    *,
    arm_id: str,
    checkpoint_step: int,
    device: Any,
) -> tuple[dict[str, Any], str, str]:
    training = read_json_object(
        training_result_path,
        error=UgiPotencyAdapterError,
        label="base training result",
    )
    arms = training.get("arms")
    if not isinstance(arms, Mapping) or arm_id not in arms:
        raise UgiPotencyAdapterError("base training result omits the declared arm")
    checkpoints = arms[arm_id].get("checkpoints")
    if not isinstance(checkpoints, list):
        raise UgiPotencyAdapterError("base training result omits checkpoint receipts")
    matches = [row for row in checkpoints if int(row.get("step", -1)) == checkpoint_step]
    if len(matches) != 1:
        raise UgiPotencyAdapterError("base checkpoint step is not unique")
    receipt = matches[0]
    member_name = f"{arm_id}/{receipt['filename']}"
    with tarfile.open(archive_path, mode="r") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as error:
            raise UgiPotencyAdapterError("base archive omits the requested checkpoint") from error
        if not member.isfile() or member.name != member_name:
            raise UgiPotencyAdapterError("base checkpoint member is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise UgiPotencyAdapterError("base checkpoint member cannot be read")
        payload = handle.read()
    member_sha256 = hashlib.sha256(payload).hexdigest()
    if member_sha256 != receipt.get("sha256"):
        raise UgiPotencyAdapterError("base checkpoint member hash changed")
    package = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != BASE_CHECKPOINT_SCHEMA
        or package.get("model_state_sha256") != receipt.get("model_state_sha256")
        or package.get("design_sha256") != str(sha256_file(design_path))
        or package.get("cache_sha256") != str(sha256_file(cache_path))
    ):
        raise UgiPotencyAdapterError("base checkpoint authentication failed")
    return package, member_name, member_sha256


def _split_rows(path: Path) -> dict[tuple[str, int, str], tuple[str, str]]:
    output: dict[tuple[str, int, str], tuple[str, str]] = {}
    for row in read_csv(path):
        scheme = row["scheme"]
        if scheme not in {"held_head_5fold", "held_aldehyde_isocyanide_pair_5fold"}:
            continue
        key = (scheme, int(row["fold"]), normalized_agile_label(row["label"]))
        if key in output:
            raise UgiPotencyAdapterError(f"duplicate potency split row: {key}")
        output[key] = (row["stage"], row["group_id"])
    return output


def _joined_observations(
    observations_path: Path,
    assignments_path: Path,
    splits_path: Path,
    cache: SynthesisProgramProductionCache,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], tuple[str, str]]]:
    observations = [
        row
        for row in read_csv(observations_path)
        if row["study_id"] == "YX_2024" and row["endpoint"] == "HeLa"
    ]
    if len(observations) != 1100:
        raise UgiPotencyAdapterError(
            "the single-compound YX_2024 HeLa view must contain 1,100 rows"
        )
    assignment_by_smiles: dict[str, str] = {}
    for row in read_csv(assignments_path):
        smiles = row["canonical_product_smiles"]
        if smiles in assignment_by_smiles:
            raise UgiPotencyAdapterError("Ugi assignments contain duplicate constitutional graphs")
        assignment_by_smiles[smiles] = row["product_id"]
    cache_index_by_id = {
        cache.record_id(int(index)): int(index) for index in cache.indices(program_id=PROGRAM_ID)
    }
    joined: list[dict[str, Any]] = []
    for row in observations:
        smiles = row["model_smiles"]
        product_id = assignment_by_smiles.get(smiles)
        cache_index = None if product_id is None else cache_index_by_id.get(product_id)
        if cache_index is None:
            raise UgiPotencyAdapterError(
                "an LNPDB HeLa graph has no independently constructed Ugi program record"
            )
        value = float(row["label_value"])
        if not np.isfinite(value):
            raise UgiPotencyAdapterError("potency observation is non-finite")
        joined.append(
            {
                "label": normalized_agile_label(row["source_lipid_name"]),
                "cache_index": cache_index,
                "model_smiles": smiles,
                "potency": value,
            }
        )
    if len({row["label"] for row in joined}) != 1100:
        raise UgiPotencyAdapterError("LNPDB HeLa labels are not one-to-one")
    splits = _split_rows(splits_path)
    expected = 2 * 5 * len(joined)
    if len(splits) != expected:
        raise UgiPotencyAdapterError("component-disjoint split coverage changed")
    return joined, splits


def _morphology_features(record: Any, *, role_count: int) -> np.ndarray:
    states = derive_role_morphology_states(record)
    values: list[float] = []
    for role in range(1, role_count):
        indices = np.flatnonzero(record.role_states == role)
        values.extend(
            [0.0] * states.shape[1]
            if not len(indices)
            else states[int(indices[0])].astype(np.float64).tolist()
        )
    values.extend((float(record.node_count), float(record.graph.closure_count)))
    return np.asarray(values, dtype=np.float64)


def _model_with_adapter(
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    config: Mapping[str, Any],
    device: Any,
    policy: PotencyAdapterPolicy,
) -> Any:
    model_config = dict(package["model_config"])
    model_config.update(
        {
            "potency_adapter_dim": int(config["adapter"]["bottleneck_dim"]),
            "potency_condition_dim": int(config["adapter"]["condition_dim"]),
        }
    )
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=model_config,
        device=device,
    )
    initialize_potency_adapter_from_base(model, package["model_state"])
    configure_potency_adapter(model, policy)
    freeze_base_parameters(model)
    model.eval()
    return model


def _clean_batch(
    cache: SynthesisProgramProductionCache, indices: Sequence[int], model: Any, device: Any
) -> dict[str, Any]:
    return move_tensors(
        collate_synthesis_program_training_batch(
            cache.records(np.asarray(indices, dtype=np.int64)),
            maximum_closures=int(model.maximum_closures),
            conditioning="program",
            vocabulary=cache.vocabulary,
        ),
        device,
    )


def _fit_adapter(
    *,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    config: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    quantiles: np.ndarray,
    policy: PotencyAdapterPolicy,
    device: Any,
    seed: int,
    steps: int,
    labelled_batch_size: int,
    retention_batch_size: int,
) -> tuple[Any, list[dict[str, float]]]:
    model = _model_with_adapter(package, cache, config, device, policy)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        optimizer_parameters(model),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    retention_support = cache.indices(program_id=PROGRAM_ID, fold="train")
    retention_weights = cache.arrays["source_weights"][retention_support].astype(np.float64)
    retention_weights /= retention_weights.sum()
    losses: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        selected = quartile_balanced_sample(quantiles, size=labelled_batch_size, rng=rng)
        labelled_indices = [int(train_rows[index]["cache_index"]) for index in selected]
        labelled_clean = _clean_batch(cache, labelled_indices, model, device)
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
        condition = PotencyConditionBatch(
            endpoint_id=policy.endpoint_id,
            target_quantiles=torch.as_tensor(
                quantiles[selected], dtype=torch.float32, device=device
            ),
            policy_id=policy.policy_id,
        )
        labelled_predictions = _synthesis_program_predict(
            model,
            labelled_clean,
            labelled_noisy,
            labelled_t,
            potency_condition=condition,
        )
        denoising, _ = synthesis_program_flow_loss(labelled_predictions, labelled_clean)

        retention_indices = rng.choice(
            retention_support,
            size=retention_batch_size,
            replace=True,
            p=retention_weights,
        )
        retention_clean = _clean_batch(cache, retention_indices, model, device)
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
                model,
                retention_clean,
                retention_noisy,
                retention_t,
            )
        student = _synthesis_program_predict(
            model,
            retention_clean,
            retention_noisy,
            retention_t,
            potency_condition=PotencyCondition(
                endpoint_id=policy.endpoint_id,
                target_quantile=0.5,
                policy_id=policy.policy_id,
            ),
        )
        retention = masked_retention_kl(student, teacher, retention_clean)
        loss = denoising + float(training["retention_weight"]) * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters(model),
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            denoising_value, retention_value, gradient_value = (
                float(value)
                for value in torch.stack(
                    (denoising.detach(), retention.detach(), gradient_norm.detach())
                ).cpu()
            )
            losses.append(
                {
                    "step": float(step),
                    "denoising": denoising_value,
                    "retention_kl": retention_value,
                    "gradient_norm": gradient_value,
                }
            )
    return model, losses


def _evaluate_fold(
    *,
    real_model: Any,
    shuffled_model: Any,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[str],
    policy: PotencyAdapterPolicy,
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
            condition = PotencyCondition(
                endpoint_id=policy.endpoint_id,
                target_quantile=0.9,
                policy_id=policy.policy_id,
            )
            with torch.inference_mode():
                null = _synthesis_program_predict(real_model, clean, noisy, t)
                real = _synthesis_program_predict(
                    real_model,
                    clean,
                    noisy,
                    t,
                    potency_condition=condition,
                )
                shuffled = _synthesis_program_predict(
                    shuffled_model,
                    clean,
                    noisy,
                    t,
                    potency_condition=condition,
                )
                null_nll = per_record_masked_nll(null, clean)
                real_nll = per_record_masked_nll(real, clean)
                shuffled_nll = per_record_masked_nll(shuffled, clean)
            real_scores = (null_nll - real_nll).cpu().tolist()
            shuffled_scores = (null_nll - shuffled_nll).cpu().tolist()
            for local_index, (row, real_score, shuffled_score) in enumerate(
                zip(local, real_scores, shuffled_scores, strict=True)
            ):
                output.append(
                    {
                        "label": row["label"],
                        "potency": float(row["potency"]),
                        "cluster_id": clusters[offset + local_index],
                        "time_bin": time_name,
                        "real_score": float(real_score),
                        "shuffled_score": float(shuffled_score),
                    }
                )
    return output


def _policy(
    config: Mapping[str, Any], intervals: Sequence[Sequence[float]]
) -> PotencyAdapterPolicy:
    return PotencyAdapterPolicy(
        policy_id=str(config["condition"]["policy_id"]),
        endpoint_id=str(config["condition"]["endpoint_id"]),
        program_id=PROGRAM_ID,
        minimum_quantile=float(config["condition"]["quantile_support"][0]),
        maximum_quantile=float(config["condition"]["quantile_support"][1]),
        active_time_intervals=tuple((float(row[0]), float(row[1])) for row in intervals),
    )


def run_potency_adapter_crossfit(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    allocated_device: str,
) -> dict[str, Any]:
    """Fit cross-fitted real/shuffled adapters and qualify flow-time signal support."""

    if torch is None:
        raise UgiPotencyAdapterError("potency adapter training requires torch")
    config = read_json_object(
        config_path,
        error=UgiPotencyAdapterError,
        label="Ugi potency adapter config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or profile not in {"smoke", "full"}:
        raise UgiPotencyAdapterError("unsupported potency adapter config or profile")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise UgiPotencyAdapterError("potency adapter authorization is missing")
    if profile == "full" and authorization.get("adapter_fit_execution_authorized") is not True:
        raise UgiPotencyAdapterError("full potency adapter fitting is not authorized")
    if authorization.get("nonzero_generation_authorized") is not False:
        raise UgiPotencyAdapterError("cross-fit config must not authorize guided generation")
    runtime = config[profile]
    if runtime["device"] != allocated_device or runtime["precision"] != "float32":
        raise UgiPotencyAdapterError("allocated device or precision differs from the contract")
    if runtime["deterministic_algorithms"] is not True:
        raise UgiPotencyAdapterError("potency adapter fitting must remain deterministic")
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiPotencyAdapterError("CUDA potency adapter fitting requested but unavailable")
    _set_determinism(int(config["seed"]), int(runtime["cpu_threads"]), device)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
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
        joined, splits = _joined_observations(
            paths["potency_observations"],
            paths["ugi_assignments"],
            paths["oracle_split_assignments"],
            cache,
        )
        all_intervals = [row["interval"] for row in config["validation"]["time_bins"]]
        fit_policy = _policy(config, all_intervals)
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
                        test_rows.append(row)
                        test_clusters.append(split[1])
                if not train_rows or not test_rows:
                    raise UgiPotencyAdapterError("component-disjoint fold has empty support")
                values = np.asarray([float(row["potency"]) for row in train_rows])
                quantiles = training_ecdf_quantiles(values)
                permutation = deterministic_permutation(
                    len(train_rows),
                    seed=int(config["seed"]) + fold,
                    namespace=scheme,
                )
                local_seed = int(config["seed"]) + 10_000 * scheme_index + 100 * fold
                real_model, real_losses = _fit_adapter(
                    package=package,
                    cache=cache,
                    config=config,
                    train_rows=train_rows,
                    quantiles=quantiles,
                    policy=fit_policy,
                    device=device,
                    seed=local_seed,
                    steps=int(runtime["optimizer_steps"]),
                    labelled_batch_size=int(runtime["labelled_batch_size"]),
                    retention_batch_size=int(runtime["retention_batch_size"]),
                )
                shuffled_model, shuffled_losses = _fit_adapter(
                    package=package,
                    cache=cache,
                    config=config,
                    train_rows=train_rows,
                    quantiles=quantiles[permutation],
                    policy=fit_policy,
                    device=device,
                    seed=local_seed,
                    steps=int(runtime["optimizer_steps"]),
                    labelled_batch_size=int(runtime["labelled_batch_size"]),
                    retention_batch_size=int(runtime["retention_batch_size"]),
                )
                rows = _evaluate_fold(
                    real_model=real_model,
                    shuffled_model=shuffled_model,
                    package=package,
                    cache=cache,
                    rows=test_rows,
                    clusters=test_clusters,
                    policy=fit_policy,
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
                    values,
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
                            "high_potency": int(
                                map_to_training_ecdf(values, [row["potency"]])[0] >= 0.75
                            ),
                        }
                    )
                evaluation_rows.extend(rows)
                fold_records.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "train_rows": len(train_rows),
                        "test_rows": len(test_rows),
                        "real_losses": real_losses,
                        "shuffled_losses": shuffled_losses,
                    }
                )

        bootstrap = config["validation"]["clustered_bootstrap"]
        thresholds = config["validation"]["gates"]
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
                truth = [int(row["high_potency"]) for row in local]
                real = [float(row["real_score"]) for row in local]
                shuffled = [float(row["shuffled_score"]) for row in local]
                residual = [float(row["residual_potency"]) for row in local]
                clusters = [f"{row['fold']}::{row['cluster_id']}" for row in local]
                real_auc = roc_auc(truth, real)
                shuffled_auc = roc_auc(truth, shuffled)
                rho = spearman(residual, real)
                seed = int(bootstrap["seed"]) + 100 * scheme_index + time_index
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
                    residual_spearman=rho,
                    residual_spearman_lower=float(rho_interval["lower_95"]),
                    shuffled_auroc=shuffled_auc,
                    shuffled_difference_lower=float(difference["lower_95"]),
                    thresholds=thresholds,
                )
                if gates["passes"]:
                    active_by_scheme[scheme].add(name)
                bin_metrics[f"{scheme}::{name}"] = {
                    "rows": len(local),
                    "real_auroc": real_auc,
                    "shuffled_auroc": shuffled_auc,
                    "residual_spearman": rho,
                    "real_auroc_bootstrap": auc_interval,
                    "residual_spearman_bootstrap": rho_interval,
                    "real_minus_shuffled_auroc_bootstrap": difference,
                    "gates": gates,
                }
        active_bins = sorted(set.intersection(*active_by_scheme.values()))
        eligible_names = {str(value) for value in config["validation"]["eligible_time_bins"]}
        active_bins = [value for value in active_bins if value in eligible_names]
        signal_pass = profile == "full" and bool(active_bins)

        checkpoints: dict[str, Any] = {}
        if signal_pass:
            intervals_by_name = {
                str(row["name"]): row["interval"] for row in config["validation"]["time_bins"]
            }
            final_policy = _policy(config, [intervals_by_name[name] for name in active_bins])
            values = np.asarray([float(row["potency"]) for row in joined])
            quantiles = training_ecdf_quantiles(values)
            permutation = deterministic_permutation(
                len(joined), seed=int(config["seed"]), namespace="final"
            )
            for arm, targets in (
                ("potency_q90", quantiles),
                ("shuffled_label_q90", quantiles[permutation]),
            ):
                model, losses = _fit_adapter(
                    package=package,
                    cache=cache,
                    config=config,
                    train_rows=joined,
                    quantiles=targets,
                    policy=final_policy,
                    device=device,
                    seed=int(config["seed"]),
                    steps=int(runtime["optimizer_steps"]),
                    labelled_batch_size=int(runtime["labelled_batch_size"]),
                    retention_batch_size=int(runtime["retention_batch_size"]),
                )
                checkpoint_path = output_dir / f"{arm}_adapter.pt"
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
                    "adapter_state": potency_adapter_state_dict(model),
                    "combined_model_state_sha256": _model_state_sha256(model),
                    "parameter_report": potency_parameter_report(model),
                    "losses": losses,
                }
                atomic_torch_save(checkpoint_path, package_out)
                checkpoints[arm] = artifact_record(checkpoint_path)

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "signal_gate_pass"
            if signal_pass
            else "smoke_complete" if profile == "smoke" else "signal_gate_fail"
        ),
        "profile": profile,
        "config": pin_record(config_path, repo),
        "inputs": {name: pin_record(path, repo) for name, path in paths.items()},
        "base_checkpoint": {
            "member_name": member_name,
            "member_sha256": member_sha256,
            "model_state_sha256": package["model_state_sha256"],
        },
        "observations": 1100,
        "folds": fold_records,
        "time_bin_metrics": bin_metrics,
        "active_time_bins": active_bins,
        "checkpoints": checkpoints,
        "gates": {
            "component_disjoint_schemes_complete": profile == "smoke" or (len(fold_records) == 10),
            "at_least_one_mid_or_late_bin_passes_both_schemes": signal_pass,
            "nonzero_generation_remains_unauthorized": True,
        },
        "nonclaims": [
            "This cross-fit result does not show that guided generation improves potency.",
            "LNPDB within-study normalized HeLa labels are not new measurements.",
            "No synthesis, proposal-engine, candidate-selection or prospective action was run.",
        ],
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    _deterministic_tar(
        [result_path, *(output_dir / f"{arm}_adapter.pt" for arm in sorted(checkpoints))],
        output_dir / "adapter_bundle.tar",
        base=output_dir,
    )
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiPotencyAdapterError",
    "run_potency_adapter_crossfit",
]
