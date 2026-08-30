"""Layerwise node-attention mTP prediction over a frozen FORGE Transformer.

The scorer keeps the node axis, learns a mixture of the final Transformer layers, attends to the
three Ugi precursor roles and the complete graph, and consumes predicted clean atom and bond
probabilities.  This module is diagnostic only and never runs guided generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.hela_potency.partial_state_value import (
    _bin_metrics,
    _frozen_base_model,
)
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
from forge.model.partial_state_value import LayerwiseRoleAttentionValueHead, direct_value_loss
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import _synthesis_program_predict
from forge.model.training_restart import atomic_torch_save
from forge.potency.adapter_failure_attribution import factorial_dataset_summary
from forge.potency.adapter_guidance import (
    deterministic_permutation,
    map_to_training_ecdf,
    morphology_residuals,
    quartile_balanced_sample,
    training_ecdf_quantiles,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.ugi_hela_partial_state_attention_value_config.v2"
RESULT_SCHEMA = "forge.ugi_hela_partial_state_attention_value_crossfit_result.v2"
LEDGER_SCHEMA = "forge.ugi_hela_partial_state_attention_value_score.v2"
CHECKPOINT_SCHEMA = "forge.ugi_hela_partial_state_attention_value_checkpoint.v2"
FOLD_PROGRESS_SCHEMA = "forge.ugi_hela_partial_state_attention_value_fold_progress.v2"


class UgiPartialStateAttentionValueError(ValueError):
    """The bounded layerwise partial-state diagnostic violated its contract."""


@dataclass(frozen=True)
class AttentionFeatureBank:
    """CPU-resident frozen features with deterministic paired indexing."""

    hidden_layers: Any
    atom_probabilities: Any
    parent_bond_probabilities: Any
    closure_bond_probabilities: Any
    node_mask: Any
    closure_mask: Any
    role_states: Any
    times: Any

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.hidden_layers.shape[:3])

    def batch(
        self,
        row_indices: Any,
        time_indices: Any,
        replica_indices: Any,
        *,
        device: Any,
    ) -> dict[str, Any]:
        rows = torch.as_tensor(row_indices, dtype=torch.long)
        times = torch.as_tensor(time_indices, dtype=torch.long)
        replicas = torch.as_tensor(replica_indices, dtype=torch.long)
        if rows.ndim != 1 or times.shape != rows.shape or replicas.shape != rows.shape:
            raise UgiPartialStateAttentionValueError("feature-bank indices are malformed")
        return {
            "hidden_layers": self.hidden_layers[rows, times, replicas].to(
                device=device, dtype=torch.float32
            ),
            "atom_probabilities": self.atom_probabilities[rows, times, replicas].to(
                device=device, dtype=torch.float32
            ),
            "parent_bond_probabilities": self.parent_bond_probabilities[rows, times, replicas].to(
                device=device, dtype=torch.float32
            ),
            "closure_bond_probabilities": self.closure_bond_probabilities[rows, times, replicas].to(
                device=device, dtype=torch.float32
            ),
            "node_mask": self.node_mask[rows].to(device=device),
            "closure_mask": self.closure_mask[rows].to(device=device),
            "role_states": self.role_states[rows].to(device=device),
            "flow_time": self.times[times].to(device=device, dtype=torch.float32),
        }


def _representation_signature(config: Mapping[str, Any]) -> str:
    return str(sha256_json(config["representation"]))


def _fold_signature(
    *,
    config: Mapping[str, Any],
    scheme: str,
    fold: int,
    seed: int,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    time_bins: Sequence[Mapping[str, Any]],
) -> str:
    return str(
        sha256_json(
            {
                "fold": fold,
                "representation": _representation_signature(config),
                "scheme": scheme,
                "seed": seed,
                "test_labels": [str(row["label"]) for row in test_rows],
                "time_bins": list(time_bins),
                "train_labels": [str(row["label"]) for row in train_rows],
            }
        )
    )


def _load_progress(
    path: Path,
    *,
    signature: str,
    test_rows: int,
    evaluation_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = read_json_object(
        path,
        error=UgiPartialStateAttentionValueError,
        label="attention-value fold progress",
    )
    record = value.get("fold_record")
    rows = value.get("evaluation_rows")
    if (
        value.get("schema_version") != FOLD_PROGRESS_SCHEMA
        or value.get("fold_signature") != signature
        or not isinstance(record, dict)
        or not isinstance(rows, list)
        or int(record.get("test_rows", -1)) != test_rows
        or len(rows) != evaluation_rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise UgiPartialStateAttentionValueError(f"attention-value progress is invalid: {path}")
    return dict(record), [dict(row) for row in rows]


def _write_progress(
    path: Path,
    *,
    signature: str,
    fold_record: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_json(
        path,
        {
            "evaluation_rows": [dict(row) for row in evaluation_rows],
            "fold_record": dict(fold_record),
            "fold_signature": signature,
            "schema_version": FOLD_PROGRESS_SCHEMA,
        },
    )


def _allocate_feature_bank(
    *,
    rows: int,
    times: int,
    replicas: int,
    layers: int,
    maximum_nodes: int,
    hidden_dim: int,
    atom_classes: int,
    maximum_closures: int,
    bond_classes: int,
    time_values: Sequence[float],
) -> AttentionFeatureBank:
    feature_shape = (rows, times, replicas)
    return AttentionFeatureBank(
        hidden_layers=torch.zeros(
            (*feature_shape, layers, maximum_nodes, hidden_dim), dtype=torch.float16
        ),
        atom_probabilities=torch.zeros(
            (*feature_shape, maximum_nodes, atom_classes), dtype=torch.float16
        ),
        parent_bond_probabilities=torch.zeros(
            (*feature_shape, maximum_nodes, bond_classes), dtype=torch.float16
        ),
        closure_bond_probabilities=torch.zeros(
            (*feature_shape, maximum_closures, bond_classes), dtype=torch.float16
        ),
        node_mask=torch.zeros((rows, maximum_nodes), dtype=torch.bool),
        closure_mask=torch.zeros((rows, maximum_closures), dtype=torch.bool),
        role_states=torch.zeros((rows, maximum_nodes), dtype=torch.long),
        times=torch.as_tensor(time_values, dtype=torch.float32),
    )


def _extract_feature_bank(
    *,
    model: Any,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    rows: Sequence[Mapping[str, Any]],
    times: Sequence[float],
    replicas: int,
    batch_size: int,
    layer_count: int,
    seed: int,
    device: Any,
) -> AttentionFeatureBank:
    """Cache frozen node-level features in float16 while all fitting remains float32."""

    if replicas < 1 or batch_size < 1 or not times or layer_count < 1:
        raise UgiPartialStateAttentionValueError("attention feature-bank geometry is empty")
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    maximum_nodes = max(cache.record(int(row["cache_index"])).node_count for row in rows)
    bank: AttentionFeatureBank | None = None
    for time_index, time_value in enumerate(times):
        if not 0.0 < time_value < 1.0:
            raise UgiPartialStateAttentionValueError("partial-state times must be interior")
        for replica in range(replicas):
            for offset in range(0, len(rows), batch_size):
                local = rows[offset : offset + batch_size]
                clean = _clean_batch(
                    cache,
                    [int(row["cache_index"]) for row in local],
                    model,
                    device,
                )
                local_nodes = int(clean["node_mask"].shape[1])
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
                        return_hidden_layers=layer_count,
                    )
                    hidden = predictions["hidden_layers"]
                    atoms = torch.softmax(predictions["nodes"], dim=-1)
                    parent_bonds = torch.softmax(predictions["parent_bonds"], dim=-1)
                    closure_bonds = torch.softmax(predictions["closure_bonds"], dim=-1)
                if bank is None:
                    bank = _allocate_feature_bank(
                        rows=len(rows),
                        times=len(times),
                        replicas=replicas,
                        layers=layer_count,
                        maximum_nodes=maximum_nodes,
                        hidden_dim=int(hidden.shape[-1]),
                        atom_classes=int(atoms.shape[-1]),
                        maximum_closures=int(closure_bonds.shape[1]),
                        bond_classes=int(parent_bonds.shape[-1]),
                        time_values=times,
                    )
                target = slice(offset, offset + len(local))
                bank.hidden_layers[target, time_index, replica, :, :local_nodes] = hidden.to(
                    device="cpu", dtype=torch.float16
                )
                bank.atom_probabilities[target, time_index, replica, :local_nodes] = atoms.to(
                    device="cpu", dtype=torch.float16
                )
                bank.parent_bond_probabilities[target, time_index, replica, :local_nodes] = (
                    parent_bonds.to(device="cpu", dtype=torch.float16)
                )
                bank.closure_bond_probabilities[target, time_index, replica] = closure_bonds.to(
                    device="cpu", dtype=torch.float16
                )
                local_node_mask = clean["node_mask"].to(device="cpu")
                local_roles = clean["role_states"].to(device="cpu")
                local_closure_mask = clean["closure_mask"].to(device="cpu")
                bank.node_mask[target, :local_nodes] = local_node_mask
                bank.role_states[target, :local_nodes] = local_roles
                bank.closure_mask[target] = local_closure_mask
    if bank is None or not bool(bank.node_mask.any(dim=1).all()):
        raise UgiPartialStateAttentionValueError("attention feature bank is incomplete")
    return bank


def _role_indices(
    cache: SynthesisProgramProductionCache, config: Mapping[str, Any]
) -> tuple[int, int, int]:
    names = tuple(str(value) for value in config["representation"]["attention_roles"])
    expected = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    if names != expected:
        raise UgiPartialStateAttentionValueError("attention-role order changed")
    vocabulary = tuple(cache.vocabulary.role_states)
    try:
        return tuple(vocabulary.index(name) for name in names)  # type: ignore[return-value]
    except ValueError as error:
        raise UgiPartialStateAttentionValueError(
            "attention role is absent from the cache"
        ) from error


def _head(
    *,
    bank: AttentionFeatureBank,
    role_count: int,
    role_indices: tuple[int, int, int],
    value_hidden_dim: int,
    seed: int,
    device: Any,
) -> Any:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return LayerwiseRoleAttentionValueHead(
        hidden_dim=int(bank.hidden_layers.shape[-1]),
        layer_count=int(bank.hidden_layers.shape[3]),
        atom_classes=int(bank.atom_probabilities.shape[-1]),
        bond_classes=int(bank.parent_bond_probabilities.shape[-1]),
        role_count=role_count,
        attention_role_indices=role_indices,
        value_hidden_dim=value_hidden_dim,
    ).to(device)


def _fit_heads(
    *,
    bank: AttentionFeatureBank,
    row_indices: np.ndarray,
    values: np.ndarray,
    node_counts: np.ndarray,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    role_count: int,
    role_indices: tuple[int, int, int],
    seed: int,
    namespace: str,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    hidden_dim = int(config["value_head"]["hidden_dim"])
    real = _head(
        bank=bank,
        role_count=role_count,
        role_indices=role_indices,
        value_hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    shuffled = _head(
        bank=bank,
        role_count=role_count,
        role_indices=role_indices,
        value_hidden_dim=hidden_dim,
        seed=seed,
        device=device,
    )
    structural = _head(
        bank=bank,
        role_count=role_count,
        role_indices=role_indices,
        value_hidden_dim=hidden_dim,
        seed=seed + 1,
        device=device,
    )
    for key, value in real.state_dict().items():
        if not torch.equal(value, shuffled.state_dict()[key]):
            raise UgiPartialStateAttentionValueError(
                "real and shuffled attention heads differ at initialization"
            )

    mean, scale = float(values.mean()), float(values.std(ddof=0))
    node_mean, node_scale = float(node_counts.mean()), float(node_counts.std(ddof=0))
    if scale <= 0.0 or node_scale <= 0.0:
        raise UgiPartialStateAttentionValueError("training-fold target variance is zero")
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
        time_indices = rng.integers(0, bank.shape[1], size=batch_size)
        replica_indices = rng.integers(0, bank.shape[2], size=batch_size)
        features = bank.batch(
            row_indices[local_indices],
            time_indices,
            replica_indices,
            device=device,
        )
        targets = {
            "real": standardized[local_indices],
            "shuffled": shuffled_targets[local_indices],
            "structural": standardized_nodes[local_indices],
        }
        for name, head in heads.items():
            prediction = head(**features)
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
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "loss": float(loss.detach().cpu()),
                        "ranking": float(parts["ranking"].detach().cpu()),
                        "regression": float(parts["regression"].detach().cpu()),
                        "step": float(step),
                    }
                )
    return (
        real.eval(),
        shuffled.eval(),
        structural.eval(),
        {
            "node_count_normalization": {"mean": node_mean, "scale": node_scale},
            "paired_control": {
                "identical_initialization": True,
                "identical_partial_state_batches": True,
                "only_difference": "real versus deterministically permuted mTP label",
                "permutation_sha256": str(sha256_json(permutation.tolist())),
            },
            "target_normalization": {"mean": mean, "scale": scale},
            "traces": traces,
        },
    )


def _score_fold(
    *,
    heads: tuple[Any, Any, Any],
    bank: AttentionFeatureBank,
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
        count = len(test_indices)
        features = bank.batch(
            test_indices,
            np.full(count, time_index, dtype=np.int64),
            np.zeros(count, dtype=np.int64),
            device=device,
        )
        with torch.inference_mode():
            real_score = real(**features).cpu().numpy()
            shuffled_score = shuffled(**features).cpu().numpy()
            structural_score = structural(**features).cpu().numpy()
        real_raw = real_score * float(normalization["scale"]) + float(normalization["mean"])
        shuffled_raw = shuffled_score * float(normalization["scale"]) + float(normalization["mean"])
        node_raw = structural_score * float(node_normalization["scale"]) + float(
            node_normalization["mean"]
        )
        for index, row in enumerate(test_rows):
            output.append(
                {
                    "cluster_id": str(test_clusters[index]),
                    "fold": fold,
                    "held_quantile": float(held_quantiles[index]),
                    "high_potency": int(float(held_quantiles[index]) >= 0.75),
                    "label": str(row["label"]),
                    "node_count": float(node_counts[index]),
                    "node_count_prediction": float(node_raw[index]),
                    "potency": float(row["potency"]),
                    "real_prediction": float(real_raw[index]),
                    "schema_version": LEDGER_SCHEMA,
                    "scheme": scheme,
                    "shuffled_prediction": float(shuffled_raw[index]),
                    "time_bin": str(time_name),
                }
            )
    return output


def _validate_config(config: Mapping[str, Any], *, profile: str, allocated_device: str) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA or profile not in {"smoke", "full"}:
        raise UgiPartialStateAttentionValueError("unsupported attention-value config or profile")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise UgiPartialStateAttentionValueError("attention-value authorization is missing")
    if authorization.get("implementation_and_smoke_authorized") is not True:
        raise UgiPartialStateAttentionValueError("attention-value implementation is not authorized")
    if profile == "full" and authorization.get("full_value_fit_execution_authorized") is not True:
        raise UgiPartialStateAttentionValueError("full attention-value fit is not authorized")
    forbidden = (
        "nonzero_generation_authorized",
        "synthesis_calls_authorized",
        "proposal_calls_authorized",
        "candidate_selection_authorized",
    )
    if any(authorization.get(name) is not False for name in forbidden):
        raise UgiPartialStateAttentionValueError(
            "attention-value fit must not authorize generation"
        )
    runtime = config[profile]
    if (
        runtime["device"] != allocated_device
        or runtime["precision"] != "float32"
        or runtime["deterministic_algorithms"] is not True
    ):
        raise UgiPartialStateAttentionValueError("attention-value runtime changed")
    representation = config.get("representation")
    if not isinstance(representation, Mapping) or representation != {
        "atom_probabilities": "predicted_clean_softmax",
        "attention_roles": [
            "amine_head",
            "oxoester_aldehyde_body_tail",
            "isocyanide_tail",
        ],
        "closure_bond_probabilities": "predicted_clean_softmax",
        "explicit_interactions": [
            "amine_x_aldehyde",
            "amine_x_isocyanide",
            "aldehyde_x_isocyanide",
        ],
        "hidden_layers": "final_four",
        "node_axis_preserved": True,
        "parent_bond_probabilities": "predicted_clean_softmax",
        "storage_dtype": "float16",
    }:
        raise UgiPartialStateAttentionValueError("attention representation contract changed")
    paired = config["training"].get("paired_control")
    if paired != {
        "identical_initialization": True,
        "identical_partial_state_batches": True,
        "only_difference": "real versus deterministically permuted mTP label",
    }:
        raise UgiPartialStateAttentionValueError("paired control contract changed")
    data_scope = config.get("data_scope")
    if not isinstance(data_scope, Mapping) or (
        int(data_scope.get("observations", -1)),
        int(data_scope.get("independent_heads", -1)),
        int(data_scope.get("independent_aldehyde_isocyanide_pairs", -1)),
        int(data_scope.get("new_measured_head_or_tail_chemotypes", -1)),
    ) != (1100, 20, 55, 0):
        raise UgiPartialStateAttentionValueError("measured-data scope changed")


def run_partial_state_attention_value_crossfit(
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
    """Cross-fit the layerwise role-attention scorer without sampling."""

    if torch is None:
        raise UgiPartialStateAttentionValueError("attention-value fitting requires torch")
    config = read_json_object(
        config_path,
        error=UgiPartialStateAttentionValueError,
        label="attention-value config",
    )
    _validate_config(config, profile=profile, allocated_device=allocated_device)
    runtime = config[profile]
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiPartialStateAttentionValueError(
            "CUDA attention-value fitting requested unavailable"
        )
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
            raise UgiPartialStateAttentionValueError("base checkpoint schema changed")
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
                    "label_value": str(row["potency"]),
                    "model_smiles": str(row["model_smiles"]),
                    "source_lipid_name": str(row["source_lipid_name"]),
                }
                for row in joined
            ]
        )
        if (
            int(scope["rows"]),
            int(scope["unique_heads"]),
            int(scope["unique_aldehyde_isocyanide_pairs"]),
        ) != (1100, 20, 55):
            raise UgiPartialStateAttentionValueError("observed measured-data support changed")
        configured_bins = list(config["validation"]["time_bins"])
        requested_names = set(str(value) for value in runtime["time_bin_names"])
        time_bins = [row for row in configured_bins if str(row["name"]) in requested_names]
        if len(time_bins) != len(requested_names):
            raise UgiPartialStateAttentionValueError("runtime time-bin selection is invalid")
        times = [float(row["evaluation_time"]) for row in time_bins]
        layer_count = int(config["value_head"]["transformer_layer_count"])
        if int(package["model_config"]["layers"]) < layer_count:
            raise UgiPartialStateAttentionValueError("base Transformer has fewer than four layers")
        train_bank = _extract_feature_bank(
            model=model,
            package=package,
            cache=cache,
            rows=joined,
            times=times,
            replicas=int(runtime["training_corruption_replicas"]),
            batch_size=int(runtime["feature_batch_size"]),
            layer_count=layer_count,
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
            layer_count=layer_count,
            seed=int(config["seed"]) + 2_000_000,
            device=device,
        )
        if _model_state_sha256(model) != base_state_before:
            raise UgiPartialStateAttentionValueError("feature extraction mutated the generator")
        role_count = len(cache.vocabulary.role_states)
        attention_roles = _role_indices(cache, config)
        schemes = list(config["validation"]["schemes"])
        folds = list(range(5))
        if profile == "smoke":
            schemes, folds = schemes[:1], folds[:1]
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
                    raise UgiPartialStateAttentionValueError("component-disjoint fold is empty")
                train_indices = np.asarray(
                    [int(row["row_index"]) for row in train_rows], dtype=np.int64
                )
                test_indices = np.asarray(
                    [int(row["row_index"]) for row in test_rows], dtype=np.int64
                )
                test_clusters = [splits[(scheme, fold, row["label"])][1] for row in test_rows]
                local_seed = int(config["seed"]) + 10_000 * scheme_index + 100 * fold
                signature = _fold_signature(
                    config=config,
                    scheme=scheme,
                    fold=fold,
                    seed=local_seed,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    time_bins=time_bins,
                )
                progress_path = progress_dir / f"{scheme}__fold_{fold}.json"
                expected_rows = len(test_rows) * len(time_bins)
                if progress_path.exists():
                    if not resume:
                        raise UgiPartialStateAttentionValueError(
                            f"unexpected attention progress without resume: {progress_path}"
                        )
                    restored_record, restored_rows = _load_progress(
                        progress_path,
                        signature=signature,
                        test_rows=len(test_rows),
                        evaluation_rows=expected_rows,
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
                real, shuffled, structural, fit = _fit_heads(
                    bank=train_bank,
                    row_indices=train_indices,
                    values=train_values,
                    node_counts=train_node_counts,
                    config=config,
                    runtime=runtime,
                    role_count=role_count,
                    role_indices=attention_roles,
                    seed=local_seed,
                    namespace=f"{scheme}:{fold}",
                    device=device,
                )
                held_quantiles = map_to_training_ecdf(
                    train_values, [float(row["potency"]) for row in test_rows]
                )
                rows = _score_fold(
                    heads=(real, shuffled, structural),
                    bank=evaluation_bank,
                    test_rows=test_rows,
                    test_indices=test_indices,
                    test_clusters=test_clusters,
                    held_quantiles=held_quantiles,
                    time_names=[str(row["name"]) for row in time_bins],
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
                    "fold": fold,
                    "scheme": scheme,
                    "test_rows": len(test_rows),
                    "train_rows": len(train_rows),
                    **fit,
                }
                _write_progress(
                    progress_path,
                    signature=signature,
                    fold_record=fold_record,
                    evaluation_rows=rows,
                )
                progress_commit()
                fold_records.append(fold_record)
                evaluation_rows.extend(rows)

        metric_config = dict(config)
        metric_config["validation"] = dict(config["validation"])
        metric_config["validation"]["time_bins"] = time_bins
        bin_metrics, active_by_scheme = _bin_metrics(evaluation_rows, config=metric_config)
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
            real, shuffled, _, fit = _fit_heads(
                bank=train_bank,
                row_indices=all_indices,
                values=all_values,
                node_counts=all_node_counts,
                config=config,
                runtime=runtime,
                role_count=role_count,
                role_indices=attention_roles,
                seed=int(config["seed"]),
                namespace="final",
                device=device,
            )
            for arm, head in (("real_mtp", real), ("shuffled_mtp", shuffled)):
                checkpoint_path = output_dir / f"{arm}_partial_state_attention_value.pt"
                atomic_torch_save(
                    checkpoint_path,
                    {
                        "active_time_bins": active_bins,
                        "arm": arm,
                        "attention_role_indices": attention_roles,
                        "base": {
                            "archive_sha256": str(sha256_file(paths["base_checkpoint_archive"])),
                            "member_name": member_name,
                            "member_sha256": member_sha256,
                            "model_state_sha256": package["model_state_sha256"],
                        },
                        "guided_generation_authorized": False,
                        "head_config": dict(config["value_head"]),
                        "head_state": {
                            key: value.detach().cpu().clone()
                            for key, value in sorted(head.state_dict().items())
                        },
                        "paired_control": fit["paired_control"],
                        "representation": dict(config["representation"]),
                        "schema_version": CHECKPOINT_SCHEMA,
                        "target_normalization": fit["target_normalization"],
                        "time_bins": time_bins,
                        "trusted_local_checkpoint": True,
                    },
                )
                checkpoint_paths.append(checkpoint_path)
                checkpoint_records[arm] = artifact_record(checkpoint_path)

    ledger_path = output_dir / "evaluation_scores.csv.gz"
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
        [
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
        ],
    )
    bundle_path = output_dir / "value_head_bundle.tar"
    _deterministic_tar(checkpoint_paths, bundle_path, base=output_dir)
    result = {
        "active_time_bins": active_bins,
        "artifacts": {
            "evaluation_scores": artifact_record(ledger_path),
            "value_head_bundle": artifact_record(bundle_path),
        },
        "base_generator_frozen": True,
        "base_model_state_sha256": base_state_before,
        "bin_metrics": bin_metrics,
        "checkpoints": checkpoint_records,
        "direct_scalar_target": "training-fold standardized HeLa mTP",
        "feature_bank": {
            "atom_classes": int(train_bank.atom_probabilities.shape[-1]),
            "bond_classes": int(train_bank.parent_bond_probabilities.shape[-1]),
            "evaluation_corruption_replicas": 1,
            "hidden_dim": int(train_bank.hidden_layers.shape[-1]),
            "layer_count": int(train_bank.hidden_layers.shape[3]),
            "maximum_nodes": int(train_bank.hidden_layers.shape[4]),
            "storage_dtype": "float16",
            "times": times,
            "training_bytes": sum(
                value.numel() * value.element_size()
                for value in (
                    train_bank.hidden_layers,
                    train_bank.atom_probabilities,
                    train_bank.parent_bond_probabilities,
                    train_bank.closure_bond_probabilities,
                    train_bank.node_mask,
                    train_bank.closure_mask,
                    train_bank.role_states,
                )
            ),
            "training_corruption_replicas": int(runtime["training_corruption_replicas"]),
        },
        "folds": fold_records,
        "guided_generation": False,
        "inputs": {name: pin_record(path, repo) for name, path in sorted(paths.items())},
        "observations": len(joined),
        "oracle_calls": 0,
        "paired_real_shuffled_controls": True,
        "profile": profile,
        "representation": dict(config["representation"]),
        "schema_version": RESULT_SCHEMA,
        "status": (
            "signal_gate_pass"
            if signal_pass
            else "smoke_complete" if profile == "smoke" else "signal_gate_fail"
        ),
        "synthesis_calls": 0,
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "FOLD_PROGRESS_SCHEMA",
    "LEDGER_SCHEMA",
    "RESULT_SCHEMA",
    "AttentionFeatureBank",
    "UgiPartialStateAttentionValueError",
    "run_partial_state_attention_value_crossfit",
]
