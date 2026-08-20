#!/usr/bin/env python3
"""Apply the frozen Phase 1 Ugi architecture/checkpoint selection policy."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski

from forge.potency.annotations import ROLE_NAMES
from forge.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.corpus.ugi_held_component_gate import (
    _canonical_molecule,
    _catalog_by_role,
    _forward_reconstructs_product,
    _reaction_contract,
)

REPO = Path(__file__).resolve().parents[3]
COMMON_TRAINING_INPUT_KEYS = (
    "assignments",
    "semantic_products",
    "semantic_atoms",
    "atom_vocabulary",
    "prepared_cache",
)
EXPECTED_EXPANDED_SOURCE_ROWS = 112_386


class SelectionPolicyError(RuntimeError):
    """Raised when the frozen selection contract cannot be applied exactly."""


DESCRIPTOR_FUNCTIONS: dict[str, Callable[[Chem.Mol], float]] = {
    "heavy_atoms": lambda molecule: float(molecule.GetNumHeavyAtoms()),
    "molecular_weight": lambda molecule: float(Descriptors.MolWt(molecule)),
    "logp": lambda molecule: float(Crippen.MolLogP(molecule)),
    "rings": lambda molecule: float(Lipinski.RingCount(molecule)),
    "rotatable_bonds": lambda molecule: float(Lipinski.NumRotatableBonds(molecule)),
    "heteroatoms": lambda molecule: float(Lipinski.NumHeteroatoms(molecule)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SelectionPolicyError(f"expected JSON object: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_selection_policy(
    path: Path, *, _seen: set[Path] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise SelectionPolicyError(f"cyclic selection-policy inheritance: {path}")
    seen.add(path)
    revision = _load_json(path)
    if "base_policy" not in revision:
        return revision, revision
    base_path = _require_hash(revision["base_policy"])
    _, base = _load_selection_policy(base_path, _seen=seen)
    overrides = revision.get("overrides")
    if not isinstance(overrides, dict):
        raise SelectionPolicyError("versioned policy has no override object")
    resolved = _deep_merge(base, overrides)
    resolved["schema_version"] = revision["schema_version"]
    resolved["base_policy"] = revision["base_policy"]
    resolved["correction"] = revision.get("correction", {})
    return revision, resolved


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute() and str(value).startswith("/root/forge_repo/"):
        return REPO / value.relative_to("/root/forge_repo")
    return value if value.is_absolute() else REPO / value


def _require_hash(specification: dict[str, Any]) -> Path:
    path = _resolve(specification["path"])
    observed = _sha256(path)
    if observed != specification["sha256"]:
        raise SelectionPolicyError(f"frozen input hash mismatch: {path}: {observed}")
    return path


def _normalized_input_specification(
    specification: dict[str, Any],
    *,
    verify_local_hash: bool,
) -> dict[str, str]:
    if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
        raise SelectionPolicyError("training input specification must contain path and sha256")
    path = _resolve(specification["path"])
    try:
        relative_path = path.resolve().relative_to(REPO.resolve())
    except ValueError as error:
        raise SelectionPolicyError(f"training input is outside repository: {path}") from error
    expected_hash = str(specification["sha256"])
    if verify_local_hash:
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise SelectionPolicyError(
                f"training input hash mismatch: {path}: {observed_hash} != {expected_hash}"
            )
    return {"path": str(relative_path), "sha256": expected_hash}


def _path_hash_specification(specification: dict[str, Any]) -> dict[str, str]:
    """Extract an immutable file identity from a metadata-bearing policy record."""

    if not isinstance(specification, dict) or not {"path", "sha256"}.issubset(specification):
        raise SelectionPolicyError("policy file specification has no path/hash identity")
    return {
        "path": str(specification["path"]),
        "sha256": str(specification["sha256"]),
    }


def _common_training_input_audit(
    architectures: tuple[str, ...],
    training_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    normalized_by_architecture: dict[str, dict[str, dict[str, str]]] = {}
    for architecture in architectures:
        payload = training_payloads[architecture]
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            raise SelectionPolicyError(f"training result has no input manifest: {architecture}")
        missing = sorted(set(COMMON_TRAINING_INPUT_KEYS) - set(inputs))
        if missing:
            raise SelectionPolicyError(
                f"training result is missing shared inputs for {architecture}: {missing}"
            )
        normalized = {
            key: _normalized_input_specification(inputs[key], verify_local_hash=True)
            for key in COMMON_TRAINING_INPUT_KEYS
        }
        mounted = payload.get("cloud_execution", {}).get("mounted_inputs")
        if not isinstance(mounted, dict):
            raise SelectionPolicyError(
                f"training result has no mounted-input audit: {architecture}"
            )
        for key, specification in normalized.items():
            if mounted.get(specification["path"]) != specification["sha256"]:
                raise SelectionPolicyError(f"mounted-input mismatch for {architecture}:{key}")
        normalized_by_architecture[architecture] = normalized

    reference_architecture = architectures[0]
    common = normalized_by_architecture[reference_architecture]
    for architecture in architectures[1:]:
        if normalized_by_architecture[architecture] != common:
            raise SelectionPolicyError(
                "architecture arms did not train on identical shared input path/hash manifests"
            )
    return common, {
        "status": "pass",
        "required_identical_inputs": list(COMMON_TRAINING_INPUT_KEYS),
        "common_inputs": common,
        "inputs_by_architecture": normalized_by_architecture,
        "local_hashes_verified": True,
        "cloud_mount_hashes_verified": True,
    }


def _require_checkpoint_inputs(
    checkpoint: dict[str, Any],
    expected: dict[str, dict[str, str]],
    *,
    architecture: str,
    step: int,
) -> None:
    inputs = checkpoint.get("inputs")
    if not isinstance(inputs, dict):
        raise SelectionPolicyError(f"checkpoint has no input manifest: {architecture}:{step}")
    missing = sorted(set(COMMON_TRAINING_INPUT_KEYS) - set(inputs))
    if missing:
        raise SelectionPolicyError(
            f"checkpoint is missing shared inputs at {architecture}:{step}: {missing}"
        )
    observed = {
        key: _normalized_input_specification(inputs[key], verify_local_hash=False)
        for key in COMMON_TRAINING_INPUT_KEYS
    }
    if observed != expected:
        raise SelectionPolicyError(
            f"checkpoint embedded input path/hash mismatch: {architecture}:{step}"
        )


def _read_assignments(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _selection_reference_rows(
    assignments: list[dict[str, str]], specification: dict[str, Any]
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Return only the folds authorized to influence checkpoint selection."""

    selection_folds = tuple(str(value) for value in specification["selection_folds"])
    if not selection_folds or len(set(selection_folds)) != len(selection_folds):
        raise SelectionPolicyError("selection reference folds are empty or duplicated")
    source_fold_counts = Counter(str(row["primary_product_fold"]) for row in assignments)
    unknown = sorted(set(selection_folds) - set(source_fold_counts))
    if unknown:
        raise SelectionPolicyError(f"selection reference contains unknown folds: {unknown}")
    selected = [row for row in assignments if str(row["primary_product_fold"]) in selection_folds]
    expected_rows = int(specification["selection_rows"])
    if len(selected) != expected_rows:
        raise SelectionPolicyError(
            "selection-visible reference does not contain the preregistered "
            f"{expected_rows:,} rows: {len(selected):,}"
        )
    return selected, source_fold_counts


def _wasserstein_1(left: np.ndarray, right: np.ndarray) -> float:
    """Exact one-dimensional empirical Wasserstein-1 distance."""

    if not len(left) or not len(right):
        raise SelectionPolicyError("Wasserstein distance requires two nonempty samples")
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    support = np.sort(np.concatenate((left, right)))
    if len(support) < 2:
        return 0.0
    intervals = np.diff(support)
    left_cdf = np.searchsorted(left, support[:-1], side="right") / len(left)
    right_cdf = np.searchsorted(right, support[:-1], side="right") / len(right)
    return float(np.sum(np.abs(left_cdf - right_cdf) * intervals))


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        raise SelectionPolicyError("KS distance requires two nonempty samples")
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    support = np.sort(np.unique(np.concatenate((left, right))))
    left_cdf = np.searchsorted(left, support, side="right") / len(left)
    right_cdf = np.searchsorted(right, support, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _effective_shannon(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    probabilities = np.asarray(tuple(counts.values()), dtype=np.float64) / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _concentration(counts: Counter[str], n: int) -> float:
    total = sum(counts.values())
    if not total:
        return 1.0
    return float(sum(sorted(counts.values(), reverse=True)[:n]) / total)


def _descriptor_values(
    smiles: list[str],
    cache: dict[str, tuple[float, ...]],
) -> dict[str, np.ndarray]:
    names = tuple(DESCRIPTOR_FUNCTIONS)
    rows = []
    for value in smiles:
        if value not in cache:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise SelectionPolicyError(f"invalid descriptor molecule: {value!r}")
            cache[value] = tuple(DESCRIPTOR_FUNCTIONS[name](molecule) for name in names)
        rows.append(cache[value])
    matrix = np.asarray(rows, dtype=np.float64)
    return {name: matrix[:, index] for index, name in enumerate(names)}


def _canonicalized_molecule(value: str) -> tuple[Chem.Mol, str]:
    molecule = _canonical_molecule(value)
    return molecule, Chem.MolToSmiles(molecule, canonical=True)


def _atom_vocabulary(path: Path) -> set[tuple[str, int, bool, int]]:
    value = _load_json(path)
    return {
        (
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row["explicit_hydrogens"]),
        )
        for row in value["atom_vocabulary"]
    }


def _support_violations(
    row: dict[str, Any],
    molecule: Chem.Mol,
    model_config: dict[str, Any],
    atom_vocabulary: set[tuple[str, int, bool, int]],
) -> list[str]:
    violations = []
    if len(Chem.GetMolFrags(molecule)) != 1:
        violations.append("disconnected_product")
    if molecule.GetNumHeavyAtoms() > int(model_config["maximum_total_atoms"]):
        violations.append("maximum_total_atoms")
    for atom in molecule.GetAtoms():
        state = (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumExplicitHs(),
        )
        if state not in atom_vocabulary:
            violations.append("atom_state_vocabulary")
            break
    allowed_bonds = {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    }
    if any(bond.GetBondType() not in allowed_bonds for bond in molecule.GetBonds()):
        violations.append("bond_state_vocabulary")

    program = row["program"]
    node_counts = tuple(int(value) for value in program["node_counts"])
    junctions = tuple(int(value) for value in program["junction_budgets"])
    cycles = tuple(int(value) for value in program["cycle_ranks"])
    attachments = tuple(int(value) for value in program["attachment_counts"])
    if len(node_counts) != 3 or any(
        value < 1 or value > int(model_config["maximum_component_atoms"]) for value in node_counts
    ):
        violations.append("component_atom_bound")
    if len(junctions) != 3 or any(
        value < 0 or value > int(model_config["maximum_junction_budget"]) for value in junctions
    ):
        violations.append("junction_budget_bound")
    if len(cycles) != 3 or any(
        value < 0 or value > int(model_config["maximum_cycle_rank"]) for value in cycles
    ):
        violations.append("cycle_rank_bound")
    if len(attachments) != 3 or any(
        value < 1 or value > int(model_config["maximum_attachment_count"]) for value in attachments
    ):
        violations.append("attachment_count_bound")
    for role, node_count, attachment_count in zip(
        ROLE_NAMES, node_counts, attachments, strict=True
    ):
        offspring = tuple(int(value) for value in row["offspring_by_role"][role])
        if len(offspring) != node_count:
            violations.append(f"{role}:offspring_length")
        if any(value < 0 or value > int(model_config["maximum_children"]) for value in offspring):
            violations.append(f"{role}:offspring_bound")
        if sum(offspring) != node_count - attachment_count:
            violations.append(f"{role}:forest_edge_identity")
    return sorted(set(violations))


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1:
        raise SelectionPolicyError(
            "paired bootstrap vectors must be aligned one-dimensional arrays"
        )
    difference = left.astype(np.float64) - right.astype(np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    batch = 250
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        estimates[start:stop] = difference[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point_difference": float(difference.mean()),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _parse_arm_specifications(values: list[str]) -> dict[str, dict[int, Path]]:
    output: dict[str, dict[int, Path]] = {}
    for value in values:
        try:
            architecture, step_text, path_text = value.split(":", 2)
            step = int(step_text)
        except ValueError as error:
            raise SelectionPolicyError("--arm-sample must be ARCHITECTURE:STEP:PATH") from error
        if step in output.setdefault(architecture, {}):
            raise SelectionPolicyError(f"duplicate candidate: {architecture}:{step}")
        output[architecture][step] = _resolve(path_text)
    return output


def _parse_training_results(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        try:
            architecture, path_text = value.split(":", 1)
        except ValueError as error:
            raise SelectionPolicyError("--training-result must be ARCHITECTURE:PATH") from error
        output[architecture] = _resolve(path_text)
    return output


def _require_sampling_contract(
    result: dict[str, Any],
    policy: dict[str, Any],
    *,
    architecture: str,
    step: int,
) -> None:
    """Bind optional decoder and feasibility settings used by a selection policy."""

    expected = policy.get("matched_sampling", {})
    sampling = result.get("sampling")
    if not isinstance(sampling, dict):
        raise SelectionPolicyError(f"{architecture}:{step} has no sampling metadata")
    if "sample_steps" in expected and int(sampling.get("sample_steps", -1)) != int(
        expected["sample_steps"]
    ):
        raise SelectionPolicyError(f"{architecture}:{step} sample-step contract changed")

    expected_branch_runs = expected.get("maximum_adjacent_branch_graph_runs_by_role")
    if expected_branch_runs is not None:
        expected_vector = [expected_branch_runs[role] for role in ROLE_NAMES]
        if sampling.get("maximum_adjacent_branch_runs") != expected_vector:
            raise SelectionPolicyError(f"{architecture}:{step} branch-run contract changed")

    if "evaluate_exact_l1_terminal_admission" in expected and bool(
        sampling.get("evaluate_exact_l1_terminal_admission")
    ) != bool(expected["evaluate_exact_l1_terminal_admission"]):
        raise SelectionPolicyError(f"{architecture}:{step} exact-L1 audit contract changed")

    if "terminal_decoder_mode" in expected:
        decoder = sampling.get("terminal_decoder")
        if not isinstance(decoder, dict):
            raise SelectionPolicyError(f"{architecture}:{step} has no terminal decoder metadata")
        if decoder.get("mode") != expected["terminal_decoder_mode"]:
            raise SelectionPolicyError(f"{architecture}:{step} terminal decoder changed")
        if decoder.get("seed") != expected.get("terminal_decoder_seed"):
            raise SelectionPolicyError(f"{architecture}:{step} terminal decoder seed changed")
        if not math.isclose(
            float(decoder.get("temperature", float("nan"))),
            float(expected.get("terminal_temperature", 1.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SelectionPolicyError(f"{architecture}:{step} terminal temperature changed")


def _expected_program_seed(policy: dict[str, Any]) -> int:
    """Return the frozen program-draw seed independently of the flow RNG seed."""

    matched = policy.get("matched_sampling", {})
    if "program_seed" in matched:
        return int(matched["program_seed"])
    return int(matched.get("seed", policy["matched_uncertainty"]["seed"]))


def _candidate_metrics(
    *,
    architecture: str,
    step: int,
    sampling_path: Path,
    checkpoint_hashes: dict[int, str],
    policy: dict[str, Any],
    prior: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    reaction: Any,
    expected_checkpoint_inputs: dict[str, dict[str, str]],
    atom_vocabulary: set[tuple[str, int, bool, int]],
    reference_smiles: set[str],
    reference_descriptors: dict[str, np.ndarray],
    descriptor_cache: dict[str, tuple[float, ...]],
) -> tuple[dict[str, Any], np.ndarray]:
    result = _load_json(sampling_path)
    samples = result["samples"]
    expected_count = int(policy["frozen_inputs"]["unconditional_program_prior"]["attempted_draws"])
    if len(samples) != expected_count:
        raise SelectionPolicyError(f"{architecture}:{step} has {len(samples)} draws")
    sampling_seed = int(
        policy.get("matched_sampling", {}).get("seed", policy["matched_uncertainty"]["seed"])
    )
    if int(result["seed"]) != sampling_seed:
        raise SelectionPolicyError(f"{architecture}:{step} sampling seed changed")
    _require_sampling_contract(
        result,
        policy,
        architecture=architecture,
        step=step,
    )
    observed_probe = _resolve(result["matched_staged_result"])
    expected_probe = _resolve(policy["frozen_inputs"]["unconditional_program_prior"]["path"])
    if observed_probe.resolve() != expected_probe.resolve():
        raise SelectionPolicyError(f"{architecture}:{step} used another program prior")
    checkpoint_path = _resolve(result["checkpoints"]["joint"])
    observed_checkpoint_hash = _sha256(checkpoint_path)
    if observed_checkpoint_hash != checkpoint_hashes[step]:
        raise SelectionPolicyError(f"checkpoint hash mismatch at {architecture}:{step}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != step:
        raise SelectionPolicyError(f"embedded checkpoint step mismatch: {architecture}:{step}")
    _require_checkpoint_inputs(
        checkpoint,
        expected_checkpoint_inputs,
        architecture=architecture,
        step=step,
    )
    expected_mode = {
        "full_morphology_program_conditioning": "full_morphology",
        "size_only_conditioning": "size_only",
    }[architecture]
    if result["sampling"]["conditioning_mode"] != expected_mode:
        raise SelectionPolicyError(f"conditioning mismatch: {architecture}:{step}")
    supplied_fields = tuple(result["sampling"]["program_fields_supplied"])
    generated_fields = tuple(result["sampling"]["program_fields_generated"])
    for index, (sample, expected) in enumerate(zip(samples, prior["samples"], strict=True)):
        if expected_mode == "full_morphology":
            aligned = sample["program"] == expected["program"]
            expected_supplied = (
                "node_counts",
                "junction_budgets",
                "cycle_ranks",
                "attachment_counts",
            )
            expected_generated: tuple[str, ...] = ()
        else:
            aligned = sample["program"]["node_counts"] == expected["program"]["node_counts"]
            expected_supplied = ("node_counts",)
            expected_generated = (
                "junction_budgets",
                "cycle_ranks",
                "attachment_counts",
            )
        if not aligned:
            raise SelectionPolicyError(f"program mismatch at {architecture}:{step}:{index}")
        if supplied_fields != expected_supplied or generated_fields != expected_generated:
            raise SelectionPolicyError(f"program-field contract mismatch at {architecture}:{step}")
    model_config = checkpoint["model_config"]

    valid_smiles = []
    unique_valid = set()
    component_counts = {role: Counter() for role in ROLE_NAMES}
    successes = np.zeros(expected_count, dtype=np.int8)
    valid = reconstruction_valid = forward_valid = handles_valid = 0
    forbidden_valid = support_invalid = exact_frozen = 0
    support_examples = []
    handle_failure_examples = []
    outside_catalog_products = 0

    for index, row in enumerate(samples):
        if not row.get("valid"):
            continue
        valid += 1
        smiles = str(row["smiles"])
        molecule, canonical_smiles = _canonicalized_molecule(smiles)
        valid_smiles.append(canonical_smiles)
        unique_valid.add(canonical_smiles)
        exact_frozen += int(canonical_smiles in reference_smiles)
        violations = _support_violations(row, molecule, model_config, atom_vocabulary)
        if violations:
            support_invalid += 1
            if len(support_examples) < 5:
                support_examples.append({"smiles": canonical_smiles, "violations": violations})
        if not row.get("component_reconstruction_valid"):
            continue
        components = row.get("component_smiles_by_role")
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            continue
        reconstruction_valid += 1
        handles_pass = True
        outside = False
        forbidden = False
        canonical_components = {}
        for role_index, role in enumerate(ROLE_NAMES):
            component_smiles = str(components[role])
            component, canonical_component_smiles = _canonicalized_molecule(component_smiles)
            canonical_components[role] = canonical_component_smiles
            component_counts[role][canonical_component_smiles] += 1
            outside = outside or canonical_component_smiles not in catalog[role]
            qualification = reaction_handle_qualification(
                component,
                query=reaction.handles[role_index],
                forbidden=reaction.forbidden[role_index],
                allowed_site_multiplicity=reaction.definition.reactant_roles[
                    role_index
                ].allowed_site_multiplicity,
            )
            forbidden = forbidden or bool(qualification["forbidden_substructure_match"])
            handles_pass = handles_pass and bool(qualification["passes_registry_handle_policy"])
        forbidden_valid += int(forbidden)
        if not handles_pass and len(handle_failure_examples) < 5:
            handle_failure_examples.append(
                {"smiles": canonical_smiles, "components": canonical_components}
            )
        handles_valid += int(handles_pass)
        outside_catalog_products += int(outside)
        forward, _, _ = _forward_reconstructs_product(
            reaction, canonical_components, canonical_smiles
        )
        forward_valid += int(forward)
        successes[index] = int(
            not violations and not forbidden and handles_pass and outside and forward
        )

    generated_descriptors = _descriptor_values(valid_smiles, descriptor_cache)
    distribution = {}
    for descriptor in policy["descriptor_distribution_realism_floors"]["descriptors"]:
        generated_values = generated_descriptors[descriptor]
        reference_values = reference_descriptors[descriptor]
        reference_quantiles = _quantiles(reference_values)
        width = max(reference_quantiles["q95"] - reference_quantiles["q05"], 1.0)
        distribution[descriptor] = {
            "generated": _quantiles(generated_values),
            "reference": reference_quantiles,
            "normalized_wasserstein_1": _wasserstein_1(reference_values, generated_values) / width,
            "kolmogorov_smirnov": _ks_distance(reference_values, generated_values),
            "generated_fraction_inside_reference_q05_q95": float(
                np.mean(
                    (generated_values >= reference_quantiles["q05"])
                    & (generated_values <= reference_quantiles["q95"])
                )
            ),
        }
    mean_wasserstein = float(
        np.mean([value["normalized_wasserstein_1"] for value in distribution.values()])
    )
    mean_ks = float(np.mean([value["kolmogorov_smirnov"] for value in distribution.values()]))

    role_metrics = {
        role: {
            "unique_reconstructed_components": len(counts),
            "effective_shannon_component_count": _effective_shannon(counts),
            "top_1_component_fraction": _concentration(counts, 1),
            "top_5_component_fraction": _concentration(counts, 5),
        }
        for role, counts in component_counts.items()
    }
    denominators = {
        "attempted": expected_count,
        "valid": valid,
        "reconstructed": reconstruction_valid,
    }
    metrics = {
        "architecture": architecture,
        "step": step,
        "sampling_result": {
            "path": str(sampling_path.relative_to(REPO)),
            "sha256": _sha256(sampling_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO)),
            "sha256": observed_checkpoint_hash,
        },
        "denominators": denominators,
        "valid_fraction_of_attempted": valid / expected_count,
        "component_reconstruction_fraction_of_valid": reconstruction_valid / max(valid, 1),
        "exact_forward_reconstruction_fraction_of_reconstructed": forward_valid
        / max(reconstruction_valid, 1),
        "all_three_handle_qualification_fraction_of_reconstructed": handles_valid
        / max(reconstruction_valid, 1),
        "valid_outputs_with_forbidden_substructure": forbidden_valid,
        "valid_outputs_outside_declared_support": support_invalid,
        "support_violation_examples": support_examples,
        "handle_failure_examples": handle_failure_examples,
        "unique_valid_products": len(unique_valid),
        "unique_fraction_among_valid": len(unique_valid) / max(valid, 1),
        "exact_frozen_products": exact_frozen,
        "exact_frozen_product_fraction_among_valid": exact_frozen / max(valid, 1),
        "products_with_outside_catalog_component": outside_catalog_products,
        "outside_catalog_component_fraction_among_reconstructed": outside_catalog_products
        / max(reconstruction_valid, 1),
        "usable_open_ended_successes": int(successes.sum()),
        "usable_open_ended_yield": float(successes.mean()),
        "usable_open_ended_success_indices": np.flatnonzero(successes).tolist(),
        "usable_open_ended_vector_sha256": hashlib.sha256(successes.tobytes()).hexdigest(),
        "per_role_component_distribution": role_metrics,
        "descriptor_distribution": distribution,
        "mean_normalized_wasserstein_1": mean_wasserstein,
        "mean_kolmogorov_smirnov": mean_ks,
    }
    return metrics, successes


def _apply_candidate_gates(metrics: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    hard = policy["candidate_hard_gates"]
    checks = {
        "valid_fraction": metrics["valid_fraction_of_attempted"]
        >= hard["minimum_valid_fraction_of_all_attempted_draws"],
        "component_inverse": metrics["component_reconstruction_fraction_of_valid"]
        >= hard["minimum_component_reconstruction_fraction_of_valid_products"],
        "exact_forward": metrics["exact_forward_reconstruction_fraction_of_reconstructed"]
        >= hard["minimum_exact_forward_reconstruction_fraction_of_reconstructed_products"],
        "handles": metrics["all_three_handle_qualification_fraction_of_reconstructed"]
        >= hard["minimum_all_three_handle_qualification_fraction_of_reconstructed_products"],
        "forbidden_substructure": metrics["valid_outputs_with_forbidden_substructure"]
        <= hard["maximum_valid_outputs_with_forbidden_substructure"],
        "declared_support": metrics["valid_outputs_outside_declared_support"]
        <= hard["maximum_valid_outputs_outside_declared_atom_bond_charge_size_cycle_support"],
        "uniqueness": metrics["unique_fraction_among_valid"]
        >= hard["minimum_unique_fraction_among_valid_products"],
        "frozen_product_reproduction": metrics["exact_frozen_product_fraction_among_valid"]
        <= hard["maximum_exact_frozen_product_fraction_among_valid_products"],
    }
    collapse = policy["per_role_component_collapse_safeguards"]
    for role, thresholds in collapse["roles"].items():
        observed = metrics["per_role_component_distribution"][role]
        checks[f"{role}:unique"] = (
            observed["unique_reconstructed_components"]
            >= thresholds["minimum_unique_reconstructed_components"]
        )
        checks[f"{role}:effective_shannon"] = (
            observed["effective_shannon_component_count"]
            >= thresholds["minimum_effective_shannon_component_count"]
        )
        checks[f"{role}:top1"] = (
            observed["top_1_component_fraction"]
            <= collapse["maximum_top_1_component_fraction_per_role"]
        )
        checks[f"{role}:top5"] = (
            observed["top_5_component_fraction"]
            <= collapse["maximum_top_5_component_fraction_per_role"]
        )

    realism = policy["descriptor_distribution_realism_floors"]
    wasserstein = realism["normalized_wasserstein_1"]
    ks = realism["kolmogorov_smirnov"]
    coverage = realism["central_reference_coverage"]
    checks["mean_normalized_wasserstein_1"] = (
        metrics["mean_normalized_wasserstein_1"] <= wasserstein["maximum_mean_across_descriptors"]
    )
    checks["mean_kolmogorov_smirnov"] = (
        metrics["mean_kolmogorov_smirnov"] <= ks["maximum_mean_across_descriptors"]
    )
    for descriptor, observed in metrics["descriptor_distribution"].items():
        checks[f"{descriptor}:wasserstein"] = (
            observed["normalized_wasserstein_1"] <= wasserstein["maximum_each_descriptor"]
        )
        checks[f"{descriptor}:ks"] = observed["kolmogorov_smirnov"] <= ks["maximum_each_descriptor"]
        checks[f"{descriptor}:central_coverage"] = (
            observed["generated_fraction_inside_reference_q05_q95"]
            >= coverage["minimum_generated_fraction_each_descriptor"]
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
    }


def _within_architecture_selection(
    candidates: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    policy: dict[str, Any],
) -> dict[str, Any]:
    passing = [candidate for candidate in candidates if candidate["gate"]["status"] == "pass"]
    if not passing:
        return {"status": "no_passing_checkpoint", "selected": None}
    top = max(passing, key=lambda value: value["usable_open_ended_yield"])
    uncertainty = policy["matched_uncertainty"]
    comparisons = {}
    noninferior = []
    for candidate in passing:
        comparison = _bootstrap_difference(
            vectors[top["candidate_id"]],
            vectors[candidate["candidate_id"]],
            seed=int(uncertainty["seed"]),
            replicates=int(uncertainty["bootstrap_replicates"]),
            confidence=float(uncertainty["confidence_interval"]),
        )
        comparisons[candidate["candidate_id"]] = comparison
        threshold = float(uncertainty["operationally_meaningful_absolute_difference"])
        if comparison["point_difference"] < threshold or (
            comparison["lower"] <= 0 <= comparison["upper"]
        ):
            noninferior.append(candidate)

    best_realism = min(value["mean_normalized_wasserstein_1"] for value in noninferior)
    realism_tied = [
        value
        for value in noninferior
        if value["mean_normalized_wasserstein_1"] <= best_realism + 0.02
    ]
    best_reproduction = min(
        value["exact_frozen_product_fraction_among_valid"] for value in realism_tied
    )
    reproduction_tied = [
        value
        for value in realism_tied
        if math.isclose(
            value["exact_frozen_product_fraction_among_valid"],
            best_reproduction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    selected = min(reproduction_tied, key=lambda value: int(value["step"]))
    return {
        "status": "selected",
        "top_primary_candidate": top["candidate_id"],
        "noninferior_candidates": [value["candidate_id"] for value in noninferior],
        "paired_comparisons_from_top": comparisons,
        "selected": selected["candidate_id"],
    }


def select(
    *,
    policy_path: Path,
    training_results: dict[str, Path],
    arm_samples: dict[str, dict[int, Path]],
    output_path: Path,
) -> dict[str, Any]:
    policy_revision, policy = _load_selection_policy(policy_path)
    architectures = tuple(policy["scope"]["architectures"])
    expected_steps = tuple(
        int(value) for value in policy["scope"]["common_serial_checkpoint_steps"]
    )
    if set(training_results) != set(architectures) or set(arm_samples) != set(architectures):
        raise SelectionPolicyError("selection inputs do not contain both frozen architectures")
    for architecture in architectures:
        if tuple(sorted(arm_samples[architecture])) != expected_steps:
            raise SelectionPolicyError(f"incomplete serial sample set for {architecture}")

    for specification in policy["frozen_inputs"].values():
        if (
            isinstance(specification, dict)
            and "path" in specification
            and "sha256" in specification
        ):
            _require_hash(specification)
    for specification in policy["frozen_inputs"]["architecture_configs"].values():
        _require_hash(specification)
    inverse_path = _require_hash(policy["nonselecting_evidence"]["held_component_outputs"])
    inverse = _load_json(inverse_path)["reference_inverse_audit"]
    inverse_policy = policy["reference_inverse_gate"]
    inverse_pass = (
        inverse["status"] == inverse_policy["required_status"]
        and inverse["exact_component_reconstruction_fraction"]
        == inverse_policy["required_exact_component_reconstruction_fraction"]
        and inverse["unique_components_covered_by_role"]
        == inverse_policy["required_unique_components_covered_by_role"]
    )
    if not inverse_pass:
        raise SelectionPolicyError("deterministic inverse-Ugi gate no longer passes")

    prior_path = _resolve(policy["frozen_inputs"]["unconditional_program_prior"]["path"])
    prior = _load_json(prior_path)
    if int(prior["seed"]) != _expected_program_seed(policy):
        raise SelectionPolicyError("frozen program-prior draw seed changed")
    if len(prior["samples"]) != int(
        policy["frozen_inputs"]["unconditional_program_prior"]["attempted_draws"]
    ):
        raise SelectionPolicyError("frozen program-prior draw count changed")
    registry_path = _resolve(policy["frozen_inputs"]["admitted_component_registry"]["path"])
    reaction_path = _resolve(policy["frozen_inputs"]["qualified_reactions"]["path"])
    catalog = _catalog_by_role(registry_path)
    reaction = _reaction_contract(reaction_path)

    training_payloads = {
        architecture: _load_json(path) for architecture, path in training_results.items()
    }
    common_training_inputs, common_input_audit = _common_training_input_audit(
        architectures, training_payloads
    )
    reference_policy = policy["frozen_inputs"]["frozen_product_reference"]
    assignments_specification = _normalized_input_specification(
        _path_hash_specification(reference_policy),
        verify_local_hash=True,
    )
    if assignments_specification != common_training_inputs["assignments"]:
        raise SelectionPolicyError(
            "selection reference is not the identical assignment corpus used by both arms"
        )
    assignments_path = _resolve(assignments_specification["path"])
    source_assignments = _read_assignments(assignments_path)
    expected_source_rows = int(reference_policy["rows"])
    if expected_source_rows != EXPECTED_EXPANDED_SOURCE_ROWS:
        raise SelectionPolicyError("expanded source-reference row contract changed")
    if len(source_assignments) != expected_source_rows:
        raise SelectionPolicyError(
            "expanded source reference does not contain the preregistered "
            f"{expected_source_rows:,} rows: {len(source_assignments):,}"
        )
    assignments, fold_counts = _selection_reference_rows(source_assignments, reference_policy)
    expected_fold_counts = {
        str(key): int(value) for key, value in reference_policy["source_fold_counts"].items()
    }
    if fold_counts != Counter(expected_fold_counts):
        raise SelectionPolicyError(f"unexpected expanded-reference folds: {fold_counts}")
    reference_smiles_list = [str(row["canonical_product_smiles"]) for row in assignments]
    reference_smiles = set(reference_smiles_list)
    if len(reference_smiles) != int(reference_policy["selection_unique_products"]):
        raise SelectionPolicyError("selection-visible product reference contains duplicates")
    descriptor_cache: dict[str, tuple[float, ...]] = {}
    reference_descriptors = _descriptor_values(reference_smiles_list, descriptor_cache)

    checkpoint_hashes = {
        architecture: {
            int(row["step"]): str(row["sha256"]) for row in payload["checkpoint_snapshots"]
        }
        for architecture, payload in training_payloads.items()
    }
    for architecture in architectures:
        config = policy["frozen_inputs"]["architecture_configs"][architecture]
        mounted = training_payloads[architecture]["cloud_execution"]["mounted_inputs"]
        if mounted.get(config["path"]) != config["sha256"]:
            raise SelectionPolicyError(f"training result config mismatch: {architecture}")

    atom_path = _resolve(common_training_inputs["atom_vocabulary"]["path"])
    atom_vocabulary = _atom_vocabulary(atom_path)

    candidates_by_architecture: dict[str, list[dict[str, Any]]] = {
        architecture: [] for architecture in architectures
    }
    vectors = {}
    for architecture in architectures:
        for step in expected_steps:
            metrics, vector = _candidate_metrics(
                architecture=architecture,
                step=step,
                sampling_path=arm_samples[architecture][step],
                checkpoint_hashes=checkpoint_hashes[architecture],
                policy=policy,
                prior=prior,
                catalog=catalog,
                reaction=reaction,
                expected_checkpoint_inputs=common_training_inputs,
                atom_vocabulary=atom_vocabulary,
                reference_smiles=reference_smiles,
                reference_descriptors=reference_descriptors,
                descriptor_cache=descriptor_cache,
            )
            candidate_id = f"{architecture}:step_{step}"
            metrics["candidate_id"] = candidate_id
            metrics["gate"] = _apply_candidate_gates(metrics, policy)
            candidates_by_architecture[architecture].append(metrics)
            vectors[candidate_id] = vector

    within = {
        architecture: _within_architecture_selection(
            candidates_by_architecture[architecture], vectors, policy
        )
        for architecture in architectures
    }
    available = {
        architecture: value["selected"]
        for architecture, value in within.items()
        if value["selected"] is not None
    }
    between = None
    selected_candidate = None
    if not available:
        decision = "no_production_generator"
        reason = "Neither architecture has a checkpoint passing every frozen gate."
    elif len(available) == 1:
        architecture, selected_candidate = next(iter(available.items()))
        decision = architecture
        reason = "Only one architecture has a checkpoint passing every frozen gate."
    else:
        full_id = available["full_morphology_program_conditioning"]
        size_id = available["size_only_conditioning"]
        uncertainty = policy["matched_uncertainty"]
        between = _bootstrap_difference(
            vectors[full_id],
            vectors[size_id],
            seed=int(uncertainty["seed"]),
            replicates=int(uncertainty["bootstrap_replicates"]),
            confidence=float(uncertainty["confidence_interval"]),
        )
        threshold = float(uncertainty["operationally_meaningful_absolute_difference"])
        meaningful = abs(between["point_difference"]) >= threshold and not (
            between["lower"] <= 0 <= between["upper"]
        )
        between["meaningfully_different"] = meaningful
        between["left"] = full_id
        between["right"] = size_id
        if meaningful and between["point_difference"] > 0:
            decision = "full_morphology_program_conditioning"
            selected_candidate = full_id
            reason = "Full morphology has statistically and operationally superior usable open-ended yield."
        elif meaningful:
            decision = "size_only_conditioning"
            selected_candidate = size_id
            reason = (
                "Size-only has statistically and operationally superior usable open-ended yield."
            )
        else:
            decision = "size_only_conditioning"
            selected_candidate = size_id
            reason = "Both pass and the primary difference is not meaningful; the frozen parsimony rule selects weaker size-only conditioning."

    result = {
        "schema_version": (
            "phase1_ugi_architecture_checkpoint_selection."
            + policy["schema_version"].rsplit(".", 1)[-1]
        ),
        "status": "complete",
        "scope": {
            "product_and_l1_only": True,
            "held_component_outputs_used_for_selection": False,
            "heldout_loss_used_for_selection": False,
            "l2_routes_evaluated": False,
            "biological_guidance_evaluated": False,
            "synthesis_guidance_evaluated": False,
        },
        "policy": {
            "path": str(policy_path.relative_to(REPO)),
            "sha256": _sha256(policy_path),
            "resolved_schema_version": policy["schema_version"],
            "base_policy": policy_revision.get("base_policy"),
        },
        "reference_inverse_gate": {"status": "pass", **inverse},
        "common_training_input_audit": common_input_audit,
        "expanded_frozen_product_reference": {
            "path": str(assignments_path.relative_to(REPO)),
            "sha256": _sha256(assignments_path),
            "source_rows": len(source_assignments),
            "source_fold_counts": dict(sorted(fold_counts.items())),
            "selection_rows": len(assignments),
            "selection_unique_products": len(reference_smiles),
            "selection_folds": list(reference_policy["selection_folds"]),
            "heldout_excluded_from_selection": "heldout"
            not in set(reference_policy["selection_folds"]),
            "used_for_exact_product_reproduction": True,
            "used_for_descriptor_distribution": True,
        },
        "lineage_product_reference": {
            "path": policy["frozen_inputs"]["lineage_product_reference"]["path"],
            "sha256": policy["frozen_inputs"]["lineage_product_reference"]["sha256"],
            "used_for_exact_product_reproduction": False,
            "used_for_descriptor_distribution": False,
            "note": policy["frozen_inputs"]["lineage_product_reference"]["use"],
        },
        "inputs": {
            "training_results": {
                architecture: {"path": str(path.relative_to(REPO)), "sha256": _sha256(path)}
                for architecture, path in training_results.items()
            },
            "program_prior": {
                "path": str(prior_path.relative_to(REPO)),
                "sha256": _sha256(prior_path),
            },
        },
        "candidates_by_architecture": candidates_by_architecture,
        "within_architecture_selection": within,
        "between_architecture_comparison": between,
        "decision": {
            "selected_architecture": decision,
            "selected_candidate": selected_candidate,
            "production_generator_frozen": selected_candidate is not None,
            "reason": reason,
        },
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v1.json",
    )
    parser.add_argument("--training-result", action="append", required=True)
    parser.add_argument("--arm-sample", action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_architecture_checkpoint_selection_v1.json",
    )
    args = parser.parse_args()
    result = select(
        policy_path=args.policy.resolve(),
        training_results=_parse_training_results(args.training_result),
        arm_samples=_parse_arm_specifications(args.arm_sample),
        output_path=args.output.resolve(),
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
