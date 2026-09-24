"""Build a packed shared-model cache from exact twelve-library semantics."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import rdBase

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.combinatorial_program_semantics import (
    RESULT_SCHEMA as SEMANTIC_RESULT_SCHEMA,
)
from forge.corpus.combinatorial_program_semantics import (
    SEMANTIC_SCHEMA,
)
from forge.corpus.synthesis_program_production_cache import (
    CACHE_SCHEMA,
    SynthesisProgramProductionCache,
    SynthesisProgramProductionCacheError,
    SynthesisProgramSourceRecord,
    _atom_vocabulary_payload,
    _PackedArrays,
    _write_deterministic_npz,
)
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    tensorize_synthesis_program_product,
)
from forge.model.vocabulary import load_atom_vocabulary

CONFIG_SCHEMA = "forge.combinatorial_program_cache_config.v1"
RESULT_SCHEMA = "forge.combinatorial_program_cache.v1"
_FOLDS = ("train", "calibration", "heldout")


class CombinatorialProgramCacheError(ValueError):
    """The semantic source or packed cache violates the declared contract."""


def _vocabulary(raw: Mapping[str, Any]) -> ReactionProgramVocabulary:
    return ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in raw["program_states"]),
        role_states=tuple(str(value) for value in raw["role_states"]),
        core_position_states=tuple(str(value) for value in raw["core_position_states"]),
        maximum_steps=int(raw["maximum_steps"]),
    )


def _records(path: Path) -> Iterator[dict[str, Any]]:
    previous = ""
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            product_id = str(record.get("product_id", ""))
            if not product_id or product_id <= previous:
                raise CombinatorialProgramCacheError(
                    "semantic records must have unique increasing product identifiers"
                )
            previous = product_id
            yield record


def _artifact(path: Path, logical_path: Path, repo: Path) -> dict[str, Any]:
    return {
        "path": str(logical_path.resolve().relative_to(repo.resolve())),
        "sha256": str(sha256_file(path)),
        "bytes": path.stat().st_size,
    }


def build_combinatorial_program_cache(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Pack every admitted semantic record and revalidate it through the public cache loader."""

    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialProgramCacheError(
            "config/output must be inside the repository and output must be fresh"
        )
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CombinatorialProgramCacheError("unsupported combinatorial cache config")
    if config.get("weight_policy") != {
        "within_family": "mean_realism_weight",
        "between_families": "equal_mass",
        "semantic_abstentions": "exclude_then_renormalize_within_family",
        "raw_family_count_sampling": False,
    }:
        raise CombinatorialProgramCacheError("cache weight policy changed")
    required_folds = tuple(str(value) for value in config.get("required_folds", ()))
    if (
        not required_folds
        or len(set(required_folds)) != len(required_folds)
        or not set(required_folds).issubset(_FOLDS)
    ):
        raise CombinatorialProgramCacheError("required_folds must be unique active fold names")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "semantic_result",
        "semantic_records",
        "semantic_config",
        "atom_vocabulary",
    }:
        raise CombinatorialProgramCacheError("cache input set changed")
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in sorted(raw_inputs.items())}
    semantic = json.loads(paths["semantic_result"].read_text())
    semantic_config = json.loads(paths["semantic_config"].read_text())
    if (
        semantic.get("schema_version") != SEMANTIC_RESULT_SCHEMA
        or semantic.get("status") != "pass"
        or semantic.get("config") != raw_inputs["semantic_config"]
        or semantic.get("artifacts", {}).get("semantic_records.jsonl.gz")
        != raw_inputs["semantic_records"]
        or semantic_config.get("inputs", {}).get("atom_vocabulary") != raw_inputs["atom_vocabulary"]
    ):
        raise CombinatorialProgramCacheError(
            "semantic qualification is not authenticated and passing"
        )
    expected_counts = {
        str(program): {fold: int(folds[fold]) for fold in _FOLDS}
        for program, folds in config["expected_fold_counts"].items()
    }
    if expected_counts != semantic.get("semantic_fold_counts"):
        raise CombinatorialProgramCacheError("expected cache population changed")
    vocabulary = _vocabulary(semantic["representation"]["program_vocabulary"])
    programs = tuple(vocabulary.program_states[1:])
    if set(programs) != set(expected_counts) or set(programs) != set(semantic_config["programs"]):
        raise CombinatorialProgramCacheError("semantic and cache program vocabularies differ")
    support = {
        "maximum_heavy_atoms": int(config["support_bounds"]["maximum_heavy_atoms"]),
        "maximum_closures": int(config["support_bounds"]["maximum_closures"]),
    }
    if support != {
        "maximum_heavy_atoms": int(semantic_config["support_bounds"]["maximum_heavy_atoms"]),
        "maximum_closures": int(semantic_config["support_bounds"]["maximum_closures"]),
    }:
        raise CombinatorialProgramCacheError("cache support differs from semantic qualification")
    atom_vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    config_hash = str(sha256_file(config_path))
    source_root = Path(__file__).resolve().parents[2]
    source_names = (
        "forge/corpus/combinatorial_program_cache.py",
        "forge/corpus/combinatorial_program_semantics.py",
        "forge/corpus/synthesis_program_production_cache.py",
        "forge/model/synthesis_program_graph.py",
        "forge/model/reaction_program_conditioning.py",
        "experiments/phase1/multireaction/combinatorial_program_cache.py",
    )
    sources = {name: str(sha256_file(source_root / name)) for name in source_names}
    start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".combinatorial-cache-", dir=output.parent) as temp:
        work = Path(temp)
        packed = _PackedArrays()
        counts: Counter[tuple[str, str]] = Counter()
        maximum_atoms = maximum_closures = fixed_failures = 0
        for raw in _records(paths["semantic_records"]):
            if (
                raw.get("schema_version") != SEMANTIC_SCHEMA
                or raw.get("disposition") != "admit_transform_consistency"
                or raw.get("evidence_basis") != "computed_transform_consistency"
            ):
                raise CombinatorialProgramCacheError("semantic record disposition changed")
            product_id = str(raw["product_id"])
            smiles = str(raw["canonical_smiles"])
            program = str(raw["reaction_family"])
            fold = str(raw["fold"])
            source_weight = float(raw["source_weight"])
            training_weight = float(raw["training_sampling_weight"])
            if (
                program not in expected_counts
                or fold not in _FOLDS
                or product_id != hashlib.sha256(smiles.encode()).hexdigest()
                or not np.isfinite(source_weight)
                or source_weight <= 0
                or not np.isfinite(training_weight)
                or training_weight < 0
                or (fold == "train") != (training_weight > 0)
            ):
                raise CombinatorialProgramCacheError("semantic record identity or weight changed")
            roles = tuple(str(value) for value in raw["origin_roles"])
            positions = tuple(
                f"{program}:{value}" if value else "exterior" for value in raw["core_positions"]
            )
            fixed = tuple(int(value) for value in raw["fixed_atom_indices"])
            try:
                graph = tensorize_synthesis_program_product(
                    record_id=product_id,
                    program_id=program,
                    canonical_product_smiles=smiles,
                    atom_roles=roles,
                    atom_core_positions=positions,
                    program_depth=int(raw["program_depth"]),
                    vocabulary=vocabulary,
                    atom_vocabulary=atom_vocabulary,
                    fixed_atom_indices=fixed,
                )
            except (SynthesisProgramGraphError, ValueError) as error:
                raise CombinatorialProgramCacheError(str(error)) from error
            fixed_policy = bool(semantic_config["programs"][program]["fix_core_atoms"])
            expected_fixed = (
                graph.core_position_states > 1
                if fixed_policy
                else np.zeros(graph.node_count, dtype=np.bool_)
            )
            fixed_failures += int(not np.array_equal(graph.fixed_atom_mask, expected_fixed))
            fixed_edges = int(np.count_nonzero(graph.fixed_parent_bond_mask)) + int(
                np.count_nonzero(graph.fixed_closure_bond_mask)
            )
            fixed_failures += int(bool(expected_fixed.any()) != (fixed_edges > 0))
            packed.append(
                SynthesisProgramSourceRecord(
                    record=graph,
                    fold=fold,
                    source_weight=source_weight,
                )
            )
            counts[(program, fold)] += 1
            maximum_atoms = max(maximum_atoms, graph.node_count)
            maximum_closures = max(maximum_closures, graph.graph.closure_count)
        fold_counts = {
            program: {fold: counts[(program, fold)] for fold in _FOLDS} for program in programs
        }
        metadata = {
            "schema_version": CACHE_SCHEMA,
            "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
            "inputs": raw_inputs,
            "fold_states": list(_FOLDS),
            "fold_counts": fold_counts,
            "program_vocabulary": semantic["representation"]["program_vocabulary"],
            "atom_vocabulary": _atom_vocabulary_payload(atom_vocabulary),
            "support": {
                **support,
                "observed_maximum_heavy_atoms": maximum_atoms,
                "observed_maximum_closures": maximum_closures,
            },
            "weight_policy": config["weight_policy"],
            "weight_fields": {program: "mean_realism_weight" for program in programs},
            "model_state_excludes": [
                "component_identifiers",
                "component_smiles",
                "component_fingerprints",
                "fragment_tokens",
                "biological_labels",
            ],
        }
        gates = {
            "semantic_qualification_passed": semantic["status"] == "pass",
            "all_semantic_fold_counts_exact": fold_counts == expected_counts,
            "all_semantic_records_packed": packed.records
            == sum(sum(value.values()) for value in expected_counts.values()),
            "every_family_has_required_folds": all(
                all(folds[fold] > 0 for fold in required_folds) for folds in fold_counts.values()
            ),
            "full_heavy_atom_support_preserved": maximum_atoms <= support["maximum_heavy_atoms"],
            "full_closure_support_preserved": maximum_closures <= support["maximum_closures"],
            "fixed_state_policy_exact": fixed_failures == 0,
            "raw_family_count_sampling_absent": True,
        }
        if not all(gates.values()):
            raise CombinatorialProgramCacheError(
                f"combinatorial cache failed before publication: {gates}"
            )
        cache_path = work / "cache.npz"
        _write_deterministic_npz(cache_path, packed.arrays(metadata))
        with SynthesisProgramProductionCache(cache_path) as cache:
            masses = {program: 1.0 / len(programs) for program in programs}
            measure = cache.training_measure(masses)
            observed_mass = {
                program: float(measure[cache.indices(program_id=program, fold="train")].sum())
                for program in programs
            }
            if cache.fold_counts() != expected_counts or any(
                not np.isclose(value, 1.0 / len(programs)) for value in observed_mass.values()
            ):
                raise SynthesisProgramProductionCacheError(
                    "published cache failed count or equal-family-mass validation"
                )
        for name, pin in raw_inputs.items():
            resolve_pin(pin, repo, label=name)
        if str(sha256_file(config_path)) != config_hash or any(
            str(sha256_file(source_root / name)) != digest for name, digest in sources.items()
        ):
            raise CombinatorialProgramCacheError(
                "code or configuration changed during cache construction"
            )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "pass",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            "config": {"path": str(config_path.relative_to(repo)), "sha256": config_hash},
            "inputs": raw_inputs,
            "sources": sources,
            "artifacts": {"cache.npz": _artifact(cache_path, output / "cache.npz", repo)},
            "environment": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
            "records": packed.records,
            "fold_counts": fold_counts,
            "equal_family_training_mass": observed_mass,
            "support": metadata["support"],
            "fixed_state_failures": fixed_failures,
            "gates": gates,
            "training_calls": 0,
            "generator_sampling_calls": 0,
            "nonclaims": [
                "This cache is model-input infrastructure, not evidence of model generalization or improved realism.",
                "Transform-consistent products remain reaction-enumerated support, not route-certified products.",
                "Semantic abstentions remain excluded and visible in the pinned upstream audit.",
                "No route, oracle, candidate selection, biological label or reductive-amination substructure metric is present.",
            ],
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(work, output)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "CombinatorialProgramCacheError",
    "build_combinatorial_program_cache",
]
