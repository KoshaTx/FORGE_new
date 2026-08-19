#!/usr/bin/env python3
"""Compare native Ugi flow endpoints with terminal-logit re-decoding."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_chemistry_flow import (
    UgiChemistryFlowError,
    chemistry_sample_statistics,
    chemistry_sample_to_molecule,
)
from forge.product.ugi_end_to_end_sampling import _closure_model, _load_checkpoint
from forge.product.ugi_generated_components import (
    UgiGeneratedComponentError,
    generated_ugi_component_smiles,
)
from forge.product.ugi_held_component_gate import load_ugi_reaction_contract
from forge.product.ugi_joint_end_to_end_sampling import (
    _annotate_l1_terminal_admission,
    _load_matched_programs,
    complete_ugi_joint_terminals,
    flow_endpoint_chemistry_sample,
)
from forge.product.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    sample_ugi_joint_sparse_terminals,
)
from forge.product.ugi_terminal_decoder_challenger import (
    TAIL_ROLES,
    summarize_terminal_decoder_arm,
)
from forge.product.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _endpoint_arm(
    conditions: tuple[Any, ...],
    terminals: list[Any],
    terminal_rows: tuple[dict[str, Any], ...],
    terminal_samples: tuple[Any, ...],
    atom_vocabulary: tuple[Any, ...],
    reaction: Any,
) -> tuple[dict[str, Any], list[Any]]:
    rows: list[dict[str, Any]] = []
    samples: list[Any] = []
    for condition, terminal, terminal_row, corrected_sample in zip(
        conditions,
        terminals,
        terminal_rows,
        terminal_samples,
        strict=True,
    ):
        row = {
            key: value
            for key, value in terminal_row.items()
            if key
            in {
                "structure_id",
                "product_id",
                "program",
                "offspring_by_role",
                "source_stratum",
                "branch_class",
                "component_novelty_class",
                "held_role_class",
            }
        }
        row["terminal_decoder_mode"] = "native_flow_endpoint_shared_closure"
        if corrected_sample is None:
            sample = None
            row.update(
                {
                    "smiles": None,
                    "valid": False,
                    "failure_type": "MatchedClosureCompletionUnavailable",
                    "component_smiles_by_role": None,
                    "component_reconstruction_valid": False,
                    "component_reconstruction_error": "matched terminal decoder did not complete",
                }
            )
            _annotate_l1_terminal_admission(row, reaction)
            rows.append(row)
            samples.append(sample)
            continue
        try:
            sample = flow_endpoint_chemistry_sample(
                condition,
                terminal,
                corrected_sample.closure_bond_states,
            )
            molecule = chemistry_sample_to_molecule(condition, sample, atom_vocabulary)
            row.update(
                {
                    "smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
                    "valid": True,
                    "failure_type": None,
                }
            )
            try:
                row["component_smiles_by_role"] = generated_ugi_component_smiles(
                    condition,
                    sample,
                    atom_vocabulary,
                )
                row["component_reconstruction_valid"] = True
                row["component_reconstruction_error"] = None
            except UgiGeneratedComponentError as error:
                row["component_smiles_by_role"] = None
                row["component_reconstruction_valid"] = False
                row["component_reconstruction_error"] = str(error)
        except (
            IndexError,
            RuntimeError,
            ValueError,
            UgiChemistryFlowError,
        ) as error:
            sample = None
            row.update(
                {
                    "smiles": None,
                    "valid": False,
                    "failure_type": type(error).__name__,
                    "component_smiles_by_role": None,
                    "component_reconstruction_valid": False,
                    "component_reconstruction_error": str(error),
                }
            )
        _annotate_l1_terminal_admission(row, reaction)
        rows.append(row)
        samples.append(sample)
    statistics = chemistry_sample_statistics(conditions, samples, atom_vocabulary)
    return {
        "schema_version": "phase1_ugi_flow_endpoint_joint_chemistry_arm.v1",
        "samples": rows,
        "statistics": statistics,
    }, samples


def _tail_feature_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role in TAIL_ROLES:
        left_features = left["tail_chemotypes_exact_l1_eligible_only"][role][
            "feature_occurrence_fractions"
        ]
        right_features = right["tail_chemotypes_exact_l1_eligible_only"][role][
            "feature_occurrence_fractions"
        ]
        output[role] = {
            feature: float(right_features[feature]) - float(left_features[feature])
            for feature in sorted(left_features)
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_flow_endpoint_joint_chemistry_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_flow_endpoint_joint_chemistry_audit_v1.json",
    )
    args = parser.parse_args()

    import torch

    config = _load_json(args.config)
    if config.get("schema_version") != "phase1_ugi_flow_endpoint_joint_chemistry_audit_config.v1":
        raise ValueError("unexpected flow-endpoint audit config schema")
    verified_inputs = {}
    for name, specification in config["inputs"].items():
        path = REPO / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise ValueError(f"frozen flow-endpoint audit input changed: {name}")
        verified_inputs[name] = {"path": specification["path"], "sha256": digest}

    design = config["design"]
    joint_checkpoint = _load_checkpoint(
        REPO / config["inputs"]["joint_checkpoint"]["path"],
        "phase1_ugi_joint_sparse_checkpoint.v1",
    )
    closure_checkpoint = _load_checkpoint(
        REPO / config["inputs"]["closure_checkpoint"]["path"],
        (
            "phase1_ugi_sparse_closure_checkpoint.v1",
            "phase1_ugi_sparse_closure_checkpoint.v2",
        ),
    )
    cache_path = Path(joint_checkpoint["inputs"]["prepared_cache"]["path"])
    if not cache_path.is_file() and str(cache_path).startswith("/root/forge_repo/"):
        cache_path = REPO / cache_path.relative_to("/root/forge_repo")
    corpus, _ = load_ugi_training_cache(cache_path)
    programs, metadata = _load_matched_programs(REPO / config["inputs"]["matched_programs"]["path"])
    sample_count = int(design["programs"])
    programs = programs[:sample_count]
    metadata = metadata[:sample_count]
    architecture = dict(joint_checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(
        atom_classes=len(corpus.atom_vocabulary),
        **architecture,
    )
    model.load_state_dict(joint_checkpoint["model_state"])
    closure_model = _closure_model(closure_checkpoint)
    terminals, flow_statistics = sample_ugi_joint_sparse_terminals(
        model,
        programs,
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in joint_checkpoint["source_marginals"].items()
        },
        sample_steps=int(design["sample_steps"]),
        batch_size=int(design["batch_size"]),
        seed=int(design["flow_seed"]),
        device="cpu",
        allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
        maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
        maximum_adjacent_branch_runs=design["maximum_adjacent_branch_runs"],
    )
    reaction = load_ugi_reaction_contract(REPO / config["inputs"]["qualified_reactions"]["path"])
    completion = complete_ugi_joint_terminals(
        model,
        closure_model,
        terminals,
        corpus,
        program_metadata=metadata,
        closure_generator_state=torch.Generator()
        .manual_seed(int(design["closure_seed"]))
        .get_state(),
        allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
        maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
        l1_reaction=reaction,
        terminal_decoder_mode=str(design["terminal_decoder_mode"]),
        terminal_generator_state=torch.Generator()
        .manual_seed(int(design["terminal_seed"]))
        .get_state(),
        terminal_temperature=float(design["terminal_temperature"]),
    )
    terminal_raw = {
        "schema_version": "phase1_ugi_flow_endpoint_joint_chemistry_arm.v1",
        "samples": list(completion.rows),
        "statistics": chemistry_sample_statistics(
            completion.conditions,
            completion.samples,
            corpus.atom_vocabulary,
        ),
    }
    endpoint_raw, _ = _endpoint_arm(
        completion.conditions,
        terminals,
        completion.rows,
        completion.samples,
        corpus.atom_vocabulary,
        reaction,
    )
    terminal_summary = summarize_terminal_decoder_arm(terminal_raw)
    endpoint_summary = summarize_terminal_decoder_arm(endpoint_raw)
    reference = _load_json(REPO / config["inputs"]["balanced_training_reference"]["path"])
    output = {
        "schema_version": "phase1_ugi_flow_endpoint_joint_chemistry_audit.v1",
        "status": "complete_selection_visible_paired_diagnostic",
        "config": {
            "path": str(args.config.relative_to(REPO)),
            "sha256": sha256_file(args.config),
        },
        "audit_source_sha256": sha256_file(Path(__file__)),
        "inputs": verified_inputs,
        "flow_sampling": flow_statistics,
        "arms": {
            "terminal_logit_redraw": terminal_summary,
            "native_flow_endpoint_shared_closure": endpoint_summary,
        },
        "endpoint_minus_terminal_feature_fraction": _tail_feature_delta(
            terminal_summary,
            endpoint_summary,
        ),
        "balanced_training_target": {
            role: reference["role_component_reference"][role]["feature_occurrence_fractions"]
            for role in TAIL_ROLES
        },
        "interpretation_policy": config["decision_policy"],
        "raw_arms": {
            "terminal_logit_redraw": terminal_raw,
            "native_flow_endpoint_shared_closure": endpoint_raw,
        },
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arms": output["arms"],
                "feature_delta": output["endpoint_minus_terminal_feature_fraction"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
