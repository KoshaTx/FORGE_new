"""Evaluate bounded chemistry-plus-morphology generator challengers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rdkit import rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.design.ugi_chemistry_bias_attribution import compact_product_motif_summary
from forge.design.ugi_terminal_decoder_challenger import summarize_terminal_decoder_arm

CONFIG_SCHEMA_VERSION = "phase1_ugi_chemistry_morphology_challenger_evaluation_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_chemistry_morphology_challenger_evaluation.v1"


class UgiChemistryMorphologyChallengerError(ValueError):
    """Raised when a challenger evaluation is not reproducible."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiChemistryMorphologyChallengerError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiChemistryMorphologyChallengerError(f"{label} must be an object")
    return value


def challenger_checks(
    arm: Mapping[str, Any],
    product_motifs: Mapping[str, Any],
    *,
    unique_fraction: float,
    reproduction_fraction: float,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    """Apply only the gates frozen before the challenger sample was inspected."""

    aldehyde = arm["tail_chemotypes_exact_l1_eligible_only"]["oxoester_aldehyde_body_tail"][
        "feature_occurrence_fractions"
    ]
    branch_lower, branch_upper = [
        float(value) for value in gates["aldehyde_branch_fraction_interval"]
    ]
    motifs = product_motifs["motifs"]
    return {
        "minimum_valid_fraction": float(arm["valid_fraction"])
        >= float(gates["minimum_valid_fraction"]),
        "exact_l1_fraction_of_valid": float(arm["exact_l1_fraction_of_valid"])
        >= float(gates["exact_l1_fraction_of_valid"]),
        "minimum_unique_fraction_of_valid": unique_fraction
        >= float(gates["minimum_unique_fraction_of_valid"]),
        "maximum_exact_frozen_product_reproduction_fraction": reproduction_fraction
        <= float(gates["maximum_exact_frozen_product_reproduction_fraction"]),
        "maximum_product_alkyne_fraction": float(motifs["alkyne"]["fraction"])
        <= float(gates["maximum_product_alkyne_fraction"]),
        "maximum_product_allene_or_cumulene_fraction": float(
            motifs["allene_or_carbon_cumulene"]["fraction"]
        )
        <= float(gates["maximum_product_allene_or_cumulene_fraction"]),
        "minimum_aldehyde_double_bond_fraction": float(aldehyde["has_carbon_carbon_double_bond"])
        >= float(gates["minimum_aldehyde_double_bond_fraction"]),
        "maximum_aldehyde_triple_bond_fraction": float(aldehyde["has_carbon_carbon_triple_bond"])
        <= float(gates["maximum_aldehyde_triple_bond_fraction"]),
        "minimum_aldehyde_ester_fraction": float(aldehyde["has_ester_like_carbonyl"])
        >= float(gates["minimum_aldehyde_ester_fraction"]),
        "aldehyde_branch_fraction_interval": branch_lower
        <= float(aldehyde["has_carbon_branch"])
        <= branch_upper,
    }


def build_challenger_evaluation(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build a hash-pinned evaluation for one nonselecting challenger draw."""

    config = _load_json(config_path, label="challenger evaluation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiChemistryMorphologyChallengerError("unsupported evaluation config")
    loaded: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise UgiChemistryMorphologyChallengerError(f"input hash mismatch: {name}")
        loaded[name] = _load_json(path, label=name)
        inputs[name] = {"path": str(path.relative_to(repo)), "sha256": observed}
    contract = loaded["challenger_contract"]
    sample = loaded["challenger_sample"]
    if len(sample.get("samples", [])) != int(contract["sampling"]["attempted_draws"]):
        raise UgiChemistryMorphologyChallengerError("challenger sample count changed")
    arm = summarize_terminal_decoder_arm(sample)
    exact_rows = [
        row
        for row in sample["samples"]
        if row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]
    with rdBase.BlockLogs():
        product_motifs = compact_product_motif_summary(
            (str(row["smiles"]) for row in exact_rows), maximum_examples=5
        )
    valid = int(arm["valid_molecules"])
    unique_fraction = int(sample["statistics"]["unique_valid_molecules"]) / valid
    exact_matches = int(sample["reference_comparison"]["all_frozen_ugi"]["exact_matches"])
    reproduction_fraction = exact_matches / valid
    checks = challenger_checks(
        arm,
        product_motifs,
        unique_fraction=unique_fraction,
        reproduction_fraction=reproduction_fraction,
        gates=contract["diagnostic_gates"],
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_challenger_evaluation",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs,
        "arm": arm,
        "product_motifs_exact_l1_only": product_motifs,
        "unique_fraction_of_valid": unique_fraction,
        "exact_frozen_product_reproduction_fraction": reproduction_fraction,
        "checks": checks,
        "all_diagnostic_gates_pass": all(checks.values()),
        "adjudication": {
            "production_generator_replaced": False,
            "prospective_candidate_selection_changed": False,
            "independent_confirmation_authorized": all(checks.values()),
            "no_retries_or_seed_search": True,
        },
    }
