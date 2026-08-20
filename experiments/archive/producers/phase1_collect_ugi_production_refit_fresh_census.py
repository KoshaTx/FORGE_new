#!/usr/bin/env python3
"""Collect and audit the independent all-fold-refit generator census."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from forge.corpus.r1_prime_audit import sha256_file
from forge.potency.annotations import ROLE_NAMES
from forge.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.corpus.ugi_held_component_gate import _reaction_contract
from experiments.phase1.product_l1.evaluation.tail_chemotype import (
    compare_component_cohorts,
    summarize_component_cohort,
)

REPO = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "phase1_ugi_production_refit_fresh_census.v1"
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")


class FreshCensusError(RuntimeError):
    """Raised when the fresh-census contract or outputs drift."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FreshCensusError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
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


def _canonical(smiles: str) -> tuple[Chem.Mol, str]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise FreshCensusError(f"invalid connected SMILES in admitted row: {smiles}")
    return molecule, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _pin(repo: Path, record: dict[str, Any], *, label: str) -> Path:
    if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
        raise FreshCensusError(f"malformed input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise FreshCensusError(f"input pin escapes repository: {label}") from error
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise FreshCensusError(f"input pin changed: {label}")
    return path


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or np.any(~np.isfinite(array)):
        raise FreshCensusError("quantiles require finite observations")
    return {
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _nearest_component_similarity(generated: list[str], reference: list[str]) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reference_unique = sorted(set(reference))
    reference_fingerprints = [
        generator.GetFingerprint(_canonical(smiles)[0]) for smiles in reference_unique
    ]
    generated_unique = sorted(set(generated))
    maxima = []
    for smiles in generated_unique:
        fingerprint = generator.GetFingerprint(_canonical(smiles)[0])
        maxima.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints)))
    return {
        "generated_unique_components": len(generated_unique),
        "reference_unique_components": len(reference_unique),
        "nearest_reference_tanimoto_quantiles": _quantiles(maxima),
        "fraction_at_least_0_4": sum(value >= 0.4 for value in maxima) / len(maxima),
        "fraction_at_least_0_6": sum(value >= 0.6 for value in maxima) / len(maxima),
        "fraction_at_least_0_8": sum(value >= 0.8 for value in maxima) / len(maxima),
        "fraction_exact_reference_identity": sum(math.isclose(value, 1.0) for value in maxima)
        / len(maxima),
    }


def collect_census(repo: Path, config_path: Path) -> dict[str, Any]:
    """Verify, combine and summarize four independent census shards."""

    config = _json(config_path)
    if (
        config.get("schema_version") != "phase1_ugi_production_refit_fresh_census_config.v1"
        or config.get("status") != "frozen_before_independent_all_fold_refit_census"
    ):
        raise FreshCensusError("unsupported fresh-census config")
    inputs = {label: _pin(repo, record, label=label) for label, record in config["inputs"].items()}
    design = config["design"]
    shards = int(design["shards"])
    per_shard = int(design["programs_per_shard"])
    rows: list[dict[str, Any]] = []
    shard_receipts = []
    for shard in range(shards):
        result_path = (
            repo
            / f"results/phase1/ugi_production_refit_fresh_census_v1/shard_{shard:02d}/result.json"
        )
        result = _json(result_path)
        program_path = inputs[f"program_shard_{shard:02d}"]
        expected_program = _json(program_path)
        sampling = result.get("sampling", {})
        decoder = sampling.get("terminal_decoder", {})
        shard_rows = result.get("samples")
        if (
            int(result.get("seed", -1)) != int(design["flow_seeds"][shard])
            or decoder.get("mode") != design["terminal_decoder_mode"]
            or int(decoder.get("seed", -1)) != int(design["terminal_seeds"][shard])
            or float(decoder.get("temperature", -1.0)) != float(design["terminal_temperature"])
            or sampling.get("maximum_adjacent_branch_runs")
            != design["maximum_adjacent_branch_runs"]
            or int(sampling.get("sample_steps", -1)) != int(design["sample_steps"])
            or sampling.get("terminal_tree_repairs") != 0
            or not isinstance(shard_rows, list)
            or len(shard_rows) != per_shard
            or len(expected_program.get("samples", ())) != per_shard
        ):
            raise FreshCensusError(f"fresh-census sampling contract changed: shard {shard}")
        for index, (row, expected) in enumerate(
            zip(shard_rows, expected_program["samples"], strict=True)
        ):
            if row.get("program") != expected.get("program"):
                raise FreshCensusError(f"program alignment changed: shard {shard}:{index}")
            combined = dict(row)
            combined["census_shard"] = shard
            combined["census_index"] = shard * per_shard + index
            rows.append(combined)
        shard_receipts.append(
            {
                "shard": shard,
                "result": {
                    "path": str(result_path.relative_to(repo)),
                    "sha256": sha256_file(result_path),
                },
                "attempted": len(shard_rows),
                "valid": sum(row.get("valid") is True for row in shard_rows),
            }
        )
    if len(rows) != int(design["attempted_draws"]):
        raise FreshCensusError("combined census size changed")

    reference_products: set[str] = set()
    reference_components = {role: [] for role in ROLE_NAMES}
    with gzip.open(inputs["assignments"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            reference_products.add(str(row["canonical_product_smiles"]))
            for role in ROLE_NAMES:
                reference_components[role].append(str(row[f"{role}_smiles"]))
    if len(reference_products) != 112386:
        raise FreshCensusError("all-fold product reference changed")

    reaction = _reaction_contract(inputs["qualified_reactions"])
    valid_rows = [row for row in rows if row.get("valid") is True]
    canonical_products = []
    generated_components = {role: [] for role in ROLE_NAMES}
    handles_pass = 0
    exact_l1 = 0
    exact_refit = 0
    for row in valid_rows:
        _, canonical_product = _canonical(str(row["smiles"]))
        canonical_products.append(canonical_product)
        exact_refit += int(canonical_product in reference_products)
        components = row.get("component_smiles_by_role")
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            continue
        if (
            row.get("component_reconstruction_valid") is not True
            or (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed")
            is not True
        ):
            continue
        exact_l1 += 1
        row_handles = True
        for role_index, role in enumerate(ROLE_NAMES):
            molecule, canonical_component = _canonical(str(components[role]))
            generated_components[role].append(canonical_component)
            qualification = reaction_handle_qualification(
                molecule,
                query=reaction.handles[role_index],
                forbidden=reaction.forbidden[role_index],
                allowed_site_multiplicity=reaction.definition.reactant_roles[
                    role_index
                ].allowed_site_multiplicity,
            )
            row_handles = row_handles and bool(qualification["passes_registry_handle_policy"])
        handles_pass += int(row_handles)

    generated_tail = {
        role: summarize_component_cohort(generated_components[role]) for role in TAIL_ROLES
    }
    reference_tail = {
        role: summarize_component_cohort(reference_components[role]) for role in TAIL_ROLES
    }
    tail_comparison = {
        role: compare_component_cohorts(generated_tail[role], reference_tail[role])
        for role in TAIL_ROLES
    }
    component_similarity = {
        role: _nearest_component_similarity(generated_components[role], reference_components[role])
        for role in ROLE_NAMES
    }
    valid_count = len(valid_rows)
    reconstructed = min(len(generated_components[role]) for role in ROLE_NAMES)
    unique_valid = len(set(canonical_products))
    policy = config["decision_policy"]
    checks = {
        "valid_fraction": valid_count / len(rows) >= float(policy["minimum_valid_fraction"]),
        "exact_l1": exact_l1 / max(valid_count, 1)
        == float(policy["required_exact_l1_fraction_of_valid"]),
        "uniqueness": unique_valid / max(valid_count, 1)
        >= float(policy["minimum_unique_fraction_of_valid"]),
        "three_handles": handles_pass / max(reconstructed, 1)
        >= float(policy["minimum_three_handle_fraction_of_reconstructed"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
        "shards": shard_receipts,
        "summary": {
            "attempted_draws": len(rows),
            "valid_products": valid_count,
            "valid_fraction": valid_count / len(rows),
            "unique_valid_products": unique_valid,
            "unique_fraction_of_valid": unique_valid / max(valid_count, 1),
            "exact_l1_products": exact_l1,
            "exact_l1_fraction_of_valid": exact_l1 / max(valid_count, 1),
            "all_three_handles_qualified": handles_pass,
            "all_three_handle_fraction_of_reconstructed": handles_pass / max(reconstructed, 1),
            "exact_refit_corpus_products": exact_refit,
            "exact_refit_corpus_fraction_of_valid": exact_refit / max(valid_count, 1),
            "invalid_failure_types": dict(
                sorted(
                    Counter(
                        str(row.get("failure_type") or "invalid_unspecified")
                        for row in rows
                        if row.get("valid") is not True
                    ).items()
                )
            ),
        },
        "tail_chemotypes": {
            "generated": generated_tail,
            "all_fold_reference": reference_tail,
            "comparison": tail_comparison,
        },
        "component_nearest_reference_similarity": component_similarity,
        "gate": {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        },
        "interpretation": {
            "selection_outputs_reused": False,
            "checkpoint_or_threshold_changed": False,
            "chemotype_summaries_are_descriptive": True,
            "prospective_candidate_lock_authorized_by_this_audit_alone": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_production_refit_fresh_census_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_production_refit_fresh_census_v1/result.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    result = collect_census(REPO.resolve(), config_path.resolve())
    _atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))
    print(sha256_file(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
