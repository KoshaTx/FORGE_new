#!/usr/bin/env python3
"""Run a nonselecting chemistry-motif and candidate-eligibility audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase

from forge.potency.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_candidate_eligibility import (
    compile_smarts,
    declared_support_violations,
    heteroatom_extreme_flags,
    heteroatom_profile,
    heteroatom_thresholds,
    motif_hits,
)
from forge.product.ugi_component_expansion import reaction_handle_qualification
from forge.product.ugi_held_component_gate import (
    _canonical_molecule,
    _forward_reconstructs_product,
    _reaction_contract,
)

REPO = Path(__file__).resolve().parents[1]


class CandidateEligibilityAuditError(RuntimeError):
    """Raised when a frozen audit input or invariant changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CandidateEligibilityAuditError(f"expected JSON object: {path}")
    return value


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


def _require_input(specification: dict[str, Any]) -> Path:
    path = _resolve(specification["path"])
    observed = _sha256(path)
    if observed != specification["sha256"]:
        raise CandidateEligibilityAuditError(f"input hash mismatch: {path}: {observed}")
    return path


def _read_assignments(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _atom_vocabulary(path: Path) -> set[tuple[str, int, bool, int]]:
    payload = _load_json(path)
    return {
        (
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row["explicit_hydrogens"]),
        )
        for row in payload["atom_vocabulary"]
    }


def _canonicalized_molecule(smiles: str) -> tuple[Chem.Mol, str]:
    molecule = _canonical_molecule(smiles)
    return molecule, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def _append_example(
    examples: list[dict[str, Any]],
    value: dict[str, Any],
    maximum: int,
) -> None:
    if len(examples) < maximum:
        examples.append(value)


def audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    inputs = config["inputs"]
    resolved = {name: _require_input(specification) for name, specification in inputs.items()}

    manifest = _load_json(resolved["production_manifest"])
    selection = _load_json(resolved["selection_result"])
    sample = _load_json(resolved["selected_sample"])
    if manifest["status"] != "frozen":
        raise CandidateEligibilityAuditError("production generator is not frozen")
    if selection["decision"]["selected_candidate"] != config["selected_candidate"]:
        raise CandidateEligibilityAuditError("selection candidate changed")
    if manifest["selection"]["result"]["sha256"] != inputs["selection_result"]["sha256"]:
        raise CandidateEligibilityAuditError("manifest selection result changed")
    if (
        manifest["selection"]["fresh_selected_sample"]["sha256"]
        != inputs["selected_sample"]["sha256"]
    ):
        raise CandidateEligibilityAuditError("manifest selected sample changed")
    if int(sample["seed"]) != int(config["sampling_contract"]["seed"]):
        raise CandidateEligibilityAuditError("selected sample seed changed")
    rows = sample["samples"]
    if len(rows) != int(config["sampling_contract"]["attempted_draws"]):
        raise CandidateEligibilityAuditError("selected sample count changed")

    assignments = _read_assignments(resolved["assignments"])
    fold_field = config["selection_reference"]["fold_field"]
    visible_folds = set(config["selection_reference"]["folds"])
    source_fold_counts = Counter(str(row[fold_field]) for row in assignments)
    visible_rows = [row for row in assignments if str(row[fold_field]) in visible_folds]
    if len(visible_rows) != int(config["selection_reference"]["expected_rows"]):
        raise CandidateEligibilityAuditError("selection-visible reference count changed")
    reference_smiles = [str(row["canonical_product_smiles"]) for row in visible_rows]
    reference_set = set(reference_smiles)
    if len(reference_set) != len(reference_smiles):
        raise CandidateEligibilityAuditError("selection-visible reference is not unique")

    queries = compile_smarts(config["motif_smarts"])
    maximum_examples = int(config["reporting"]["maximum_examples_per_flag"])
    reference_motif_counts = Counter()
    reference_motif_examples = {name: [] for name in queries}
    reference_hetero_counts = []
    reference_hetero_fractions = []
    with rdBase.BlockLogs():
        for smiles in reference_smiles:
            molecule, canonical = _canonicalized_molecule(smiles)
            hits = motif_hits(molecule, queries)
            for name, present in hits.items():
                if present:
                    reference_motif_counts[name] += 1
                    _append_example(
                        reference_motif_examples[name],
                        {"smiles": canonical},
                        maximum_examples,
                    )
            count, fraction = heteroatom_profile(molecule)
            reference_hetero_counts.append(count)
            reference_hetero_fractions.append(fraction)

    lower_quantile = float(config["extreme_heteroatoms"]["lower_quantile"])
    upper_quantile = float(config["extreme_heteroatoms"]["upper_quantile"])
    thresholds = heteroatom_thresholds(
        np.asarray(reference_hetero_counts, dtype=np.float64),
        np.asarray(reference_hetero_fractions, dtype=np.float64),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    reference_extreme_counts = Counter()
    reference_extreme_examples = {
        "low_heteroatom_count": [],
        "high_heteroatom_count": [],
        "low_heteroatom_fraction": [],
        "high_heteroatom_fraction": [],
        "any_extreme_heteroatom_pattern": [],
    }
    for smiles, count, fraction in zip(
        reference_smiles,
        reference_hetero_counts,
        reference_hetero_fractions,
        strict=True,
    ):
        flags = heteroatom_extreme_flags(count, fraction, thresholds)
        for name, present in flags.items():
            if present:
                reference_extreme_counts[name] += 1
                _append_example(
                    reference_extreme_examples[name],
                    {"smiles": smiles, "heteroatoms": count, "fraction": fraction},
                    maximum_examples,
                )

    checkpoint = torch.load(resolved["checkpoint"], map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    atom_vocabulary = _atom_vocabulary(resolved["atom_vocabulary"])
    reaction = _reaction_contract(resolved["qualified_reactions"])

    generated_motif_counts = Counter()
    generated_motif_examples = {name: [] for name in queries}
    generated_extreme_counts = Counter()
    generated_extreme_examples = {name: [] for name in reference_extreme_examples}
    support_violation_counts = Counter()
    support_violation_products = 0
    support_violation_examples = []
    failure_types = Counter()
    valid_products = reconstructed = forward_passes = handle_passes = 0
    forbidden_products = exact_reference_products = 0
    generated_hetero_counts = []
    generated_hetero_fractions = []

    with rdBase.BlockLogs():
        for index, row in enumerate(rows):
            if not row.get("valid"):
                failure_types[str(row.get("failure_type") or "invalid_unspecified")] += 1
                continue
            valid_products += 1
            molecule, canonical = _canonicalized_molecule(str(row["smiles"]))
            exact_reference = canonical in reference_set
            exact_reference_products += int(exact_reference)
            hits = motif_hits(molecule, queries)
            for name, present in hits.items():
                if present:
                    generated_motif_counts[name] += 1
                    _append_example(
                        generated_motif_examples[name],
                        {
                            "sample_index": index,
                            "smiles": canonical,
                            "exact_selection_visible_product": exact_reference,
                        },
                        maximum_examples,
                    )
            count, fraction = heteroatom_profile(molecule)
            generated_hetero_counts.append(count)
            generated_hetero_fractions.append(fraction)
            extreme_flags = heteroatom_extreme_flags(count, fraction, thresholds)
            for name, present in extreme_flags.items():
                if present:
                    generated_extreme_counts[name] += 1
                    _append_example(
                        generated_extreme_examples[name],
                        {
                            "sample_index": index,
                            "smiles": canonical,
                            "heteroatoms": count,
                            "fraction": fraction,
                            "exact_selection_visible_product": exact_reference,
                        },
                        maximum_examples,
                    )

            violations = declared_support_violations(row, molecule, model_config, atom_vocabulary)
            if violations:
                support_violation_products += 1
                support_violation_counts.update(violations)
                _append_example(
                    support_violation_examples,
                    {"sample_index": index, "smiles": canonical, "violations": violations},
                    maximum_examples,
                )

            if not row.get("component_reconstruction_valid"):
                continue
            components = row.get("component_smiles_by_role")
            if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
                continue
            reconstructed += 1
            canonical_components = {}
            handles_pass = True
            forbidden = False
            for role_index, role in enumerate(ROLE_NAMES):
                component, component_smiles = _canonicalized_molecule(str(components[role]))
                canonical_components[role] = component_smiles
                qualification = reaction_handle_qualification(
                    component,
                    query=reaction.handles[role_index],
                    forbidden=reaction.forbidden[role_index],
                    allowed_site_multiplicity=reaction.definition.reactant_roles[
                        role_index
                    ].allowed_site_multiplicity,
                )
                handles_pass = handles_pass and bool(qualification["passes_registry_handle_policy"])
                forbidden = forbidden or bool(qualification["forbidden_substructure_match"])
            handle_passes += int(handles_pass)
            forbidden_products += int(forbidden)
            forward, _, _ = _forward_reconstructs_product(reaction, canonical_components, canonical)
            forward_passes += int(forward)

    attempted = len(rows)
    motif_results = {}
    for name, smarts in config["motif_smarts"].items():
        reference_count = reference_motif_counts[name]
        generated_count = generated_motif_counts[name]
        reference_fraction = reference_count / len(reference_smiles)
        generated_fraction = generated_count / valid_products
        motif_results[name] = {
            "smarts": smarts,
            "reference": {
                **_rate(reference_count, len(reference_smiles)),
                "examples": reference_motif_examples[name],
            },
            "generated_valid": {
                **_rate(generated_count, valid_products),
                "examples": generated_motif_examples[name],
            },
            "generated_attempted": _rate(generated_count, attempted),
            "absolute_fraction_difference_generated_minus_reference": (
                generated_fraction - reference_fraction
            ),
            "fraction_ratio_generated_over_reference": (
                generated_fraction / reference_fraction if reference_fraction else None
            ),
            "reference_zero": reference_count == 0,
        }

    extreme_results = {}
    for name in reference_extreme_examples:
        extreme_results[name] = {
            "reference": {
                **_rate(reference_extreme_counts[name], len(reference_smiles)),
                "examples": reference_extreme_examples[name],
            },
            "generated_valid": {
                **_rate(generated_extreme_counts[name], valid_products),
                "examples": generated_extreme_examples[name],
            },
            "generated_attempted": _rate(generated_extreme_counts[name], attempted),
        }

    result = {
        "schema_version": "phase1_ugi_candidate_eligibility_audit.v1",
        "status": "complete_nonselecting",
        "scope": {
            "can_change_architecture_or_checkpoint_selection": False,
            "candidate_eligibility_diagnostic_only": True,
            "heldout_fold_evaluated": False,
            "l2_routes_evaluated": False,
            "biological_guidance_evaluated": False,
            "synthesis_guidance_evaluated": False,
        },
        "config": {"path": str(config_path.relative_to(REPO)), "sha256": _sha256(config_path)},
        "inputs": {
            name: {"path": str(path.relative_to(REPO)), "sha256": _sha256(path)}
            for name, path in resolved.items()
        },
        "selection_reference": {
            "folds": sorted(visible_folds),
            "products": len(reference_smiles),
            "source_fold_counts": dict(sorted(source_fold_counts.items())),
            "heldout_rows_skipped_without_molecule_evaluation": sum(
                count for fold, count in source_fold_counts.items() if fold not in visible_folds
            ),
        },
        "generated_denominators": {
            "attempted": attempted,
            "valid": valid_products,
            "invalid": attempted - valid_products,
            "component_reconstructed": reconstructed,
        },
        "invalid_failure_types": dict(sorted(failure_types.items())),
        "motifs": motif_results,
        "extreme_heteroatoms": {
            "definition": config["extreme_heteroatoms"],
            "thresholds_from_selection_reference": thresholds,
            "reference_count_summary": {
                "minimum": int(min(reference_hetero_counts)),
                "maximum": int(max(reference_hetero_counts)),
            },
            "generated_count_summary": {
                "minimum": int(min(generated_hetero_counts)),
                "maximum": int(max(generated_hetero_counts)),
            },
            "flags": extreme_results,
        },
        "adapter_and_declared_support": {
            "valid_product_declared_support_pass": _rate(
                valid_products - support_violation_products,
                valid_products,
            ),
            "declared_support_violation_counts": dict(sorted(support_violation_counts.items())),
            "declared_support_violation_examples": support_violation_examples,
            "component_reconstruction_pass": _rate(reconstructed, valid_products),
            "exact_forward_reconstruction_pass": _rate(forward_passes, reconstructed),
            "all_three_handle_qualification_pass": _rate(handle_passes, reconstructed),
            "forbidden_substructure_products": _rate(forbidden_products, reconstructed),
            "exact_selection_visible_products": _rate(exact_reference_products, valid_products),
        },
        "interpretation_contract": {
            "automatic_candidate_ineligibility": [
                "declared graph-support violation",
                "failed deterministic component reconstruction",
                "failed exact Ugi forward reconstruction",
                "failed frozen handle policy",
                "forbidden substructure match",
            ],
            "manual_review_flags_not_automatic_rejections": [
                "motif SMARTS match",
                "selection-reference q01-q99 heteroatom excursion",
            ],
            "no_synthesis_claim": "A motif absence or presence is not a synthesis-success or synthesis-failure result.",
        },
    }
    _atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_candidate_eligibility_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_candidate_eligibility_audit_v1.json",
    )
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "generated_denominators": result["generated_denominators"],
                "motif_counts": {
                    name: value["generated_valid"]["count"]
                    for name, value in result["motifs"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
