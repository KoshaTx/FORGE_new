"""Descriptive post-selection evaluation on held-family morphology programs.

The component-family fold was inspected before the production checkpoint was
frozen.  This module therefore reports a stress test, not a pristine held-out
estimate and not a model-selection gate.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import write_json as _atomic_json
from forge.data.r0_splits import sha256_file
from forge.design.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.design.corpus.ugi_held_component_gate import (
    _canonical_molecule,
    _catalog_by_role,
    _forward_reconstructs_product,
    _reaction_contract,
    analyze_generated_components,
    held_role_class,
)
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

SCHEMA_VERSION = "phase1_ugi_product_l1_postselection_held_component_stress.v1"


class UgiPostselectionHeldComponentError(RuntimeError):
    """Raised when the descriptive stress-test contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiPostselectionHeldComponentError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() and str(path).startswith("/root/forge_repo/"):
        return repository / path.relative_to("/root/forge_repo")
    return path if path.is_absolute() else repository / path


def _require_input(repository: Path, specification: Mapping[str, Any]) -> Path:
    path = _resolve(repository, str(specification["path"]))
    if not path.is_file():
        raise UgiPostselectionHeldComponentError(f"missing input: {path}")
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise UgiPostselectionHeldComponentError(
            f"input hash mismatch: {path}: {observed} != {specification['sha256']}"
        )
    return path


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def component_membership_class(
    components: Mapping[str, str], catalog: Mapping[str, Mapping[str, str]]
) -> str:
    """Classify generated precursors against the frozen 424-component registry."""

    folds = [
        catalog[role].get(str(components[role]), "outside_admitted_catalog") for role in ROLE_NAMES
    ]
    if "outside_admitted_catalog" in folds:
        return "genuinely_generated_component"
    if "heldout" in folds:
        return "exact_heldout_component"
    if "calibration" in folds:
        return "exact_calibration_component"
    return "train_catalog_only"


def _validate_program_alignment(
    probe_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    assignments: Mapping[str, Mapping[str, str]],
) -> Counter[str]:
    if len(probe_rows) != len(sample_rows):
        raise UgiPostselectionHeldComponentError("probe and generated sample counts differ")
    held_classes: Counter[str] = Counter()
    for index, (probe, sample) in enumerate(zip(probe_rows, sample_rows, strict=True)):
        product_id = str(probe["product_id"])
        if str(sample.get("product_id")) != product_id:
            raise UgiPostselectionHeldComponentError(
                f"probe/sample product order differs at row {index}"
            )
        if sample.get("program") != probe.get("program"):
            raise UgiPostselectionHeldComponentError(
                f"probe/sample morphology program differs for {product_id}"
            )
        assignment = assignments.get(product_id)
        if assignment is None:
            raise UgiPostselectionHeldComponentError(f"probe product is unassigned: {product_id}")
        if assignment["primary_product_fold"] != "heldout":
            raise UgiPostselectionHeldComponentError(
                f"probe product is not in the heldout product fold: {product_id}"
            )
        expected_class = held_role_class(assignment)
        if probe.get("held_role_class") != expected_class:
            raise UgiPostselectionHeldComponentError(
                f"probe held-role class differs from frozen assignments: {product_id}"
            )
        if sample.get("held_role_class") != expected_class:
            raise UgiPostselectionHeldComponentError(
                f"sample held-role class differs from frozen assignments: {product_id}"
            )
        held_classes[expected_class] += 1
    return held_classes


def _stratified_generation_summary(
    rows: list[dict[str, Any]],
    *,
    catalog: Mapping[str, Mapping[str, str]],
    reaction: Any,
    product_fold_by_smiles: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
    unique: defaultdict[str, set[str]] = defaultdict(set)
    novelty: defaultdict[str, Counter[str]] = defaultdict(Counter)
    product_membership: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        label = str(row["held_role_class"])
        groups = (label, "all")
        for group in groups:
            counters[group]["attempted"] += 1
        if not row.get("valid"):
            for group in groups:
                counters[group]["invalid"] += 1
            continue
        smiles = str(row["smiles"])
        with rdBase.BlockLogs():
            molecule = _canonical_molecule(smiles)
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        membership = product_fold_by_smiles.get(canonical, "outside_frozen_product_corpus")
        for group in groups:
            counters[group]["valid"] += 1
            unique[group].add(canonical)
            product_membership[group][membership] += 1

        components = row.get("component_smiles_by_role")
        if not row.get("component_reconstruction_valid") or not isinstance(components, dict):
            continue
        if set(components) != set(ROLE_NAMES):
            raise UgiPostselectionHeldComponentError(
                f"reconstructed component roles are incomplete: {sorted(components)}"
            )
        component_class = component_membership_class(components, catalog)
        handles_pass = True
        for role_index, role in enumerate(ROLE_NAMES):
            qualification = reaction_handle_qualification(
                _canonical_molecule(str(components[role])),
                query=reaction.handles[role_index],
                forbidden=reaction.forbidden[role_index],
                allowed_site_multiplicity=reaction.definition.reactant_roles[
                    role_index
                ].allowed_site_multiplicity,
            )
            handles_pass = handles_pass and bool(qualification["passes_registry_handle_policy"])
        forward_pass, saturated, _ = _forward_reconstructs_product(
            reaction,
            components,
            canonical,
        )
        for group in groups:
            counters[group]["component_reconstructed"] += 1
            counters[group]["all_handles_pass"] += int(handles_pass)
            counters[group]["exact_forward_reconstruction"] += int(forward_pass)
            counters[group]["forward_outcome_saturated"] += int(saturated)
            novelty[group][component_class] += 1
            if component_class == "genuinely_generated_component" and handles_pass and forward_pass:
                counters[group]["usable_structural_open_ended"] += 1

    def summarize(label: str) -> dict[str, Any]:
        counts = counters[label]
        attempted = counts["attempted"]
        valid = counts["valid"]
        reconstructed = counts["component_reconstructed"]
        return {
            "attempted_programs": attempted,
            "valid_products": _rate(valid, attempted),
            "unique_valid_products": len(unique[label]),
            "unique_fraction_of_valid_products": len(unique[label]) / valid if valid else None,
            "component_reconstruction": _rate(reconstructed, attempted),
            "all_three_handles_qualified": _rate(counts["all_handles_pass"], attempted),
            "exact_forward_reconstruction": _rate(
                counts["exact_forward_reconstruction"], attempted
            ),
            "usable_structural_open_ended": _rate(
                counts["usable_structural_open_ended"], attempted
            ),
            "component_membership_counts": dict(sorted(novelty[label].items())),
            "product_membership_counts": dict(sorted(product_membership[label].items())),
            "invalid_products": counts["invalid"],
            "forward_outcome_saturated": counts["forward_outcome_saturated"],
        }

    return summarize("all"), {
        label: summarize(label) for label in sorted(key for key in counters if key != "all")
    }


def evaluate_postselection_held_component_stress(
    *, config_path: Path, output_path: Path, repository: Path
) -> dict[str, Any]:
    """Run the hash-pinned, nonselecting held-family program stress test."""

    config = _load_json(config_path)
    resolved = {
        name: _require_input(repository, specification)
        for name, specification in config["inputs"].items()
    }
    manifest = _load_json(resolved["production_manifest"])
    probe = _load_json(resolved["held_component_probe"])
    sample = _load_json(resolved["selected_checkpoint_sample"])
    earlier_gate = _load_json(resolved["earlier_nonselecting_gate"])

    if manifest.get("status") != "frozen":
        raise UgiPostselectionHeldComponentError("production generator is not frozen")
    selected_step = int(config["selected_checkpoint"]["step"])
    if int(manifest["identity"]["checkpoint_step"]) != selected_step:
        raise UgiPostselectionHeldComponentError("selected checkpoint step changed")
    checkpoint_spec = manifest["model"]["checkpoint"]
    if checkpoint_spec["sha256"] != config["selected_checkpoint"]["sha256"]:
        raise UgiPostselectionHeldComponentError("selected checkpoint hash changed")
    checkpoint_path = _resolve(repository, sample["checkpoints"]["joint"])
    if sha256_file(checkpoint_path) != config["selected_checkpoint"]["sha256"]:
        raise UgiPostselectionHeldComponentError(
            "sample was not produced by the selected checkpoint"
        )
    if int(sample["seed"]) != int(config["sampling_contract"]["seed"]):
        raise UgiPostselectionHeldComponentError("sampling seed changed")
    if sample["matched_staged_result"] != config["inputs"]["held_component_probe"]["path"]:
        raise UgiPostselectionHeldComponentError(
            "sample does not reference the frozen program probe"
        )

    probe_rows = list(probe["samples"])
    sample_rows = list(sample["samples"])
    expected_attempts = int(config["sampling_contract"]["attempted_programs"])
    if len(probe_rows) != expected_attempts or len(sample_rows) != expected_attempts:
        raise UgiPostselectionHeldComponentError("attempted program count changed")
    forbidden_probe_fields = {
        field
        for row in probe_rows
        for field in row
        if field.endswith("_smiles") or field in {"smiles", "component_graphs"}
    }
    if forbidden_probe_fields:
        raise UgiPostselectionHeldComponentError(
            f"held program probe exposes molecular identity: {sorted(forbidden_probe_fields)}"
        )

    assignment_rows = _read_csv(resolved["assignments"])
    assignments = {row["product_id"]: row for row in assignment_rows}
    if len(assignments) != len(assignment_rows):
        raise UgiPostselectionHeldComponentError("product assignments are not unique")
    held_class_counts = _validate_program_alignment(probe_rows, sample_rows, assignments)

    catalog = _catalog_by_role(resolved["component_registry"])
    reaction = _reaction_contract(resolved["qualified_reactions"])
    product_fold_by_smiles = {
        row["canonical_product_smiles"]: row["primary_product_fold"] for row in assignment_rows
    }
    if len(product_fold_by_smiles) != len(assignment_rows):
        raise UgiPostselectionHeldComponentError("frozen product constitutions are not unique")
    overall, by_class = _stratified_generation_summary(
        sample_rows,
        catalog=catalog,
        reaction=reaction,
        product_fold_by_smiles=product_fold_by_smiles,
    )

    recomputed = analyze_generated_components(
        resolved["selected_checkpoint_sample"],
        resolved["component_registry"],
        resolved["qualified_reactions"],
    )
    earlier_step = earlier_gate["checkpoints_by_step"].get(str(selected_step))
    if earlier_step is None:
        raise UgiPostselectionHeldComponentError("earlier gate lacks the selected checkpoint")
    earlier_generated = earlier_step["generated_component_analysis"]
    protected_keys = (
        "valid_products",
        "component_reconstruction_valid",
        "product_component_novelty_counts",
        "all_three_handles_pass_fraction",
        "exact_forward_product_reconstruction_fraction",
    )
    if any(recomputed[key] != earlier_generated[key] for key in protected_keys):
        raise UgiPostselectionHeldComponentError("recomputed generation metrics changed")

    full_loss = earlier_step["heldout_loss"]["full_heldout"]
    if full_loss["checkpoint"]["sha256"] != config["selected_checkpoint"]["sha256"]:
        raise UgiPostselectionHeldComponentError("heldout loss used a different checkpoint")
    if int(full_loss["evaluated_records"]) != 30_122:
        raise UgiPostselectionHeldComponentError("full heldout loss coverage changed")
    if float(full_loss["heldout_coverage_fraction"]) != 1.0:
        raise UgiPostselectionHeldComponentError("full heldout loss is incomplete")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_descriptive_postselection_evaluation",
        "config": {
            "path": str(config_path.relative_to(repository)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            name: {"path": config["inputs"][name]["path"], "sha256": sha256_file(path)}
            for name, path in sorted(resolved.items())
        },
        "selected_checkpoint": {
            "step": selected_step,
            "path": checkpoint_spec["path"],
            "sha256": checkpoint_spec["sha256"],
            "selection_remains_frozen": True,
        },
        "evaluation_timing": {
            "raw_probe_and_samples_existed_and_were_inspected_before_production_freeze": True,
            "current_analysis_performed_after_production_freeze": True,
            "heldout_fold_is_pristine": False,
            "interpretation": "descriptive post-selection component-family stress test",
        },
        "program_contract": {
            "seed": int(sample["seed"]),
            "attempted_programs": len(probe_rows),
            "component_identities_or_graphs_exposed_to_sampler": False,
            "program_fields": sorted(probe_rows[0]["program"]),
            "held_role_class_counts": dict(sorted(held_class_counts.items())),
        },
        "generation": {
            "overall": overall,
            "by_held_role_class": by_class,
            "by_generated_component_role": recomputed["by_role"],
        },
        "single_corruption_denoising_loss": {
            "description": (
                "One deterministic corruption draw per record across all 30,122 component-family "
                "heldout products; descriptive denoising stress metric, not exact likelihood"
            ),
            "seed": full_loss["seed"],
            "records": full_loss["evaluated_records"],
            "record_weighted_metrics": full_loss["record_weighted_metrics"],
            "by_held_role_class": full_loss["by_held_role_class"],
        },
        "claim_boundary": {
            "supports": [
                "generation remains valid and exactly L1-forward-consistent under coarse morphology programs associated with held component families",
                "the selected generator emits structurally novel precursor graphs under that descriptive stress condition",
                "performance can be stratified by which precursor role families supplied the held-associated program",
            ],
            "does_not_support": [
                "a pristine or untouched heldout estimate",
                "recovery of the exact held component represented by each source product",
                "unseen reaction-family or non-Ugi generalization",
                "L2 or L3 route closure, synthesis feasibility or synthesis success",
                "biological activity or biological guidance",
            ],
        },
        "decision": {
            "architecture_or_checkpoint_changed": False,
            "thresholds_applied": False,
            "broad_pretraining_decision": "not_made_by_this_descriptive_evaluation",
        },
    }
    _atomic_json(output_path, result)
    return result
