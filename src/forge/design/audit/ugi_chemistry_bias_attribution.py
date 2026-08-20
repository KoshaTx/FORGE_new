"""Attribute generated Ugi chemistry biases to morphology or terminal chemistry.

This audit is deliberately nonselecting.  It compares a fresh, exact-L1
generated pool with selection-visible Ugi records, admitted component graphs,
and the broad structure-only lipid corpus.  It does not infer biological value,
synthesis success, or candidate eligibility from motif frequency alone.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.design.audit.ugi_tail_chemotype_audit import component_chemotype_metrics
from forge.design.guidance.ugi_candidate_eligibility import compile_smarts, motif_hits
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

CONFIG_SCHEMA_VERSION = "phase1_ugi_chemistry_bias_attribution_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_chemistry_bias_attribution.v1"
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")

PRODUCT_MOTIFS = {
    "alkyne": "[#6]#[#6]",
    "allene_or_carbon_cumulene": "[!#1]=[#6]=[!#1]",
    "peroxide": "[#8X2]-[#8X2]",
    "nitrogen_nitrogen_bond": "[#7]~[#7]",
    "product_aldehyde": "[CX3H1](=O)[#6]",
}

TAIL_FEATURES = (
    "has_carbon_branch",
    "has_adjacent_carbon_branches",
    "has_carbon_carbon_double_bond",
    "has_carbon_carbon_triple_bond",
    "has_ester_like_carbonyl",
    "has_ether_oxygen",
    "has_ring",
)


class UgiChemistryBiasAttributionError(ValueError):
    """Raised when a frozen attribution input or invariant changes."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiChemistryBiasAttributionError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiChemistryBiasAttributionError(f"{label} must be a JSON object")
    return value


def _canonical_molecule(smiles: str) -> tuple[Chem.Mol, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiChemistryBiasAttributionError(f"invalid connected molecule: {smiles}")
    return molecule, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def compact_tail_summary(smiles_values: Sequence[str]) -> dict[str, Any]:
    """Return occurrence- and unique-graph feature rates without a huge ledger."""

    if not smiles_values:
        raise UgiChemistryBiasAttributionError("tail summary requires components")
    canonical = [_canonical_molecule(str(smiles))[1] for smiles in smiles_values]
    occurrence_metrics = [component_chemotype_metrics(smiles) for smiles in canonical]
    unique_metrics = [component_chemotype_metrics(smiles) for smiles in sorted(set(canonical))]
    predicates = {
        "has_carbon_branch": lambda row: row["carbon_branch_atoms"] > 0,
        "has_adjacent_carbon_branches": lambda row: row["adjacent_carbon_branch_edges"] > 0,
        "has_carbon_carbon_double_bond": lambda row: row["carbon_carbon_double_bonds"] > 0,
        "has_carbon_carbon_triple_bond": lambda row: row["carbon_carbon_triple_bonds"] > 0,
        "has_ester_like_carbonyl": lambda row: row["ester_like_carbonyl_count"] > 0,
        "has_ether_oxygen": lambda row: row["ether_oxygen_count"] > 0,
        "has_ring": lambda row: row["ring_count"] > 0,
    }

    def rates(rows: Sequence[Mapping[str, int]]) -> dict[str, float]:
        return {
            name: sum(bool(predicate(row)) for row in rows) / len(rows)
            for name, predicate in predicates.items()
        }

    return {
        "component_occurrences": len(canonical),
        "unique_exact_components": len(set(canonical)),
        "occurrence_feature_fractions": rates(occurrence_metrics),
        "unique_component_feature_fractions": rates(unique_metrics),
    }


def compact_product_motif_summary(
    smiles_values: Iterable[str],
    *,
    maximum_examples: int,
) -> dict[str, Any]:
    """Return product motif rates and bounded examples for a declared cohort."""

    queries = compile_smarts(PRODUCT_MOTIFS)
    counts = Counter()
    examples: dict[str, list[str]] = {name: [] for name in queries}
    denominator = 0
    unique: set[str] = set()
    for smiles in smiles_values:
        molecule, canonical = _canonical_molecule(str(smiles))
        denominator += 1
        unique.add(canonical)
        for name, present in motif_hits(molecule, queries).items():
            if not present:
                continue
            counts[name] += 1
            if len(examples[name]) < maximum_examples:
                examples[name].append(canonical)
    if denominator == 0:
        raise UgiChemistryBiasAttributionError("product motif summary requires molecules")
    return {
        "product_occurrences": denominator,
        "unique_products": len(unique),
        "motifs": {
            name: {
                "smarts": PRODUCT_MOTIFS[name],
                "count": counts[name],
                "fraction": counts[name] / denominator,
                "examples": examples[name],
            }
            for name in PRODUCT_MOTIFS
        },
    }


def prior_positive_junction_fractions(program_prior: Mapping[str, Any]) -> dict[str, float]:
    """Return role-wise prior mass assigned to positive junction budgets."""

    output = {}
    for role in ROLE_NAMES:
        records = program_prior.get("role_priors", {}).get(role)
        if not isinstance(records, list) or not records:
            raise UgiChemistryBiasAttributionError(f"program prior lacks role: {role}")
        output[role] = sum(
            float(record["probability"]) for record in records if int(record["junction_budget"]) > 0
        )
    return output


def sampled_positive_junction_fractions(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return positive-junction rates from generated morphology programs."""

    if not rows:
        raise UgiChemistryBiasAttributionError("generated program summary requires rows")
    counts = Counter()
    for row in rows:
        program = row.get("program")
        if not isinstance(program, dict):
            raise UgiChemistryBiasAttributionError("generated row lacks its program")
        values = program.get("junction_budgets")
        if not isinstance(values, (list, tuple)) or len(values) != len(ROLE_NAMES):
            raise UgiChemistryBiasAttributionError("generated junction budget is malformed")
        for role, value in zip(ROLE_NAMES, values, strict=True):
            counts[role] += int(value) > 0
    return {role: counts[role] / len(rows) for role in ROLE_NAMES}


def _input_paths(repo: Path, config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    paths: dict[str, Path] = {}
    records: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise UgiChemistryBiasAttributionError(f"input hash mismatch: {name}")
        paths[name] = path
        records[name] = {
            "path": str(path.relative_to(repo)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return paths, records


def build_chemistry_bias_attribution(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build the frozen, nonselecting fresh-pool chemistry attribution audit."""

    config = _load_json(config_path, label="chemistry-bias attribution config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiChemistryBiasAttributionError("unsupported attribution config")
    paths, inputs = _input_paths(repo, config)
    sample = _load_json(paths["fresh_pool_sample"], label="fresh pool sample")
    program_prior = _load_json(paths["program_prior"], label="program prior")
    rows = sample.get("samples")
    if not isinstance(rows, list) or len(rows) != int(config["fresh_pool"]["attempted_draws"]):
        raise UgiChemistryBiasAttributionError("fresh-pool row count changed")
    exact_rows = [
        row
        for row in rows
        if row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]
    if len(exact_rows) != int(config["fresh_pool"]["expected_exact_l1_products"]):
        raise UgiChemistryBiasAttributionError("fresh-pool exact-L1 count changed")

    assignments = _read_csv(paths["selection_reference_assignments"])
    fold_field = str(config["selection_reference"]["fold_field"])
    visible_folds = set(config["selection_reference"]["folds"])
    visible = [row for row in assignments if row[fold_field] in visible_folds]
    if len(visible) != int(config["selection_reference"]["expected_products"]):
        raise UgiChemistryBiasAttributionError("selection-visible reference count changed")
    registry = [
        row
        for row in _read_csv(paths["component_registry"])
        if row["l1_structural_admission"].lower() == "true"
    ]
    broad = [
        row
        for row in _read_csv(paths["broad_lipid_corpus"])
        if row["r0_pretraining_eligible"].lower() == "true"
    ]
    if len(broad) != int(config["broad_reference"]["expected_products"]):
        raise UgiChemistryBiasAttributionError("broad reference count changed")

    generated_components: dict[str, list[str]] = {role: [] for role in TAIL_ROLES}
    for row in exact_rows:
        mapping = row.get("component_smiles_by_role")
        if not isinstance(mapping, dict) or set(mapping) != set(ROLE_NAMES):
            raise UgiChemistryBiasAttributionError("exact-L1 row lacks component mapping")
        for role in TAIL_ROLES:
            generated_components[role].append(str(mapping[role]))
    visible_components = {
        role: [str(row[f"{role}_smiles"]) for row in visible] for role in TAIL_ROLES
    }
    registry_components = {
        role: [str(row["canonical_smiles"]) for row in registry if row["role"] == role]
        for role in TAIL_ROLES
    }

    maximum_examples = int(config["reporting"]["maximum_examples_per_motif"])
    with rdBase.BlockLogs():
        product_motifs = {
            "fresh_generated_exact_l1": compact_product_motif_summary(
                (str(row["smiles"]) for row in exact_rows),
                maximum_examples=maximum_examples,
            ),
            "selection_visible_ugi": compact_product_motif_summary(
                (str(row["canonical_product_smiles"]) for row in visible),
                maximum_examples=maximum_examples,
            ),
            "broad_observed_lipid_corpus": compact_product_motif_summary(
                (str(row["canonical_constitutional_smiles"]) for row in broad),
                maximum_examples=maximum_examples,
            ),
        }
        tail_chemistry = {
            role: {
                "fresh_generated_exact_l1": compact_tail_summary(generated_components[role]),
                "selection_visible_ugi_occurrences": compact_tail_summary(visible_components[role]),
                "selection_visible_ugi_unique_components": compact_tail_summary(
                    sorted(set(visible_components[role]))
                ),
                "admitted_registry_unique_components": compact_tail_summary(
                    sorted(set(registry_components[role]))
                ),
            }
            for role in TAIL_ROLES
        }

    branch_prior = prior_positive_junction_fractions(program_prior)
    branch_sample = sampled_positive_junction_fractions(rows)
    generated_aldehyde_branch = tail_chemistry["oxoester_aldehyde_body_tail"][
        "fresh_generated_exact_l1"
    ]["occurrence_feature_fractions"]["has_carbon_branch"]
    generated_aldehyde_junction = branch_sample["oxoester_aldehyde_body_tail"]
    branch_tracking_gap = abs(generated_aldehyde_branch - generated_aldehyde_junction)

    generated_motifs = product_motifs["fresh_generated_exact_l1"]["motifs"]
    ugi_motifs = product_motifs["selection_visible_ugi"]["motifs"]
    broad_motifs = product_motifs["broad_observed_lipid_corpus"]["motifs"]
    absent_both_reference_but_generated = [
        name
        for name in PRODUCT_MOTIFS
        if generated_motifs[name]["count"] > 0
        and ugi_motifs[name]["count"] == 0
        and broad_motifs[name]["count"] == 0
    ]

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_attribution",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs,
        "denominators": {
            "fresh_attempted": len(rows),
            "fresh_exact_l1": len(exact_rows),
            "selection_visible_ugi_products": len(visible),
            "broad_observed_lipid_products": len(broad),
            "admitted_registry_components": len(registry),
        },
        "morphology_program_attribution": {
            "positive_junction_fraction_in_frozen_prior": branch_prior,
            "positive_junction_fraction_in_fresh_draw": branch_sample,
            "fresh_aldehyde_carbon_branch_fraction": generated_aldehyde_branch,
            "fresh_aldehyde_positive_junction_fraction": generated_aldehyde_junction,
            "absolute_branch_to_program_tracking_gap": branch_tracking_gap,
            "interpretation": (
                "A small tracking gap supports treating aldehyde branching primarily as a "
                "coarse-program-prior calibration problem; it does not establish that every "
                "junction is a carbon branch."
            ),
        },
        "tail_chemistry": tail_chemistry,
        "product_motif_audit": product_motifs,
        "attribution": {
            "morphology_prior_lane": ["aldehyde and isocyanide branching incidence"],
            "terminal_chemistry_lane": [
                "carbon-carbon double versus triple bond calibration",
                "ester versus non-ester oxygen/carbonyl coordination",
                "ether oxygen calibration",
            ],
            "manual_support_policy_lane": absent_both_reference_but_generated,
            "alkyne_policy": (
                "Alkynes are reference-supported and must be calibrated rather than hard-banned."
            ),
        },
        "adjudication": {
            "frozen_v2_generator_replaced": False,
            "prospective_candidate_selection_authorized": False,
            "hard_motif_exclusions_authorized_by_frequency_alone": False,
            "bounded_prior_and_terminal_chemistry_challengers_warranted": True,
        },
    }
