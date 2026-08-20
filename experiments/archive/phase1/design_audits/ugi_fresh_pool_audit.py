"""Hash-pinned audit of a fresh Ugi product-plus-L1 sampling pool."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.corpus.r1_prime_audit import sha256_file
from experiments.phase1.product_l1.evaluation.tail_chemotype import summarize_component_cohort
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA_VERSION = "phase1_ugi_product_l1_v2_fresh_pool_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_product_l1_v2_fresh_pool_audit.v1"
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")


class UgiFreshPoolAuditError(ValueError):
    """Raised when the fresh-pool audit contract is inconsistent."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiFreshPoolAuditError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiFreshPoolAuditError(f"{label} must be a JSON object")
    return value


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiFreshPoolAuditError(f"invalid generated SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def validate_sampling_metadata(sample: Mapping[str, Any], sampling: Mapping[str, Any]) -> None:
    """Fail closed when a sampled pool differs from its frozen sampling contract."""

    if int(sample.get("seed", -1)) != int(sampling["flow_seed"]):
        raise UgiFreshPoolAuditError("flow seed changed")
    terminal_decoder = sample.get("sampling", {}).get("terminal_decoder", {})
    if terminal_decoder.get("mode") != sampling["terminal_decoder_mode"]:
        raise UgiFreshPoolAuditError("terminal decoder changed")
    if int(terminal_decoder.get("seed", -1)) != int(sampling["terminal_seed"]):
        raise UgiFreshPoolAuditError("terminal decoder seed changed")
    if float(terminal_decoder.get("temperature", -1.0)) != float(sampling["terminal_temperature"]):
        raise UgiFreshPoolAuditError("terminal decoder temperature changed")


def summarize_fresh_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_visible_products: set[str],
    catalog: Mapping[tuple[str, str], bool],
) -> dict[str, Any]:
    """Summarize exact L1, novelty and tail diversity without route inference."""

    valid = [row for row in rows if row.get("valid") is True]
    eligible = [
        row
        for row in valid
        if row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]
    failure_types = Counter(
        str(row.get("failure_type") or "invalid_unspecified")
        for row in rows
        if row.get("valid") is not True
    )
    canonical_products = [_canonical(str(row["smiles"])) for row in valid]
    exact_reference = sum(product in selection_visible_products for product in canonical_products)
    components: dict[str, list[str]] = {role: [] for role in ROLE_NAMES}
    provenance_occurrences = Counter()
    product_provenance = Counter()
    for row in eligible:
        mapping = row.get("component_smiles_by_role")
        if not isinstance(mapping, dict) or set(mapping) != set(ROLE_NAMES):
            raise UgiFreshPoolAuditError("exact-L1 row lacks all three component roles")
        row_strata: list[str] = []
        for role in ROLE_NAMES:
            component = _canonical(str(mapping[role]))
            components[role].append(component)
            current_catalog = catalog.get((role, component))
            if current_catalog is True:
                stratum = "original_ugi_component"
            elif current_catalog is False:
                stratum = "admitted_transferred_or_expanded_known_component"
            else:
                stratum = "graph_absent_from_frozen_424_component_catalog"
            provenance_occurrences[(role, stratum)] += 1
            row_strata.append(stratum)
        if "graph_absent_from_frozen_424_component_catalog" in row_strata:
            product_provenance["product_contains_catalog_absent_component"] += 1
        elif "admitted_transferred_or_expanded_known_component" in row_strata:
            product_provenance[
                "product_contains_admitted_transferred_or_expanded_known_component"
            ] += 1
        else:
            product_provenance["product_uses_only_original_ugi_components"] += 1

    unique_products = len(set(canonical_products))
    valid_count = len(valid)
    eligible_count = len(eligible)
    strata = (
        "original_ugi_component",
        "admitted_transferred_or_expanded_known_component",
        "graph_absent_from_frozen_424_component_catalog",
    )
    return {
        "attempted_draws": len(rows),
        "valid_products": valid_count,
        "valid_fraction": valid_count / len(rows),
        "unique_valid_products": unique_products,
        "unique_fraction_of_valid": unique_products / valid_count,
        "exact_l1_products": eligible_count,
        "exact_l1_fraction_of_valid": eligible_count / valid_count,
        "invalid_failure_types": dict(sorted(failure_types.items())),
        "exact_selection_visible_products": exact_reference,
        "exact_selection_visible_product_fraction": exact_reference / valid_count,
        "products_by_structural_provenance": {
            name: product_provenance[name]
            for name in (
                "product_uses_only_original_ugi_components",
                "product_contains_admitted_transferred_or_expanded_known_component",
                "product_contains_catalog_absent_component",
            )
        },
        "component_occurrences_by_role_and_structural_provenance": {
            role: {stratum: provenance_occurrences[(role, stratum)] for stratum in strata}
            for role in ROLE_NAMES
        },
        "tail_chemotypes_exact_l1_only": {
            role: summarize_component_cohort(components[role]) for role in TAIL_ROLES
        },
    }


def build_fresh_pool_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    """Build a reproducible audit from a frozen pool and reference artifacts."""

    config = _load_json(config_path, label="fresh-pool audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiFreshPoolAuditError("unexpected fresh-pool audit config schema")
    inputs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name, specification in config["inputs"].items():
        path = (repo / specification["path"]).resolve()
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise UgiFreshPoolAuditError(f"input hash mismatch: {name}")
        paths[name] = path
        inputs[name] = {"path": str(path.relative_to(repo)), "sha256": observed}

    manifest = _load_json(paths["production_generator_v2"], label="production manifest")
    pool_contract = _load_json(paths["fresh_pool_contract"], label="fresh-pool contract")
    sample = _load_json(paths["fresh_pool_sample"], label="fresh-pool sample")
    if manifest.get("status") != "frozen_after_independent_decoder_confirmation":
        raise UgiFreshPoolAuditError("v2 production generator is not frozen")
    if pool_contract.get("status") != "frozen_before_fresh_program_draw":
        raise UgiFreshPoolAuditError("fresh-pool contract was not frozen before sampling")
    sampling = pool_contract["sampling"]
    validate_sampling_metadata(sample, sampling)
    rows = sample.get("samples")
    if not isinstance(rows, list) or len(rows) != int(sampling["programs"]):
        raise UgiFreshPoolAuditError("fresh-pool row count changed")

    selection_folds = set(config["selection_reference"]["folds"])
    visible_products: set[str] = set()
    with gzip.open(paths["selection_reference_assignments"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row[config["selection_reference"]["fold_field"]] in selection_folds:
                visible_products.add(str(row["canonical_product_smiles"]))
    if len(visible_products) != int(config["selection_reference"]["expected_products"]):
        raise UgiFreshPoolAuditError("selection-visible reference size changed")

    catalog: dict[tuple[str, str], bool] = {}
    with gzip.open(paths["component_registry"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["l1_structural_admission"].lower() != "true":
                continue
            key = (str(row["role"]), _canonical(str(row["canonical_smiles"])))
            if key in catalog:
                raise UgiFreshPoolAuditError("duplicate role-specific catalog identity")
            catalog[key] = row["is_current_catalog"].lower() == "true"

    with rdBase.BlockLogs():
        summary = summarize_fresh_pool_rows(
            rows,
            selection_visible_products=visible_products,
            catalog=catalog,
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_fresh_pool_audit",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": inputs,
        "summary": summary,
        "adjudication": {
            "prospective_candidate_selection_changed": False,
            "synthesis_guidance_authorized": False,
            "biological_guidance_authorized": False,
            "pool_may_advance_to_nonselecting_oracle_and_route_assessment": (
                summary["exact_l1_fraction_of_valid"] == 1.0
                and summary["valid_fraction"] >= float(config["minimum_valid_fraction"])
            ),
        },
    }
