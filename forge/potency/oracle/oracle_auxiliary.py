"""Audit auxiliary Ugi and related 3CR supervision for the AGILE oracle.

The audit is intentionally separate from model fitting. It verifies assay
semantics, structures, component overlap, and source-review corrections before
any auxiliary record may enter a shared encoder or transfer-learning study.
Raw labels from separate studies are never concatenated as one regression
target.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.core.io import read_json_object
from forge.synthesis.engine.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)

CONFIG_SCHEMA_VERSION = "m0_07_auxiliary_supervision_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_auxiliary_supervision.v1"

LEDGER_FIELDS = (
    "study_id",
    "pmid",
    "source_record_id",
    "source_lipid_name",
    "model_type",
    "label_value",
    "label_semantics",
    "raw_source_product_smiles",
    "model_smiles",
    "model_identity_action",
    "chemistry_relation",
    "component_ids_json",
    "component_smiles_json",
    "source_review_overrides_json",
    "forward_unique_product_count",
    "forward_source_product_verified",
    "forward_model_product_verified",
    "paired_supervision_eligible",
    "exact_agile_product_overlap",
)


class AuxiliarySupervisionError(ValueError):
    """Raised when auxiliary supervision violates the frozen audit contract."""


def _canonical(smiles: str, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip() or smiles.strip() == "NA":
        raise AuxiliarySupervisionError(f"{label} is not a molecular SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise AuxiliarySupervisionError(f"{label} is invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=AuxiliarySupervisionError, label=label)


def _load_csv(path: Path, label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise AuxiliarySupervisionError(f"{label} not found: {path}") from exc


def _verify_input(repo_root: Path, spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    relative = spec.get("path")
    expected = spec.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise AuxiliarySupervisionError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise AuxiliarySupervisionError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise AuxiliarySupervisionError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json(path, "auxiliary supervision config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AuxiliarySupervisionError("unsupported auxiliary supervision config schema")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise AuxiliarySupervisionError("config lacks policy")
    if policy.get("allow_cross_study_raw_label_pooling") is not False:
        raise AuxiliarySupervisionError("cross-study raw label pooling must remain disabled")
    if policy.get("prediction_head_key") != ["study_id", "model_type"]:
        raise AuxiliarySupervisionError(
            "auxiliary prediction heads must be keyed by study_id and model_type"
        )
    auxiliary_filter = policy.get("model_selection_auxiliary_filter")
    if not isinstance(auxiliary_filter, dict) or set(auxiliary_filter) != {
        "all_splits",
        "scaffold_holdout",
        "component_holdout",
        "component_pair_holdout",
    }:
        raise AuxiliarySupervisionError("config must declare per-fold auxiliary leakage filters")
    studies = config.get("studies")
    if not isinstance(studies, list) or not studies:
        raise AuxiliarySupervisionError("config must declare at least one auxiliary study")
    return config


def _review_overrides(
    source_reviews: Mapping[str, Any],
    review_id: str | None,
) -> dict[tuple[str, str], str]:
    if review_id is None:
        return {}
    reviews = source_reviews.get("reviews")
    if not isinstance(reviews, list):
        raise AuxiliarySupervisionError("source review artifact lacks reviews")
    review = next(
        (item for item in reviews if isinstance(item, dict) and item.get("review_id") == review_id),
        None,
    )
    if review is None:
        raise AuxiliarySupervisionError(f"source review {review_id!r} not found")

    overrides: dict[tuple[str, str], str] = {}
    for family in review.get("l2_route_families", []):
        if not isinstance(family, dict):
            continue
        target = family.get("target")
        if (
            isinstance(target, dict)
            and family.get("route_family_id") == "jc_2023_b5_branched_hexyl_aldehyde_exact_route"
        ):
            overrides[("aldehyde", "B5")] = _canonical(
                str(target.get("smiles", "")),
                label=f"{review_id} aldehyde B5",
            )
        for member in family.get("members", []):
            if not isinstance(member, dict):
                continue
            component_label = member.get("label")
            product_smiles = member.get("product_smiles")
            if isinstance(component_label, str) and isinstance(product_smiles, str):
                overrides[("isocyanide", component_label)] = _canonical(
                    product_smiles,
                    label=f"{review_id} isocyanide {component_label}",
                )
    return overrides


def _label_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise AuxiliarySupervisionError("cannot summarize an empty label group")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "population_standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _render_csv_gzip(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in LEDGER_FIELDS})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(buffer.getvalue().encode())
    return output.getvalue()


def _artifact_metadata(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def filter_auxiliary_training_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    exact_test_products: set[str],
    held_scaffolds: set[str],
    held_components: Mapping[str, set[str]],
    held_component_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Apply the frozen per-fold auxiliary leakage exclusions."""

    retained: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    for source_row in rows:
        row = dict(source_row)
        product = _canonical(row.get("model_smiles", ""), label="auxiliary model product")
        if product in exact_test_products:
            reasons["exact_test_product"] += 1
            continue
        molecule = Chem.MolFromSmiles(product)
        if molecule is None:
            raise AuxiliarySupervisionError("auxiliary model product is invalid")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=False,
        )
        if scaffold in held_scaffolds:
            reasons["held_scaffold"] += 1
            continue
        try:
            components = json.loads(row.get("component_smiles_json", ""))
        except json.JSONDecodeError as exc:
            raise AuxiliarySupervisionError("auxiliary component_smiles_json is invalid") from exc
        if not isinstance(components, dict):
            raise AuxiliarySupervisionError("auxiliary component_smiles_json must be an object")
        canonical_components = {
            role: _canonical(smiles, label=f"auxiliary {role}")
            for role, smiles in components.items()
            if isinstance(role, str) and isinstance(smiles, str)
        }
        if any(
            canonical_components.get(role) in structures
            for role, structures in held_components.items()
        ):
            reasons["held_component"] += 1
            continue
        pair_leak = False
        for (left_role, right_role), held_pairs in held_component_pairs.items():
            observed = (
                canonical_components.get(left_role),
                canonical_components.get(right_role),
            )
            if None not in observed and observed in held_pairs:
                pair_leak = True
                break
        if pair_leak:
            reasons["held_component_pair"] += 1
            continue
        retained.append(row)
    reasons["retained"] = len(retained)
    return retained, dict(sorted(reasons.items()))


def run_auxiliary_supervision_audit(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the deterministic source and compatibility audit."""

    config = _load_config(config_path)
    try:
        config_relative = str(config_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        config_relative = str(config_path.resolve())
    config_input = {
        "path": config_relative,
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise AuxiliarySupervisionError("config lacks inputs")
    verified = {
        name: _verify_input(repo_root, spec, name)
        for name, spec in sorted(inputs.items())
        if isinstance(spec, dict)
    }
    if set(verified) != {
        "agile_curated",
        "lnpdb",
        "qualified_reactions",
        "source_reviews",
        "ugi_variant",
    }:
        raise AuxiliarySupervisionError(
            "config must verify AGILE, LNPDB, qualified reactions, source reviews, "
            "and the Ugi variant"
        )

    lnpdb = _load_csv(repo_root / verified["lnpdb"]["path"], "LNPDB")
    agile = _load_csv(repo_root / verified["agile_curated"]["path"], "curated AGILE")
    source_reviews = _load_json(
        repo_root / verified["source_reviews"]["path"],
        "source reviews",
    )
    agile_products = {_canonical(row["model_smiles"], label="AGILE product") for row in agile}
    agile_components = {
        "A": {_canonical(row["A_smiles"], label="AGILE A") for row in agile},
        "B": {_canonical(row["B_smiles"], label="AGILE B") for row in agile},
        "C": {_canonical(row["C_smiles"], label="AGILE C") for row in agile},
    }

    label_semantics = config["policy"]["label_semantics"]
    ledger: list[dict[str, Any]] = []
    study_summaries: dict[str, Any] = {}
    for study in config["studies"]:
        if not isinstance(study, dict):
            raise AuxiliarySupervisionError("study configuration must be an object")
        study_id = study.get("study_id")
        if not isinstance(study_id, str):
            raise AuxiliarySupervisionError("study lacks study_id")
        rows = [row for row in lnpdb if row.get("Experiment_ID") == study_id]
        if len(rows) != study.get("expected_rows"):
            raise AuxiliarySupervisionError(
                f"{study_id} row count {len(rows)} != {study.get('expected_rows')}"
            )
        model_counts = Counter(row.get("Model_type") for row in rows)
        if dict(sorted(model_counts.items())) != study.get("expected_model_counts"):
            raise AuxiliarySupervisionError(
                f"{study_id} model counts changed: {dict(sorted(model_counts.items()))}"
            )
        if {row.get("Publication_PMID") for row in rows} != {study.get("pmid")}:
            raise AuxiliarySupervisionError(f"{study_id} PMID mismatch")

        component_roles = study.get("component_roles")
        if not isinstance(component_roles, dict) or not component_roles:
            raise AuxiliarySupervisionError(f"{study_id} lacks component roles")
        overrides = _review_overrides(source_reviews, study.get("source_review_id"))
        forward_spec = study.get("forward_verification")
        compiled_forward = None
        role_order: tuple[str, ...] = ()
        if forward_spec is not None:
            if not isinstance(forward_spec, dict):
                raise AuxiliarySupervisionError(
                    f"{study_id} forward_verification must be an object"
                )
            configured_roles = forward_spec.get("role_order")
            if (
                not isinstance(configured_roles, list)
                or not configured_roles
                or any(role not in component_roles for role in configured_roles)
            ):
                raise AuxiliarySupervisionError(f"{study_id} has an invalid forward role order")
            role_order = tuple(configured_roles)
            try:
                compiled_forward = load_qualified_forward_reaction(
                    repo_root / verified["qualified_reactions"]["path"],
                    repo_root / verified["ugi_variant"]["path"],
                    reaction_id=forward_spec.get("reaction_id", ""),
                )
            except QualifiedForwardError as exc:
                raise AuxiliarySupervisionError(
                    f"{study_id} qualified forward transform failed: {exc}"
                ) from exc
            if len(role_order) != len(compiled_forward.role_names):
                raise AuxiliarySupervisionError(
                    f"{study_id} forward role order has the wrong length"
                )
        role_components: dict[str, dict[str, str]] = defaultdict(dict)
        label_groups: dict[str, list[float]] = defaultdict(list)
        raw_product_set: set[str] = set()
        product_set: set[str] = set()
        overlap_products: set[str] = set()
        task_products: set[tuple[str, str]] = set()
        identity_actions: Counter[str] = Counter()
        identity_change_components: set[tuple[str, str]] = set()

        for row in rows:
            raw_product = _canonical(
                row.get("IL_SMILES", ""),
                label=f"{study_id} product",
            )
            raw_product_set.add(raw_product)
            try:
                label_value = float(row.get("Experiment_value", ""))
            except ValueError as exc:
                raise AuxiliarySupervisionError(f"{study_id} has nonnumeric label") from exc
            if not math.isfinite(label_value):
                raise AuxiliarySupervisionError(f"{study_id} has nonfinite label")
            model_type = row.get("Model_type", "")
            label_groups[model_type].append(label_value)

            component_ids: dict[str, str] = {}
            component_smiles: dict[str, str] = {}
            applied_overrides: dict[str, str] = {}
            for role, role_spec in component_roles.items():
                name_field = role_spec.get("name_field")
                smiles_field = role_spec.get("smiles_field")
                if not isinstance(name_field, str) or not isinstance(smiles_field, str):
                    raise AuxiliarySupervisionError(f"{study_id} role {role} is incomplete")
                component_id = row.get(name_field, "")
                raw_smiles = row.get(smiles_field, "")
                raw_component = _canonical(
                    raw_smiles,
                    label=f"{study_id} {role} {component_id}",
                )
                key = (role, component_id)
                if key in overrides:
                    component = overrides[key]
                    if component != raw_component:
                        applied_overrides[role] = component
                        identity_change_components.add(key)
                else:
                    component = raw_component
                previous = role_components[role].setdefault(component_id, component)
                if previous != component:
                    raise AuxiliarySupervisionError(
                        f"{study_id} {role} {component_id} maps to multiple structures"
                    )
                component_ids[role] = component_id
                component_smiles[role] = component

            product = raw_product
            identity_action = "source_product_retained"
            forward_unique_product_count = ""
            forward_source_product_verified = ""
            forward_model_product_verified = ""
            paired_supervision_eligible = False
            if compiled_forward is not None and forward_spec is not None:
                try:
                    products = unique_forward_products(
                        compiled_forward,
                        tuple(component_smiles[role] for role in role_order),
                        max_products=forward_spec["max_products"],
                        isomeric_smiles=False,
                    )
                except QualifiedForwardError as exc:
                    raise AuxiliarySupervisionError(
                        f"{study_id}/{row.get('IL_name', '')} forward verification failed: {exc}"
                    ) from exc
                forward_unique_product_count = str(len(products))
                source_verified = raw_product in products
                forward_source_product_verified = str(source_verified).lower()
                if source_verified:
                    paired_supervision_eligible = True
                else:
                    rebuild = forward_spec.get("source_component_rebuild")
                    if not isinstance(rebuild, dict):
                        raise AuxiliarySupervisionError(
                            f"{study_id}/{row.get('IL_name', '')} failed exact forward "
                            "reconstruction without a correction policy"
                        )
                    allowed_role = rebuild.get("allowed_component_role")
                    allowed_ids = rebuild.get("allowed_component_ids")
                    correction_allowed = (
                        rebuild.get("require_unique_forward_product") is True
                        and len(products) == 1
                        and isinstance(allowed_role, str)
                        and isinstance(allowed_ids, list)
                        and component_ids.get(allowed_role) in allowed_ids
                    )
                    if not correction_allowed:
                        raise AuxiliarySupervisionError(
                            f"{study_id}/{row.get('IL_name', '')} failed exact forward "
                            "reconstruction outside the frozen correction policy"
                        )
                    product = products[0]
                    identity_action = "source_component_forward_reconstruction"
                    paired_supervision_eligible = True
                forward_model_product_verified = str(product in products).lower()
                if not paired_supervision_eligible or product not in products:
                    raise AuxiliarySupervisionError(
                        f"{study_id}/{row.get('IL_name', '')} did not qualify paired supervision"
                    )
            else:
                identity_action = "database_product_unverified_related_chemistry"

            product_set.add(product)
            if product in agile_products:
                overlap_products.add(product)
            task_product = (model_type, product)
            if task_product in task_products:
                raise AuxiliarySupervisionError(
                    f"{study_id}/{model_type} contains duplicate product supervision"
                )
            task_products.add(task_product)
            identity_actions[identity_action] += 1

            ledger.append(
                {
                    "study_id": study_id,
                    "pmid": study["pmid"],
                    "source_record_id": row.get("LNP_ID", ""),
                    "source_lipid_name": row.get("IL_name", ""),
                    "model_type": model_type,
                    "label_value": format(label_value, ".12g"),
                    "label_semantics": label_semantics,
                    "raw_source_product_smiles": raw_product,
                    "model_smiles": product,
                    "model_identity_action": identity_action,
                    "chemistry_relation": study["chemistry_relation"],
                    "component_ids_json": json.dumps(component_ids, sort_keys=True),
                    "component_smiles_json": json.dumps(component_smiles, sort_keys=True),
                    "source_review_overrides_json": json.dumps(
                        applied_overrides,
                        sort_keys=True,
                    ),
                    "forward_unique_product_count": forward_unique_product_count,
                    "forward_source_product_verified": forward_source_product_verified,
                    "forward_model_product_verified": forward_model_product_verified,
                    "paired_supervision_eligible": str(paired_supervision_eligible).lower(),
                    "exact_agile_product_overlap": str(product in agile_products).lower(),
                }
            )

        if len(raw_product_set) != study.get("expected_unique_products"):
            raise AuxiliarySupervisionError(
                f"{study_id} raw unique product count {len(raw_product_set)} "
                f"!= {study.get('expected_unique_products')}"
            )
        if len(product_set) != study.get("expected_unique_products"):
            raise AuxiliarySupervisionError(
                f"{study_id} unique product count {len(product_set)} "
                f"!= {study.get('expected_unique_products')}"
            )
        if forward_spec is not None:
            expected_actions = {
                "source_product_retained": forward_spec["expected_source_exact_products"],
                "source_component_forward_reconstruction": forward_spec[
                    "expected_source_component_rebuilds"
                ],
            }
            if dict(sorted(identity_actions.items())) != expected_actions:
                raise AuxiliarySupervisionError(
                    f"{study_id} forward identity actions changed: "
                    f"{dict(sorted(identity_actions.items()))}"
                )
        stats = {model: _label_stats(values) for model, values in sorted(label_groups.items())}
        for model, summary in stats.items():
            if (
                abs(float(summary["mean"])) > 1e-6
                or abs(float(summary["population_standard_deviation"]) - 1.0) > 1e-6
            ):
                raise AuxiliarySupervisionError(
                    f"{study_id}/{model} no longer has within-endpoint z-score semantics"
                )

        component_summary: dict[str, Any] = {}
        for role, components in sorted(role_components.items()):
            role_spec = component_roles[role]
            agile_role = role_spec.get("agile_role")
            structures = set(components.values())
            overlap = structures & agile_components[agile_role] if agile_role else set()
            component_summary[role] = {
                "source_component_ids": len(components),
                "unique_structures": len(structures),
                "agile_role_comparison": agile_role,
                "exact_agile_component_overlap": len(overlap) if agile_role else None,
                "new_relative_to_agile_role": len(structures - overlap) if agile_role else None,
            }

        study_summaries[study_id] = {
            "pmid": study["pmid"],
            "chemistry_relation": study["chemistry_relation"],
            "rows": len(rows),
            "unique_products": len(product_set),
            "exact_agile_product_overlap": len(overlap_products),
            "model_counts": dict(sorted(model_counts.items())),
            "label_statistics": stats,
            "components": component_summary,
            "source_review_override_definitions": len(overrides),
            "source_review_identity_changes": len(identity_change_components),
            "model_identity_actions": dict(sorted(identity_actions.items())),
            "paired_supervision_eligible_rows": sum(
                row["study_id"] == study_id and row["paired_supervision_eligible"] == "true"
                for row in ledger
            ),
        }

    ledger.sort(
        key=lambda row: (
            row["study_id"],
            row["model_type"],
            row["source_lipid_name"],
            row["source_record_id"],
        )
    )
    ledger_payload = _render_csv_gzip(ledger)
    ledger_name = "oracle_auxiliary_records.csv.gz"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_auxiliary_supervision_audit",
        "seed": config["seed"],
        "inputs": {"config": config_input, **verified},
        "studies": study_summaries,
        "policy": config["policy"],
        "decision": {
            "raw_label_pooling_allowed": False,
            "jc_2023_disposition": (
                "eligible_for_shared_encoder_study_specific_head_source_pretraining_"
                "and_external_transfer_comparison"
            ),
            "lm_2019_disposition": (
                "eligible_for_related_chemistry_encoder_pretraining_and_external_transfer_"
                "comparison_not_direct_ugi_target_pooling"
            ),
            "retain_auxiliary_data_only_if": config["policy"]["retention_gate"],
            "head_diversity_note": (
                "JC_2023 adds no exact amine-head structures beyond AGILE; its direct value "
                "is new native-Ugi tail chemistry and an external study shift."
            ),
        },
        "artifacts": {ledger_name: _artifact_metadata(ledger_payload)},
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / ledger_name, ledger_payload)
    _atomic_write(output_dir / "oracle_auxiliary_supervision.json", result_payload)
    return result
