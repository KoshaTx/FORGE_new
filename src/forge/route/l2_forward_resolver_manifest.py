"""Deterministically normalize admitted development L2 scope for the resolver."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.route.l2_forward_resolver import (
    L2_FORWARD_RESOLVER_CONFIG_SCHEMA_VERSION,
    L2ForwardResolverError,
    transform_receipt_sha256,
)
from forge.route.qualified_forward import load_qualified_forward_reaction

_UPSTREAM_REGISTRY = Path("configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json")
_UPSTREAM_LEDGER = Path(
    "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz"
)
_C16_REGISTRY = Path("configs/route/phase1_ugi3_exact_c16_qualified_reactions_v1.json")
_C16_LEDGER = Path("results/phase1/ugi3_exact_c16_route_v1/step_verification_ledger.json.gz")
_C18_REGISTRY = Path("configs/route/phase1_ugi3_exact_c18_qualified_reactions_v1.json")
_C18_LEDGER = Path("results/phase1/ugi3_exact_c18_route_v1/step_verification_ledger.json.gz")
_TARGETED_EVIDENCE = Path("configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json")
_TARGETED_LEDGER = Path(
    "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/route_verification_ledger.csv.gz"
)


def _artifact(repo_root: Path, relative: Path) -> dict[str, Any]:
    path = repo_root / relative
    if not path.is_file():
        raise L2ForwardResolverError(f"resolver manifest input is missing: {path}")
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise L2ForwardResolverError(f"invalid resolver manifest input: {path}") from exc
    if not isinstance(value, dict):
        raise L2ForwardResolverError(f"resolver manifest input must be an object: {path}")
    return value


def _csv_gzip_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise L2ForwardResolverError(f"resolver scope ledger has no header: {path}")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise L2ForwardResolverError(f"could not read resolver scope ledger: {path}") from exc


def _json_gzip_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise L2ForwardResolverError(f"could not read resolver scope ledger: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise L2ForwardResolverError(f"resolver JSON scope ledger is malformed: {path}")
    rows = value["rows"]
    if any(not isinstance(row, dict) for row in rows):
        raise L2ForwardResolverError(f"resolver JSON scope row is malformed: {path}")
    return rows


def _as_reactant_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item for item in parsed)
    ):
        raise L2ForwardResolverError(f"{label} must contain a nonempty reactant list")
    return tuple(parsed)


def _as_product_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise L2ForwardResolverError(f"{label} must contain a product list")
    return tuple(parsed)


class _AdmissionAccumulator:
    def __init__(self) -> None:
        self._records: dict[
            str,
            dict[tuple[tuple[str, ...], str], set[str]],
        ] = defaultdict(lambda: defaultdict(set))

    def add(
        self,
        *,
        reaction_id: str,
        reactants: tuple[str, ...],
        target: str,
        source_record: dict[str, Any],
    ) -> None:
        self._records[reaction_id][(reactants, target)].add(_sha256_payload(source_record))

    def admissions_for(self, reaction_id: str) -> list[dict[str, Any]]:
        records = self._records.get(reaction_id, {})
        output = []
        for (reactants, target), source_hashes in sorted(records.items()):
            identity = {
                "reaction_id": reaction_id,
                "role_ordered_reactants": list(reactants),
                "target_smiles": target,
            }
            output.append(
                {
                    "admission_id": "exact-pair-" + _sha256_payload(identity)[:20],
                    "role_ordered_reactants": list(reactants),
                    "target_smiles": target,
                    "source_record_sha256s": sorted(source_hashes),
                }
            )
        return output


def _collect_upstream_exact_source(repo_root: Path, accumulator: _AdmissionAccumulator) -> None:
    rows = _csv_gzip_rows(repo_root / _UPSTREAM_LEDGER)
    for row in rows:
        expected = row.get("expected_product")
        products = _as_product_tuple(
            row.get("forward_products_json"),
            label="upstream forward_products_json",
        )
        if (
            row.get("verification_status") != "verified_exact_product_unique"
            or row.get("expected_product_in_outputs") != "true"
            or row.get("forward_product_count") != "1"
            or products != (expected,)
        ):
            raise L2ForwardResolverError("upstream exact-source ledger contains an unverified row")
        accumulator.add(
            reaction_id=row["qualified_reaction_id"],
            reactants=_as_reactant_tuple(row["reactants_json"], label="upstream reactants_json"),
            target=str(expected),
            source_record=row,
        )


def _collect_exact_chain(
    repo_root: Path,
    ledger_path: Path,
    accumulator: _AdmissionAccumulator,
) -> None:
    rows = _json_gzip_rows(repo_root / ledger_path)
    for row in rows:
        product = row.get("product_smiles")
        products = tuple(row.get("forward_products", ()))
        if (
            row.get("verified_exact_product_unique") is not True
            or row.get("forward_product_count") != 1
            or products != (product,)
        ):
            raise L2ForwardResolverError("exact-chain ledger contains an unverified row")
        if isinstance(row.get("reactants"), list):
            reactants = tuple(value["canonical_smiles"] for value in row["reactants"])
        else:
            reactants = (str(row.get("reactant_smiles")),)
        accumulator.add(
            reaction_id=str(row["reaction_id"]),
            reactants=reactants,
            target=str(product),
            source_record=row,
        )


def _collect_targeted_exact(repo_root: Path, accumulator: _AdmissionAccumulator) -> None:
    evidence = _load_json(repo_root / _TARGETED_EVIDENCE)
    exact_routes = evidence.get("exact_routes")
    if not isinstance(exact_routes, list):
        raise L2ForwardResolverError("targeted exact evidence contains no routes")
    reaction_by_route = {
        str(record["route_id"]): str(record["reaction_id"])
        for record in exact_routes
        if isinstance(record, dict) and record.get("disposition") == "admit_exact"
    }
    rows = _csv_gzip_rows(repo_root / _TARGETED_LEDGER)
    for row in rows:
        target = row.get("target_canonical_smiles")
        products = _as_product_tuple(
            row.get("forward_products_json"),
            label="targeted forward_products_json",
        )
        if (
            row.get("forward_verified_exact_unique") != "True"
            or row.get("forward_product_count") != "1"
            or products != (target,)
            or row.get("route_id") not in reaction_by_route
        ):
            raise L2ForwardResolverError("targeted exact ledger contains an unverified row")
        accumulator.add(
            reaction_id=reaction_by_route[row["route_id"]],
            reactants=(row["reactant_canonical_smiles"],),
            target=str(target),
            source_record=row,
        )


def build_l2_forward_resolver_config(repo_root: Path) -> dict[str, Any]:
    """Build the frozen inactive resolver manifest from development evidence only."""

    accumulator = _AdmissionAccumulator()
    _collect_upstream_exact_source(repo_root, accumulator)
    _collect_exact_chain(repo_root, _C16_LEDGER, accumulator)
    _collect_exact_chain(repo_root, _C18_LEDGER, accumulator)
    _collect_targeted_exact(repo_root, accumulator)

    registry_specs = (
        ("l2_exact_source_registry", _UPSTREAM_REGISTRY),
        ("l2_exact_c16_registry", _C16_REGISTRY),
        ("l2_exact_c18_registry", _C18_REGISTRY),
    )
    artifacts: dict[str, dict[str, Any]] = {
        label: _artifact(repo_root, path) for label, path in registry_specs
    }
    for label, path in (
        ("l2_exact_source_step_ledger", _UPSTREAM_LEDGER),
        ("l2_exact_c16_step_ledger", _C16_LEDGER),
        ("l2_exact_c18_step_ledger", _C18_LEDGER),
        ("targeted_aldehyde_exact_evidence", _TARGETED_EVIDENCE),
        ("targeted_aldehyde_step_ledger", _TARGETED_LEDGER),
    ):
        artifacts[label] = _artifact(repo_root, path)

    transforms: list[dict[str, Any]] = []
    for registry_label, registry_relative in registry_specs:
        registry_path = repo_root / registry_relative
        registry = _load_json(registry_path)
        reactions = registry.get("reactions")
        if not isinstance(reactions, list):
            raise L2ForwardResolverError(f"registry has no reactions: {registry_path}")
        for definition in reactions:
            if not isinstance(definition, dict):
                raise L2ForwardResolverError(f"registry reaction is malformed: {registry_path}")
            reaction_id = str(definition["reaction_id"])
            variant_relative = Path("configs/route/variants") / f"{reaction_id}.json"
            variant_label = f"variant__{reaction_id}"
            artifacts[variant_label] = _artifact(repo_root, variant_relative)
            compiled = load_qualified_forward_reaction(
                registry_path,
                repo_root / variant_relative,
                reaction_id=reaction_id,
            )
            admissions = accumulator.admissions_for(reaction_id)
            if not admissions:
                raise L2ForwardResolverError(
                    f"admitted L2 reaction has no verified exact scope: {reaction_id}"
                )
            transforms.append(
                {
                    "reaction_id": reaction_id,
                    "registry_artifact": registry_label,
                    "variant_artifact": variant_label,
                    "transform_sha256": transform_receipt_sha256(
                        reaction_id=reaction_id,
                        registry_sha256=artifacts[registry_label]["sha256"],
                        variant_sha256=artifacts[variant_label]["sha256"],
                        role_names=compiled.role_names,
                        qualification_status=str(definition["status"]),
                    ),
                    "admissions": admissions,
                }
            )

    transforms.sort(key=lambda value: value["reaction_id"])
    return {
        "schema_version": L2_FORWARD_RESOLVER_CONFIG_SCHEMA_VERSION,
        "status": "frozen_inactive_not_benchmarked",
        "activation_authorized": False,
        "resolver_id": "forge-independent-exact-scope-upstream-l2",
        "resolver_version": "1",
        "purpose": (
            "Independently screen proposal-only reactants against every exact-scope-admitted "
            "upstream L2 transform and role assignment."
        ),
        "scope_guards": {
            "ugi_l1_registry_admitted": False,
            "proposal_model_class_may_select_verifier": False,
            "proposal_model_score_may_select_verifier": False,
            "model_score_may_enter_v_syn": False,
            "exact_forward_implies_scope_evidence": False,
            "sealed_holdouts_accessed": False,
            "frozen_benchmark_executed": False,
        },
        "budget": {
            "maximum_forward_calls_per_proposal": 50,
            "maximum_products_per_assignment": 64,
            "budget_exhaustion_policy": "censor_before_partial_forward_execution",
        },
        "scientific_authority": {
            "proposal_screen_only": True,
            "may_create_route_evidence": False,
            "may_set_substrate_scope": False,
            "may_set_synthesis_success_probability": False,
            "may_close_route": False,
            "may_enter_synthesis_value": False,
        },
        "source_scope": {
            "development_inputs_only": True,
            "admission_unit": "exact_role_ordered_reactant_tuple_and_exact_target",
            "family_or_analogue_projection_admitted": False,
            "duplicate_evidence_records_merge_without_duplicating_assignments": True,
        },
        "artifacts": dict(sorted(artifacts.items())),
        "transforms": transforms,
        "summary": {
            "transform_count": len(transforms),
            "exact_pair_admission_count": sum(len(value["admissions"]) for value in transforms),
            "registry_count": len(registry_specs),
        },
    }


__all__ = ["build_l2_forward_resolver_config"]
