"""Forward-verify exact-source Ugi component programs conservatively.

Only transformations bound to a hash-pinned qualified reaction and variant are
executed.  Source extraction, structural projection and procurement closure are
never interpreted as forward verification.  Unsupported transformations remain
explicit, step-level negative results.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.synthesis.engine.qualified_forward import (
    QualifiedForwardError,
    QualifiedForwardReaction,
    load_qualified_forward_reaction,
    unique_forward_products,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_exact_source_forward_verification_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_exact_source_forward_verification.v1"
SOURCE_ROUTE_SCHEMA_VERSION = "m0_09_agile_component_routes.v1"
EXACT_PROGRAM_STATUS = "exact_source_program"
VERIFIED_STATUS = "verified_exact_product_unique"
AMBIGUOUS_MATCH_STATUS = "ambiguous_expected_product_among_multiple"
MISSING_BINDING_STATUS = "unverified_missing_qualified_transform_binding"
NO_PRODUCTS_STATUS = "unverified_no_forward_products"
EXPECTED_ABSENT_STATUS = "unverified_expected_product_absent"
INVALID_INPUT_STATUS = "error_invalid_step_input"
EXECUTION_ERROR_STATUS = "error_qualified_transform_execution"
LEDGER_FIELDS = (
    "component_id",
    "component_role",
    "component_smiles",
    "program_family",
    "source_route_id",
    "step_index",
    "transformation",
    "reactants_json",
    "expected_product",
    "qualified_reaction_id",
    "verification_status",
    "reason",
    "forward_product_count",
    "forward_products_json",
    "expected_product_in_outputs",
)


class Ugi3ExactSourceForwardVerificationError(ValueError):
    """Raised when the gate's hash-pinned input contract is invalid."""


@dataclass(frozen=True)
class TransformBinding:
    """One executable upstream transformation binding."""

    transformation: str
    reaction_id: str
    variant_path: Path
    reactant_order: tuple[int, ...]
    compiled: QualifiedForwardReaction


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3ExactSourceForwardVerificationError, label=label)


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    observed = sha256_file(path)
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected:
        raise Ugi3ExactSourceForwardVerificationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _read_gzip_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3ExactSourceForwardVerificationError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3ExactSourceForwardVerificationError(f"could not read {label}: {path}") from exc


def _json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3ExactSourceForwardVerificationError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3ExactSourceForwardVerificationError(f"{label} must be a list")
    return parsed


def _canonical_smiles(smiles: Any, *, label: str, isomeric_smiles: bool) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3ExactSourceForwardVerificationError(f"{label} must be a non-empty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3ExactSourceForwardVerificationError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=isomeric_smiles,
    )


def _source_routes_by_id(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if artifact.get("schema_version") != SOURCE_ROUTE_SCHEMA_VERSION:
        raise Ugi3ExactSourceForwardVerificationError("source routes have an unsupported schema")
    routes = artifact.get("routes")
    if not isinstance(routes, list) or any(
        not isinstance(route, dict) or not isinstance(route.get("route_id"), str)
        for route in routes
    ):
        raise Ugi3ExactSourceForwardVerificationError("source routes contain malformed records")
    by_id = {route["route_id"]: route for route in routes}
    if len(by_id) != len(routes):
        raise Ugi3ExactSourceForwardVerificationError("source route identifiers are not unique")
    return by_id


def _normalized_source_step(step: Mapping[str, Any]) -> dict[str, Any]:
    reactants = step.get("reactants")
    product = step.get("product")
    if not isinstance(reactants, list) or not isinstance(product, dict):
        raise Ugi3ExactSourceForwardVerificationError(
            "source route contains a malformed reaction step"
        )
    canonical_reactants: list[str] = []
    for reactant in reactants:
        if not isinstance(reactant, dict) or not isinstance(reactant.get("canonical_smiles"), str):
            raise Ugi3ExactSourceForwardVerificationError(
                "source route contains a malformed reactant"
            )
        canonical_reactants.append(reactant["canonical_smiles"])
    if not isinstance(product.get("canonical_smiles"), str):
        raise Ugi3ExactSourceForwardVerificationError("source route contains a malformed product")
    return {
        "step_index": step.get("step_index"),
        "transformation": step.get("transformation"),
        "reactants": sorted(canonical_reactants),
        "product": product["canonical_smiles"],
    }


def _normalized_program_step(step: Mapping[str, Any]) -> dict[str, Any]:
    reactants = step.get("reactants")
    if (
        isinstance(step.get("step_index"), bool)
        or not isinstance(step.get("step_index"), int)
        or not isinstance(step.get("transformation"), str)
        or not isinstance(reactants, list)
        or any(not isinstance(value, str) for value in reactants)
        or not isinstance(step.get("product"), str)
    ):
        raise Ugi3ExactSourceForwardVerificationError(
            "component ledger contains a malformed program step"
        )
    return {
        "step_index": step["step_index"],
        "transformation": step["transformation"],
        "reactants": list(reactants),
        "product": step["product"],
    }


def _registry_reaction_ids(registry: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise Ugi3ExactSourceForwardVerificationError(f"{label} must contain a reactions list")
    identifiers = [
        reaction.get("reaction_id") for reaction in reactions if isinstance(reaction, dict)
    ]
    if any(not isinstance(identifier, str) for identifier in identifiers):
        raise Ugi3ExactSourceForwardVerificationError(
            f"{label} contains a malformed reaction identifier"
        )
    if len(set(identifiers)) != len(identifiers):
        raise Ugi3ExactSourceForwardVerificationError(
            f"{label} contains duplicate reaction identifiers"
        )
    return tuple(sorted(identifiers))


def _validate_exact_source_qualification_metadata(
    registry: Mapping[str, Any],
    bindings: Mapping[str, TransformBinding],
    transformation_counts: Mapping[str, int],
    *,
    source_routes_path: Path,
) -> None:
    if registry.get("scope") != "exact_source_forward_verification_only":
        raise Ugi3ExactSourceForwardVerificationError(
            "upstream registry is not limited to exact-source forward verification"
        )
    source = registry.get("source_route_artifact")
    if not isinstance(source, dict) or source.get("sha256") != sha256_file(source_routes_path):
        raise Ugi3ExactSourceForwardVerificationError(
            "upstream registry does not pin the exact source-route artifact"
        )
    claims = registry.get("claims_boundary")
    required_false_claims = {
        "qualified_for_general_enumeration",
        "qualified_for_substrate_scope_extrapolation",
        "conditions_encoded_in_graph_transform",
        "exact_graph_reconstruction_is_experimental_success",
    }
    if not isinstance(claims, dict) or any(
        claims.get(field) is not False for field in required_false_claims
    ):
        raise Ugi3ExactSourceForwardVerificationError(
            "upstream registry lacks conservative claim boundaries"
        )
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise Ugi3ExactSourceForwardVerificationError("upstream registry has no reactions")
    by_id = {
        reaction.get("reaction_id"): reaction
        for reaction in reactions
        if isinstance(reaction, dict) and isinstance(reaction.get("reaction_id"), str)
    }
    if set(bindings) != set(transformation_counts):
        raise Ugi3ExactSourceForwardVerificationError(
            "exact-source qualification requires one binding per observed transformation"
        )
    for transformation, binding in bindings.items():
        definition = by_id.get(binding.reaction_id)
        if (
            not isinstance(definition, dict)
            or definition.get("status") != "qualified_for_exact_source_forward_verification_only"
            or definition.get("source_transformation") != transformation
            or definition.get("qualification_pair_count") != transformation_counts[transformation]
        ):
            raise Ugi3ExactSourceForwardVerificationError(
                f"binding {transformation!r} lacks matching exact-source qualification metadata"
            )


def _portable_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _load_bindings(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    executable_registry_path: Path,
) -> dict[str, TransformBinding]:
    raw_bindings = config.get("qualified_transform_bindings")
    if not isinstance(raw_bindings, dict):
        raise Ugi3ExactSourceForwardVerificationError(
            "qualified_transform_bindings must be an object"
        )
    bindings: dict[str, TransformBinding] = {}
    for transformation, definition in sorted(raw_bindings.items()):
        if not isinstance(transformation, str) or not isinstance(definition, dict):
            raise Ugi3ExactSourceForwardVerificationError(
                "qualified transform binding is malformed"
            )
        reaction_id = definition.get("reaction_id")
        variant_asset = definition.get("variant_asset")
        expected_sha256 = definition.get("variant_expected_sha256")
        reactant_order = definition.get("reactant_order")
        if not isinstance(reaction_id, str) or not isinstance(variant_asset, str):
            raise Ugi3ExactSourceForwardVerificationError(
                f"binding {transformation!r} lacks reaction_id or variant_asset"
            )
        if (
            not isinstance(reactant_order, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int) for index in reactant_order
            )
            or sorted(reactant_order) != list(range(len(reactant_order)))
        ):
            raise Ugi3ExactSourceForwardVerificationError(
                f"binding {transformation!r} reactant_order must be a permutation"
            )
        variant_path = (config_path.parent.parent.parent / variant_asset).resolve()
        _verify_hash(
            variant_path,
            expected_sha256,
            label=f"{transformation} variant",
        )
        try:
            compiled = load_qualified_forward_reaction(
                executable_registry_path,
                variant_path,
                reaction_id=reaction_id,
            )
        except QualifiedForwardError as exc:
            raise Ugi3ExactSourceForwardVerificationError(
                f"could not load binding {transformation!r}: {exc}"
            ) from exc
        if len(reactant_order) != len(compiled.role_names):
            raise Ugi3ExactSourceForwardVerificationError(
                f"binding {transformation!r} reactant_order length does not match roles"
            )
        bindings[transformation] = TransformBinding(
            transformation=transformation,
            reaction_id=reaction_id,
            variant_path=variant_path,
            reactant_order=tuple(reactant_order),
            compiled=compiled,
        )
    return bindings


def _verify_step(
    step: Mapping[str, Any],
    binding: TransformBinding | None,
    *,
    max_products: int,
    isomeric_smiles: bool,
) -> dict[str, Any]:
    transformation = step["transformation"]
    reactants = list(step["reactants"])
    expected = step["product"]
    if binding is None:
        return {
            "qualified_reaction_id": "",
            "verification_status": MISSING_BINDING_STATUS,
            "reason": (f"no qualified executable transform is bound for {transformation}"),
            "forward_product_count": 0,
            "forward_products": [],
            "expected_product_in_outputs": False,
        }
    try:
        ordered_reactants = [reactants[index] for index in binding.reactant_order]
        canonical_expected = _canonical_smiles(
            expected,
            label=f"{transformation} expected product",
            isomeric_smiles=isomeric_smiles,
        )
        for index, reactant in enumerate(ordered_reactants):
            _canonical_smiles(
                reactant,
                label=f"{transformation} reactant {index}",
                isomeric_smiles=isomeric_smiles,
            )
    except Ugi3ExactSourceForwardVerificationError as exc:
        return {
            "qualified_reaction_id": binding.reaction_id,
            "verification_status": INVALID_INPUT_STATUS,
            "reason": str(exc),
            "forward_product_count": 0,
            "forward_products": [],
            "expected_product_in_outputs": False,
        }
    try:
        products = unique_forward_products(
            binding.compiled,
            ordered_reactants,
            max_products=max_products,
            isomeric_smiles=isomeric_smiles,
        )
    except QualifiedForwardError as exc:
        return {
            "qualified_reaction_id": binding.reaction_id,
            "verification_status": EXECUTION_ERROR_STATUS,
            "reason": str(exc),
            "forward_product_count": 0,
            "forward_products": [],
            "expected_product_in_outputs": False,
        }
    matched = canonical_expected in products
    if matched and len(products) == 1:
        status = VERIFIED_STATUS
        reason = "qualified forward execution reproduced the exact expected product"
    elif matched:
        status = AMBIGUOUS_MATCH_STATUS
        reason = (
            "qualified forward execution reproduced the expected product but also "
            "produced additional unique products"
        )
    elif not products:
        status = NO_PRODUCTS_STATUS
        reason = "qualified forward execution produced no sanitized product"
    else:
        status = EXPECTED_ABSENT_STATUS
        reason = "qualified forward execution did not reproduce the exact expected product"
    return {
        "qualified_reaction_id": binding.reaction_id,
        "verification_status": status,
        "reason": reason,
        "forward_product_count": len(products),
        "forward_products": list(products),
        "expected_product_in_outputs": matched,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3ExactSourceForwardVerificationError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif observed != expected:
        raise Ugi3ExactSourceForwardVerificationError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def build_exact_source_forward_verification(
    config_path: Path,
    component_program_ledger_path: Path,
    source_routes_path: Path,
    executable_registry_path: Path,
    family_registry_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Return a hash-pinned exact-source step audit and deterministic ledger."""

    config = _load_json(config_path, label="forward-verification config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ExactSourceForwardVerificationError("unsupported config schema")
    paths = {
        "component_program_ledger": component_program_ledger_path,
        "source_routes": source_routes_path,
        "executable_registry": executable_registry_path,
        "family_registry": family_registry_path,
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(paths):
        raise Ugi3ExactSourceForwardVerificationError("config inputs mismatch")
    for name, path in paths.items():
        record = inputs[name]
        if not isinstance(record, dict):
            raise Ugi3ExactSourceForwardVerificationError(f"config input {name} must be an object")
        _verify_hash(path, record.get("expected_sha256"), label=name)

    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise Ugi3ExactSourceForwardVerificationError("policy must be an object")
    max_products = policy.get("max_products_per_step")
    isomeric_smiles = policy.get("isomeric_smiles")
    require_qualification_metadata = policy.get(
        "require_exact_source_qualification_metadata",
        False,
    )
    if (
        isinstance(max_products, bool)
        or not isinstance(max_products, int)
        or max_products <= 0
        or not isinstance(isomeric_smiles, bool)
        or not isinstance(require_qualification_metadata, bool)
    ):
        raise Ugi3ExactSourceForwardVerificationError(
            "invalid max_products_per_step or isomeric_smiles policy"
        )

    source_routes = _source_routes_by_id(_load_json(source_routes_path, label="source routes"))
    executable_registry = _load_json(
        executable_registry_path,
        label="executable qualified registry",
    )
    family_registry = _load_json(
        family_registry_path,
        label="qualified family registry",
    )
    executable_reaction_ids = _registry_reaction_ids(
        executable_registry,
        label="executable qualified registry",
    )
    family_reaction_ids = _registry_reaction_ids(
        family_registry,
        label="qualified family registry",
    )
    bindings = _load_bindings(
        config,
        config_path=config_path,
        executable_registry_path=executable_registry_path,
    )
    repo_root = config_path.resolve().parents[2]

    program_rows = _read_gzip_csv(
        component_program_ledger_path,
        label="component program ledger",
    )
    exact_rows = [row for row in program_rows if row.get("program_status") == EXACT_PROGRAM_STATUS]
    ledger_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    transformation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_route_ids: set[str] = set()
    verified_components = 0

    for row in sorted(exact_rows, key=lambda item: item["component_id"]):
        component_id = row.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise Ugi3ExactSourceForwardVerificationError(
                "exact-source component lacks component_id"
            )
        route_ids = _json_list(
            row.get("exact_source_route_ids_json", ""),
            label=f"{component_id} source route identifiers",
        )
        if len(route_ids) != 1 or not isinstance(route_ids[0], str):
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} must reference exactly one source route"
            )
        route_id = route_ids[0]
        try:
            route = source_routes[route_id]
        except KeyError as exc:
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} references unknown source route {route_id!r}"
            ) from exc
        if route_id in source_route_ids:
            raise Ugi3ExactSourceForwardVerificationError(
                f"source route {route_id!r} is assigned to multiple components"
            )
        source_route_ids.add(route_id)
        target = route.get("target")
        source_steps = route.get("steps")
        if (
            not isinstance(target, dict)
            or target.get("canonical_smiles") != row.get("canonical_smiles")
            or not isinstance(source_steps, list)
        ):
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} source route target or steps mismatch"
            )
        raw_program_steps = _json_list(
            row.get("program_steps_json", ""),
            label=f"{component_id} program steps",
        )
        if any(not isinstance(step, dict) for step in raw_program_steps):
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} contains a malformed program step"
            )
        program_steps = [_normalized_program_step(step) for step in raw_program_steps]
        normalized_source_steps = [_normalized_source_step(step) for step in source_steps]
        normalized_program_steps = [
            {**step, "reactants": sorted(step["reactants"])} for step in program_steps
        ]
        if normalized_source_steps != normalized_program_steps:
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} program does not exactly reproduce source route {route_id!r}"
            )
        expected_indices = list(range(1, len(program_steps) + 1))
        if (
            not program_steps
            or [step["step_index"] for step in program_steps] != expected_indices
            or program_steps[-1]["product"] != row["canonical_smiles"]
        ):
            raise Ugi3ExactSourceForwardVerificationError(
                f"{component_id} program steps are not a complete ordered route"
            )

        family_counts[row["program_family"]] += 1
        role_counts[row["role"]] += 1
        component_statuses: list[str] = []
        for step in program_steps:
            transformation = step["transformation"]
            outcome = _verify_step(
                step,
                bindings.get(transformation),
                max_products=max_products,
                isomeric_smiles=isomeric_smiles,
            )
            status = outcome["verification_status"]
            status_counts[status] += 1
            transformation_counts[transformation] += 1
            component_statuses.append(status)
            ledger_rows.append(
                {
                    "component_id": component_id,
                    "component_role": row["role"],
                    "component_smiles": row["canonical_smiles"],
                    "program_family": row["program_family"],
                    "source_route_id": route_id,
                    "step_index": step["step_index"],
                    "transformation": transformation,
                    "reactants_json": json.dumps(step["reactants"], separators=(",", ":")),
                    "expected_product": step["product"],
                    "qualified_reaction_id": outcome["qualified_reaction_id"],
                    "verification_status": status,
                    "reason": outcome["reason"],
                    "forward_product_count": outcome["forward_product_count"],
                    "forward_products_json": json.dumps(
                        outcome["forward_products"], separators=(",", ":")
                    ),
                    "expected_product_in_outputs": str(
                        outcome["expected_product_in_outputs"]
                    ).lower(),
                }
            )
        if component_statuses and all(status == VERIFIED_STATUS for status in component_statuses):
            verified_components += 1

    summary = {
        "exact_source_component_programs": len(exact_rows),
        "unique_source_routes": len(source_route_ids),
        "upstream_steps": len(ledger_rows),
        "forward_verified_steps": status_counts[VERIFIED_STATUS],
        "steps_by_transformation": dict(sorted(transformation_counts.items())),
        "steps_by_verification_status": dict(sorted(status_counts.items())),
        "component_programs_by_family": dict(sorted(family_counts.items())),
        "component_programs_by_role": dict(sorted(role_counts.items())),
        "fully_forward_verified_component_programs": verified_components,
    }
    if require_qualification_metadata:
        _validate_exact_source_qualification_metadata(
            executable_registry,
            bindings,
            transformation_counts,
            source_routes_path=source_routes_path,
        )
    _validate_expected(summary, config.get("expected_counts"), label="summary")
    ledger_bytes = _csv_bytes(ledger_rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config.get("task"),
        "inputs": {
            "config": {
                "path": _portable_path(config_path, repo_root=repo_root),
                "sha256": sha256_file(config_path),
            },
            **{
                name: {
                    "path": _portable_path(path, repo_root=repo_root),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "bound_variants": {
                transformation: {
                    "path": _portable_path(binding.variant_path, repo_root=repo_root),
                    "sha256": sha256_file(binding.variant_path),
                }
                for transformation, binding in sorted(bindings.items())
            },
        },
        "policy": policy,
        "registry_inventory": {
            "executable_reaction_ids": list(executable_reaction_ids),
            "qualified_family_reaction_ids": list(family_reaction_ids),
            "bound_upstream_transformations": sorted(bindings),
            "observed_unbound_transformations": sorted(set(transformation_counts) - set(bindings)),
        },
        "summary": summary,
        "claims_boundary": {
            "route_extraction_is_step_forward_verification": False,
            "structural_program_reproduction_is_step_forward_verification": False,
            "procurement_closure_is_step_forward_verification": False,
            "unbound_transform_is_forward_verified": False,
            "multiproduct_match_is_unambiguous_verification": False,
            "exact_source_pair_reconstruction_is_substrate_scope_qualification": False,
            "graph_transform_encodes_reported_reagents_or_conditions": False,
            "exact_graph_reconstruction_is_experimental_success": False,
            "complete_forward_verified_dossier_claimed": False,
        },
        "safe_claim": (
            f"Exact canonical products were uniquely reconstructed for "
            f"{status_counts[VERIFIED_STATUS]}/{len(ledger_rows)} extracted upstream "
            f"steps spanning {verified_components}/{len(exact_rows)} exact-source Ugi "
            "component programs. This establishes deterministic reconstruction only "
            "for these source-resolved pairs; it does not qualify general substrate "
            "scope, encode reagents or conditions, establish experimental success, "
            "or by itself close complete product dossiers."
        ),
        "artifacts": {
            "step_ledger_sha256": sha256_bytes(ledger_bytes),
        },
    }
    return result, ledger_bytes
