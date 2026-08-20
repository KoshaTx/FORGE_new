"""Qualify Ugi-3 reactive-site semantics against all measured AGILE products."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from forge.chem.reactive_sites import (
    SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES,
    audit_reactive_site_multiplicity,
)
from forge.core.io import read_json_object
from forge.route.sources.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_ugi3_assembly_qualification_config.v1"
RESULT_SCHEMA_VERSION = "m0_09_ugi3_assembly_qualification.v1"
LABEL_PATTERN = re.compile(r"^A([1-9][0-9]*)B([1-9][0-9]*)C([1-9][0-9]*)$")
REQUIRED_COLUMNS = {
    "id",
    "label",
    "combined_mol_SMILES",
    "A_smiles",
    "B_smiles",
    "C_smiles",
}


class Ugi3AssemblyQualificationError(ValueError):
    """Raised when AGILE assembly qualification violates its frozen contract."""


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3AssemblyQualificationError, label=label)


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise Ugi3AssemblyQualificationError(f"{label} not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise Ugi3AssemblyQualificationError(
            f"{label} is not valid YAML: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise Ugi3AssemblyQualificationError(f"{label} must contain a YAML mapping")
    return value


def _verify_hash(path: Path, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3AssemblyQualificationError(
            f"{label} expected_sha256 must be a 64-character string"
        )
    if not path.exists():
        raise Ugi3AssemblyQualificationError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise Ugi3AssemblyQualificationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _expect_count(observed: int, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise Ugi3AssemblyQualificationError(f"{label} expected count must be an integer")
    if observed != expected:
        raise Ugi3AssemblyQualificationError(
            f"{label} count mismatch: expected {expected}, observed {observed}"
        )


def _validated_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Ugi3AssemblyQualificationError(f"{label} must be a nonempty string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Ugi3AssemblyQualificationError(
            f"{label} is not valid ISO-8601: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise Ugi3AssemblyQualificationError(f"{label} must include a timezone")
    return parsed.isoformat()


def _canonical_smiles(value: Any, *, label: str) -> tuple[str, Chem.Mol]:
    if not isinstance(value, str) or not value:
        raise Ugi3AssemblyQualificationError(f"{label} must be a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise Ugi3AssemblyQualificationError(f"{label} is not valid SMILES: {value!r}")
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        molecule,
    )


def _read_agile_rows(path: Path) -> list[dict[str, str]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise Ugi3AssemblyQualificationError(
            f"AGILE measured library not found: {path}"
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or ()))
            raise Ugi3AssemblyQualificationError(
                f"AGILE measured library is missing columns: {missing}"
            )
        return list(reader)


def _load_ugi_reaction(
    registry: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Chem.Mol,
    tuple[Chem.Mol, ...],
    rdChemReactions.ChemicalReaction,
]:
    scope = config["scope"]
    reaction_id = scope["reaction_id"]
    if variant.get("variant") != reaction_id or variant.get("reaction_id") != reaction_id:
        raise Ugi3AssemblyQualificationError(
            "Ugi variant and qualification reaction_id disagree"
        )
    compatibility = variant.get("compatibility")
    if not isinstance(compatibility, dict):
        raise Ugi3AssemblyQualificationError(
            "Ugi variant is missing compatibility semantics"
        )
    semantics = compatibility.get("amine_head_site_multiplicity_semantics")
    if semantics != scope["multiplicity_semantics"]:
        raise Ugi3AssemblyQualificationError(
            "Ugi variant and qualification multiplicity semantics disagree"
        )
    if semantics != SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES:
        raise Ugi3AssemblyQualificationError(
            f"unsupported Ugi amine multiplicity semantics {semantics!r}"
        )
    if compatibility.get("deduplicate_forward_products") is not True:
        raise Ugi3AssemblyQualificationError(
            "Ugi variant must require forward-product deduplication"
        )
    if (
        compatibility.get("multiple_unique_products_require_explicit_site_selection")
        is not True
    ):
        raise Ugi3AssemblyQualificationError(
            "Ugi variant must require explicit selection for multiple unique products"
        )

    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise Ugi3AssemblyQualificationError("qualified registry has no reactions")
    matches = [
        reaction
        for reaction in reactions
        if isinstance(reaction, dict) and reaction.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise Ugi3AssemblyQualificationError(
            f"registry reaction {reaction_id!r} must resolve exactly once"
        )
    reaction = matches[0]
    roles = reaction.get("reactant_roles")
    if not isinstance(roles, list) or len(roles) != variant.get(
        "expected_reactant_count"
    ):
        raise Ugi3AssemblyQualificationError(
            "registry and variant reactant counts disagree"
        )
    role_names = [role.get("name") for role in roles if isinstance(role, dict)]
    variant_role_names = [
        role.get("name")
        for role in variant.get("roles", [])
        if isinstance(role, dict)
    ]
    if role_names != variant_role_names:
        raise Ugi3AssemblyQualificationError(
            "registry and variant reactant role order disagree"
        )
    amine_matches = [
        role for role in roles if role.get("name") == scope["amine_role"]
    ]
    if len(amine_matches) != 1:
        raise Ugi3AssemblyQualificationError(
            "qualified registry must contain exactly one Ugi amine role"
        )
    amine_role = amine_matches[0]
    handle = Chem.MolFromSmarts(amine_role.get("required_handle_smarts", ""))
    if handle is None:
        raise Ugi3AssemblyQualificationError(
            "Ugi amine required-handle SMARTS did not compile"
        )
    if scope.get("require_single_atom_amine_handle") is True and handle.GetNumAtoms() != 1:
        raise Ugi3AssemblyQualificationError(
            "symmetry qualification requires a single-atom Ugi amine handle"
        )
    forbidden = tuple(
        Chem.MolFromSmarts(smarts)
        for smarts in amine_role.get("forbidden_smarts", [])
    )
    if any(pattern is None for pattern in forbidden):
        raise Ugi3AssemblyQualificationError(
            "Ugi amine forbidden SMARTS did not compile"
        )
    forward = rdChemReactions.ReactionFromSmarts(
        reaction.get("atom_mapped_reaction_smarts", "")
    )
    if forward is None or forward.GetNumReactantTemplates() != len(roles):
        raise Ugi3AssemblyQualificationError(
            "Ugi forward transform did not compile with the declared role count"
        )
    return (
        reaction,
        amine_role,
        handle,
        tuple(pattern for pattern in forbidden if pattern is not None),
        forward,
    )


def _forward_products(
    reaction: rdChemReactions.ChemicalReaction,
    reactants: tuple[Chem.Mol, ...],
    *,
    max_outcomes: int,
    row_label: str,
) -> tuple[int, set[str]]:
    with rdBase.BlockLogs():
        outcomes = reaction.RunReactants(reactants, maxProducts=max_outcomes)
    if len(outcomes) >= max_outcomes:
        raise Ugi3AssemblyQualificationError(
            f"{row_label} reached max_forward_outcomes={max_outcomes}"
        )
    products: list[str] = []
    for outcome_index, outcome in enumerate(outcomes):
        if len(outcome) != 1:
            raise Ugi3AssemblyQualificationError(
                f"{row_label} outcome {outcome_index} did not contain one product"
            )
        molecule = outcome[0]
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise Ugi3AssemblyQualificationError(
                f"{row_label} outcome {outcome_index} could not be sanitized"
            ) from exc
        products.append(
            Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
        )
    return len(outcomes), set(products)


def _row_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_ugi3_assembly_qualification(
    config_path: Path,
    agile_measured_library_path: Path,
    qualified_reactions_path: Path,
    ugi_variant_path: Path,
) -> dict[str, Any]:
    """Build the deterministic 1,200-product Ugi assembly qualification."""

    config = _load_json(config_path, label="Ugi assembly qualification config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3AssemblyQualificationError(
            f"config schema must be {CONFIG_SCHEMA_VERSION!r}"
        )
    generated_utc = _validated_timestamp(
        config.get("generated_utc"),
        label="generated_utc",
    )
    scope = config.get("scope")
    expected = config.get("expected_counts")
    if not isinstance(scope, dict) or not isinstance(expected, dict):
        raise Ugi3AssemblyQualificationError(
            "qualification config must define scope and expected_counts objects"
        )
    max_outcomes = scope.get("max_forward_outcomes_per_record")
    if (
        isinstance(max_outcomes, bool)
        or not isinstance(max_outcomes, int)
        or max_outcomes <= 0
    ):
        raise Ugi3AssemblyQualificationError(
            "max_forward_outcomes_per_record must be a positive integer"
        )
    paths = {
        "agile_measured_library": agile_measured_library_path,
        "qualified_reactions": qualified_reactions_path,
        "ugi_variant": ugi_variant_path,
    }
    for name, path in paths.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name.replace("_", " "),
        )

    registry = _load_json(qualified_reactions_path, label="qualified reaction registry")
    variant = _load_yaml(ugi_variant_path, label="Ugi variant")
    reaction, amine_role, amine_handle, amine_forbidden, forward = (
        _load_ugi_reaction(registry, variant, config)
    )
    allowed = amine_role.get("allowed_site_multiplicity")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in allowed
        )
    ):
        raise Ugi3AssemblyQualificationError(
            "Ugi amine allowed_site_multiplicity must be positive integers"
        )
    allowed_multiplicity = tuple(sorted(set(allowed)))
    rows = _read_agile_rows(agile_measured_library_path)
    _expect_count(len(rows), expected["measured_products"], label="measured products")

    ids: set[str] = set()
    product_labels: set[str] = set()
    combinations: set[tuple[str, str, str]] = set()
    components: dict[str, dict[str, str]] = {
        "A": {},
        "B": {},
        "C": {},
    }
    row_audit: list[dict[str, Any]] = []
    head_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    forward_profile: Counter[str] = Counter()
    exact_reconstructed = 0
    raw_pass = 0
    qualified_pass = 0
    for position, row in enumerate(rows):
        row_id = row["id"]
        label = row["label"]
        if not row_id or row_id in ids:
            raise Ugi3AssemblyQualificationError(
                f"row {position} has a missing or duplicate id"
            )
        if not label or label in product_labels:
            raise Ugi3AssemblyQualificationError(
                f"row {position} has a missing or duplicate label"
            )
        label_match = LABEL_PATTERN.fullmatch(label)
        if label_match is None:
            raise Ugi3AssemblyQualificationError(
                f"row {position} has malformed component label {label!r}"
            )
        ids.add(row_id)
        product_labels.add(label)
        a_label, b_label, c_label = (
            f"A{label_match.group(1)}",
            f"B{label_match.group(2)}",
            f"C{label_match.group(3)}",
        )
        combinations.add((a_label, b_label, c_label))
        canonical_components: list[str] = []
        reactants: list[Chem.Mol] = []
        for prefix, column, component_label in (
            ("A", "A_smiles", a_label),
            ("B", "B_smiles", b_label),
            ("C", "C_smiles", c_label),
        ):
            canonical, molecule = _canonical_smiles(
                row[column],
                label=f"{label} {component_label}",
            )
            prior = components[prefix].setdefault(component_label, canonical)
            if prior != canonical:
                raise Ugi3AssemblyQualificationError(
                    f"{component_label} maps to multiple structures"
                )
            canonical_components.append(canonical)
            reactants.append(molecule)
        expected_product, _ = _canonical_smiles(
            row["combined_mol_SMILES"],
            label=f"{label} product",
        )

        site_audit = audit_reactive_site_multiplicity(
            reactants[0],
            amine_handle,
        )
        raw_accepted = (
            site_audit.raw_match_count in allowed_multiplicity
            and not any(reactants[0].HasSubstructMatch(pattern) for pattern in amine_forbidden)
        )
        qualified_accepted = (
            site_audit.symmetry_distinct_match_count in allowed_multiplicity
            and not any(reactants[0].HasSubstructMatch(pattern) for pattern in amine_forbidden)
        )
        raw_outcomes, unique_products = _forward_products(
            forward,
            tuple(reactants),
            max_outcomes=max_outcomes,
            row_label=label,
        )
        reconstructed = expected_product in unique_products
        profile_key = f"raw_{raw_outcomes}_unique_{len(unique_products)}"
        forward_profile[profile_key] += 1
        raw_pass += int(raw_accepted)
        qualified_pass += int(qualified_accepted)
        exact_reconstructed += int(reconstructed)
        record = {
            "id": row_id,
            "label": label,
            "component_labels": [a_label, b_label, c_label],
            "canonical_component_smiles": canonical_components,
            "raw_required_handle_matches": site_audit.raw_match_count,
            "symmetry_distinct_required_handle_matches": (
                site_audit.symmetry_distinct_match_count
            ),
            "passes_raw_amine_multiplicity": raw_accepted,
            "passes_qualified_amine_multiplicity": qualified_accepted,
            "raw_forward_outcomes": raw_outcomes,
            "unique_forward_products": len(unique_products),
            "expected_product_reconstructed": reconstructed,
        }
        row_audit.append(record)
        head_rows[a_label].append(record)

    _expect_count(
        len(product_labels),
        expected["unique_product_labels"],
        label="unique product labels",
    )
    _expect_count(
        len(components["A"]),
        expected["amine_heads"],
        label="amine heads",
    )
    _expect_count(
        len(components["B"]),
        expected["aldehyde_components"],
        label="aldehyde components",
    )
    _expect_count(
        len(components["C"]),
        expected["isocyanide_components"],
        label="isocyanide components",
    )
    expected_combinations = {
        (a_label, b_label, c_label)
        for a_label in components["A"]
        for b_label in components["B"]
        for c_label in components["C"]
    }
    if combinations != expected_combinations:
        raise Ugi3AssemblyQualificationError(
            "AGILE rows do not form the complete observed A x B x C combination set"
        )
    _expect_count(
        len(combinations),
        expected["complete_component_combinations"],
        label="complete component combinations",
    )

    head_audit: list[dict[str, Any]] = []
    for head_label in sorted(head_rows, key=lambda value: int(value[1:])):
        records = head_rows[head_label]
        raw_counts = {record["raw_required_handle_matches"] for record in records}
        symmetry_counts = {
            record["symmetry_distinct_required_handle_matches"] for record in records
        }
        raw_outcome_counts = {record["raw_forward_outcomes"] for record in records}
        unique_product_counts = {
            record["unique_forward_products"] for record in records
        }
        if any(
            len(values) != 1
            for values in (
                raw_counts,
                symmetry_counts,
                raw_outcome_counts,
                unique_product_counts,
            )
        ):
            raise Ugi3AssemblyQualificationError(
                f"{head_label} qualification metrics vary across component partners"
            )
        head_audit.append(
            {
                "component_label": head_label,
                "canonical_smiles": components["A"][head_label],
                "rows": len(records),
                "raw_required_handle_matches": next(iter(raw_counts)),
                "symmetry_distinct_required_handle_matches": next(
                    iter(symmetry_counts)
                ),
                "passes_raw_amine_multiplicity": all(
                    record["passes_raw_amine_multiplicity"] for record in records
                ),
                "passes_qualified_amine_multiplicity": all(
                    record["passes_qualified_amine_multiplicity"]
                    for record in records
                ),
                "raw_forward_outcomes_per_record": next(iter(raw_outcome_counts)),
                "unique_forward_products_per_record": next(
                    iter(unique_product_counts)
                ),
                "products_reconstructed_exactly": sum(
                    record["expected_product_reconstructed"] for record in records
                ),
                "equivalent_site_collapse": (
                    next(iter(raw_counts)) > next(iter(symmetry_counts))
                ),
                "requires_explicit_site_selection": (
                    next(iter(unique_product_counts)) > 1
                ),
            }
        )

    raw_fail = len(rows) - raw_pass
    qualified_fail = len(rows) - qualified_pass
    heads_raw_fail = sum(
        not record["passes_raw_amine_multiplicity"] for record in head_audit
    )
    heads_qualified_fail = sum(
        not record["passes_qualified_amine_multiplicity"] for record in head_audit
    )
    heads_equivalent_collapse = sum(
        record["equivalent_site_collapse"] for record in head_audit
    )
    rows_equivalent_collapse = sum(
        record["raw_required_handle_matches"]
        > record["symmetry_distinct_required_handle_matches"]
        for record in row_audit
    )
    heads_multiple_products = sum(
        record["requires_explicit_site_selection"] for record in head_audit
    )
    rows_multiple_products = sum(
        record["unique_forward_products"] > 1 for record in row_audit
    )
    count_checks = {
        "products_reconstructed_exactly": exact_reconstructed,
        "products_passing_raw_amine_multiplicity": raw_pass,
        "products_failing_raw_amine_multiplicity": raw_fail,
        "products_passing_qualified_amine_multiplicity": qualified_pass,
        "products_failing_qualified_amine_multiplicity": qualified_fail,
        "heads_failing_raw_amine_multiplicity": heads_raw_fail,
        "heads_failing_qualified_amine_multiplicity": heads_qualified_fail,
        "heads_with_equivalent_site_collapse": heads_equivalent_collapse,
        "products_with_equivalent_site_collapse": rows_equivalent_collapse,
        "heads_with_multiple_unique_forward_products": heads_multiple_products,
        "products_with_multiple_unique_forward_products": rows_multiple_products,
    }
    for name, observed in count_checks.items():
        _expect_count(observed, expected[name], label=name.replace("_", " "))
    if dict(sorted(forward_profile.items())) != config[
        "expected_forward_outcome_profile"
    ]:
        raise Ugi3AssemblyQualificationError(
            "forward outcome profile does not match the frozen expectation"
        )

    head_by_label = {record["component_label"]: record for record in head_audit}
    for head_label, finding in config["required_head_findings"].items():
        observed = head_by_label.get(head_label)
        if observed is None:
            raise Ugi3AssemblyQualificationError(
                f"required head finding {head_label} is missing"
            )
        for field, expected_value in finding.items():
            observed_field = (
                "raw_forward_outcomes_per_record"
                if field == "raw_forward_outcomes_per_record"
                else field
            )
            if observed.get(observed_field) != expected_value:
                raise Ugi3AssemblyQualificationError(
                    f"{head_label}.{field} mismatch: expected {expected_value!r}, "
                    f"observed {observed.get(observed_field)!r}"
                )

    failed_rows = [
        record["label"]
        for record in row_audit
        if not record["passes_qualified_amine_multiplicity"]
        or not record["expected_product_reconstructed"]
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": generated_utc,
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": _portable_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                config_path,
                agile_measured_library_path,
                qualified_reactions_path,
                ugi_variant_path,
            )
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "measured_products": len(rows),
            "unique_product_labels": len(product_labels),
            "amine_heads": len(components["A"]),
            "aldehyde_components": len(components["B"]),
            "isocyanide_components": len(components["C"]),
            "complete_component_combinations": len(combinations),
            **count_checks,
        },
        "multiplicity_policy": {
            "reaction_id": reaction["reaction_id"],
            "role": amine_role["name"],
            "required_handle_smarts": amine_role["required_handle_smarts"],
            "registry_allowed_site_multiplicity": list(allowed_multiplicity),
            "semantics": config["scope"]["multiplicity_semantics"],
            "atom_equivalence_method": config["scope"]["atom_equivalence_method"],
            "raw_match_count_role": config["scope"]["raw_match_count_role"],
            "selectivity_policy": reaction["selectivity_policy"],
        },
        "forward_outcome_profile": dict(sorted(forward_profile.items())),
        "row_qualification_digest_sha256": _row_digest(row_audit),
        "failed_row_labels": failed_rows,
        "head_audit": head_audit,
        "qa_flags": [
            {
                "flag_id": "agile_a5_raw_multiplicity_conflict_resolved",
                "status": "qualified_by_symmetry_and_exact_forward_reconstruction",
                "component_label": "A5",
                "raw_required_handle_matches": head_by_label["A5"][
                    "raw_required_handle_matches"
                ],
                "symmetry_distinct_required_handle_matches": head_by_label["A5"][
                    "symmetry_distinct_required_handle_matches"
                ],
                "measured_products_reconstructed": head_by_label["A5"][
                    "products_reconstructed_exactly"
                ],
            },
            {
                "flag_id": "agile_multi_product_heads_require_site_selection",
                "status": "bounded_regioisomeric_ambiguity",
                "component_labels": [
                    record["component_label"]
                    for record in head_audit
                    if record["requires_explicit_site_selection"]
                ],
                "policy": (
                    "A generated or decomposed product must identify the reacting "
                    "amine site when a head yields multiple unique products."
                ),
            },
        ],
        "decision": {
            "raw_match_count_is_hard_gate": False,
            "qualified_site_multiplicity_is_hard_gate": True,
            "all_measured_products_pass": not failed_rows,
            "model_built": False,
            "conclusion": (
                "The registry multiplicity values are retained and interpreted as "
                "symmetry-distinct Ugi amine sites. This admits A5 without an "
                "identity exception and reconstructs all 1,200 measured products."
            ),
        },
    }


def write_ugi3_assembly_qualification(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Atomically write the validated Ugi assembly qualification artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
