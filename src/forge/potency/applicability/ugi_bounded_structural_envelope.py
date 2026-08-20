"""Label-free, lipid-native structural envelope for bounded Ugi biology mode.

The envelope is derived only from unique measured AGILE molecular graphs.  It
uses a small set of integer graph features and never reads biological labels,
pKa, formulated-particle measurements or generated structures while fitting.
The controller is identity outside its explicitly selected bounded mode.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.design.audit.ugi_tail_chemotype_audit import component_chemotype_metrics
from forge.potency.applicability import ugi_distributional_applicability as applicability

DERIVATION_CONFIG_SCHEMA = "phase1_ugi_bounded_structural_envelope_derivation_config.v1"
ENVELOPE_SCHEMA = "phase1_ugi_bounded_structural_envelope.v1"
CENSUS_CONFIG_SCHEMA = "phase1_ugi_bounded_structural_envelope_census_config.v1"
CENSUS_RESULT_SCHEMA = "phase1_ugi_bounded_structural_envelope_census.v1"
CENSUS_LEDGER_SCHEMA = "phase1_ugi_bounded_structural_envelope_census_ledger.v1"

VIEWS = ("product", "amine", "aldehyde", "isocyanide")
VIEW_FIELDS = {
    "product": "model_smiles",
    "amine": "A_smiles",
    "aldehyde": "B_smiles",
    "isocyanide": "C_smiles",
}
GENERATED_FIELDS = {
    "product": "product_smiles",
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}
FEATURES = (
    "heavy_atoms",
    "carbon_atoms",
    "nitrogen_atoms",
    "oxygen_atoms",
    "other_hetero_atoms",
    "carbon_subgraph_diameter",
    "carbon_branch_atoms",
    "adjacent_carbon_branch_edges",
    "carbon_unsaturation_count",
    "ring_count",
    "ester_or_ether_count",
)
MODE_OPEN = "open"
MODE_BOUNDED = "bounded_structural_envelope_v1"

DERIVATION_SCOPE = {
    "reference": "unique_measured_AGILE_graphs_by_view",
    "biological_labels_read": False,
    "generated_structures_read": False,
    "pka_used": False,
    "lnp_size_or_formulation_measurements_used": False,
    "quantile_method": "linear_type7",
    "fence": "outward_rounded_Tukey_1.5_IQR",
    "feature_values": "nonnegative_integer_graph_descriptors",
}
CENSUS_SCOPE = {
    "read_only": True,
    "valid_exact_l1_terminals_only": True,
    "oracle_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "generator_trajectories_advanced": False,
    "nonzero_guidance_execution": False,
    "candidate_selection": False,
    "fixed_head_diagnostic_only": True,
    "open_mode_unchanged": True,
}


class UgiBoundedStructuralEnvelopeError(RuntimeError):
    """Raised when the bounded structural-envelope contract changes."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiBoundedStructuralEnvelopeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiBoundedStructuralEnvelopeError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiBoundedStructuralEnvelopeError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiBoundedStructuralEnvelopeError(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiBoundedStructuralEnvelopeError(f"{label} hash changed")
    return path


def _quantile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise UgiBoundedStructuralEnvelopeError("quantile input is invalid")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def lipid_native_features(smiles: str) -> dict[str, int]:
    """Return the frozen compact graph-only feature vector."""

    raw = component_chemotype_metrics(smiles)
    output = {
        "heavy_atoms": raw["heavy_atoms"],
        "carbon_atoms": raw["carbon_atoms"],
        "nitrogen_atoms": raw["nitrogen_atoms"],
        "oxygen_atoms": raw["oxygen_atoms"],
        "other_hetero_atoms": (
            raw["sulfur_atoms"] + raw["phosphorus_atoms"] + raw["halogen_atoms"]
        ),
        "carbon_subgraph_diameter": raw["carbon_subgraph_diameter"],
        "carbon_branch_atoms": raw["carbon_branch_atoms"],
        "adjacent_carbon_branch_edges": raw["adjacent_carbon_branch_edges"],
        "carbon_unsaturation_count": (
            raw["carbon_carbon_double_bonds"] + raw["carbon_carbon_triple_bonds"]
        ),
        "ring_count": raw["ring_count"],
        "ester_or_ether_count": (raw["ester_like_carbonyl_count"] + raw["ether_oxygen_count"]),
    }
    if tuple(output) != FEATURES or any(
        not isinstance(value, int) or value < 0 for value in output.values()
    ):
        raise UgiBoundedStructuralEnvelopeError("lipid-native feature contract changed")
    return output


def _feature_fences(rows: Sequence[Mapping[str, int]]) -> dict[str, dict[str, float | int]]:
    output = {}
    for feature in FEATURES:
        values = [int(row[feature]) for row in rows]
        q25 = _quantile(values, 0.25)
        median = _quantile(values, 0.5)
        q75 = _quantile(values, 0.75)
        iqr = q75 - q25
        lower = max(0, math.floor(q25 - 1.5 * iqr))
        upper = math.ceil(q75 + 1.5 * iqr)
        output[feature] = {
            "minimum_inclusive": lower,
            "maximum_inclusive": upper,
            "q25": q25,
            "median": median,
            "q75": q75,
            "iqr": iqr,
            "observed_minimum": min(values),
            "observed_maximum": max(values),
        }
    return output


def _inside(features: Mapping[str, int], bounds: Mapping[str, Any]) -> tuple[str, ...]:
    violations = []
    for feature in FEATURES:
        record = bounds.get(feature)
        if not isinstance(record, Mapping):
            raise UgiBoundedStructuralEnvelopeError("envelope feature bound is missing")
        value = int(features[feature])
        if not int(record["minimum_inclusive"]) <= value <= int(record["maximum_inclusive"]):
            violations.append(feature)
    return tuple(violations)


@dataclass(frozen=True)
class EnvelopeAssessment:
    """Terminal-only structural-envelope assessment."""

    mode: str
    action: str
    inside: bool
    active: bool
    incremental_potential: float
    identity_multiplier: float
    violations_by_view: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "action": self.action,
            "inside": self.inside,
            "active": self.active,
            "incremental_potential": self.incremental_potential,
            "identity_multiplier": self.identity_multiplier,
            "violations_by_view": {
                view: list(violations) for view, violations in self.violations_by_view
            },
        }


class BoundedStructuralEnvelopeController:
    """Mode-scoped gate; identity outside bounded mode."""

    def __init__(self, envelope: Mapping[str, Any]) -> None:
        if envelope.get("schema_version") != ENVELOPE_SCHEMA:
            raise UgiBoundedStructuralEnvelopeError("unsupported envelope schema")
        content = {key: value for key, value in envelope.items() if key != "result_sha256"}
        if envelope.get("result_sha256") != _sha256_payload(content):
            raise UgiBoundedStructuralEnvelopeError("envelope identity changed")
        bounds = envelope.get("bounds_by_view")
        if not isinstance(bounds, Mapping) or set(bounds) != set(VIEWS):
            raise UgiBoundedStructuralEnvelopeError("envelope views changed")
        self.bounds = bounds

    def assess_terminal(
        self,
        *,
        mode: Literal["open", "bounded_structural_envelope_v1"],
        valid_terminal: bool,
        exact_l1: bool,
        structures: Mapping[str, str] | None,
    ) -> EnvelopeAssessment:
        if mode == MODE_OPEN:
            return EnvelopeAssessment(
                mode=mode,
                action="identity_noop",
                inside=True,
                active=False,
                incremental_potential=0.0,
                identity_multiplier=1.0,
                violations_by_view=tuple((view, ()) for view in VIEWS),
            )
        if mode != MODE_BOUNDED:
            raise UgiBoundedStructuralEnvelopeError("unknown biology mode")
        if not valid_terminal or not exact_l1 or structures is None:
            return EnvelopeAssessment(
                mode=mode,
                action="abstain_invalid_or_nonexact_terminal",
                inside=False,
                active=False,
                incremental_potential=0.0,
                identity_multiplier=1.0,
                violations_by_view=tuple((view, ()) for view in VIEWS),
            )
        if set(structures) != set(VIEWS):
            raise UgiBoundedStructuralEnvelopeError("terminal structure views changed")
        violations = tuple(
            (view, _inside(lipid_native_features(structures[view]), self.bounds[view]))
            for view in VIEWS
        )
        inside = not any(items for _, items in violations)
        return EnvelopeAssessment(
            mode=mode,
            action="inside_structural_envelope" if inside else "abstain_outside_envelope",
            inside=inside,
            active=inside,
            incremental_potential=0.0,
            identity_multiplier=1.0,
            violations_by_view=violations,
        )


def build_bounded_structural_envelope(repo: Path, config_path: Path) -> dict[str, Any]:
    """Freeze the envelope without reading generated structures or outcomes."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="structural-envelope derivation config")
    if config.get("schema_version") != DERIVATION_CONFIG_SCHEMA:
        raise UgiBoundedStructuralEnvelopeError("unsupported derivation config")
    if config.get("scope") != DERIVATION_SCOPE or tuple(config.get("features", ())) != FEATURES:
        raise UgiBoundedStructuralEnvelopeError("derivation policy changed")
    inputs = config.get("inputs")
    expected = {"curated_agile", "feature_source", "runner", "source", "tests"}
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise UgiBoundedStructuralEnvelopeError("derivation input set changed")
    paths = {name: _pin(repo, record, label=name) for name, record in inputs.items()}
    try:
        with gzip.open(paths["curated_agile"], "rt", newline="") as handle:
            measured = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise UgiBoundedStructuralEnvelopeError("cannot read measured AGILE structures") from error
    if not measured:
        raise UgiBoundedStructuralEnvelopeError("measured AGILE structure table is empty")
    unique_by_view: dict[str, list[str]] = {}
    bounds = {}
    for view, field in VIEW_FIELDS.items():
        unique = sorted({applicability._canonical(row[field]) for row in measured})
        unique_by_view[view] = unique
        bounds[view] = _feature_fences([lipid_native_features(smiles) for smiles in unique])
    reference_pass_by_view = {
        view: sum(
            not _inside(lipid_native_features(smiles), bounds[view])
            for smiles in unique_by_view[view]
        )
        for view in VIEWS
    }
    row_pass = 0
    for row in measured:
        if all(
            not _inside(lipid_native_features(row[field]), bounds[view])
            for view, field in VIEW_FIELDS.items()
        ):
            row_pass += 1
    content = {
        "schema_version": ENVELOPE_SCHEMA,
        "status": "frozen_measured_only_label_free_structural_envelope",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "scope": DERIVATION_SCOPE,
        "features": list(FEATURES),
        "reference": {
            "measured_rows": len(measured),
            "unique_graphs_by_view": {view: len(values) for view, values in unique_by_view.items()},
            "unique_graphs_inside_by_view": reference_pass_by_view,
            "measured_rows_inside_all_views": row_pass,
        },
        "bounds_by_view": bounds,
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "adjudication": {
            "bounded_mode_only": True,
            "open_mode_changed": False,
            "nonzero_guidance_authorized": False,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise UgiBoundedStructuralEnvelopeError(f"cannot read ledger: {path}") from error


def _bounded_census_action(
    unseen: tuple[str, ...],
    *,
    envelope_inside: bool,
    current_policy_action: str,
) -> str:
    if unseen == ("amine", "aldehyde", "isocyanide"):
        return "abstain_all_three_new_unsupported"
    if not envelope_inside:
        return "abstain_outside_structural_envelope"
    if current_policy_action == "applicability_supported_no_score":
        return "active_contrast_supported_no_score"
    if current_policy_action == "neutral":
        return "neutral_exact_measured_combination"
    return "abstain_current_v3_or_role_policy"


def build_bounded_structural_envelope_census(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Apply the frozen envelope once to valid exact-L1 generated terminals."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="bounded-envelope census config")
    if config.get("schema_version") != CENSUS_CONFIG_SCHEMA:
        raise UgiBoundedStructuralEnvelopeError("unsupported census config")
    if config.get("scope") != CENSUS_SCOPE:
        raise UgiBoundedStructuralEnvelopeError("census scope changed")
    inputs = config.get("inputs")
    expected = {
        "applicability_ledger",
        "envelope",
        "fresh_pool_sample",
        "generated_role_census_ledger",
        "generated_role_census_result",
        "runner",
        "source",
        "tests",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise UgiBoundedStructuralEnvelopeError("census input set changed")
    paths = {name: _pin(repo, record, label=name) for name, record in inputs.items()}
    envelope = _load_json(paths["envelope"], label="frozen structural envelope")
    controller = BoundedStructuralEnvelopeController(envelope)
    prior = _load_json(paths["generated_role_census_result"], label="generated-role census")
    if prior.get("scope", {}).get("nonzero_guidance_execution") is not False:
        raise UgiBoundedStructuralEnvelopeError("prior census scope changed")
    applicability_rows = _csv_rows(paths["applicability_ledger"])
    prior_rows = _csv_rows(paths["generated_role_census_ledger"])
    prior_by_index = {int(row["sample_index"]): row for row in prior_rows}
    sample = _load_json(paths["fresh_pool_sample"], label="fresh terminal sample")
    samples = sample.get("samples")
    if not isinstance(samples, list):
        raise UgiBoundedStructuralEnvelopeError("fresh sample records changed")
    open_assessment = controller.assess_terminal(
        mode=MODE_OPEN,
        valid_terminal=False,
        exact_l1=False,
        structures=None,
    )
    if open_assessment.action != "identity_noop" or open_assessment.identity_multiplier != 1.0:
        raise UgiBoundedStructuralEnvelopeError("open mode is not identity")

    fields = [
        "sample_index",
        "product_id",
        "exact_unseen_roles_json",
        "v3_overall_distribution_bin",
        "v3_current_policy_action",
        "envelope_inside",
        "envelope_violations_json",
        "bounded_census_action",
    ]
    output = []
    actions: Counter[str] = Counter()
    envelope_by_v3: defaultdict[str, Counter[str]] = defaultdict(Counter)
    all_three_by_envelope: Counter[str] = Counter()
    view_violations: Counter[str] = Counter()
    for row in applicability_rows:
        index = int(row["sample_index"])
        if index >= len(samples) or index not in prior_by_index:
            raise UgiBoundedStructuralEnvelopeError("terminal index alignment changed")
        terminal = samples[index]
        forward = terminal.get("l1_forward_verification")
        components = terminal.get("component_smiles_by_role")
        valid = terminal.get("valid") is True
        exact = isinstance(forward, Mapping) and forward.get("exact_product_reconstructed") is True
        if not valid or not exact or not isinstance(components, Mapping):
            raise UgiBoundedStructuralEnvelopeError("census input includes nonexact terminal")
        structures = {
            "product": row["product_smiles"],
            "amine": row["amine_smiles"],
            "aldehyde": row["aldehyde_smiles"],
            "isocyanide": row["isocyanide_smiles"],
        }
        assessment = controller.assess_terminal(
            mode=MODE_BOUNDED,
            valid_terminal=True,
            exact_l1=True,
            structures=structures,
        )
        prior_row = prior_by_index[index]
        unseen = tuple(json.loads(row["exact_unseen_roles_json"]))
        current_action = prior_row["current_policy_action"]
        action = _bounded_census_action(
            unseen,
            envelope_inside=assessment.inside,
            current_policy_action=current_action,
        )
        violations = assessment.to_dict()["violations_by_view"]
        for view, names in violations.items():
            if names:
                view_violations[view] += 1
        overall = row["overall_distribution_bin"]
        envelope_by_v3[overall]["inside" if assessment.inside else "outside"] += 1
        if unseen == ("amine", "aldehyde", "isocyanide"):
            all_three_by_envelope["inside" if assessment.inside else "outside"] += 1
        actions[action] += 1
        output.append(
            {
                "sample_index": index,
                "product_id": row["product_id"],
                "exact_unseen_roles_json": row["exact_unseen_roles_json"],
                "v3_overall_distribution_bin": overall,
                "v3_current_policy_action": current_action,
                "envelope_inside": assessment.inside,
                "envelope_violations_json": json.dumps(
                    violations, sort_keys=True, separators=(",", ":")
                ),
                "bounded_census_action": action,
            }
        )
    active = actions["active_contrast_supported_no_score"]
    ledger = _csv_bytes(output, fields)
    content = {
        "schema_version": CENSUS_RESULT_SCHEMA,
        "status": "complete_nonselecting_bounded_envelope_census_nonzero_blocked",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "scope": CENSUS_SCOPE,
        "open_mode_identity_check": open_assessment.to_dict(),
        "terminal_records": len(output),
        "envelope_inside": sum(str(row["envelope_inside"]).lower() == "true" for row in output),
        "envelope_outside": sum(str(row["envelope_inside"]).lower() != "true" for row in output),
        "actions": dict(sorted(actions.items())),
        "envelope_by_v3_distribution_bin": {
            name: dict(sorted(counts.items())) for name, counts in sorted(envelope_by_v3.items())
        },
        "all_three_new_by_envelope": dict(sorted(all_three_by_envelope.items())),
        "view_violation_counts": dict(sorted(view_violations.items())),
        "artifacts": {
            "ledger.csv.gz": {
                "schema_version": CENSUS_LEDGER_SCHEMA,
                "records": len(output),
                "sha256": sha256_bytes(ledger),
            }
        },
        "adjudication": {
            "nontrivial_active_contrast": active > 0,
            "active_contrast_records": active,
            "nonzero_guidance_authorized": False,
            "lambda_zero_identity_required_before_nonzero": True,
            "all_three_new_authorized": False,
            "fixed_head_primary_path": False,
            "next_gate": (
                "review census and, only if active contrast is nontrivial, freeze a separate "
                "bounded-mode lambda-zero identity experiment"
            ),
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}, ledger


__all__ = [
    "BoundedStructuralEnvelopeController",
    "EnvelopeAssessment",
    "MODE_BOUNDED",
    "MODE_OPEN",
    "UgiBoundedStructuralEnvelopeError",
    "_bounded_census_action",
    "build_bounded_structural_envelope",
    "build_bounded_structural_envelope_census",
    "lipid_native_features",
]
