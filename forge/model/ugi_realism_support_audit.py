"""Attribute measured Ugi train coverage under the existing decoder program policy.

This is an empirical reference control. Admission of a coarse program is a necessary
condition in the current sampler, not a proof that its particular molecular graph can
be generated, and a distance between empirical subsets is not an attainable bound.
"""

from __future__ import annotations

import math
import platform
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import rdBase

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_json
from forge.core.io import iter_csv, read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.common_lipid_realism import (
    RealismPolicy,
    UgiDevelopmentReference,
    build_ugi_development_realism_reference,
    fit_robust_descriptor_scale,
)
from forge.model.ugi_development_realism import (
    ROLE_DESCRIPTOR_NAMES,
    _continuous_summary,
    ugi_role_descriptor_vector,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_measured_joint_program_prior import (
    program_admitted_by_ester_policy,
    program_from_layout_record,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA = "forge.ugi_realism_support_audit_config.v1"
RESULT_SCHEMA = "forge.ugi_realism_support_audit.v1"


class UgiRealismSupportAuditError(ValueError):
    """A support audit input or conservation invariant failed."""


def select_measured_training_rows(
    rows: Iterable[Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Mask fold and provenance before touching any structure or component field."""

    selected: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        counts["assignment_rows"] += 1
        if row["primary_product_fold"] != "train":
            counts["non_train_rows_masked_before_structure_access"] += 1
            continue
        if str(row["is_source_adjudicated_measured_product"]).strip().lower() not in {
            "1",
            "true",
        }:
            counts["non_measured_train_rows_masked_before_structure_access"] += 1
            continue
        product_id = str(row["product_id"]).strip()
        canonical = str(row["canonical_product_smiles"]).strip()
        components = {role: str(row[f"{role}_smiles"]).strip() for role in ROLE_NAMES}
        families = {role: str(row[f"{role}_family_id"]).strip() for role in ROLE_NAMES}
        if not all((product_id, canonical, *components.values(), *families.values())):
            raise UgiRealismSupportAuditError("measured train row has missing identity or roles")
        if product_id in selected:
            raise UgiRealismSupportAuditError(f"duplicate measured train product: {product_id}")
        selected[product_id] = {
            "product_id": product_id,
            "canonical_product_smiles": canonical,
            "components_by_role": components,
            "families_by_role": families,
            "group_id": "|".join(families[role] for role in ROLE_NAMES),
        }
        counts["measured_training_rows"] += 1
    if not selected:
        raise UgiRealismSupportAuditError("no source-adjudicated measured train rows")
    if len({row["canonical_product_smiles"] for row in selected.values()}) != len(selected):
        raise UgiRealismSupportAuditError("duplicate measured train molecular constitution")
    return dict(sorted(selected.items())), dict(sorted(counts.items()))


def program_exclusion_reasons(
    program: UgiMorphologyProgram, policy: UgiEsterChemotypePolicy
) -> list[str]:
    """Attribute count restrictions while keeping the existing admission function authoritative.

    The topology fallback calls the existing gate with count checks bypassed by a
    delegating view. It does not restate its chemistry-specific topology constants.
    """

    reasons: list[str] = []
    for index, role in enumerate(ROLE_NAMES):
        count = program.node_counts[index]
        if count < policy.minimum_topology_exterior_atoms(role):
            reasons.append(f"{role}.below_minimum_exterior_atoms")
        if count > policy.maximum_exterior_atoms(role):
            reasons.append(f"{role}.above_maximum_exterior_atoms")

    class CountNeutralPolicy:
        def __getattr__(self, name: str) -> Any:
            return getattr(policy, name)

        def minimum_topology_exterior_atoms(self, role: str) -> int:
            return program.node_counts[ROLE_NAMES.index(role)]

        def maximum_exterior_atoms(self, role: str) -> int:
            return program.node_counts[ROLE_NAMES.index(role)]

    if not program_admitted_by_ester_policy(program, CountNeutralPolicy()):  # type: ignore[arg-type]
        reasons.append("existing_decoder_non_count_topology_restriction")
    admitted = program_admitted_by_ester_policy(program, policy)
    if admitted != (not reasons):
        raise UgiRealismSupportAuditError("reason attribution disagrees with existing decoder gate")
    return sorted(reasons)


def summarize_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report overlapping reasons plus a mutually exclusive partition conserving every row."""

    if not rows or len({row["product_id"] for row in rows}) != len(rows):
        raise UgiRealismSupportAuditError("support ledger is empty or contains duplicate products")
    reason_counts: Counter[str] = Counter()
    reason_patterns: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    component_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        role: defaultdict(list) for role in ROLE_NAMES
    }
    family_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        role: defaultdict(list) for role in ROLE_NAMES
    }
    for row in rows:
        reasons = list(row["exclusion_reasons"])
        if bool(row["program_admitted"]) != (not reasons) or len(reasons) != len(set(reasons)):
            raise UgiRealismSupportAuditError(
                "inconsistent admission or repeated exclusion reasons"
            )
        category = "admitted" if not reasons else "excluded"
        reason_counts.update(reasons)
        reason_patterns[" & ".join(sorted(reasons)) if reasons else "admitted"] += 1
        groups[str(row["group_id"])][category] += 1
        for role in ROLE_NAMES:
            component_rows[role][row["components_by_role"][role]].append(row)
            family_rows[role][row["families_by_role"][role]].append(row)

    def subset_summary(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        admitted = sum(bool(row["program_admitted"]) for row in subset)
        return {
            "measured_products": len(subset),
            "admitted_products": admitted,
            "excluded_products": len(subset) - admitted,
            "exclusion_reasons_overlapping": dict(
                sorted(
                    Counter(reason for row in subset for reason in row["exclusion_reasons"]).items()
                )
            ),
        }

    unique_components = {}
    unique_families = {}
    for role in ROLE_NAMES:
        for source, target in ((component_rows, unique_components), (family_rows, unique_families)):
            entries = {
                identity: subset_summary(subset)
                for identity, subset in sorted(source[role].items())
            }
            target[role] = {
                "observed": len(entries),
                "with_any_admitted_product": sum(
                    v["admitted_products"] > 0 for v in entries.values()
                ),
                "entirely_excluded": sum(v["admitted_products"] == 0 for v in entries.values()),
                "partially_excluded": sum(
                    v["admitted_products"] > 0 and v["excluded_products"] > 0
                    for v in entries.values()
                ),
                "entries": entries,
            }
    if sum(reason_patterns.values()) != len(rows):
        raise UgiRealismSupportAuditError(
            "mutually exclusive reason counts do not conserve products"
        )
    admitted = sum(bool(row["program_admitted"]) for row in rows)
    return {
        "measured_training_products": len(rows),
        "admitted_products": admitted,
        "excluded_products": len(rows) - admitted,
        "admitted_product_fraction": admitted / len(rows),
        "exclusion_reasons_overlapping": dict(sorted(reason_counts.items())),
        "mutually_exclusive_reason_patterns": dict(sorted(reason_patterns.items())),
        "reason_pattern_conservation": True,
        "unique_components_by_role": unique_components,
        "unique_families_by_role": unique_families,
        "family_triples": {
            "observed": len(groups),
            "with_any_admitted_product": sum(group["admitted"] > 0 for group in groups.values()),
            "entirely_excluded": sum(group["admitted"] == 0 for group in groups.values()),
            "entries": {
                group: dict(sorted(counts.items())) for group, counts in sorted(groups.items())
            },
        },
    }


def _equal_group_matrix(
    rows: Sequence[Mapping[str, Any]], matrix: np.ndarray, *, maximum_rows: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Represent exact equal group/product mass by bounded integer replication, without RNG."""

    groups = Counter(str(row["group_id"]) for row in rows)
    mass_per_group = math.lcm(*groups.values())
    expanded_rows = mass_per_group * len(groups)
    if expanded_rows > maximum_rows:
        raise UgiRealismSupportAuditError(
            f"exact group-balanced replication needs {expanded_rows} rows, above {maximum_rows}"
        )
    repeats = [mass_per_group // groups[str(row["group_id"])] for row in rows]
    return np.repeat(matrix, repeats, axis=0), {
        "groups": len(groups),
        "unique_products": len(rows),
        "deterministic_replicated_rows": expanded_rows,
        "interpretation": "equal family-triple mass then equal measured-product mass within group",
        "independent_observations_added": 0,
    }


def assess_support_reference_controls(
    rows: Sequence[Mapping[str, Any]],
    reference: UgiDevelopmentReference,
    policy: RealismPolicy,
    *,
    maximum_replicated_rows: int,
) -> dict[str, Any]:
    """Retain full reference metrics and add explicitly conditional, train-only subset views."""

    admitted = [row for row in rows if row["program_admitted"]]
    if not admitted:
        raise UgiRealismSupportAuditError("no admitted measured products for reference control")
    admitted_ids = {row["product_id"] for row in admitted}
    scaling_raw = np.asarray(
        [ugi_role_descriptor_vector(row.components()) for row in reference.scaling]
    )
    scale = fit_robust_descriptor_scale(scaling_raw)
    scaling = scale.transform(scaling_raw)
    evaluation = scale.transform(
        np.asarray([ugi_role_descriptor_vector(row.components()) for row in reference.evaluation])
    )
    observed = scale.transform(
        np.asarray([ugi_role_descriptor_vector(row["components_by_role"]) for row in rows])
    )
    admitted_matrix = observed[[bool(row["program_admitted"]) for row in rows]]
    balanced_admitted, balanced_audit = _equal_group_matrix(
        admitted, admitted_matrix, maximum_rows=maximum_replicated_rows
    )
    balanced_observed, balanced_observed_audit = _equal_group_matrix(
        rows, observed, maximum_rows=maximum_replicated_rows
    )
    evaluation_mask = np.asarray([row.structure_id in admitted_ids for row in reference.evaluation])
    conditional_evaluation = evaluation[evaluation_mask]

    def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        return _continuous_summary(left, right, chunk_size=policy.distance_chunk_size)

    full_views = {
        "full_measured_scaling_vs_full_evaluation": metrics(scaling, evaluation),
        "full_measured_product_uniform_vs_full_evaluation": metrics(observed, evaluation),
        "full_measured_group_balanced_vs_full_evaluation": metrics(balanced_observed, evaluation),
        "admitted_measured_product_uniform_vs_full_evaluation": metrics(
            admitted_matrix, evaluation
        ),
        "admitted_measured_group_balanced_vs_full_evaluation": metrics(
            balanced_admitted, evaluation
        ),
    }
    conditional_views = (
        {
            "status": "measured",
            "admitted_measured_product_uniform_vs_admitted_evaluation": metrics(
                admitted_matrix, conditional_evaluation
            ),
            "admitted_measured_group_balanced_vs_admitted_evaluation": metrics(
                balanced_admitted, conditional_evaluation
            ),
        }
        if len(conditional_evaluation)
        else {"status": "abstained_empty_admitted_evaluation_subset"}
    )
    return {
        "features": list(ROLE_DESCRIPTOR_NAMES),
        "full_reference_diagnostic": full_views,
        "supplemental_admitted_reference_diagnostic": conditional_views,
        "full_reference": reference.realism.audit,
        "counts": {
            "full_measured_products": len(rows),
            "admitted_measured_products": len(admitted),
            "full_scaling_rows": len(scaling),
            "full_evaluation_rows": len(evaluation),
            "admitted_evaluation_rows": len(conditional_evaluation),
            "admitted_products_overlapping_full_evaluation": int(evaluation_mask.sum()),
        },
        "admitted_population_weighting": balanced_audit,
        "full_population_weighting": balanced_observed_audit,
        "scale": {
            "fit_population": "unchanged full measured train-development scaling partition",
            "center": scale.center.tolist(),
            "scale": scale.scale.tolist(),
        },
        "conditional_reference_rebalanced": False,
        "frozen_reference_or_gate_changed": False,
        "possible_generator_best_case_bound": False,
        "interpretation": (
            "Empirical reference controls with disclosed train-set overlap. Coarse-program admission "
            "does not certify complete graph attainability. The admitted evaluation subset preserves "
            "the original full-reference scale but is not necessarily group-balanced; it is a "
            "supplemental diagnostic and never replaces the original full-reference assessment."
        ),
    }


def run_ugi_realism_support_audit(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Execute one pinned local coverage audit without training or molecular generation."""

    repo = repo_root.resolve()
    config_path = (
        (repo / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    )
    output = (repo / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    if output.exists():
        raise UgiRealismSupportAuditError(f"output already exists: {output}")
    config = read_json_object(
        config_path, error=UgiRealismSupportAuditError, label="support audit config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "sources",
        "seed",
        "policy",
        "nonclaims",
    }:
        raise UgiRealismSupportAuditError("support audit config schema changed")
    if (
        isinstance(config["seed"], bool)
        or not isinstance(config["seed"], int)
        or config["seed"] < 0
    ):
        raise UgiRealismSupportAuditError("seed must be a non-negative integer")
    expected_policy = {
        "fold": "train",
        "measured_only": True,
        "admission": "existing_program_admitted_by_ester_policy",
        "full_reference_preserved": True,
        "maximum_replicated_rows": 10000,
        "morphology_quantile": 0.25,
        "training_generation_route_or_oracle_calls": 0,
    }
    if config["policy"] != expected_policy:
        raise UgiRealismSupportAuditError("support audit policy changed")
    if not isinstance(config["inputs"], Mapping) or set(config["inputs"]) != {
        "production_cache",
        "ugi_assignments",
        "qualified_reactions",
        "program_draw",
        "development_realism_config",
    }:
        raise UgiRealismSupportAuditError("support audit input contract changed")
    if not isinstance(config["sources"], Mapping) or not config["sources"]:
        raise UgiRealismSupportAuditError("support audit requires implementation source pins")
    inputs = {
        name: resolve_pin(pin, repo, label=name) for name, pin in sorted(config["inputs"].items())
    }
    sources = {
        name: resolve_pin(pin, repo, label=name) for name, pin in sorted(config["sources"].items())
    }
    draw = read_json_object(
        inputs["program_draw"], error=UgiRealismSupportAuditError, label="program draw"
    )
    for label in ("production_cache", "ugi_assignments", "qualified_reactions"):
        if draw["inputs"][label]["sha256"] != config["inputs"][label]["sha256"]:
            raise UgiRealismSupportAuditError(f"historical program draw {label} input differs")
    development = read_json_object(
        inputs["development_realism_config"],
        error=UgiRealismSupportAuditError,
        label="development reference config",
    )
    if (
        development.get("schema_version") != "forge.ugi_development_lipid_realism_config.v1"
        or development["reference"]["kind"]
        != "source_adjudicated_measured_ugi_train_group_balanced"
        or development["inputs"]["ugi_assignments"] != config["inputs"]["ugi_assignments"]
        or development["inputs"]["qualified_ugi_reactions"]
        != config["inputs"]["qualified_reactions"]
    ):
        raise UgiRealismSupportAuditError(
            "development reference contract differs from audit inputs"
        )
    selected, selection_counts = select_measured_training_rows(iter_csv(inputs["ugi_assignments"]))
    ester_policy = UgiEsterChemotypePolicy.from_qualified_registry(
        inputs["qualified_reactions"],
        training_assignments_path=inputs["ugi_assignments"],
        expected_sha256=config["inputs"]["qualified_reactions"]["sha256"],
        expected_training_assignments_sha256=config["inputs"]["ugi_assignments"]["sha256"],
        reaction_id=draw["reaction_id"],
        morphology_quantile=config["policy"]["morphology_quantile"],
    )
    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    ledger = []
    observed_ids: set[str] = set()
    try:
        for raw_index in cache.indices(program_id=draw["reaction_id"], fold="train"):
            index = int(raw_index)
            product_id = cache.record_id(index)
            if product_id not in selected:
                continue
            if product_id in observed_ids:
                raise UgiRealismSupportAuditError(f"duplicate cache product: {product_id}")
            observed_ids.add(product_id)
            program = program_from_layout_record(cache.record(index), vocabulary=cache.vocabulary)
            reasons = program_exclusion_reasons(program, ester_policy)
            ledger.append(
                {
                    **selected[product_id],
                    "program": asdict(program),
                    "program_admitted": not reasons,
                    "exclusion_reasons": reasons,
                }
            )
    finally:
        cache.close()
    if observed_ids != set(selected):
        raise UgiRealismSupportAuditError(
            f"cache lacks {len(set(selected) - observed_ids)} measured train products"
        )
    ledger.sort(key=lambda row: row["product_id"])
    summary = summarize_support(ledger)
    historical = draw["prior_audit"]
    agreement = {
        "measured_training_rows": summary["measured_training_products"]
        == historical["measured_training_rows"],
        "eligible_rows": summary["admitted_products"] == historical["eligible_rows"],
        "excluded_rows": summary["excluded_products"]
        == historical["excluded_by_existing_decoder_program_support"],
    }
    if not all(agreement.values()):
        raise UgiRealismSupportAuditError(
            f"current gate disagrees with pinned draw census: {agreement}"
        )
    policy = RealismPolicy.from_mapping(development["policy"])
    reference = build_ugi_development_realism_reference(
        inputs["ugi_assignments"],
        policy,
        rows_per_group_per_partition=development["reference"]["rows_per_group_per_partition"],
        split_seed=development["reference"]["split_seed"],
    )
    controls = assess_support_reference_controls(
        ledger,
        reference,
        policy,
        maximum_replicated_rows=config["policy"]["maximum_replicated_rows"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "measured",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": config["scientific_question"],
        "seed": config["seed"],
        "random_sampling_used": False,
        "reference_split_seed": development["reference"]["split_seed"],
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in inputs.items()},
        "sources": {
            **{label: pin_record(path, repo) for label, path in sources.items()},
            "audit_implementation": pin_record(Path(__file__), repo),
            "audit_cli": pin_record(
                repo / "experiments/phase1/multireaction/ugi_realism_support_audit.py", repo
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "policy": config["policy"],
        "ester_policy": ester_policy.to_mapping(),
        "selection_counts": selection_counts,
        "historical_draw_census_agreement": agreement,
        "support": summary,
        "reference_controls": controls,
        "ledger_sha256": str(sha256_json(ledger)),
        "calibration_or_heldout_structures_accessed": False,
        "training_generation_route_or_oracle_calls": 0,
        "nonclaims": config["nonclaims"],
    }
    # Build complete artifacts in a sibling staging directory; expose a final directory only
    # after all calculations and writes succeed. An exception never publishes partial metrics.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.partial-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        ledger_path = staging / "measured_train_ledger.json"
        write_json(
            ledger_path, {"schema_version": "forge.ugi_realism_support_ledger.v1", "rows": ledger}
        )
        result["artifacts"] = {
            "measured_train_ledger": artifact_record(
                ledger_path, logical_path="measured_train_ledger.json"
            )
        }
        write_json(staging / "result.json", result)
        staging.rename(output)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiRealismSupportAuditError",
    "select_measured_training_rows",
    "program_exclusion_reasons",
    "summarize_support",
    "assess_support_reference_controls",
    "run_ugi_realism_support_audit",
]
