"""Qualify bounded Ugi decoder and retro-assessment interventions on frozen evidence."""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_failure_attribution import (
    _COMMON_MEMBER,
    _SAMPLING_MEMBER,
    _json_member,
    _jsonl_member,
    _native_forward_exact,
    unsupported_edge_signatures,
)
from forge.assembly import Ugi3AssemblyAdapter, Ugi3TransformConsistentCandidate
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.local_chemistry_support import LocalChemistrySupport

CONFIG_SCHEMA = "forge.ugi_tree_transformer_intervention_readiness_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_intervention_readiness.v1"
RETRO_ATTEMPT_SCHEMA = "forge.ugi_transform_consistency_diagnostic_attempt.v1"
EXPECTED_SEEDS = (20260905, 20260906, 20260907)
ATTEMPTS_PER_SEED = 3072
EXPECTED_RETRO_ABSTENTIONS = 190
EXPECTED_LOCAL_UNSUPPORTED = 677


class UgiTreeTransformerInterventionReadinessError(ValueError):
    """Frozen evidence cannot qualify the proposed bounded intervention."""


def _canonical_components(value: object, roles: Sequence[str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != set(roles):
        raise UgiTreeTransformerInterventionReadinessError(
            "native component trace roles changed"
        )
    output = []
    with rdBase.BlockLogs():
        for role in roles:
            molecule = Chem.MolFromSmiles(str(value[role]))
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise UgiTreeTransformerInterventionReadinessError(
                    f"native {role} component is invalid"
                )
            output.append(
                (
                    role,
                    Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
                )
            )
    return tuple(output)


def classify_transform_consistent_candidates(
    candidates: Sequence[Ugi3TransformConsistentCandidate],
    native_components: tuple[tuple[str, str], ...],
) -> tuple[str, Ugi3TransformConsistentCandidate | None]:
    """Classify a method-blind reverse result against a hidden native trace."""

    matching = [candidate for candidate in candidates if candidate.components == native_components]
    if not candidates:
        return "no_transform_consistent_candidate", None
    if not matching:
        return "transform_consistent_candidate_native_trace_not_recovered", None
    candidate = matching[0]
    if candidate.registry_handle_qualified:
        return "native_trace_registry_handle_qualified", candidate
    return "native_trace_transform_consistent_handle_rejected", candidate


def _config(config_path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = read_json_object(
        config_path,
        error=UgiTreeTransformerInterventionReadinessError,
        label="tree-Transformer intervention-readiness config",
    )
    required = {
        "schema_version",
        "status",
        "expected_training_seeds",
        "attempts_per_seed",
        "expected_retro_abstentions",
        "expected_local_unsupported_exact_l1",
        "inputs",
        "policy",
        "nonclaims",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != required:
        raise UgiTreeTransformerInterventionReadinessError("readiness config fields changed")
    if (
        config.get("status")
        != "frozen_after_failure_attribution_before_constrained_resampling"
        or config.get("expected_training_seeds") != list(EXPECTED_SEEDS)
        or config.get("attempts_per_seed") != ATTEMPTS_PER_SEED
        or config.get("expected_retro_abstentions") != EXPECTED_RETRO_ABSTENTIONS
        or config.get("expected_local_unsupported_exact_l1") != EXPECTED_LOCAL_UNSUPPORTED
    ):
        raise UgiTreeTransformerInterventionReadinessError("readiness contract changed")
    expected_policy = {
        "registry_handle_gate_unchanged": True,
        "transform_consistency_reported_separately_from_exact_l1": True,
        "native_component_traces_used_for_diagnostic_comparison_only": True,
        "role_local_policy_is_training_fold_only": True,
        "component_identities_in_decoder_policy": False,
        "production_thresholds_changed": False,
        "candidate_selection": False,
        "generation_calls": 0,
        "route_calls": 0,
        "oracle_calls": 0,
    }
    if config.get("policy") != expected_policy:
        raise UgiTreeTransformerInterventionReadinessError("readiness policy changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "failure_attribution",
        "attempt_attributions",
        "qualified_reactions",
        "role_local_policy",
        "evaluation_details",
    }:
        raise UgiTreeTransformerInterventionReadinessError("readiness inputs changed")
    paths = {
        label: resolve_pin(inputs[label], repo, label=label)
        for label in (
            "failure_attribution",
            "attempt_attributions",
            "qualified_reactions",
            "role_local_policy",
        )
    }
    return config, paths


def _attribution_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            values = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise UgiTreeTransformerInterventionReadinessError(
            "attempt-attribution ledger is invalid"
        ) from error
    if not values or values.pop(0) != {
        "schema_version": "forge.ugi_tree_transformer_failure_attribution_attempt.v1",
        "rows": len(values),
    }:
        raise UgiTreeTransformerInterventionReadinessError(
            "attempt-attribution ledger header changed"
        )
    return values


def diagnose_tree_transformer_interventions(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Close saved-attempt diagnostics before any new constrained sampling run."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise UgiTreeTransformerInterventionReadinessError(
            f"intervention-readiness output already exists: {output_dir}"
        )
    config, paths = _config(config_path, repo)
    failure = read_json_object(
        paths["failure_attribution"],
        error=UgiTreeTransformerInterventionReadinessError,
        label="failure attribution",
    )
    if (
        failure.get("status") != "complete_negative_result_attributed"
        or failure.get("attempts") != len(EXPECTED_SEEDS) * ATTEMPTS_PER_SEED
    ):
        raise UgiTreeTransformerInterventionReadinessError(
            "readiness requires the completed frozen failure attribution"
        )
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            paths["role_local_policy"],
            error=UgiTreeTransformerInterventionReadinessError,
            label="role-local chemistry policy",
        )
    )
    if not support.enforces_role_cycles:
        raise UgiTreeTransformerInterventionReadinessError(
            "role-local policy lacks complete training-fold cycle support"
        )
    adapter = Ugi3AssemblyAdapter.from_registry(paths["qualified_reactions"])

    raw_archives = config["inputs"]["evaluation_details"]
    if not isinstance(raw_archives, list) or len(raw_archives) != len(EXPECTED_SEEDS):
        raise UgiTreeTransformerInterventionReadinessError("evaluation archive set changed")
    retro_rows: list[dict[str, Any]] = []
    retro_classes: Counter[str] = Counter()
    handle_rejections: Counter[str] = Counter()
    archive_inputs: dict[str, Any] = {}
    for seed, raw in zip(EXPECTED_SEEDS, raw_archives, strict=True):
        if not isinstance(raw, Mapping) or raw.get("seed") != seed:
            raise UgiTreeTransformerInterventionReadinessError(
                "evaluation archive seed order changed"
            )
        archive_path = resolve_pin(
            {"path": raw.get("path"), "sha256": raw.get("sha256")},
            repo,
            label=f"evaluation details {seed}",
        )
        archive_inputs[str(seed)] = pin_record(archive_path, repo)
        with tarfile.open(archive_path, "r") as archive:
            sampling = _json_member(archive, _SAMPLING_MEMBER)
            samples = sampling.get("samples")
            common = _jsonl_member(
                archive,
                _COMMON_MEMBER,
                schema_version="forge.common_ugi_assessed_attempts.v1",
                expected_rows=ATTEMPTS_PER_SEED,
            )
        if not isinstance(samples, list) or len(samples) != ATTEMPTS_PER_SEED:
            raise UgiTreeTransformerInterventionReadinessError(
                f"sampling denominator changed for seed {seed}"
            )
        for attempt_index, (sample, assessed) in enumerate(
            zip(samples, common, strict=True)
        ):
            if not (
                sample.get("valid") is True
                and assessed.get("exact_l1_program") is not True
                and _native_forward_exact(sample)
            ):
                continue
            smiles = str(sample["smiles"])
            native = _canonical_components(sample.get("component_smiles_by_role"), adapter.roles)
            candidates = adapter.transform_consistent_decomposition_candidates(
                smiles, maximum_outcomes=512
            )
            classification, matched = classify_transform_consistent_candidates(
                candidates, native
            )
            retro_classes[classification] += 1
            rejected = []
            if matched is not None:
                for assessment in matched.handle_assessments:
                    if not assessment.passes_registry_handle_policy:
                        label = (
                            f"{assessment.role}:symmetry_distinct_sites="
                            f"{assessment.symmetry_distinct_handle_sites}"
                        )
                        handle_rejections[label] += 1
                        rejected.append(assessment.to_mapping())
            retro_rows.append(
                {
                    "schema_version": RETRO_ATTEMPT_SCHEMA,
                    "seed": seed,
                    "attempt_index": attempt_index,
                    "classification": classification,
                    "transform_consistent_candidate_count": len(candidates),
                    "native_trace_recovered": matched is not None,
                    "registry_handle_qualified": (
                        matched.registry_handle_qualified if matched is not None else False
                    ),
                    "rejected_handle_assessments": rejected,
                }
            )
    if len(retro_rows) != EXPECTED_RETRO_ABSTENTIONS:
        raise UgiTreeTransformerInterventionReadinessError(
            "valid native-forward retro-abstention denominator changed"
        )

    attributions = _attribution_rows(paths["attempt_attributions"])
    local_rows = [
        row
        for row in attributions
        if row.get("primary_attribution") == "exact_l1_unsupported_local_chemistry"
    ]
    if len(local_rows) != EXPECTED_LOCAL_UNSUPPORTED:
        raise UgiTreeTransformerInterventionReadinessError(
            "local-unsupported exact-L1 denominator changed"
        )
    reproduced = 0
    v2_signatures: Counter[str] = Counter()
    for row in local_rows:
        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            raise UgiTreeTransformerInterventionReadinessError(
                "local-unsupported attribution lacks a product"
            )
        signatures = unsupported_edge_signatures(smiles, support)
        if signatures:
            reproduced += 1
            v2_signatures.update(set(signatures))
    if reproduced != len(local_rows):
        raise UgiTreeTransformerInterventionReadinessError(
            "v2 role-local policy no longer detects every frozen unsupported-edge product"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        ledger = partial / "retro_attempts.jsonl.gz"
        write_jsonl(
            ledger,
            [
                {"schema_version": RETRO_ATTEMPT_SCHEMA, "rows": len(retro_rows)},
                *retro_rows,
            ],
        )
        source_root = repo / "forge/model/ugi_chemistry_flow.py"
        sampler_source = (
            repo
            / "experiments/phase1/product_l1/sampling/ugi_joint_end_to_end_sampling.py"
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "ready_for_new_constrained_sampling_preflight",
            "decision": "do_not_retrain_before_constrained_checkpoint_resampling",
            "retro_diagnostic": {
                "target_attempts": len(retro_rows),
                "classifications": dict(sorted(retro_classes.items())),
                "handle_rejections": dict(sorted(handle_rejections.items())),
                "registry_exact_l1_promotions": sum(
                    row["registry_handle_qualified"] for row in retro_rows
                ),
                "transform_consistency_is_not_exact_l1": True,
                "registry_handle_gate_unchanged": True,
            },
            "role_local_constraint_qualification": {
                "frozen_unsupported_products": len(local_rows),
                "detected_by_v2_policy": reproduced,
                "top_product_level_signatures": [
                    {"signature": signature, "products": count}
                    for signature, count in v2_signatures.most_common(20)
                ],
                "training_records": support.training_records[adapter.reaction_id],
                "role_edge_signatures": len(support.role_edges[adapter.reaction_id]),
                "role_cycle_signatures": len(support.role_cycles[adapter.reaction_id]),
                "complete_component_identities_stored": False,
                "decoder_masks_before_categorical_selection": True,
            },
            "telemetry_qualification": {
                "typed_terminal_decode_failures_implemented": True,
                "molecule_construction_subtypes_implemented": True,
                "empirical_failure_subtypes_available_only_after_new_sampling": True,
            },
            "next_experiment": {
                "operation": "paired_seed0_frozen_checkpoint_resampling",
                "retraining": False,
                "same_program_draw": True,
                "same_flow_and_terminal_seeds": True,
                "repairs_or_retries": False,
                "production_result_replaced": False,
            },
            "inputs": {
                "config": pin_record(config_path, repo),
                **{label: pin_record(path, repo) for label, path in sorted(paths.items())},
                "evaluation_details": archive_inputs,
            },
            "retro_attempts": artifact_record(
                ledger, logical_path="retro_attempts.jsonl.gz"
            ),
            "implementation": {
                "diagnostic": pin_record(Path(__file__), repo),
                "terminal_decoder": pin_record(source_root, repo),
                "sampler": pin_record(sampler_source, repo),
            },
            "production_thresholds_changed": False,
            "candidate_selection": False,
            "calls": {"generation": 0, "route": 0, "oracle": 0},
            "nonclaims": list(config["nonclaims"]),
        }
        write_json(partial / "result.json", result)
        partial.rename(output_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTreeTransformerInterventionReadinessError",
    "classify_transform_consistent_candidates",
    "diagnose_tree_transformer_interventions",
]
