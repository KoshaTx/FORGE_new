"""Render manuscript prose numbers from the final hash-pinned GEM evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json
from forge_paper.completed_evidence_v1 import (
    CONFIG_SCHEMA as COMMON_CONFIG_SCHEMA,
)
from forge_paper.completed_evidence_v1 import (
    _common_optional_values,
    _load_common_assessments,
    load_lipid_realism_aggregate,
)
from forge_paper.gem_table8 import RESULT_SCHEMA as MECHANISM_RESULT_SCHEMA
from forge_paper.gem_table9 import RESULT_SCHEMA as CATALOGUE_RESULT_SCHEMA

CONFIG_SCHEMA = "forge.gem_prose_results_config.v1"
RESULT_SCHEMA = "forge.gem_prose_results_render.v1"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
ATTEMPTS_PER_SEED = 3072
PROGRAM_KEYS = (
    "ugi_3cr_agile",
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)


class GemProseResultsError(ValueError):
    """A prose result is missing, stale, inadmissible, or inconsistent with its table."""


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GemProseResultsError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GemProseResultsError(f"{label} must be finite")
    return parsed


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GemProseResultsError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GemProseResultsError(f"{label} must be a mapping")
    return value


def _pin_without_size(value: Any, *, label: str) -> dict[str, str]:
    record = _mapping(value, label=label)
    if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise GemProseResultsError(f"{label} lacks a path or sha256")
    return {"path": record["path"], "sha256": record["sha256"]}


def _load_pin(pin: Any, repo: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_pin(_mapping(pin, label=label), repo, label=label)
    return path, read_json_object(path, error=GemProseResultsError, label=label)


def _mean_sd_tex(values: Sequence[float], *, digits: int = 1, scale: float = 1.0) -> str:
    if len(values) != len(EXPECTED_SEEDS):
        raise GemProseResultsError("a prose summary does not contain exactly three seeds")
    scaled = [scale * value for value in values]
    return rf"${statistics.fmean(scaled):.{digits}f}\pm{statistics.stdev(scaled):.{digits}f}$"


def _format_count(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _summary_mean(result: Mapping[str, Any], arm: str, metric: str) -> float:
    summaries = _mapping(result.get("summaries"), label="table summaries")
    arm_summary = _mapping(summaries.get(arm), label=f"summary {arm}")
    metric_summary = _mapping(arm_summary.get(metric), label=f"summary {arm}.{metric}")
    return _number(metric_summary.get("mean"), label=f"summary {arm}.{metric}.mean")


def _realism_metric(aggregate: Mapping[str, Any], method: str, metric: str) -> Mapping[str, Any]:
    methods = _mapping(aggregate.get("methods"), label="realism methods")
    method_result = _mapping(methods.get(method), label=f"realism {method}")
    metrics = _mapping(method_result.get("metrics"), label=f"realism {method} metrics")
    result = _mapping(metrics.get(metric), label=f"realism {method}.{metric}")
    if (
        result.get("status") != "estimated"
        or result.get("available_seeds") != len(EXPECTED_SEEDS)
        or result.get("required_seeds") != len(EXPECTED_SEEDS)
    ):
        raise GemProseResultsError(
            f"realism metric is not a three-seed estimate: {method}.{metric}"
        )
    return result


def _realism_tex(
    aggregate: Mapping[str, Any], method: str, metric: str, *, scale: float = 1.0
) -> str:
    summary = _realism_metric(aggregate, method, metric)
    mean = scale * _number(summary.get("mean"), label=f"realism {method}.{metric}.mean")
    sd = scale * _number(
        summary.get("sample_standard_deviation"), label=f"realism {method}.{metric}.sd"
    )
    return rf"${mean:.1f}\pm{sd:.1f}$"


def _validate_result(result: Mapping[str, Any], schema: str, *, label: str) -> None:
    gates = _mapping(result.get("gates"), label=f"{label} gates")
    if (
        result.get("schema_version") != schema
        or result.get("status") != "complete"
        or result.get("candidate_selection") is not False
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise GemProseResultsError(f"{label} is not complete and passing")


def render_gem_prose_results(
    config_path: Path,
    repo: Path,
    macro_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render every computed number cited in GEM result prose from final evidence."""

    config = read_json_object(config_path, error=GemProseResultsError, label="GEM prose config")
    expected_fields = {
        "schema_version",
        "status",
        "common_assessment_config",
        "lipid_realism_aggregate",
        "mechanism_table_result",
        "catalogue_table_result",
        "catalogue_adjudication",
        "novelty_audit",
        "synthesis_guidance",
        "attempts_per_method_per_seed",
        "expected_seeds",
        "candidate_selection",
    }
    if (
        set(config) != expected_fields
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_final_core_saturation_evidence"
        or config.get("attempts_per_method_per_seed") != ATTEMPTS_PER_SEED
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or config.get("candidate_selection") is not False
    ):
        raise GemProseResultsError("GEM prose config changed")

    common_config_path, common_config = _load_pin(
        config["common_assessment_config"], repo, label="common-assessment config"
    )
    if (
        common_config.get("schema_version") != COMMON_CONFIG_SCHEMA
        or common_config.get("candidate_selection") is not False
        or common_config.get("expected_seeds") != list(EXPECTED_SEEDS)
    ):
        raise GemProseResultsError("common-assessment config is inadmissible")
    try:
        common, _payloads, common_sources = _load_common_assessments(
            common_config.get("common_assessments"), repo, EXPECTED_SEEDS
        )
    except ValueError as error:
        raise GemProseResultsError(str(error)) from error

    realism, realism_source = load_lipid_realism_aggregate(config["lipid_realism_aggregate"], repo)
    mechanism_path, mechanism = _load_pin(
        config["mechanism_table_result"], repo, label="mechanism table result"
    )
    catalogue_table_path, catalogue_table = _load_pin(
        config["catalogue_table_result"], repo, label="catalogue table result"
    )
    catalogue_path, catalogue = _load_pin(
        config["catalogue_adjudication"], repo, label="catalogue adjudication"
    )
    novelty_path, novelty = _load_pin(config["novelty_audit"], repo, label="novelty audit")
    guidance_path, guidance = _load_pin(
        config["synthesis_guidance"], repo, label="synthesis-guidance audit"
    )
    _validate_result(mechanism, MECHANISM_RESULT_SCHEMA, label="mechanism table")
    _validate_result(catalogue_table, CATALOGUE_RESULT_SCHEMA, label="catalogue table")
    if (
        catalogue.get("schema_version") != "forge.final_bl_core_production_adjudication.v1"
        or catalogue.get("status") != "complete"
        or catalogue.get("candidate_selection") is not False
    ):
        raise GemProseResultsError("catalogue adjudication is inadmissible")
    if (
        novelty.get("schema_version") != "phase1_ugi_novelty_audit.v1"
        or novelty.get("status") != "complete"
    ):
        raise GemProseResultsError("novelty audit is inadmissible")
    if (
        guidance.get("schema_version") != "phase1_ugi_synthesis_guidance_failure_audit.v1"
        or guidance.get("status") != "proposal_augmented_synthesis_guidance_failure_audited"
    ):
        raise GemProseResultsError("synthesis-guidance audit is inadmissible")

    def common_tex(method: str, metric: str) -> str:
        values = _common_optional_values(common[method], metric)
        if any(value is None for value in values):
            raise GemProseResultsError(f"common metric is not estimable: {method}.{metric}")
        return _mean_sd_tex([float(value) for value in values if value is not None])

    forge_common = common["forge_transformer"]
    held_total = sum(
        _integer(row["metrics"]["held_component_exact_l1_products"], label="FORGE held count")
        for row in forge_common
    )
    held_attempts = sum(_integer(row["attempts"], label="FORGE attempts") for row in forge_common)

    c2st_means = []
    for method_result in _mapping(realism.get("methods"), label="realism methods").values():
        metrics = _mapping(method_result, label="realism method").get("metrics")
        metric = _mapping(metrics, label="realism metrics").get("grouped_c2st_auc")
        if isinstance(metric, Mapping) and metric.get("status") == "estimated":
            c2st_means.append(_number(metric.get("mean"), label="realism C2ST mean"))
    if not c2st_means:
        raise GemProseResultsError("no structural-realism C2ST estimate is available")

    mechanism_sources = _mapping(mechanism.get("sources"), label="mechanism sources").get(
        "training_results"
    )
    if not isinstance(mechanism_sources, list) or len(mechanism_sources) != len(EXPECTED_SEEDS):
        raise GemProseResultsError("mechanism training sources changed")
    parameter_counts: dict[str, set[int]] = {
        "full_transformer": set(),
        "fact_matched": set(),
        "fact_generous": set(),
    }
    parameter_sources = []
    for index, source in enumerate(mechanism_sources):
        pin = _pin_without_size(source, label=f"mechanism training source {index}")
        source_path = resolve_pin(pin, repo, label=f"mechanism training source {index}")
        payload = read_json_object(
            source_path,
            error=GemProseResultsError,
            label=f"mechanism training source {index}",
        )
        arms = _mapping(payload.get("arms"), label="mechanism training arms")
        if (
            payload.get("schema_version") != "forge.synthesis_program_production_training_result.v1"
            or payload.get("status") != "pass"
            or int(payload.get("seed", -1)) != EXPECTED_SEEDS[index]
        ):
            raise GemProseResultsError("mechanism training source is inadmissible")
        for arm in parameter_counts:
            parameter_counts[arm].add(
                _integer(
                    _mapping(arms.get(arm), label=f"mechanism arm {arm}").get("parameter_count"),
                    label=f"mechanism {arm} parameter count",
                )
            )
        parameter_sources.append(pin_record(source_path, repo))
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise GemProseResultsError("mechanism parameter counts vary across seeds")
    shared_parameters = next(iter(parameter_counts["full_transformer"]))
    if parameter_counts["fact_matched"] != {shared_parameters}:
        raise GemProseResultsError("FACT-matched no longer matches the Transformer parameter count")
    generous_parameters = next(iter(parameter_counts["fact_generous"]))

    catalogue_summaries = _mapping(catalogue_table.get("summaries"), label="catalogue summaries")

    def catalogue_value(program: str, method: str, metric: str) -> float:
        program_summary = _mapping(
            catalogue_summaries.get(program), label=f"catalogue summary {program}"
        )
        method_summary = _mapping(
            program_summary.get(method), label=f"catalogue summary {program}.{method}"
        )
        return _number(
            method_summary.get(metric), label=f"catalogue summary {program}.{method}.{metric}"
        )

    catalogue_programs = _mapping(
        _mapping(
            catalogue.get("finite_component_catalogue_comparison"),
            label="finite catalogue comparison",
        ).get("programs"),
        label="finite catalogue programs",
    )
    ugi_catalogue = _mapping(
        _mapping(catalogue_programs.get(PROGRAM_KEYS[0]), label="Ugi catalogue program").get(
            "catalogue"
        ),
        label="Ugi catalogue",
    )
    ugi_roles = _mapping(ugi_catalogue.get("role_component_counts"), label="Ugi role counts")
    fold_coverage = _mapping(
        ugi_catalogue.get("source_product_coverage_by_fold"), label="Ugi fold coverage"
    )
    for fold, expected in (("train", 1.0), ("calibration", 0.0), ("heldout", 0.0)):
        observed = _number(
            _mapping(fold_coverage.get(fold), label=f"Ugi {fold} coverage").get(
                "coverage_fraction"
            ),
            label=f"Ugi {fold} coverage",
        )
        if observed != expected:
            raise GemProseResultsError(f"Ugi catalogue {fold} coverage changed")

    novelty_full = _mapping(
        _mapping(novelty.get("generator_novelty"), label="generator novelty").get("full_corpus"),
        label="full-corpus novelty",
    )
    guidance_diagnostics = _mapping(
        guidance.get("controller_diagnostics"), label="guidance controller diagnostics"
    )

    macros = {
        "ForgeProseAttemptsPerProgram": _format_count(ATTEMPTS_PER_SEED),
        "ForgeProseConditionedTotalAttempts": _format_count(
            ATTEMPTS_PER_SEED * len(EXPECTED_SEEDS) * len(PROGRAM_KEYS)
        ),
        "ForgeProseCommonSelectorExactPerThousand": common_tex(
            "learned_inventory_selector", "exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonForgeExactPerThousand": common_tex(
            "forge_transformer", "exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonForgeDistinctPerThousand": common_tex(
            "forge_transformer", "unique_exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonForgeHeldPerThousand": common_tex(
            "forge_transformer", "held_component_exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonDefogExactPerThousand": common_tex(
            "defog_unconditional", "exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonDefogHeldPerThousand": common_tex(
            "defog_unconditional", "held_component_exact_l1_products_per_1000_attempts"
        ),
        "ForgeProseCommonGenmolValidPerThousand": common_tex(
            "genmol_safe", "valid_products_per_1000_attempts"
        ),
        "ForgeProseCommonHeldTotal": str(held_total),
        "ForgeProseCommonHeldAttempts": _format_count(held_attempts),
        "ForgeProseRealismCtwoSTMin": f"{min(c2st_means):.3f}",
        "ForgeProseRealismCtwoSTMax": f"{max(c2st_means):.3f}",
        "ForgeProseRealismGenmolEffective": _realism_tex(
            realism, "genmol_safe", "effective_molecule_count"
        ),
        "ForgeProseRealismForgeEffective": _realism_tex(
            realism, "forge_transformer", "effective_molecule_count"
        ),
        "ForgeProseRealismForgeDescriptorPrecision": _realism_tex(
            realism,
            "forge_transformer",
            "descriptor_manifold_precision_per_attempt",
            scale=1000.0,
        ),
        "ForgeProseMechanismFullUgi": f"{_summary_mean(mechanism, 'full_transformer', 'ugi_l1_per_1000'):.1f}",
        "ForgeProseMechanismFullBL": f"{_summary_mean(mechanism, 'full_transformer', 'bl_l1_per_1000'):.1f}",
        "ForgeProseMechanismFullLX": f"{_summary_mean(mechanism, 'full_transformer', 'lx_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactMatchedUgi": f"{_summary_mean(mechanism, 'fact_matched', 'ugi_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactMatchedBL": f"{_summary_mean(mechanism, 'fact_matched', 'bl_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactMatchedLX": f"{_summary_mean(mechanism, 'fact_matched', 'lx_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactGenerousUgi": f"{_summary_mean(mechanism, 'fact_generous', 'ugi_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactGenerousBL": f"{_summary_mean(mechanism, 'fact_generous', 'bl_l1_per_1000'):.1f}",
        "ForgeProseMechanismFactGenerousLX": f"{_summary_mean(mechanism, 'fact_generous', 'lx_l1_per_1000'):.1f}",
        "ForgeProseMechanismSharedParameterMillions": f"{shared_parameters / 1_000_000.0:.2f}",
        "ForgeProseMechanismGenerousParameterMillions": f"{generous_parameters / 1_000_000.0:.2f}",
        "ForgeProseNoCrossAttentionUgiReduction": f"{_summary_mean(mechanism, 'full_transformer', 'ugi_l1_per_1000') - _summary_mean(mechanism, 'input_only_program', 'ugi_l1_per_1000'):.1f}",
        "ForgeProseNoCrossAttentionBLIncrease": f"{_summary_mean(mechanism, 'input_only_program', 'bl_l1_per_1000') - _summary_mean(mechanism, 'full_transformer', 'bl_l1_per_1000'):.1f}",
        "ForgeProseNoCrossAttentionLXReduction": f"{_summary_mean(mechanism, 'full_transformer', 'lx_l1_per_1000') - _summary_mean(mechanism, 'input_only_program', 'lx_l1_per_1000'):.1f}",
        "ForgeProseFullReferenceSize": _format_count(
            _integer(novelty_full.get("reference_size"), label="full reference size")
        ),
        "ForgeProseCatalogueUgiAmineCount": str(
            _integer(ugi_roles.get("amine_head"), label="Ugi amine count")
        ),
        "ForgeProseCatalogueUgiAldehydeCount": str(
            _integer(ugi_roles.get("oxoester_aldehyde_body_tail"), label="Ugi aldehyde count")
        ),
        "ForgeProseCatalogueUgiIsocyanideCount": str(
            _integer(ugi_roles.get("isocyanide_tail"), label="Ugi isocyanide count")
        ),
        "ForgeProseCatalogueUgiTupleCeiling": _format_count(
            _integer(ugi_catalogue.get("component_tuple_space_upper_bound"), label="Ugi ceiling")
        ),
        "ForgeProseCatalogueBLTupleCeiling": _format_count(
            _integer(
                _mapping(
                    _mapping(
                        catalogue_programs.get(PROGRAM_KEYS[1]), label="BL catalogue program"
                    ).get("catalogue"),
                    label="BL catalogue",
                ).get("component_tuple_space_upper_bound"),
                label="BL ceiling",
            )
        ),
        "ForgeProseCatalogueLXTupleCeiling": _format_count(
            _integer(
                _mapping(
                    _mapping(
                        catalogue_programs.get(PROGRAM_KEYS[2]), label="LX catalogue program"
                    ).get("catalogue"),
                    label="LX catalogue",
                ).get("component_tuple_space_upper_bound"),
                label="LX ceiling",
            )
        ),
        "ForgeProseCatalogueTrainCoveragePercent": "100",
        "ForgeProseCatalogueHeldCoveragePercent": "0",
        "ForgeProseGuidanceMixedGroups": str(
            _integer(guidance_diagnostics.get("mixed_utility_groups"), label="mixed groups")
        ),
        "ForgeProseGuidanceTotalGroups": str(
            _integer(guidance_diagnostics.get("checkpoint_groups"), label="guidance groups")
        ),
    }
    for prefix, method in (("Forge", "forge"), ("Catalogue", "finite_catalogue")):
        for label, program in zip(("Ugi", "BL", "LX"), PROGRAM_KEYS, strict=True):
            metrics = [
                ("ValidPerThousand", "valid_per_1000"),
                ("ExactPerThousand", "exact_l1_per_1000"),
                ("EffectiveComponentCount", "effective_component_count"),
            ]
            if method == "forge":
                metrics.append(("ComponentNovelPerThousand", "component_novel_per_1000"))
            for metric_label, metric in metrics:
                macros[f"ForgeProseCatalogue{prefix}{label}{metric_label}"] = (
                    f"{catalogue_value(program, method, metric):.1f}"
                )

    rendered = "\n".join(_macro(name, macros[name]) for name in sorted(macros)) + "\n"
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(macro_path, rendered.encode("utf-8"))

    sources = [
        pin_record(common_config_path, repo),
        *common_sources,
        realism_source,
        pin_record(mechanism_path, repo),
        *parameter_sources,
        pin_record(catalogue_table_path, repo),
        pin_record(catalogue_path, repo),
        pin_record(novelty_path, repo),
        pin_record(guidance_path, repo),
    ]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": sources,
        "candidate_selection": False,
        "macros": macros,
        "artifact": artifact_record(macro_path, logical_path=macro_path.name),
        "gates": {
            "all_prose_numbers_generated": True,
            "common_benchmark_matches_final_assessments": True,
            "held_component_count_uses_final_common_benchmark": held_total == 13,
            "structural_realism_matches_table_7": True,
            "mechanism_study_matches_table_8": True,
            "catalogue_comparison_matches_table_9": True,
            "guidance_diagnostic_hash_pinned": True,
            "three_independent_training_seeds": True,
            "candidate_selection_absent": True,
        },
    }
    if not all(result["gates"].values()):
        raise GemProseResultsError("GEM prose evidence gate failed")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemProseResultsError", "render_gem_prose_results"]
