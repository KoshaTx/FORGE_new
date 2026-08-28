"""Strict aggregation and machine-generated v1 paper table/figure inputs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import (
    atomic_write,
    iter_jsonl,
    read_json_object,
    write_csv,
    write_json,
    write_jsonl,
)

CONFIG_SCHEMA = "forge.natbiotech_v1_renderer_config.v1"
LEDGER_SCHEMA = "forge.natbiotech_v1_result_rows.v1"
COMMON_ROW_SCHEMA = "forge.natbiotech_v1_common_seed_row.v2"
MECHANISM_ROW_SCHEMA = "forge.natbiotech_v1_mechanism_seed_row.v1"
HELD_FAMILY_ROW_SCHEMA = "forge.natbiotech_v1_held_family_seed_row.v1"

COMMON_METRICS = (
    "valid_per_1000",
    "exact_l1_per_1000",
    "unique_l1_per_1000",
    "open_ended_per_1000",
    "held_component_per_1000",
    "diversity",
    "decomposition_coverage",
    "replay_precision",
    "ambiguity_fraction",
    "verified_upstream_fraction",
    "terminal_evidence_fraction",
    "complete_dossier_fraction",
    "route_abstention_fraction",
)
MECHANISM_METRICS = (
    "held_family_loss",
    "ugi_l1_per_1000",
    "bl_l1_per_1000",
    "lx_l1_per_1000",
    "held_component_per_1000",
    "cross_role_fidelity",
    "diversity",
)


class PaperResultsV1Error(ValueError):
    """Verified experiment rows are incomplete or inconsistent for paper rendering."""


def common_assessment_seed_row(
    result_path: Path, repo: Path, *, route_union_result: Path | None = None
) -> dict[str, Any]:
    """Extract one common row from a verified assessor result without manual transcription."""

    result = read_json_object(
        result_path, error=PaperResultsV1Error, label="common Ugi assessment result"
    )
    if result.get("status") != "pass" or result.get("schema_version") not in {
        "forge.common_ugi_complete_assessment.v1",
        "forge.learned_inventory_selector_result.v1",
    }:
        raise PaperResultsV1Error("common Ugi result is not a passing supported assessment")
    common = result.get("common_assessment")
    route = result.get("route_evidence_assessment")
    if not isinstance(common, Mapping) or not isinstance(route, Mapping):
        raise PaperResultsV1Error("common or route assessment is missing")
    route_source = result_path
    route_scope = "bounded_forge_candidate_index"
    if route_union_result is not None:
        union = read_json_object(
            route_union_result,
            error=PaperResultsV1Error,
            label="method-blind route-union adjudication",
        )
        if (
            union.get("schema_version") != "forge.common_ugi_method_blind_route_adjudication.v1"
            or union.get("status") != "pass"
        ):
            raise PaperResultsV1Error("method-blind route-union adjudication is not passing")
        matches = [
            row
            for row in union.get("method_seed_results", [])
            if isinstance(row, Mapping)
            and row.get("method_id") == common.get("method_id")
            and row.get("seed") == common.get("seed")
        ]
        if len(matches) != 1 or not isinstance(
            matches[0].get("route_evidence_assessment"), Mapping
        ):
            raise PaperResultsV1Error("route-union adjudication lacks the common method/seed row")
        route = matches[0]["route_evidence_assessment"]
        scope = route.get("evidence_scope")
        if (
            not isinstance(scope, Mapping)
            or scope.get("method_blind_cross_method_union") is not True
            or scope.get("route_evidence_closure_comparison_authorized") is not True
            or scope.get("synthesis_success_comparison_authorized") is not False
        ):
            raise PaperResultsV1Error("route-union evidence scope is not admissible")
        route_source = route_union_result
        route_scope = "method_blind_cross_method_union"
    metrics = common.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PaperResultsV1Error("common assessment metrics are missing")
    valid = int(metrics["valid"])
    eligible = int(route["eligible_unique_exact_l1"])

    def fraction(numerator: object, denominator: int) -> float | None:
        if isinstance(numerator, bool) or not isinstance(numerator, (int, float)):
            raise PaperResultsV1Error("route assessment count is not numeric")
        return float(numerator) / denominator if denominator else None

    return {
        "schema_version": COMMON_ROW_SCHEMA,
        "method_id": str(common["method_id"]),
        "seed": int(common["seed"]),
        "metrics": {
            "valid_per_1000": float(metrics["valid_products_per_1000_attempts"]),
            "exact_l1_per_1000": float(metrics["exact_l1_products_per_1000_attempts"]),
            "unique_l1_per_1000": float(metrics["unique_exact_l1_products_per_1000_attempts"]),
            "open_ended_per_1000": float(
                metrics["unique_method_visible_open_ended_exact_l1_products_per_1000_attempts"]
            ),
            "held_component_per_1000": float(
                metrics["held_component_exact_l1_products_per_1000_attempts"]
            ),
            "diversity": metrics["mean_pairwise_ecfp4_distance"],
            "decomposition_coverage": metrics["retro_decomposition_coverage_among_valid"],
            "replay_precision": metrics["retro_transform_precision"],
            "ambiguity_fraction": fraction(metrics["ambiguous_exact_decompositions"], valid),
            "verified_upstream_fraction": fraction(route["verified_upstream"], eligible),
            "terminal_evidence_fraction": fraction(route["terminal_evidence"], eligible),
            "complete_dossier_fraction": fraction(route["complete_dossier"], eligible),
            "route_abstention_fraction": fraction(route["abstentions"], eligible),
        },
        "source": pin_record(result_path, repo),
        "route_source": pin_record(route_source, repo),
        "route_scope": route_scope,
    }


def mechanism_seed_rows(evaluation_path: Path, repo: Path) -> list[dict[str, Any]]:
    """Extract all architecture rows from one passing mechanism-study evaluation."""

    result = read_json_object(
        evaluation_path, error=PaperResultsV1Error, label="mechanism evaluation result"
    )
    if (
        result.get("schema_version") != "forge.synthesis_program_production_evaluation_result.v1"
        or result.get("status") != "pass"
    ):
        raise PaperResultsV1Error("mechanism evaluation is not a passing production evaluation")
    checkpoints = result["checkpoint_metrics"]
    component = result["component_disjoint_metrics"]
    held = result["ugi_held_component_metrics"]
    fidelity = result["cross_role_fidelity"]
    final_step = str(max(int(step) for arm in checkpoints.values() for step in arm))
    program_fields = {
        "ugi_l1_per_1000": "ugi_3cr_agile",
        "bl_l1_per_1000": "bl_2023_repeated_aza_michael",
        "lx_l1_per_1000": "lx_2024_repeated_reductive_amination",
    }
    rows = []
    for arm_id in sorted(checkpoints):
        final = checkpoints[arm_id][final_step]["heldout"]
        if set(final) != set(program_fields.values()):
            raise PaperResultsV1Error(f"mechanism arm {arm_id} does not cover all three programs")
        losses = [
            float(component[arm_id][program]["heldout_denoising_loss_at_t_0_5"])
            for program in program_fields.values()
        ]
        diversities = [final[program]["internal_diversity"] for program in final]
        rows.append(
            {
                "schema_version": MECHANISM_ROW_SCHEMA,
                "arm_id": arm_id,
                "seed": int(result["seed"]),
                "metrics": {
                    "held_family_loss": float(np.mean(losses)),
                    **{
                        field: 1000.0 * float(final[program]["exact_l1_yield_per_attempt"])
                        for field, program in program_fields.items()
                    },
                    "held_component_per_1000": float(
                        held[arm_id]["exact_l1_products_with_held_component_per_1000_attempts"]
                    ),
                    "cross_role_fidelity": fidelity[arm_id][
                        "residual_correlation_frobenius_distance"
                    ],
                    "diversity": (
                        float(np.mean([float(value) for value in diversities]))
                        if all(value is not None for value in diversities)
                        else None
                    ),
                },
                "source": pin_record(evaluation_path, repo),
            }
        )
    return rows


def held_family_seed_rows(evaluation_path: Path, repo: Path) -> list[dict[str, Any]]:
    """Extract secondary leave-one-family-out rows without turning them into a hard gate."""

    result = read_json_object(
        evaluation_path, error=PaperResultsV1Error, label="held-family evaluation result"
    )
    if (
        result.get("schema_version") != "forge.synthesis_program_production_evaluation_result.v1"
        or result.get("status") != "pass"
    ):
        raise PaperResultsV1Error("held-family evaluation is not passing")
    checkpoints = result["checkpoint_metrics"]
    final_step = str(max(int(step) for arm in checkpoints.values() for step in arm))
    rows = []
    for arm_id, arm in sorted(checkpoints.items()):
        if not arm_id.startswith("leave_out_"):
            raise PaperResultsV1Error(f"unexpected held-family arm: {arm_id}")
        values = arm[final_step]["heldout"]
        if len(values) != 1:
            raise PaperResultsV1Error(f"held-family arm {arm_id} evaluates more than one family")
        held_family, metrics = next(iter(values.items()))
        rows.append(
            {
                "schema_version": HELD_FAMILY_ROW_SCHEMA,
                "held_family": held_family,
                "seed": int(result["seed"]),
                "metrics": {
                    "exact_l1_per_1000": 1000.0 * float(metrics["exact_l1_yield_per_attempt"]),
                    "decomposition_coverage": metrics["exact_l1_decomposition_coverage"],
                    "replay_precision": metrics["exact_forward_replay_precision"],
                },
                "source": pin_record(evaluation_path, repo),
            }
        )
    return rows


def write_v1_result_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic result ledger consumed by :func:`render_v1_results`."""

    write_jsonl(path, [{"schema_version": LEDGER_SCHEMA, "rows": len(rows)}, *rows])


def _finite_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperResultsV1Error(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PaperResultsV1Error(f"{label} must be finite")
    return number


def _validate_seed_row(
    raw: object,
    *,
    schema: str,
    id_field: str,
    metrics: Sequence[str],
    route_fields: bool = False,
) -> dict[str, Any]:
    fields = {"schema_version", id_field, "seed", "metrics", "source"}
    if route_fields:
        fields.update({"route_source", "route_scope"})
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise PaperResultsV1Error(f"{schema} row must define exactly {sorted(fields)}")
    if raw["schema_version"] != schema:
        raise PaperResultsV1Error(f"unsupported result-row schema: {raw['schema_version']!r}")
    identifier = raw[id_field]
    seed = raw["seed"]
    values = raw["metrics"]
    source = raw["source"]
    if not isinstance(identifier, str) or not identifier:
        raise PaperResultsV1Error(f"{id_field} must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PaperResultsV1Error("seed must be a non-negative integer")
    if not isinstance(values, Mapping) or set(values) != set(metrics):
        raise PaperResultsV1Error(f"{identifier} metrics must define exactly {sorted(metrics)}")
    normalized = {
        name: _finite_number(values[name], label=f"{identifier}.{name}") for name in metrics
    }
    if (
        not isinstance(source, Mapping)
        or set(source) != {"path", "sha256", "bytes"}
        or isinstance(source["bytes"], bool)
        or not isinstance(source["bytes"], int)
        or source["bytes"] < 0
    ):
        raise PaperResultsV1Error("result row source must be an exact path/SHA-256/size pin")
    normalized_row = {
        id_field: identifier,
        "seed": seed,
        "metrics": normalized,
        "source": dict(source),
    }
    if route_fields:
        route_source = raw["route_source"]
        route_scope = raw["route_scope"]
        if (
            not isinstance(route_source, Mapping)
            or set(route_source) != {"path", "sha256", "bytes"}
            or isinstance(route_source["bytes"], bool)
            or not isinstance(route_source["bytes"], int)
            or route_source["bytes"] < 0
            or route_scope
            not in {"bounded_forge_candidate_index", "method_blind_cross_method_union"}
        ):
            raise PaperResultsV1Error("common route source or scope is invalid")
        normalized_row["route_source"] = dict(route_source)
        normalized_row["route_scope"] = route_scope
    return normalized_row


def _load_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(iter_jsonl(path))
    if not records or records[0] != {"schema_version": LEDGER_SCHEMA, "rows": len(records) - 1}:
        raise PaperResultsV1Error("paper result ledger header is missing or inconsistent")
    common: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for raw in records[1:]:
        if not isinstance(raw, Mapping):
            raise PaperResultsV1Error("paper result row must be an object")
        schema = raw.get("schema_version")
        if schema == COMMON_ROW_SCHEMA:
            common.append(
                _validate_seed_row(
                    raw,
                    schema=COMMON_ROW_SCHEMA,
                    id_field="method_id",
                    metrics=COMMON_METRICS,
                    route_fields=True,
                )
            )
        elif schema == MECHANISM_ROW_SCHEMA:
            mechanism.append(
                _validate_seed_row(
                    raw,
                    schema=MECHANISM_ROW_SCHEMA,
                    id_field="arm_id",
                    metrics=MECHANISM_METRICS,
                )
            )
        elif schema == HELD_FAMILY_ROW_SCHEMA:
            held.append(
                _validate_seed_row(
                    raw,
                    schema=HELD_FAMILY_ROW_SCHEMA,
                    id_field="held_family",
                    metrics=("exact_l1_per_1000", "decomposition_coverage", "replay_precision"),
                )
            )
        else:
            raise PaperResultsV1Error(f"unsupported paper result row: {schema!r}")
    return common, mechanism, held


def _group_rows(
    rows: Sequence[Mapping[str, Any]], id_field: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[id_field])].append(row)
    for identifier, values in grouped.items():
        seeds = [int(value["seed"]) for value in values]
        if len(seeds) != len(set(seeds)):
            raise PaperResultsV1Error(f"duplicate seed for {identifier}")
        values.sort(key=lambda value: int(value["seed"]))
    return dict(grouped)


def _bootstrap_summary(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], *, seed: int, resamples: int
) -> dict[str, dict[str, float | int | None]]:
    if not rows:
        raise PaperResultsV1Error("cannot aggregate an empty seed set")
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, float | int | None]] = {}
    for metric in metrics:
        values = [row["metrics"][metric] for row in rows]
        available = [float(value) for value in values if value is not None]
        if len(available) != len(values):
            output[metric] = {
                "mean": None,
                "sd": None,
                "ci95_low": None,
                "ci95_high": None,
                "available_seeds": len(available),
            }
            continue
        vector = np.asarray(available, dtype=np.float64)
        indices = rng.integers(0, len(vector), size=(resamples, len(vector)))
        bootstraps = vector[indices].mean(axis=1)
        output[metric] = {
            "mean": float(vector.mean()),
            # Sample sd, matching the mean $\pm$ sd the other tables report. The bootstrap
            # bounds are still carried in the ledger: at three seeds a 95% percentile interval
            # is pinned to [min, max], since the all-minimum resample already holds 1/27.
            "sd": float(vector.std(ddof=1)) if len(vector) > 1 else None,
            "ci95_low": float(np.quantile(bootstraps, 0.025)),
            "ci95_high": float(np.quantile(bootstraps, 0.975)),
            "available_seeds": len(available),
        }
    return output


def _tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _tex_metric(value: Mapping[str, float | int | None]) -> str:
    if value["mean"] is None:
        return "N/R"
    if value["sd"] is None:
        # One seed carries no spread; printing $\pm0.0$ would assert a variance we did not measure.
        return rf"${value['mean']:.1f}$"
    return rf"${value['mean']:.1f}\pm{value['sd']:.1f}$"


def _write_tex_rows(
    path: Path,
    order: Sequence[str],
    display: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    metrics: Sequence[str],
) -> None:
    lines = [
        " & ".join(
            [_tex_escape(str(display[identifier]))]
            + [_tex_metric(summaries[identifier][metric]) for metric in metrics]
        )
        + r" \\"
        for identifier in order
    ]
    # Same convention as the completed-evidence tables: tint the row reporting our own model.
    # \cellcolor, not \rowcolor: see the note in completed_evidence_v1._highlight_forge_rows.
    def _tint(line: str) -> str:
        if not line.split("&")[0].strip().startswith("FORGE"):
            return line
        body, _, tail = line.rpartition(r"\\")
        return "&".join(r"\cellcolor{forgerow}" + c for c in body.split("&")) + r"\\" + tail

    lines = [_tint(line) for line in lines]
    # A rule where the family changes, and before our own model. Never on the first line: these
    # files are \input straight after an \hline, whose lookahead would reject a leading \noalign.
    separated: list[str] = []
    for index, line in enumerate(lines):
        label = line.split("&")[0]
        if index and ("FACT" in label or r"\cellcolor" in label):
            if not separated[-1].startswith(r"\midrule") and "FACT" not in separated[-1].split("&")[0]:
                separated.append(r"\midrule")
            elif r"\cellcolor" in label:
                separated.append(r"\midrule")
        separated.append(line)
    lines = separated
    atomic_write(path, ("\n".join(lines) + "\n").encode())


def render_v1_results(
    config_path: Path,
    repo: Path,
    result_rows_path: Path,
    output_dir: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Render verified aggregate rows; never infer a missing experiment value."""

    config = read_json_object(
        config_path, error=PaperResultsV1Error, label="v1 result renderer config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PaperResultsV1Error("unsupported v1 result renderer config")
    expected_config_fields = {
        "schema_version",
        "bootstrap_seed",
        "bootstrap_resamples",
        "expected_seeds",
        "common_method_order",
        "mechanism_arm_order",
        "display_names",
        "held_reaction_family_is_secondary_not_hard_gate",
        "candidate_selection",
    }
    if set(config) != expected_config_fields:
        raise PaperResultsV1Error("v1 result renderer config fields changed")
    expected_seed_values = config["expected_seeds"]
    if (
        not isinstance(expected_seed_values, list)
        or not expected_seed_values
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in expected_seed_values
        )
        or len(expected_seed_values) != len(set(expected_seed_values))
    ):
        raise PaperResultsV1Error("v1 renderer expected seeds are invalid")
    if (
        config["held_reaction_family_is_secondary_not_hard_gate"] is not True
        or config["candidate_selection"] is not False
    ):
        raise PaperResultsV1Error("v1 renderer scientific safeguards changed")
    common, mechanism, held = _load_rows(result_rows_path)
    for index, row in enumerate([*common, *mechanism, *held]):
        source = row["source"]
        source_path = resolve_pin(
            {"path": source["path"], "sha256": source["sha256"]},
            repo,
            label=f"paper result row {index} source",
        )
        if source_path.stat().st_size != int(source["bytes"]):
            raise PaperResultsV1Error(f"paper result row {index} source size changed")
        if "route_source" in row:
            route_source = row["route_source"]
            route_path = resolve_pin(
                {"path": route_source["path"], "sha256": route_source["sha256"]},
                repo,
                label=f"paper result row {index} route source",
            )
            if route_path.stat().st_size != int(route_source["bytes"]):
                raise PaperResultsV1Error(f"paper result row {index} route source size changed")
    common_groups = _group_rows(common, "method_id")
    mechanism_groups = _group_rows(mechanism, "arm_id")
    held_groups = _group_rows(held, "held_family")
    expected_seed_set = set(int(value) for value in expected_seed_values)
    expected_seeds = len(expected_seed_set)
    common_order = [str(value) for value in config["common_method_order"]]
    mechanism_order = [str(value) for value in config["mechanism_arm_order"]]
    if strict:
        bounded_route_rows = [
            f"{row['method_id']}:{row['seed']}"
            for row in common
            if row["route_scope"] != "method_blind_cross_method_union"
        ]
        if bounded_route_rows:
            raise PaperResultsV1Error(
                "strict route table requires the method-blind union adjudication: "
                f"{bounded_route_rows}"
            )
        missing_common = set(common_order) - set(common_groups)
        missing_mechanism = set(mechanism_order) - set(mechanism_groups)
        if missing_common or missing_mechanism:
            raise PaperResultsV1Error(
                "paper result ledger is incomplete: "
                f"common={sorted(missing_common)}, mechanism={sorted(missing_mechanism)}"
            )
        wrong = {
            identifier: len(common_groups.get(identifier, ()))
            for identifier in common_order
            if len(common_groups.get(identifier, ())) != expected_seeds
        }
        wrong.update(
            {
                identifier: len(mechanism_groups.get(identifier, ()))
                for identifier in mechanism_order
                if len(mechanism_groups.get(identifier, ())) != expected_seeds
            }
        )
        wrong.update(
            {
                f"held:{identifier}": len(rows)
                for identifier, rows in held_groups.items()
                if len(rows) != expected_seeds
            }
        )
        if wrong:
            raise PaperResultsV1Error(f"paper rows do not have {expected_seeds} seeds: {wrong}")
        seed_sets = {
            f"common:{identifier}": {int(row["seed"]) for row in rows}
            for identifier, rows in common_groups.items()
        }
        seed_sets.update(
            {
                f"mechanism:{identifier}": {int(row["seed"]) for row in rows}
                for identifier, rows in mechanism_groups.items()
            }
        )
        seed_sets.update(
            {
                f"held:{identifier}": {int(row["seed"]) for row in rows}
                for identifier, rows in held_groups.items()
            }
        )
        changed_seeds = {
            identifier: sorted(values)
            for identifier, values in seed_sets.items()
            if values != expected_seed_set
        }
        if changed_seeds:
            raise PaperResultsV1Error(f"paper rows changed the paired seed set: {changed_seeds}")
        if set(held_groups) != {
            "ugi_3cr_agile",
            "bl_2023_repeated_aza_michael",
            "lx_2024_repeated_reductive_amination",
        }:
            raise PaperResultsV1Error(
                "held-family rows must cover exactly the three admitted programs"
            )
    common_order = [value for value in common_order if value in common_groups]
    mechanism_order = [value for value in mechanism_order if value in mechanism_groups]
    bootstrap_seed = int(config["bootstrap_seed"])
    resamples = int(config["bootstrap_resamples"])
    common_summary = {
        identifier: _bootstrap_summary(
            rows, COMMON_METRICS, seed=bootstrap_seed, resamples=resamples
        )
        for identifier, rows in common_groups.items()
    }
    mechanism_summary = {
        identifier: _bootstrap_summary(
            rows, MECHANISM_METRICS, seed=bootstrap_seed, resamples=resamples
        )
        for identifier, rows in mechanism_groups.items()
    }
    held_summary = {
        identifier: _bootstrap_summary(
            rows,
            ("exact_l1_per_1000", "decomposition_coverage", "replay_precision"),
            seed=bootstrap_seed,
            resamples=resamples,
        )
        for identifier, rows in held_groups.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    display = config["display_names"]
    _write_tex_rows(
        output_dir / "common_ugi_benchmark_rows.tex",
        common_order,
        display,
        common_summary,
        COMMON_METRICS[:6],
    )
    _write_tex_rows(
        output_dir / "architecture_ablation_rows.tex",
        mechanism_order,
        display,
        mechanism_summary,
        MECHANISM_METRICS,
    )
    _write_tex_rows(
        output_dir / "route_evidence_rows.tex",
        common_order,
        display,
        common_summary,
        COMMON_METRICS[9:13],
    )
    tradeoff_path = output_dir / "common_ugi_tradeoff.csv"
    write_csv(
        tradeoff_path,
        [
            {
                "method_id": identifier,
                **{metric: common_summary[identifier][metric]["mean"] for metric in COMMON_METRICS},
            }
            for identifier in common_order
        ],
        ["method_id", *COMMON_METRICS],
    )
    mechanism_path = output_dir / "mechanism_ablation_metrics.csv"
    write_csv(
        mechanism_path,
        [
            {
                "arm_id": identifier,
                **{
                    metric: mechanism_summary[identifier][metric]["mean"]
                    for metric in MECHANISM_METRICS
                },
            }
            for identifier in mechanism_order
        ],
        ["arm_id", *MECHANISM_METRICS],
    )
    held_path = output_dir / "held_reaction_family_metrics.csv"
    held_metrics = ("exact_l1_per_1000", "decomposition_coverage", "replay_precision")
    write_csv(
        held_path,
        [
            {
                "held_family": identifier,
                **{metric: held_summary[identifier][metric]["mean"] for metric in held_metrics},
            }
            for identifier in sorted(held_summary)
        ],
        ["held_family", *held_metrics],
    )
    summary_path = output_dir / "summary.json"
    result = {
        "schema_version": "forge.natbiotech_v1_rendered_results.v1",
        "status": "complete" if strict else "partial_nonpublication_preview",
        "config": pin_record(config_path, repo),
        "result_rows": pin_record(result_rows_path, repo),
        "bootstrap": {
            "seed": bootstrap_seed,
            "resamples": resamples,
            "independent_unit": "training_seed",
        },
        "common_methods": common_summary,
        "mechanism_arms": mechanism_summary,
        "held_reaction_families": held_summary,
        "held_reaction_family_is_secondary_not_hard_gate": True,
        "candidate_selection": False,
    }
    write_json(summary_path, result)
    artifacts = {
        path.name: artifact_record(path)
        for path in (
            output_dir / "common_ugi_benchmark_rows.tex",
            output_dir / "architecture_ablation_rows.tex",
            output_dir / "route_evidence_rows.tex",
            tradeoff_path,
            mechanism_path,
            held_path,
            summary_path,
        )
    }
    receipt = {**result, "artifacts": artifacts}
    write_json(output_dir / "result.json", receipt)
    return receipt


__all__ = [
    "COMMON_ROW_SCHEMA",
    "HELD_FAMILY_ROW_SCHEMA",
    "LEDGER_SCHEMA",
    "MECHANISM_ROW_SCHEMA",
    "PaperResultsV1Error",
    "common_assessment_seed_row",
    "held_family_seed_rows",
    "mechanism_seed_rows",
    "render_v1_results",
    "write_v1_result_rows",
]
