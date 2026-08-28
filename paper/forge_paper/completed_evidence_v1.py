"""Render manuscript fragments from already completed, hash-pinned FORGE evidence.

This renderer is deliberately separate from the strict cross-method renderer.  It may expose only
results that are already complete under their native frozen contracts.  It never fabricates a
missing external method, promotes missing evidence to a negative synthesis result, or turns an
optional held-family experiment into a paper gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_csv, write_json

CONFIG_SCHEMA = "forge.natbiotech_v1_completed_evidence_config.v1"
RESULT_SCHEMA = "forge.natbiotech_v1_completed_evidence_render.v1"

PROGRAMS = (
    "ugi_3cr_agile",
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)
PROGRAM_NAMES = {
    "ugi_3cr_agile": "Ugi",
    "bl_2023_repeated_aza_michael": "Aza-Michael",
    "lx_2024_repeated_reductive_amination": "Reductive amination",
}
INTERNAL_COMMON_METHOD_ORDER = (
    "finite_catalogue_oracle",
    "shared_null_posthoc",
    "fact_matched",
    "fact_generous",
    "forge_transformer",
)
REACTION_SPACE_EXTERNAL_METHOD_ORDER = ("rgfn",)
GENERIC_EXTERNAL_METHOD_ORDER = ("defog_unconditional", "genmol_safe")
FORMULATION_METHOD_ORDER = ("learned_inventory_selector",)
COMMON_METHOD_ORDER = (
    *INTERNAL_COMMON_METHOD_ORDER,
    *REACTION_SPACE_EXTERNAL_METHOD_ORDER,
    *GENERIC_EXTERNAL_METHOD_ORDER,
    *FORMULATION_METHOD_ORDER,
)
COMMON_METHOD_NAMES = {
    "finite_catalogue_oracle": "Finite catalogue oracle",
    "shared_null_posthoc": "Shared-null + post-hoc",
    "fact_matched": "FACT-matched",
    "fact_generous": "FACT-generous",
    "forge_transformer": "FORGE Transformer",
    "rgfn": "RGFN",
    "defog_unconditional": "DeFoG unconditional",
    "genmol_safe": "GenMol/SAFE",
    "learned_inventory_selector": "Learned inventory selector",
}
NATIVE_EXTERNAL_METHOD_ORDER = (
    *REACTION_SPACE_EXTERNAL_METHOD_ORDER,
    *GENERIC_EXTERNAL_METHOD_ORDER,
)
PAPER_COMMON_METHOD_ORDER = (
    *NATIVE_EXTERNAL_METHOD_ORDER,
    *FORMULATION_METHOD_ORDER,
    *INTERNAL_COMMON_METHOD_ORDER,
)
MECHANISM_NAMES = {
    "input_only_program": "Input-only program",
    "no_role_loss": "No role loss",
    "no_core_loss": "No core loss",
    "no_routed_adapters": "No routed adapters",
    "no_gradient_conflict_control": "No gradient conflict control",
    "fact_matched": "FACT-matched",
    "fact_generous": "FACT-generous",
    "full_transformer": "FORGE Transformer",
}


class CompletedEvidenceV1Error(ValueError):
    """Completed evidence is missing, changed, or incompatible with manuscript rendering."""


def _highlight_forge_rows(text: str) -> str:
    """Tint the rows reporting our own model so it is findable in a dense comparison.

    Applied to every ``*_rows.tex`` artifact so the highlight cannot drift between tables. The
    method name is the first cell in most tables and the second where a program is reported
    first, so both are checked. Requires ``colortbl`` and a ``forgerow`` colour in the preamble.
    """

    lines = []
    for line in text.split("\n"):
        cells = line.split("&")
        if not any(cell.strip().startswith("FORGE") for cell in cells[:2]):
            lines.append(line)
            continue
        # \cellcolor rather than \rowcolor: \rowcolor expands to \noalign, and these files are
        # \input immediately after an \hline, whose lookahead pulls the first token of the file
        # and then rejects it as a misplaced \noalign. \cellcolor is safe anywhere in a row.
        body, _, tail = line.rpartition(r"\\")
        painted = "&".join(r"\cellcolor{forgerow}" + cell for cell in body.split("&"))
        lines.append(painted + r"\\" + tail)
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    if path.name.endswith("_rows.tex"):
        text = _highlight_forge_rows(text)
    atomic_write(path, text.encode("utf-8"))


def _load_pinned_json(pin: object, repo: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
        raise CompletedEvidenceV1Error(f"{label} must be an exact path/SHA-256 pin")
    path = resolve_pin(pin, repo, label=label)
    return path, read_json_object(path, error=CompletedEvidenceV1Error, label=label)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompletedEvidenceV1Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CompletedEvidenceV1Error(f"{label} must be finite")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompletedEvidenceV1Error(f"{label} must be a non-negative integer")
    return value


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise CompletedEvidenceV1Error("cannot summarize an empty or non-vector metric")
    return float(vector.mean()), float(vector.std(ddof=1)) if len(vector) > 1 else 0.0


def _tex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def _mean_sd_tex(values: Sequence[float], *, digits: int = 1) -> str:
    mean, sd = _mean_sd(values)
    return rf"${mean:.{digits}f}\pm{sd:.{digits}f}$"


def _optional_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label=label)


def _optional_mean_sd_tex(
    values: Sequence[float | None], *, digits: int = 1, scale: float = 1.0
) -> str:
    scaled = []
    for value in values:
        if value is None:
            return "N/E"
        scaled.append(scale * value)
    return _mean_sd_tex(scaled, digits=digits)


def _metric_summary(row: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    value = row.get(metric)
    if not isinstance(value, Mapping):
        raise CompletedEvidenceV1Error(f"production metric is missing: {metric}")
    by_seed = value.get("by_seed")
    if not isinstance(by_seed, list) or len(by_seed) != 3:
        raise CompletedEvidenceV1Error(f"production metric lacks three seeds: {metric}")
    return value


def _production_row(program: str, arm: str, metrics: Mapping[str, Any]) -> str:
    def mean(name: str) -> float:
        return _number(_metric_summary(metrics, name)["mean"], label=f"{arm}.{program}.{name}")

    exact = _metric_summary(metrics, "exact_l1_yield_per_attempt")
    exact_mean = 100.0 * _number(exact["mean"], label="exact mean")
    exact_sd = 100.0 * _number(exact["sample_standard_deviation"], label="exact sd")
    cells = (
        PROGRAM_NAMES[program],
        arm,
        f"{100.0 * mean('raw_valid_fraction'):.1f}/{100.0 * mean('connected_fraction'):.1f}",
        f"{100.0 * mean('exact_l1_decomposition_coverage'):.1f}/"
        f"{100.0 * mean('exact_forward_replay_precision'):.1f}",
        f"{100.0 * mean('decomposition_abstention_fraction'):.1f}/"
        f"{100.0 * mean('decomposition_ambiguity_fraction'):.1f}",
        rf"${exact_mean:.1f}\pm{exact_sd:.1f}$",
        f"{100.0 * mean('whole_lipid_novelty_fraction'):.1f}/"
        f"{100.0 * mean('component_novelty_fraction'):.1f}",
        f"{mean('internal_diversity'):.3f}/{mean('effective_component_count'):.1f}",
    )
    return (
        " & ".join(_tex_escape(value) if index < 2 else value for index, value in enumerate(cells))
        + r" \\"
    )


def _catalogue_row(
    program: str, method: str, metrics: Mapping[str, Any], *, catalogue: bool
) -> str:
    def mean(name: str) -> float:
        return _number(_metric_summary(metrics, name)["mean"], label=f"{method}.{program}.{name}")

    whole_name = "whole_product_novelty_fraction" if catalogue else "whole_lipid_novelty_fraction"
    cells = (
        PROGRAM_NAMES[program],
        method,
        f"{1000.0 * mean('raw_valid_fraction'):.1f}",
        f"{1000.0 * mean('exact_l1_yield_per_attempt'):.1f}",
        f"{mean('unique_exact_l1_products_per_1000_attempts'):.1f}",
        f"{mean('unique_open_ended_exact_l1_products_per_1000_attempts'):.1f}",
        f"{100.0 * mean(whole_name):.1f}",
        f"{100.0 * mean('component_novelty_fraction'):.1f}",
        f"{mean('internal_diversity'):.3f}/{mean('effective_component_count'):.1f}",
    )
    return (
        " & ".join(_tex_escape(value) if index < 2 else value for index, value in enumerate(cells))
        + r" \\"
    )


def _load_common_assessments(raw: object, repo: Path, expected_seeds: Sequence[int]) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    if not isinstance(raw, Mapping) or set(raw) != set(COMMON_METHOD_ORDER):
        raise CompletedEvidenceV1Error("common-assessment methods changed")
    output: dict[str, list[dict[str, Any]]] = {}
    payloads: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    expected_method_ids = {
        "shared_null_posthoc": "shared_three_program_null",
        **{method: method for method in COMMON_METHOD_ORDER if method != "shared_null_posthoc"},
    }
    for method in COMMON_METHOD_ORDER:
        pins = raw[method]
        if not isinstance(pins, list) or len(pins) != len(expected_seeds):
            raise CompletedEvidenceV1Error(f"{method} must define one assessment per seed")
        rows: list[dict[str, Any]] = []
        method_payloads: list[dict[str, Any]] = []
        for pin in pins:
            path, result = _load_pinned_json(pin, repo, label=f"common assessment {method}")
            expected_schema = (
                "forge.learned_inventory_selector_result.v1"
                if method == "learned_inventory_selector"
                else "forge.common_ugi_complete_assessment.v1"
            )
            if (
                result.get("schema_version") != expected_schema
                or result.get("status") != "pass"
                or result.get("candidate_selection") is not False
            ):
                raise CompletedEvidenceV1Error(f"{method} assessment is not passing")
            common = result.get("common_assessment")
            route = result.get("route_evidence_assessment")
            if not isinstance(common, Mapping) or not isinstance(route, Mapping):
                raise CompletedEvidenceV1Error(f"{method} common assessment is incomplete")
            if common.get("method_id") != expected_method_ids[method]:
                raise CompletedEvidenceV1Error(f"{method} assessment changed method identity")
            if route.get("route_or_oracle_calls") != 0:
                raise CompletedEvidenceV1Error(f"{method} assessment made route or oracle calls")
            rows.append(dict(common))
            method_payloads.append(result)
            sources.append(pin_record(path, repo))
        paired = sorted(
            zip(rows, method_payloads, strict=True), key=lambda item: int(item[0]["seed"])
        )
        rows = [row for row, _ in paired]
        method_payloads = [payload for _, payload in paired]
        if [int(row["seed"]) for row in rows] != list(expected_seeds):
            raise CompletedEvidenceV1Error(f"{method} changed the paired seed set")
        output[method] = rows
        payloads[method] = method_payloads
    return output, payloads, sources


def _common_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    output = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise CompletedEvidenceV1Error("common-assessment metrics are missing")
        output.append(_number(metrics[name], label=f"common.{name}"))
    return output


def _common_optional_values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float | None]:
    output = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or name not in metrics:
            raise CompletedEvidenceV1Error(f"common-assessment metric is missing: {name}")
        output.append(_optional_number(metrics[name], label=f"common.{name}"))
    return output


def _common_mean_row(method: str, rows: Sequence[Mapping[str, Any]]) -> str:
    names = (
        "valid_products_per_1000_attempts",
        "exact_l1_products_per_1000_attempts",
        "unique_exact_l1_products_per_1000_attempts",
        "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts",
        "held_component_exact_l1_products_per_1000_attempts",
        "mean_pairwise_ecfp4_distance",
    )
    metric_cells = [
        _optional_mean_sd_tex(
            _common_optional_values(rows, name),
            digits=3 if name.endswith("distance") else 1,
        )
        for name in names
    ]
    if method in {"finite_catalogue_oracle", "rgfn", "learned_inventory_selector"}:
        open_ended = _common_values(
            rows, "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts"
        )
        if any(value != 0.0 for value in open_ended):
            raise CompletedEvidenceV1Error(
                f"finite method escaped its component inventory: {method}"
            )
        metric_cells[3] = r"$0.0\pm0.0^{\dagger}$"
    cells = [COMMON_METHOD_NAMES[method], *metric_cells]
    return " & ".join([_tex_escape(cells[0]), *cells[1:]]) + r" \\"


def _common_seed_rows(method: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = (
        "valid_products_per_1000_attempts",
        "exact_l1_products_per_1000_attempts",
        "unique_exact_l1_products_per_1000_attempts",
        "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts",
        "held_component_exact_l1_products_per_1000_attempts",
        "mean_pairwise_ecfp4_distance",
    )
    lines = []
    for row in rows:
        metrics = row["metrics"]
        values = []
        for name in names:
            value = _optional_number(metrics[name], label=name)
            values.append(
                "N/E"
                if value is None
                else f"{value:.3f}" if name.endswith("distance") else f"{value:.1f}"
            )
        lines.append(
            " & ".join([_tex_escape(COMMON_METHOD_NAMES[method]), str(row["seed"]), *values])
            + r" \\"
        )
    return lines


def _common_decomposition_row(method: str, rows: Sequence[Mapping[str, Any]]) -> str:
    valid = _common_values(rows, "valid")
    coverage = _common_optional_values(rows, "retro_decomposition_coverage_among_valid")
    precision = _common_optional_values(rows, "retro_transform_precision")
    abstention = [
        1.0 - value / valid[index] if valid[index] else None
        for index, value in enumerate(_common_values(rows, "exact_l1_program"))
    ]
    ambiguity = [
        value / valid[index] if valid[index] else None
        for index, value in enumerate(_common_values(rows, "ambiguous_exact_decompositions"))
    ]
    cells = (
        COMMON_METHOD_NAMES[method],
        _mean_sd_tex(valid, digits=1),
        _optional_mean_sd_tex(coverage, digits=1, scale=100.0),
        _optional_mean_sd_tex(precision, digits=1, scale=100.0),
        _optional_mean_sd_tex(abstention, digits=1, scale=100.0),
        _optional_mean_sd_tex(ambiguity, digits=1, scale=100.0),
    )
    return " & ".join([_tex_escape(cells[0]), *cells[1:]]) + r" \\"


def _load_mechanism_training(
    raw: object, repo: Path, expected_seeds: Sequence[int]
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list) or len(raw) != len(expected_seeds):
        raise CompletedEvidenceV1Error("mechanism training results changed")
    by_seed: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for pin in raw:
        path, result = _load_pinned_json(pin, repo, label="mechanism training result")
        if result.get("status") != "pass" or result.get("candidate_selection") is True:
            # Candidate selection is stored per arm in this schema, so only an explicit true fails.
            raise CompletedEvidenceV1Error("mechanism training result is not passing")
        if result.get("schema_version") != "forge.synthesis_program_production_training_result.v1":
            raise CompletedEvidenceV1Error("mechanism training schema changed")
        by_seed.append(result)
        sources.append(pin_record(path, repo))
    by_seed.sort(key=lambda result: int(result["seed"]))
    if [int(result["seed"]) for result in by_seed] != list(expected_seeds):
        raise CompletedEvidenceV1Error("mechanism training changed the paired seed set")
    arms = by_seed[0].get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(MECHANISM_NAMES):
        raise CompletedEvidenceV1Error("mechanism training arms changed")
    for result in by_seed:
        current = result.get("arms")
        if not isinstance(current, Mapping) or set(current) != set(arms):
            raise CompletedEvidenceV1Error("mechanism training arms differ across seeds")
        for arm_id in arms:
            fields = current[arm_id]
            reference = arms[arm_id]
            for name in ("parameter_count", "optimizer_steps", "route_calls", "oracle_calls"):
                if fields.get(name) != reference.get(name):
                    raise CompletedEvidenceV1Error(f"{arm_id}.{name} differs across seeds")
            if fields.get("candidate_selection") is not False:
                raise CompletedEvidenceV1Error(f"{arm_id} performed candidate selection")
    return dict(arms), sources


def _mechanism_compute_rows(arms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    for arm_id in MECHANISM_NAMES:
        arm = arms[arm_id]
        cells = (
            MECHANISM_NAMES[arm_id],
            f"{_integer(arm['parameter_count'], label='parameter_count'):,}",
            f"{_integer(arm['optimizer_steps'], label='optimizer_steps'):,}",
            "N/R",
            "N/R",
            r"$3\times3{,}072$",
            "N/R",
            str(_integer(arm["route_calls"], label="route_calls")),
            str(_integer(arm["oracle_calls"], label="oracle_calls")),
        )
        lines.append(" & ".join([_tex_escape(cells[0]), *cells[1:]]) + r" \\ ")
    return lines


def _load_native_compute_requests(
    raw: object, repo: Path, expected_seeds: Sequence[int]
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, Mapping) or set(raw) != set(NATIVE_EXTERNAL_METHOD_ORDER):
        raise CompletedEvidenceV1Error("native compute-request methods changed")
    output: dict[str, Mapping[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for method in NATIVE_EXTERNAL_METHOD_ORDER:
        pins = raw[method]
        if not isinstance(pins, list) or len(pins) != len(expected_seeds):
            raise CompletedEvidenceV1Error(f"{method} must define one native request per seed")
        requests: list[dict[str, Any]] = []
        for pin in pins:
            path, request = _load_pinned_json(pin, repo, label=f"native request {method}")
            if (
                request.get("schema_version") != "forge.external_ugi_native_request.v1"
                or request.get("profile") != "full"
                or request.get("requested_attempts") != 3072
                or request.get("method", {}).get("method_id") != method
                or request.get("parameters", {}).get("candidate_selection") is not False
            ):
                raise CompletedEvidenceV1Error(f"{method} native request changed its full contract")
            requests.append(request)
            sources.append(pin_record(path, repo))
        requests.sort(key=lambda request: int(request["seed"]))
        if [int(request["seed"]) for request in requests] != list(expected_seeds):
            raise CompletedEvidenceV1Error(f"{method} native requests changed the paired seed set")
        parameters = requests[0].get("parameters")
        if not isinstance(parameters, Mapping) or any(
            request.get("parameters") != parameters for request in requests[1:]
        ):
            raise CompletedEvidenceV1Error(f"{method} native parameters differ across seeds")
        output[method] = parameters
    return output, sources


def _common_call_cell(rows: Sequence[Mapping[str, Any]], name: str) -> str:
    values = []
    for row in rows:
        calls = row.get("calls")
        if not isinstance(calls, Mapping):
            raise CompletedEvidenceV1Error("common-assessment call accounting is missing")
        values.append(_integer(calls[name], label=f"common.calls.{name}"))
    if len(set(values)) != 1:
        return f"{sum(values):,}"
    value = values[0]
    return "0" if value == 0 else rf"$3\times{value:,}$".replace(",", "{,}")


def _external_compute_rows(
    common: Mapping[str, Sequence[Mapping[str, Any]]],
    payloads: Mapping[str, Sequence[Mapping[str, Any]]],
    native_parameters: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines = []
    for method in NATIVE_EXTERNAL_METHOD_ORDER:
        parameters = native_parameters[method]
        if method == "rgfn":
            train_steps = (
                f"{_integer(parameters['training_iterations'], label='iterations'):,} iter."
            )
            sample_steps = "N/R"
        else:
            train_steps = f"{_integer(parameters['training_steps'], label='training_steps'):,}"
            sample_steps = (
                f"{_integer(parameters['sampling_steps'], label='sampling_steps'):,}"
                if method == "defog_unconditional"
                else "N/R"
            )
        cells = (
            COMMON_METHOD_NAMES[method],
            "N/R",
            train_steps,
            "N/R",
            sample_steps,
            r"$3\times3{,}072$",
            _common_call_cell(common[method], "reaction_calls"),
            _common_call_cell(common[method], "route_calls"),
            _common_call_cell(common[method], "oracle_calls"),
        )
        lines.append(" & ".join([_tex_escape(cells[0]), *cells[1:]]) + r" \\")

    selector_payloads = payloads["learned_inventory_selector"]
    training = selector_payloads[0].get("training")
    if not isinstance(training, Mapping) or any(
        payload.get("training", {}).get("parameter_count") != training.get("parameter_count")
        or payload.get("training", {}).get("steps") != training.get("steps")
        for payload in selector_payloads[1:]
    ):
        raise CompletedEvidenceV1Error("learned-selector training metadata differ across seeds")
    selector_cells = (
        COMMON_METHOD_NAMES["learned_inventory_selector"],
        f"{_integer(training['parameter_count'], label='selector parameter_count'):,}",
        f"{_integer(training['steps'], label='selector steps'):,}",
        "N/R",
        "N/P",
        r"$3\times3{,}072$",
        _common_call_cell(common["learned_inventory_selector"], "reaction_calls"),
        _common_call_cell(common["learned_inventory_selector"], "route_calls"),
        _common_call_cell(common["learned_inventory_selector"], "oracle_calls"),
    )
    lines.append(" & ".join([_tex_escape(selector_cells[0]), *selector_cells[1:]]) + r" \\")
    return lines


def _load_method_blind_route_adjudication(
    pin: object,
    repo: Path,
    expected_seeds: Sequence[int],
    common_payloads: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    path, result = _load_pinned_json(pin, repo, label="method-blind route adjudication")
    required_gates = {
        "every_union_component_dispositioned": True,
        "no_out_of_union_evidence": True,
        "same_evidence_index_for_every_method": True,
        "planner_calls_during_scoring_zero": True,
        "synthesis_success_probability_claim_absent": True,
    }
    if (
        result.get("schema_version") != "forge.common_ugi_method_blind_route_adjudication.v1"
        or result.get("status") != "pass"
        or result.get("candidate_selection") is not False
        or result.get("gates") != required_gates
    ):
        raise CompletedEvidenceV1Error("method-blind route adjudication is not passing")
    rows = result.get("method_seed_results")
    if not isinstance(rows, list):
        raise CompletedEvidenceV1Error("method-blind route adjudication rows are missing")
    expected_method_ids = {
        "shared_null_posthoc": "shared_three_program_null",
        **{method: method for method in COMMON_METHOD_ORDER if method != "shared_null_posthoc"},
    }
    internal_method_ids = {value: key for key, value in expected_method_ids.items()}
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in COMMON_METHOD_ORDER}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CompletedEvidenceV1Error("method-blind route row is malformed")
        internal = raw.get("method_id")
        method = internal_method_ids.get(str(internal))
        if method is None:
            raise CompletedEvidenceV1Error(f"unexpected method-blind route method: {internal}")
        seed = _integer(raw.get("seed"), label=f"route.{method}.seed")
        route = raw.get("route_evidence_assessment")
        if (
            not isinstance(route, Mapping)
            or route.get("schema_version") != "forge.common_ugi_route_evidence_assessment.v1"
            or route.get("attempts") != 3072
            or route.get("route_or_oracle_calls") != 0
            or route.get("assessment_mode")
            != "exact_role_constitution_lookup_in_frozen_method_blind_union"
            or route.get("evidence_scope", {}).get("method_blind_cross_method_union") is not True
            or route.get("evidence_scope", {}).get("synthesis_success_comparison_authorized")
            is not False
        ):
            raise CompletedEvidenceV1Error(f"method-blind route row is inadmissible: {method}")
        matching_payload = next(
            (
                payload
                for payload in common_payloads[method]
                if int(payload["common_assessment"]["seed"]) == seed
            ),
            None,
        )
        if matching_payload is None or raw.get("source_sha256") != matching_payload.get(
            "assessed_attempts", {}
        ).get("sha256"):
            raise CompletedEvidenceV1Error(
                f"method-blind route source changed from common assessment: {method}/{seed}"
            )
        grouped[method].append(dict(route, seed=seed))
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in method_rows] != list(expected_seeds):
            raise CompletedEvidenceV1Error(f"method-blind route seed set changed: {method}")
    return grouped, result, pin_record(path, repo)


def _route_evidence_rows(
    common: Mapping[str, Sequence[Mapping[str, Any]]],
    routes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    def exact(method: str) -> str:
        return _optional_mean_sd_tex(
            _common_optional_values(common[method], "exact_l1_products_per_1000_attempts"),
            digits=1,
        )

    def route_metric(method: str, name: str) -> str:
        values = [
            1000.0
            * _number(row[name], label=f"route.{method}.{name}")
            / _number(row["attempts"], label=f"route.{method}.attempts")
            for row in routes[method]
        ]
        return _mean_sd_tex(values, digits=1)

    def row(method: str, identity: str, finite: str) -> tuple[str, ...]:
        return (
            COMMON_METHOD_NAMES[method],
            identity,
            finite,
            exact(method),
            route_metric(method, "verified_upstream"),
            route_metric(method, "terminal_evidence"),
            route_metric(method, "complete_dossier"),
            route_metric(method, "abstentions"),
        )

    rows = (
        row("rgfn", r"reaction $\times$ building block", "yes"),
        row("defog_unconditional", "generated atoms and bonds", "no"),
        row("genmol_safe", "generated fragment sequence", "no"),
        row("finite_catalogue_oracle", "train components", "yes"),
        row("learned_inventory_selector", "selected train components", "yes"),
        row("shared_null_posthoc", "generated atoms and bonds", "no"),
        row(
            "fact_matched",
            "role-isolated denoising on shared count-only layout",
            "no",
        ),
        row("forge_transformer", "generated atoms and bonds", "no"),
    )
    return [
        " & ".join(
            _tex_escape(value) if index < 3 and "\\" not in value else value
            for index, value in enumerate(row)
        )
        + r" \\"
        for row in rows
    ]


def _load_lipid_realism(pin: object, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path, result = _load_pinned_json(pin, repo, label="lipid realism aggregate")
    if (
        result.get("schema_version") != "forge.common_lipid_realism_aggregate.v1"
        or result.get("status") != "pass"
        or result.get("candidate_selection") is not False
    ):
        raise CompletedEvidenceV1Error("lipid-realism aggregate is not passing")
    methods = result.get("methods")
    expected = {
        "finite_catalogue_oracle",
        "forge_transformer",
        "genmol_safe",
        "learned_inventory_selector",
        "rgfn",
        "defog_unconditional",
    }
    if not isinstance(methods, Mapping) or set(methods) != expected:
        raise CompletedEvidenceV1Error("lipid-realism method set changed")
    return result, pin_record(path, repo)


def _realism_summary_tex(
    metrics: Mapping[str, Any], name: str, *, digits: int, scale: float = 1.0
) -> str:
    summary = metrics.get(name)
    if not isinstance(summary, Mapping):
        raise CompletedEvidenceV1Error(f"lipid-realism metric is missing: {name}")
    if summary.get("status") == "not_estimable":
        return "N/E"
    if summary.get("status") != "estimated":
        raise CompletedEvidenceV1Error(f"lipid-realism metric has invalid status: {name}")
    mean = scale * _number(summary["mean"], label=f"realism.{name}.mean")
    sd = scale * _number(
        summary["sample_standard_deviation"],
        label=f"realism.{name}.sample_standard_deviation",
    )
    return rf"${mean:.{digits}f}\pm{sd:.{digits}f}$"


def _internal_common_lines(common: Mapping[str, Any]) -> list[str]:
    """Internal comparison rows, with a rule set immediately before our own model.

    FORMULATION_METHOD_ORDER holds only the learned selector, so a rule placed between the two
    orders splits the matched controls rather than setting FORGE apart. The rule is anchored to
    the FORGE row itself so it stays correct if either order changes.
    """

    lines: list[str] = []
    for method in INTERNAL_COMMON_METHOD_ORDER:
        if method == "forge_transformer":
            lines.append(r"\midrule")
        lines.append(_common_mean_row(method, common[method]))
    return lines


def _lipid_realism_rows(result: Mapping[str, Any]) -> list[str]:
    order = (
        "rgfn",
        "defog_unconditional",
        "genmol_safe",
        "finite_catalogue_oracle",
        "learned_inventory_selector",
        "forge_transformer",
    )
    lines = []
    for method in order:
        metrics = result["methods"][method]["metrics"]
        cells = (
            COMMON_METHOD_NAMES[method],
            # Six metrics, not nine. within_declared_support tracked connected_fraction to within a
            # rounding step in every row but the catalogue oracle, and normalized_descriptor_
            # wasserstein and unique_fraction_among_connected are cited nowhere in the manuscript.
            # Ten columns forced \resizebox to shrink the table past legibility; these three were
            # the ones carrying no argument. They remain in the result mapping and can be restored
            # here if a later claim needs them.
            _realism_summary_tex(metrics, "connected_fraction_per_attempt", digits=1, scale=1000.0),
            _realism_summary_tex(
                metrics, "fingerprint_manifold_precision_per_attempt", digits=1, scale=1000.0
            )
            + "/"
            + _realism_summary_tex(metrics, "fingerprint_manifold_coverage", digits=2, scale=100.0),
            _realism_summary_tex(
                metrics, "descriptor_manifold_precision_per_attempt", digits=1, scale=1000.0
            )
            + "/"
            + _realism_summary_tex(metrics, "descriptor_manifold_coverage", digits=2, scale=100.0),
            _realism_summary_tex(metrics, "grouped_c2st_auc", digits=3),
            _realism_summary_tex(metrics, "effective_molecule_count", digits=1),
            _realism_summary_tex(metrics, "internal_diversity", digits=3),
        )
        lines.append(" & ".join([_tex_escape(cells[0]), *cells[1:]]) + r" \\")
    return lines


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def render_completed_evidence_v1(
    config_path: Path, repo: Path, output_dir: Path, *, result_path: Path | None = None
) -> dict[str, Any]:
    """Render all currently completed paper evidence without filling missing methods."""

    config = read_json_object(
        config_path, error=CompletedEvidenceV1Error, label="completed-evidence config"
    )
    expected_fields = {
        "schema_version",
        "completed_results",
        "common_assessments",
        "method_blind_route_adjudication",
        "method_blind_route_evidence",
        "native_compute_requests",
        "lipid_realism_aggregate",
        "mechanism_training_results",
        "mechanism_seed_rows",
        "expected_seeds",
        "candidate_selection",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != expected_fields:
        raise CompletedEvidenceV1Error("completed-evidence config schema or fields changed")
    if config.get("candidate_selection") is not False:
        raise CompletedEvidenceV1Error("completed-evidence rendering cannot select candidates")
    expected_seeds = config.get("expected_seeds")
    if expected_seeds != [20260825, 20260826, 20260827]:
        raise CompletedEvidenceV1Error("completed-evidence seed set changed")

    completed = config.get("completed_results")
    expected_completed = {
        "production",
        "semantic_intervention",
        "ugi_novelty",
        "role_novelty",
        "route_cascade",
        "synthesis_guidance",
        "potency_guidance",
        "sample_figure",
        "sample_atlas",
    }
    if not isinstance(completed, Mapping) or set(completed) != expected_completed:
        raise CompletedEvidenceV1Error("completed result set changed")
    loaded: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for name in sorted(completed):
        path, result = _load_pinned_json(completed[name], repo, label=name)
        loaded[name] = result
        sources.append(pin_record(path, repo))

    production = loaded["production"]
    if production.get("status") != "complete" or production.get("candidate_selection") is not False:
        raise CompletedEvidenceV1Error("production adjudication is not complete and nonselecting")
    arms = production.get("arm_summaries")
    catalogue = production.get("finite_component_catalogue_arm_summary")
    if not isinstance(arms, Mapping) or not isinstance(catalogue, Mapping):
        raise CompletedEvidenceV1Error("production summaries are missing")
    sample_figure = loaded["sample_figure"]
    if (
        sample_figure.get("schema_version") != "forge.paper.forge_generated_sample_figure.v2"
        or sample_figure.get("status") != "complete"
        or sample_figure.get("figure_mode") != "semantic_map_v2"
        or sample_figure.get("display_only") is not True
        or sample_figure.get("candidate_selection") is not False
        or sample_figure.get("selected_attempt_indices") != [2256, 1194, 2699]
    ):
        raise CompletedEvidenceV1Error("generated-sample figure is not admissible")
    sample_atlas = loaded["sample_atlas"]
    if (
        sample_atlas.get("schema_version") != "forge.paper.forge_generated_sample_atlas.v2"
        or sample_atlas.get("status") != "complete"
        or sample_atlas.get("row_count") != 12
        or sample_atlas.get("display_only") is not True
        or sample_atlas.get("candidate_selection") is not False
        or sample_atlas.get("selected_attempt_indices")
        != [2256, 1194, 2699, 2919, 329, 1776, 1861, 382, 2078, 2546, 1152, 1059]
    ):
        raise CompletedEvidenceV1Error("generated-sample atlas is not admissible")
    final_arm = arms.get("bl_core_constrained_repeat_aware")
    if not isinstance(final_arm, Mapping) or set(final_arm) != set(PROGRAMS):
        raise CompletedEvidenceV1Error("final production program set changed")

    common, common_payloads, common_sources = _load_common_assessments(
        config["common_assessments"], repo, expected_seeds
    )
    sources.extend(common_sources)
    method_blind_routes, route_union_result, route_union_source = (
        _load_method_blind_route_adjudication(
            config["method_blind_route_adjudication"],
            repo,
            expected_seeds,
            common_payloads,
        )
    )
    sources.append(route_union_source)
    route_evidence_path, route_evidence_build = _load_pinned_json(
        config["method_blind_route_evidence"], repo, label="method-blind route evidence"
    )
    route_evidence_gates = route_evidence_build.get("gates")
    route_dispositions = route_evidence_build.get("dispositions")
    expected_route_evidence_gates = {
        "every_union_component_dispositioned": True,
        "family_projection_cannot_close": True,
        "private_membership_read": False,
        "proposal_only_route_cannot_close": True,
        "public_worklist_only": True,
        "unknown_components_explicitly_abstain": True,
    }
    if (
        route_evidence_build.get("schema_version")
        != "forge.common_ugi_method_blind_route_evidence.v1"
        or route_evidence_build.get("status") != "complete_with_explicit_abstentions"
        or route_evidence_build.get("candidate_selection") is not False
        or route_evidence_gates != expected_route_evidence_gates
        or not isinstance(route_dispositions, Mapping)
        or sum(_integer(value, label="route disposition") for value in route_dispositions.values())
        != route_union_result.get("components")
        or route_evidence_build.get("component_evidence", {}).get("sha256")
        != route_union_result.get("evidence", {}).get("sha256")
    ):
        raise CompletedEvidenceV1Error("method-blind route evidence build is inadmissible")
    sources.append(pin_record(route_evidence_path, repo))
    native_parameters, native_sources = _load_native_compute_requests(
        config["native_compute_requests"], repo, expected_seeds
    )
    sources.extend(native_sources)
    lipid_realism, realism_source = _load_lipid_realism(config["lipid_realism_aggregate"], repo)
    sources.append(realism_source)
    mechanism_arms, training_sources = _load_mechanism_training(
        config["mechanism_training_results"], repo, expected_seeds
    )
    sources.extend(training_sources)
    seed_rows_path = resolve_pin(config["mechanism_seed_rows"], repo, label="mechanism seed rows")
    sources.append(pin_record(seed_rows_path, repo))
    full_held_values = []
    for record in iter_jsonl(seed_rows_path):
        if record.get("arm_id") == "full_transformer":
            full_held_values.append(
                _number(record["metrics"]["held_component_per_1000"], label="held component")
            )
    if len(full_held_values) != 3:
        raise CompletedEvidenceV1Error("full Transformer held-component rows are incomplete")

    semantic = loaded["semantic_intervention"]
    if semantic.get("status") != "complete":
        raise CompletedEvidenceV1Error("semantic intervention is not complete")
    interventions = {str(row["condition"]): row for row in semantic.get("interventions", [])}
    expected_interventions = {
        "mismatched_program__bl_2023_repeated_aza_michael",
        "mismatched_program__lx_2024_repeated_reductive_amination",
        "mismatched_roles__forward_cycle",
        "mismatched_roles__reverse_cycle",
        "null_all_program_coordinates",
    }
    if set(interventions) != expected_interventions:
        raise CompletedEvidenceV1Error("semantic interventions changed")

    route = loaded["route_cascade"]
    route_summary = route.get("summary")
    if route.get("status") != "bounded_hybrid_route_cascade_complete" or not isinstance(
        route_summary, Mapping
    ):
        raise CompletedEvidenceV1Error("route cascade is not complete")
    synthesis = loaded["synthesis_guidance"]
    synthesis_match = synthesis.get("matched_result")
    if not isinstance(synthesis_match, Mapping):
        raise CompletedEvidenceV1Error("synthesis-guidance matched result is missing")
    potency = loaded["potency_guidance"]
    potency_gate = potency.get("fresh_matched_terminal_gate")
    potency_signal = potency.get("preterminal_signal")
    if not isinstance(potency_gate, Mapping) or not isinstance(potency_signal, Mapping):
        raise CompletedEvidenceV1Error("potency-guidance result is incomplete")
    novelty = loaded["ugi_novelty"]
    role_novelty = loaded["role_novelty"]

    output_dir.mkdir(parents=True, exist_ok=True)
    production_lines = [
        _production_row("ugi_3cr_agile", "Ugi-only", arms["ugi_only_conditioned"]["ugi_3cr_agile"]),
        _production_row("ugi_3cr_agile", "FORGE conditioned", final_arm["ugi_3cr_agile"]),
        _production_row(
            "ugi_3cr_agile", "Shared null", arms["shared_three_program_null"]["ugi_3cr_agile"]
        ),
        _production_row(
            "ugi_3cr_agile",
            "Cyclic program",
            arms["shared_three_program_program_id_cyclic"]["ugi_3cr_agile"],
        ),
    ]
    for program in PROGRAMS[1:]:
        production_lines.append(r"\midrule")
        production_lines.extend(
            [
                _production_row(program, "FORGE conditioned", final_arm[program]),
                _production_row(program, "Shared null", arms["shared_three_program_null"][program]),
                _production_row(
                    program,
                    "Cyclic program",
                    arms["shared_three_program_program_id_cyclic"][program],
                ),
            ]
        )
    _write_text(output_dir / "production_comparison_rows.tex", "\n".join(production_lines) + "\n")

    def production_summary_tex(metrics: Mapping[str, Any], name: str, scale: float) -> str:
        values = [
            scale * _number(value, label=f"figure.{name}")
            for value in _metric_summary(metrics, name)["by_seed"]
        ]
        return _mean_sd_tex(values, digits=1)

    figure2_lines = []
    for program in PROGRAMS:
        coverage = _number(
            _metric_summary(final_arm[program], "exact_l1_decomposition_coverage")["mean"],
            label="figure coverage",
        )
        replay = _number(
            _metric_summary(final_arm[program], "exact_forward_replay_precision")["mean"],
            label="figure replay",
        )
        figure2_lines.append(
            " & ".join(
                (
                    PROGRAM_NAMES[program],
                    production_summary_tex(final_arm[program], "exact_l1_yield_per_attempt", 100.0),
                    production_summary_tex(
                        arms["shared_three_program_null"][program],
                        "exact_l1_yield_per_attempt",
                        100.0,
                    ),
                    production_summary_tex(
                        arms["shared_three_program_program_id_cyclic"][program],
                        "exact_l1_yield_per_attempt",
                        100.0,
                    ),
                    f"{100.0 * coverage:.1f}/{100.0 * replay:.1f}",
                )
            )
            + r" \\"
        )
    _write_text(output_dir / "shared_program_figure_rows.tex", "\n".join(figure2_lines) + "\n")

    catalogue_lines = []
    for index, program in enumerate(PROGRAMS):
        if index:
            catalogue_lines.append(r"\midrule")
        catalogue_lines.extend(
            [
                _catalogue_row(program, "FORGE", final_arm[program], catalogue=False),
                _catalogue_row(program, "Finite catalogue", catalogue[program], catalogue=True),
            ]
        )
    _write_text(output_dir / "catalogue_comparison_rows.tex", "\n".join(catalogue_lines) + "\n")
    figure3_lines = []
    for program in PROGRAMS:
        figure3_lines.append(
            " & ".join(
                (
                    PROGRAM_NAMES[program],
                    f"{1000.0 * _number(_metric_summary(final_arm[program], 'exact_l1_yield_per_attempt')['mean'], label='figure forge exact'):.1f}",
                    f"{1000.0 * _number(_metric_summary(catalogue[program], 'exact_l1_yield_per_attempt')['mean'], label='figure catalogue exact'):.1f}",
                    f"{_number(_metric_summary(final_arm[program], 'unique_open_ended_exact_l1_products_per_1000_attempts')['mean'], label='figure open ended'):.1f}",
                    "0.0",
                )
            )
            + r" \\"
        )
    _write_text(output_dir / "catalogue_figure_rows.tex", "\n".join(figure3_lines) + "\n")

    _write_text(
        output_dir / "common_ugi_completed_rows.tex",
        "\n".join(
            _common_mean_row(method, common[method]) for method in INTERNAL_COMMON_METHOD_ORDER
        )
        + "\n",
    )
    _write_text(
        output_dir / "common_ugi_reaction_external_rows.tex",
        "\n".join(
            _common_mean_row(method, common[method])
            for method in REACTION_SPACE_EXTERNAL_METHOD_ORDER
        )
        + "\n",
    )
    _write_text(
        output_dir / "common_ugi_generic_external_rows.tex",
        "\n".join(
            _common_mean_row(method, common[method]) for method in GENERIC_EXTERNAL_METHOD_ORDER
        )
        + "\n",
    )
    _write_text(
        output_dir / "common_ugi_formulation_rows.tex",
        "\n".join(_common_mean_row(method, common[method]) for method in FORMULATION_METHOD_ORDER)
        + "\n",
    )
    # A rule before each group, and before our own model so it reads as its own block. Never
    # before the first heading: that line is the first token of the \input file, and the \hline
    # preceding the \input would pull it into its lookahead and reject it as a misplaced \noalign.
    common_benchmark_lines = [
        r"\textbf{Reaction-space baselines} & & & & & & \\",
        *(
            _common_mean_row(method, common[method])
            for method in REACTION_SPACE_EXTERNAL_METHOD_ORDER
        ),
        r"\midrule",
        r"\textbf{Whole-molecule baselines} & & & & & & \\",
        *(_common_mean_row(method, common[method]) for method in GENERIC_EXTERNAL_METHOD_ORDER),
        r"\midrule",
        r"\textbf{Matched controls} & & & & & & \\",
        *(_common_mean_row(method, common[method]) for method in FORMULATION_METHOD_ORDER),
        *_internal_common_lines(common),
    ]
    _write_text(
        output_dir / "common_ugi_benchmark_completed_rows.tex",
        "\n".join(common_benchmark_lines) + "\n",
    )
    _write_text(
        output_dir / "common_ugi_seed_rows.tex",
        "\n".join(
            line
            for method in PAPER_COMMON_METHOD_ORDER
            for line in _common_seed_rows(method, common[method])
        )
        + "\n",
    )
    _write_text(
        output_dir / "common_ugi_decomposition_rows.tex",
        "\n".join(
            _common_decomposition_row(method, common[method])
            for method in PAPER_COMMON_METHOD_ORDER
        )
        + "\n",
    )
    _write_text(
        output_dir / "compute_parity_external_rows.tex",
        "\n".join(_external_compute_rows(common, common_payloads, native_parameters)) + "\n",
    )
    _write_text(
        output_dir / "compute_parity_completed_rows.tex",
        "\n".join(_mechanism_compute_rows(mechanism_arms)) + "\n",
    )
    _write_text(
        output_dir / "route_evidence_completed_rows.tex",
        "\n".join(_route_evidence_rows(common, method_blind_routes)) + "\n",
    )
    _write_text(
        output_dir / "lipid_realism_rows.tex",
        "\n".join(_lipid_realism_rows(lipid_realism)) + "\n",
    )

    ugi_catalogue = production["finite_component_catalogue_comparison"]["programs"][
        "ugi_3cr_agile"
    ]["catalogue"]
    role_counts = ugi_catalogue["role_component_counts"]
    fold_coverage = ugi_catalogue["source_product_coverage_by_fold"]

    def inventory_row(method: str) -> str:
        return (
            f"{method} & "
            f"{role_counts['amine_head']} & {role_counts['oxoester_aldehyde_body_tail']} & "
            f"{role_counts['isocyanide_tail']} & "
            f"{100.0 * fold_coverage['train']['coverage_fraction']:.1f} & "
            f"{100.0 * fold_coverage['calibration']['coverage_fraction']:.1f} & "
            f"{100.0 * fold_coverage['heldout']['coverage_fraction']:.1f} \\\\"
        )

    selector_inventories = [
        payload.get("inventory") for payload in common_payloads["learned_inventory_selector"]
    ]
    expected_inventory = {
        "amine_head": role_counts["amine_head"],
        "oxoester_aldehyde_body_tail": role_counts["oxoester_aldehyde_body_tail"],
        "isocyanide_tail": role_counts["isocyanide_tail"],
    }
    if any(inventory != expected_inventory for inventory in selector_inventories):
        raise CompletedEvidenceV1Error(
            "learned-selector inventory changed from the train catalogue"
        )
    if native_parameters["rgfn"].get("component_inventory") != "common_train_only":
        raise CompletedEvidenceV1Error("RGFN component inventory changed")
    inventory_lines = [
        inventory_row("RGFN"),
        "DeFoG unconditional & N/P & N/P & N/P & N/P & N/P & N/P \\\\ ",
        "GenMol/SAFE & N/P & N/P & N/P & N/P & N/P & N/P \\\\ ",
        inventory_row("Finite catalogue oracle"),
        inventory_row("Learned inventory selector"),
    ]
    _write_text(
        output_dir / "inventory_coverage_completed_rows.tex",
        "\n".join(inventory_lines) + "\n",
    )

    paired = production["paired_seed_comparisons"]
    retention_interval = paired["final_vs_ugi_only"]["ugi_3cr_agile"]["exact_l1_yield_per_attempt"][
        "paired_seed_difference_interval"
    ]

    def paired_values(comparison: str, program: str) -> tuple[float, float, float]:
        interval = paired[comparison][program]["exact_l1_yield_per_attempt"][
            "paired_seed_difference_interval"
        ]
        return (
            _number(interval["point_estimate"], label="paired point"),
            _number(interval["lower_bound"], label="paired low"),
            _number(interval["upper_bound"], label="paired high"),
        )

    null_values = {
        program: paired_values("final_vs_shared_null_posthoc", program) for program in PROGRAMS
    }
    cyclic_values = {
        program: paired_values("final_vs_cyclic_program_id", program) for program in PROGRAMS
    }
    semantic_rows = []
    for condition, row in interventions.items():
        low, high = row["paired_seed_95_interval"]
        semantic_rows.append(
            {
                "condition": condition,
                "exact_l1_yield": row["exact_l1_yield_mean"],
                "factual_minus_intervention": row["factual_minus_intervention_exact_l1"],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    write_csv(
        output_dir / "semantic_intervention.csv",
        semantic_rows,
        ["condition", "exact_l1_yield", "factual_minus_intervention", "ci95_low", "ci95_high"],
    )
    write_csv(
        output_dir / "production_metrics.csv",
        [
            {
                "program": program,
                "arm": arm_id,
                "exact_l1_yield": _metric_summary(metrics, "exact_l1_yield_per_attempt")["mean"],
                "valid_fraction": _metric_summary(metrics, "raw_valid_fraction")["mean"],
                "component_novelty_fraction": _metric_summary(
                    metrics, "component_novelty_fraction"
                )["mean"],
                "diversity": _metric_summary(metrics, "internal_diversity")["mean"],
            }
            for arm_id, program_rows in arms.items()
            for program, metrics in program_rows.items()
        ],
        [
            "program",
            "arm",
            "exact_l1_yield",
            "valid_fraction",
            "component_novelty_fraction",
            "diversity",
        ],
    )
    write_csv(
        output_dir / "catalogue_tradeoff.csv",
        [
            {
                "program": program,
                "method": method,
                "exact_l1_per_1000": 1000.0
                * _metric_summary(metrics, "exact_l1_yield_per_attempt")["mean"],
                "open_ended_per_1000": _metric_summary(
                    metrics, "unique_open_ended_exact_l1_products_per_1000_attempts"
                )["mean"],
                "diversity": _metric_summary(metrics, "internal_diversity")["mean"],
            }
            for program in PROGRAMS
            for method, metrics in (
                ("FORGE", final_arm[program]),
                ("Finite catalogue", catalogue[program]),
            )
        ],
        ["program", "method", "exact_l1_per_1000", "open_ended_per_1000", "diversity"],
    )

    route_states = route_summary["product_route_states"]
    component_states = route_summary["component_states"]
    potency_diff = 1000.0 * _number(
        potency_gate["authorized_tail_lane"]["difference_per_generator_call"],
        label="potency difference",
    )
    potency_ci = potency_gate["authorized_tail_lane"]["difference_ci95"]

    def route_per_thousand(method: str, name: str) -> list[float]:
        return [
            1000.0
            * _number(row[name], label=f"route macro.{method}.{name}")
            / _number(row["attempts"], label=f"route macro.{method}.attempts")
            for row in method_blind_routes[method]
        ]

    macros = {
        "ForgeUgiRetentionDifferencePP": f"{100.0 * retention_interval['point_estimate']:.2f}",
        "ForgeUgiRetentionCILowPP": f"{100.0 * retention_interval['lower_bound']:.2f}",
        "ForgeUgiRetentionCIHighPP": f"{100.0 * retention_interval['upper_bound']:.2f}",
        "ForgeHeldComponentPerThousand": f"{_mean_sd(full_held_values)[0]:.2f}",
        "ForgeHeldComponentTotal": f"{round(sum(value * 3072.0 / 1000.0 for value in full_held_values))}",
        "ForgeHeldComponentAttempts": "9{,}216",
        "ForgeNovelAgainstFullCount": f"{novelty['generator_novelty']['full_corpus']['absent']:,}".replace(
            ",", "{,}"
        ),
        "ForgeNovelAgainstFullDenominator": f"{novelty['generator_novelty']['full_corpus']['generated_distinct']:,}".replace(
            ",", "{,}"
        ),
        "ForgeNovelAgainstFullPercent": f"{100.0 * novelty['generator_novelty']['full_corpus']['novelty']:.1f}",
        "ForgeOffRegistryCount": f"{role_novelty['admitted']['main']['any_role_off_registry']['count']:,}".replace(
            ",", "{,}"
        ),
        "ForgeOffRegistryDenominator": f"{role_novelty['admitted']['main']['n_distinct_products']:,}".replace(
            ",", "{,}"
        ),
        "ForgeOffRegistryPercent": f"{100.0 * role_novelty['admitted']['main']['any_role_off_registry']['fraction']:.1f}",
        "ForgeRouteCandidates": str(route_summary["candidates"]),
        "ForgeRouteAdmitted": str(route_summary["route_assessment_admitted_candidates"]),
        "ForgeRouteExactLone": str(route_summary["candidates"]),
        "ForgeRouteCompleteProducts": str(route_states["complete"]),
        "ForgeRouteUnresolvedProducts": str(route_states["unresolved"]),
        "ForgeRouteCensoredProducts": str(route_states["search_censored"]),
        "ForgeRouteComponents": str(route_summary["components"]),
        "ForgeRouteCompleteComponents": str(component_states["complete"]),
        "ForgeRouteUnresolvedComponents": str(component_states["unresolved"]),
        "ForgeRouteCensoredComponents": str(component_states["search_censored"]),
        "ForgeCommonRouteUnionComponents": f"{route_union_result['components']:,}".replace(
            ",", "{,}"
        ),
        "ForgeCommonRouteEvidenceCompleteComponents": str(route_dispositions["complete"]),
        "ForgeCommonRouteForgeExactPerThousand": _mean_sd_tex(
            route_per_thousand("forge_transformer", "eligible_unique_exact_l1"), digits=1
        ),
        "ForgeCommonRouteForgeCompletePerThousand": _mean_sd_tex(
            route_per_thousand("forge_transformer", "complete_dossier"), digits=1
        ),
        "ForgeCommonRouteForgeAbstentionPerThousand": _mean_sd_tex(
            route_per_thousand("forge_transformer", "abstentions"), digits=1
        ),
        "ForgeCommonRouteSelectorCompletePerThousand": _mean_sd_tex(
            route_per_thousand("learned_inventory_selector", "complete_dossier"), digits=1
        ),
        "ForgeSynthesisGuidanceStrength": f"{synthesis_match['guidance_strength']:.2f}",
        "ForgeSynthesisTerminalParticles": str(synthesis_match["terminal_particles"]),
        "ForgeSynthesisCanonicalRepresentatives": str(
            synthesis_match["guided_canonical_representatives"]
        ),
        "ForgeSynthesisGuidedReady": str(synthesis_match["guided_unique_route_ready_final"]),
        "ForgeSynthesisPosthocReady": str(synthesis_match["post_hoc_unique_route_ready_final"]),
        "ForgeSynthesisChangedIdentities": str(synthesis_match["changed_terminal_identities"]),
        "ForgePotencyAUC": f"{potency_signal['group_disjoint_roc_auc']:.3f}",
        "ForgePotencyAUCCILow": f"{potency_signal['roc_auc_ci95'][0]:.3f}",
        "ForgePotencyAUCCIHigh": f"{potency_signal['roc_auc_ci95'][1]:.3f}",
        "ForgePotencyTopQuartileGain": f"{potency_signal['top_quartile_observed_gain']:.3f}",
        "ForgePotencyAttemptsPerArm": f"{potency_gate['generator_calls_per_arm']:,}".replace(
            ",", "{,}"
        ),
        "ForgePotencySupportExactLone": str(potency_gate["valid_exact_l1"]["support_enriched"]),
        "ForgePotencyNestedExactLone": str(potency_gate["valid_exact_l1"]["nested_potency"]),
        "ForgePotencySupportEligible": str(
            potency_gate["authorized_tail_lane"]["support_eligible_before_budget"]
        ),
        "ForgePotencyNestedEligible": str(
            potency_gate["authorized_tail_lane"]["potency_eligible_before_budget"]
        ),
        "ForgePotencyOracleBudget": str(potency_gate["equal_oracle_budget"]),
        "ForgePotencySupportHigh": str(
            potency_gate["authorized_tail_lane"]["support_unique_conservative_high"]
        ),
        "ForgePotencyNestedHigh": str(
            potency_gate["authorized_tail_lane"]["potency_unique_conservative_high"]
        ),
        "ForgePotencyDifferencePerThousand": f"{potency_diff:.2f}",
        "ForgePotencyDifferenceCILowPerThousand": f"{1000.0 * potency_ci[0]:.2f}",
        "ForgePotencyDifferenceCIHighPerThousand": f"{1000.0 * potency_ci[1]:.2f}",
    }
    for prefix, values in (("Null", null_values), ("Cyclic", cyclic_values)):
        for program, label in zip(PROGRAMS, ("Ugi", "BL", "LX"), strict=True):
            point, low, high = values[program]
            macros[f"Forge{prefix}{label}DifferencePP"] = f"{100.0 * point:.2f}"
            macros[f"Forge{prefix}{label}CILowPP"] = f"{100.0 * low:.2f}"
            macros[f"Forge{prefix}{label}CIHighPP"] = f"{100.0 * high:.2f}"
    factual = semantic["factual"]
    macros["ForgeSemanticFactualPercent"] = f"{100.0 * factual['exact_l1_yield_mean']:.2f}"
    for condition, label in (
        ("mismatched_program__bl_2023_repeated_aza_michael", "BLToken"),
        ("mismatched_program__lx_2024_repeated_reductive_amination", "LXToken"),
        ("mismatched_roles__forward_cycle", "ForwardRole"),
        ("mismatched_roles__reverse_cycle", "ReverseRole"),
        ("null_all_program_coordinates", "NullCoordinates"),
    ):
        row = interventions[condition]
        macros[f"ForgeSemantic{label}Percent"] = f"{100.0 * row['exact_l1_yield_mean']:.2f}"
        macros[f"ForgeSemantic{label}DifferencePP"] = (
            f"{100.0 * row['factual_minus_intervention_exact_l1']:.2f}"
        )
        macros[f"ForgeSemantic{label}CILowPP"] = f"{100.0 * row['paired_seed_95_interval'][0]:.2f}"
        macros[f"ForgeSemantic{label}CIHighPP"] = f"{100.0 * row['paired_seed_95_interval'][1]:.2f}"
    macro_path = output_dir / "completed_evidence_macros.tex"
    _write_text(macro_path, "\n".join(_macro(name, macros[name]) for name in sorted(macros)) + "\n")

    artifact_paths = [
        output_dir / "production_comparison_rows.tex",
        output_dir / "shared_program_figure_rows.tex",
        output_dir / "catalogue_comparison_rows.tex",
        output_dir / "catalogue_figure_rows.tex",
        output_dir / "common_ugi_completed_rows.tex",
        output_dir / "common_ugi_reaction_external_rows.tex",
        output_dir / "common_ugi_generic_external_rows.tex",
        output_dir / "common_ugi_formulation_rows.tex",
        output_dir / "common_ugi_benchmark_completed_rows.tex",
        output_dir / "common_ugi_seed_rows.tex",
        output_dir / "common_ugi_decomposition_rows.tex",
        output_dir / "compute_parity_external_rows.tex",
        output_dir / "compute_parity_completed_rows.tex",
        output_dir / "inventory_coverage_completed_rows.tex",
        output_dir / "route_evidence_completed_rows.tex",
        output_dir / "lipid_realism_rows.tex",
        output_dir / "semantic_intervention.csv",
        output_dir / "production_metrics.csv",
        output_dir / "catalogue_tradeoff.csv",
        macro_path,
    ]
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_for_currently_available_evidence",
        "config": pin_record(config_path, repo),
        "sources": sources,
        "available_common_methods": list(COMMON_METHOD_ORDER),
        "missing_common_methods": [],
        "method_blind_route_union_included": True,
        "held_reaction_family_included": False,
        "candidate_selection": False,
        "macros": macros,
    }
    write_json(output_dir / "completed_evidence_summary.json", summary)
    artifact_paths.append(output_dir / "completed_evidence_summary.json")
    receipt = {
        **summary,
        "artifacts": {path.name: artifact_record(path) for path in artifact_paths},
    }
    write_json(output_dir / "completed_evidence_result.json", receipt)
    if result_path is not None:
        write_json(result_path, receipt)
    return receipt


__all__ = ["CompletedEvidenceV1Error", "render_completed_evidence_v1"]
