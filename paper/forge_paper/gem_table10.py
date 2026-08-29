"""Render GEM Table 10 from the final method-blind common-Ugi route assessment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json
from forge_paper.completed_evidence_v1 import CONFIG_SCHEMA as COMPLETED_CONFIG_SCHEMA
from forge_paper.completed_evidence_v1 import (
    _common_optional_values,
    _highlight_forge_rows,
    _load_common_assessments,
    _load_method_blind_route_adjudication,
    _mean_sd_tex,
    _number,
    _route_evidence_rows,
)

CONFIG_SCHEMA = "forge.gem_table10_route_evidence_config.v1"
RESULT_SCHEMA = "forge.gem_table10_route_evidence_render.v1"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
METHOD_ORDER = (
    "rgfn",
    "defog_unconditional",
    "genmol_safe",
    "finite_catalogue_oracle",
    "learned_inventory_selector",
    "shared_null_posthoc",
    "fact_matched",
    "forge_transformer",
)


class GemTable10Error(ValueError):
    """The final common-Ugi route evidence is incomplete, stale, or inadmissible."""


def _per_thousand(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    return [
        1000.0
        * _number(row[name], label=f"route.{name}")
        / _number(row["attempts"], label="route.attempts")
        for row in rows
    ]


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_standard_deviation": stdev(values),
    }


def _renewcommand(name: str, value: str) -> str:
    return rf"\renewcommand{{\{name}}}{{{value}}}"


def render_gem_table10_route_evidence(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render Table 10 and its narrative macros from one final route adjudication."""

    config = read_json_object(config_path, error=GemTable10Error, label="GEM Table 10 config")
    expected_fields = {
        "schema_version",
        "status",
        "common_assessment_config",
        "route_adjudication",
        "route_evidence",
        "attempts_per_method_per_seed",
        "expected_seeds",
        "method_order",
        "candidate_selection",
    }
    if (
        set(config) != expected_fields
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_final_common_ugi_adjudication"
        or config.get("attempts_per_method_per_seed") != 3072
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or config.get("method_order") != list(METHOD_ORDER)
        or config.get("candidate_selection") is not False
    ):
        raise GemTable10Error("GEM Table 10 config changed")

    common_config_path = resolve_pin(
        config["common_assessment_config"],
        repo,
        label="GEM Table 10 common-assessment config",
    )
    common_config = read_json_object(
        common_config_path,
        error=GemTable10Error,
        label="GEM Table 10 common-assessment config",
    )
    common_pins = common_config.get("common_assessments")
    if (
        common_config.get("schema_version") != COMPLETED_CONFIG_SCHEMA
        or common_config.get("candidate_selection") is not False
        or common_config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or not isinstance(common_pins, Mapping)
    ):
        raise GemTable10Error("common-assessment config is incompatible")
    try:
        common, common_payloads, common_sources = _load_common_assessments(
            common_pins,
            repo,
            EXPECTED_SEEDS,
        )
        routes, adjudication, adjudication_source = _load_method_blind_route_adjudication(
            config["route_adjudication"],
            repo,
            EXPECTED_SEEDS,
            common_payloads,
        )
    except ValueError as error:
        raise GemTable10Error(str(error)) from error

    evidence_path = resolve_pin(config["route_evidence"], repo, label="GEM Table 10 evidence")
    evidence = read_json_object(evidence_path, error=GemTable10Error, label="GEM Table 10 evidence")
    evidence_gates = {
        "every_union_component_dispositioned": True,
        "family_projection_cannot_close": True,
        "private_membership_read": False,
        "proposal_only_route_cannot_close": True,
        "public_worklist_only": True,
        "unknown_components_explicitly_abstain": True,
    }
    dispositions = evidence.get("dispositions")
    if (
        evidence.get("schema_version") != "forge.common_ugi_method_blind_route_evidence.v1"
        or evidence.get("status") != "complete_with_explicit_abstentions"
        or evidence.get("candidate_selection") is not False
        or evidence.get("gates") != evidence_gates
        or not isinstance(dispositions, Mapping)
        or sum(int(value) for value in dispositions.values()) != adjudication.get("components")
        or evidence.get("component_evidence", {}).get("sha256")
        != adjudication.get("evidence", {}).get("sha256")
    ):
        raise GemTable10Error("method-blind route evidence is incompatible with adjudication")

    rows = _route_evidence_rows(common, routes)
    if len(rows) != len(METHOD_ORDER):
        raise GemTable10Error("Table 10 row count changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "route_evidence_completed_rows.tex"
    rendered_rows = _highlight_forge_rows("\n".join(rows) + "\n")
    atomic_write(row_path, rendered_rows.encode("utf-8"))

    forge_exact = _per_thousand(routes["forge_transformer"], "eligible_unique_exact_l1")
    forge_complete = _per_thousand(routes["forge_transformer"], "complete_dossier")
    forge_abstention = _per_thousand(routes["forge_transformer"], "abstentions")
    selector_complete = _per_thousand(
        routes["learned_inventory_selector"],
        "complete_dossier",
    )
    macros = {
        "ForgeCommonRouteUnionComponents": f"{int(adjudication['components']):,}".replace(
            ",", "{,}"
        ),
        "ForgeCommonRouteEvidenceCompleteComponents": str(int(dispositions["complete"])),
        "ForgeCommonRouteForgeExactPerThousand": _mean_sd_tex(forge_exact, digits=1),
        "ForgeCommonRouteForgeCompletePerThousand": _mean_sd_tex(forge_complete, digits=1),
        "ForgeCommonRouteForgeAbstentionPerThousand": _mean_sd_tex(
            forge_abstention,
            digits=1,
        ),
        "ForgeCommonRouteSelectorCompletePerThousand": _mean_sd_tex(
            selector_complete,
            digits=1,
        ),
    }
    macro_path = output_dir / "gem_table10_macros.tex"
    atomic_write(
        macro_path,
        ("\n".join(_renewcommand(name, value) for name, value in macros.items()) + "\n").encode(
            "utf-8"
        ),
    )

    summaries = {}
    for method in METHOD_ORDER:
        exact = [
            float(value)
            for value in _common_optional_values(
                common[method],
                "exact_l1_products_per_1000_attempts",
            )
            if value is not None
        ]
        if len(exact) != len(EXPECTED_SEEDS):
            raise GemTable10Error(f"exact-L1 values are not estimable for {method}")
        summaries[method] = {
            "verified_exact_l1_per_1000": _summary(exact),
            "verified_upstream_per_1000": _summary(
                _per_thousand(routes[method], "verified_upstream")
            ),
            "terminal_evidence_per_1000": _summary(
                _per_thousand(routes[method], "terminal_evidence")
            ),
            "complete_dossier_per_1000": _summary(
                _per_thousand(routes[method], "complete_dossier")
            ),
            "abstention_per_1000": _summary(_per_thousand(routes[method], "abstentions")),
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": [
            pin_record(common_config_path, repo),
            *common_sources,
            adjudication_source,
            pin_record(evidence_path, repo),
        ],
        "attempts_per_method_per_seed": 3072,
        "training_seeds": list(EXPECTED_SEEDS),
        "method_order": list(METHOD_ORDER),
        "candidate_selection": False,
        "route_union_components": int(adjudication["components"]),
        "route_evidence_dispositions": dict(dispositions),
        "method_summaries": summaries,
        "macros": macros,
        "artifacts": {
            "table_10_rows": artifact_record(row_path, logical_path=row_path.name),
            "table_10_macros": artifact_record(macro_path, logical_path=macro_path.name),
        },
        "gates": {
            "all_final_common_ledgers_match_adjudication": True,
            "same_evidence_index_for_every_method": True,
            "planner_calls_during_scoring_zero": True,
            "every_union_component_dispositioned": True,
            "family_projection_cannot_close": True,
            "candidate_selection_absent": True,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable10Error", "render_gem_table10_route_evidence"]
