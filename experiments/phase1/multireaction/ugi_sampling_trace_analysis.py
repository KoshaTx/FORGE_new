"""Describe actual Ugi selection laws from an authenticated passive replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.ugi_sampling_trace import _authenticate_record, _digest, _read
from forge.core.hashing import pin_record
from forge.core.io import read_json, write_json


class TraceAnalysisError(ValueError):
    pass


def _decision(event: dict[str, Any]) -> dict[str, Any] | None:
    """Interpret the actual draw support, keeping conditional count/position laws separate."""
    kind = event["kind"]
    if kind == "terminal_assignment" and event["final_probabilities"] is None:
        return None  # This commits a choice already recorded by the categorical event.
    if kind == "tail_count_group_choice":
        scores = np.asarray(event["group_model_scores"], dtype=float)
        selected = event["selected_group_index"]
        population = list(range(len(scores)))
    elif kind in {"categorical_choice", "terminal_assignment"}:
        candidates = event["candidates"]
        if [row["candidate_index"] for row in candidates] != list(range(len(candidates))):
            raise TraceAnalysisError("candidate indices are not contiguous")
        scores = np.asarray([row["neural_score"] for row in candidates], dtype=float)
        population = event.get("draw_candidate_indices", list(range(len(scores))))
        selected = event["selected_candidate_index"]
    else:
        return None
    probabilities = np.asarray(event["final_probabilities"], dtype=float)
    if (
        not len(scores)
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or probabilities.shape != (len(population),)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0)
        or len(set(population)) != len(population)
        or any(type(index) is not int or not 0 <= index < len(scores) for index in population)
        or type(selected) is not int
        or selected not in population
        or probabilities[population.index(selected)] <= 0
    ):
        raise TraceAnalysisError("invalid captured categorical distribution")
    local_scores = scores[population]
    best = local_scores.max()
    best_mask = local_scores == best
    return {
        "event_index": event["event_index"],
        "attempt_index": event["context"]["attempt_index"],
        "selector": event["selector"],
        "role": event["context"].get("role"),
        "kind": kind,
        "distribution_scope": event["distribution_scope"],
        "count_strategy": event.get("count_strategy"),
        "legal_candidates": len(scores),
        "actual_draw_candidates": len(population),
        "multiple_candidates": len(population) > 1,
        "positive_probability_candidates": int(np.count_nonzero(probabilities > 0)),
        "stochastic_draw": bool(np.count_nonzero(probabilities > 0) > 1),
        "selected_is_neural_maximum_in_draw_population": bool(scores[selected] == best),
        "selected_is_neural_maximum_in_legal_population": bool(scores[selected] == scores.max()),
        "selected_neural_score_gap_in_draw_population": float(best - scores[selected]),
        "probability_on_neural_maxima_in_draw_population": float(probabilities[best_mask].sum()),
        "fraction_of_draw_candidates_with_higher_neural_score": float(
            np.mean(local_scores > scores[selected])
        ),
    }


def describe_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [decision for event in events if (decision := _decision(event)) is not None]
    if not decisions:
        raise TraceAnalysisError("trace contains no captured decisions")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        groups[(decision["selector"], decision["role"], decision["distribution_scope"])].append(
            decision
        )
    summaries = []
    for (selector, role, scope), group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        nontrivial = [row for row in group if row["multiple_candidates"]]
        summaries.append(
            {
                "selector": selector,
                "role": role,
                "distribution_scope": scope,
                "decisions": len(group),
                "stochastic_decisions": sum(row["stochastic_draw"] for row in group),
                "multiple_candidate_decisions": len(nontrivial),
                "multiple_candidate_neural_maximum_selected": sum(
                    row["selected_is_neural_maximum_in_draw_population"] for row in nontrivial
                ),
                "multiple_candidate_mean_probability_on_neural_maxima": (
                    float(
                        np.mean(
                            [
                                row["probability_on_neural_maxima_in_draw_population"]
                                for row in nontrivial
                            ]
                        )
                    )
                    if nontrivial
                    else None
                ),
                "multiple_candidate_mean_fraction_candidates_with_higher_neural_score": (
                    float(
                        np.mean(
                            [
                                row["fraction_of_draw_candidates_with_higher_neural_score"]
                                for row in nontrivial
                            ]
                        )
                    )
                    if nontrivial
                    else None
                ),
                "neural_count_bypass_decisions": sum(
                    row["count_strategy"] == "frequency_resampled" for row in group
                ),
            }
        )
    return {
        "event_counts": dict(Counter(event["kind"] for event in events)),
        "decision_count_without_duplicate_commit_events": len(decisions),
        "groups": summaries,
        "decisions": decisions,
    }


def preservation_metrics(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    result = {
        field: {
            "baseline": baseline["metrics"][field],
            "treatment": treatment["metrics"][field],
            "no_observed_loss": treatment["metrics"][field] >= baseline["metrics"][field],
        }
        for field in (
            "valid_fraction",
            "exact_l1_yield_per_attempt",
            "effective_component_count",
            "unique_exact_l1_products_per_attempt",
            "mean_pairwise_ecfp4_distance",
            "unique_open_ended_exact_l1_products_per_attempt",
        )
    }
    for field in ("decomposed_products_with_any_novel_component", "whole_product_novel_to_train"):
        left, right = (
            baseline["all_attempt_metrics"][field],
            treatment["all_attempt_metrics"][field],
        )
        result[f"{field}_all_attempt_incidence"] = {
            "baseline": left,
            "treatment": right,
            "no_observed_loss": right["fraction"] >= left["fraction"],
        }
    return result


def analyze_trace(repo: Path, trace_result_path: Path, output: Path) -> dict[str, Any]:
    repo, trace_result_path, output = repo.resolve(), trace_result_path.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(repo) or output == repo:
        raise TraceAnalysisError("use a fresh output directory within the repository")
    source = _read(trace_result_path)
    if source.get("status") != "complete_diagnostic_no_improvement_claim":
        raise TraceAnalysisError("only completed authenticated replays can be analyzed")
    _authenticate_record(source["config"], repo, label="trace config")
    snapshot_path = _authenticate_record(
        source["source_snapshot"], repo, label="trace source snapshot"
    )
    if _digest(_read(snapshot_path)) != source["source_snapshot_sha256"]:
        raise TraceAnalysisError("source snapshot digest differs")
    for name, record in {**source["inputs"], **source["comparison_inputs"]}.items():
        _authenticate_record(record, repo, label=name)
    pins = {
        name: _authenticate_record(record, repo, label=name)
        for name, record in source["artifacts"].items()
    }
    descriptions = {}
    role_counts = {}
    for arm in ("baseline", "treatment"):
        if source["arms"][arm]["assessment"] != _read(pins[f"{arm}/assessment.json"]):
            raise TraceAnalysisError("embedded assessment differs from authenticated artifact")
        if source["arms"][arm]["equivalence"] != _read(pins[f"{arm}/equivalence.json"])["checks"]:
            raise TraceAnalysisError("embedded equivalence differs from authenticated artifact")
        if not source["arms"][arm]["equivalence"] or not all(
            source["arms"][arm]["equivalence"].values()
        ):
            raise TraceAnalysisError("replay equivalence failed")
        events = read_json(pins[f"{arm}_observed/events.json"])
        descriptions[arm] = describe_events(events)
        assessment = source["arms"][arm]["assessment"]
        rows = read_json(pins[f"{arm}/assessed_attempts.json"])
        role_counts[arm] = {
            role: {
                "distinct_components_in_unique_decompositions": len(
                    {
                        row["exact_l1_traces"][0]["components_by_role"][role]
                        for row in rows
                        if len(row.get("exact_l1_traces", [])) == 1
                    }
                ),
                "effective_count": metric["effective_count"],
                "novel_all_attempt_incidence": assessment["all_attempt_metrics"][
                    "component_novel_to_train_by_role"
                ][role],
            }
            for role, metric in assessment["metrics"]["component_metrics_by_role"].items()
        }
    preservation = preservation_metrics(
        source["arms"]["baseline"]["assessment"], source["arms"]["treatment"]["assessment"]
    )
    result = {
        "schema_version": "forge.ugi_sampling_trace_analysis.v1",
        "status": "descriptive_attribution_not_improvement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "trace_result": pin_record(trace_result_path, repo),
            "analysis_source": pin_record(Path(__file__).resolve(), repo),
        },
        "authenticated_artifact_count": len(pins),
        "arms": descriptions,
        "observed_preservation": preservation,
        "component_counts_by_role": role_counts,
        "all_checked_preservation_metrics_have_no_observed_loss": all(
            row["no_observed_loss"] for row in preservation.values()
        ),
        "nonclaims": [
            "Candidate logits and their sums are not calibrated whole-molecule probabilities or chemical realism scores.",
            "Candidate count is a serialization count; it does not measure distinct molecular identities or diversity.",
            "Count-group and conditional position laws remain separate; no unobserved marginal law was reconstructed.",
            "Rank agreement is measured inside the actual captured support, not among candidates removed before selection.",
            "Candidate-internal atom scoring and later commit events are excluded from decision counts to prevent double counting.",
            "This summary covers topology and joint terminal candidate selections; coordinate choices/argmax and terminal-pattern group draws are excluded. It is not a count of every terminal operation.",
            "Draw population includes every supplied categorical alternative, including zero-probability candidates. Stochastic decisions require more than one positive-probability alternative.",
            "The checked preservation metrics are not the complete promotion protocol; appearance, pathology, support and operational invariants retain their separate gates.",
            "Selecting a lower neural score does not establish that guidance worsens realism.",
            "One prespecified 16-attempt trace and 128-attempt output batch do not establish a causal mechanism or across-seed effect.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--trace-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_trace(args.repo_root, args.trace_result, args.output_dir)["status"])


if __name__ == "__main__":
    main()
