from copy import deepcopy

import pytest

from experiments.phase1.multireaction.ugi_sampling_trace_analysis import (
    TraceAnalysisError,
    _decision,
    describe_events,
    preservation_metrics,
)


def _event():
    return {
        "kind": "categorical_choice",
        "event_index": 1,
        "context": {"attempt_index": 0, "role": "tail"},
        "selector": "tail",
        "candidates": [
            {"candidate_index": index, "neural_score": score}
            for index, score in enumerate([3.0, 6.0, 5.0])
        ],
        "draw_candidate_indices": [2, 0],
        "final_probabilities": [0.75, 0.25],
        "selected_candidate_index": 2,
        "distribution_scope": "conditional_group_candidates",
    }


def test_conditional_probabilities_follow_actual_draw_indices():
    decision = _decision(_event())
    assert decision["actual_draw_candidates"] == 2
    assert decision["legal_candidates"] == 3
    assert decision["selected_is_neural_maximum_in_draw_population"]
    assert not decision["selected_is_neural_maximum_in_legal_population"]
    assert decision["probability_on_neural_maxima_in_draw_population"] == 0.75
    changed = _event()
    changed["selected_candidate_index"] = 0
    decision = _decision(changed)
    assert decision["selected_neural_score_gap_in_draw_population"] == 2
    assert decision["fraction_of_draw_candidates_with_higher_neural_score"] == 0.5


def test_commit_events_are_not_counted_as_additional_random_decisions():
    commit = {**_event(), "kind": "terminal_assignment", "final_probabilities": None}
    internal = {**_event(), "kind": "candidate_internal_atom_argmax"}
    result = describe_events([_event(), commit, internal])
    assert result["decision_count_without_duplicate_commit_events"] == 1
    assert result["groups"][0]["multiple_candidate_decisions"] == 1
    with pytest.raises(TraceAnalysisError, match="no captured decisions"):
        describe_events([commit, internal])


def test_neural_ties_and_count_frequency_law_are_reported_without_reweighting():
    event = {
        "kind": "tail_count_group_choice",
        "event_index": 2,
        "context": {"attempt_index": 0, "role": "tail"},
        "selector": "tail",
        "distribution_scope": "unsaturation_count_groups",
        "group_model_scores": [4.0, 4.0, 3.0],
        "selected_group_index": 2,
        "final_probabilities": [0.2, 0.3, 0.5],
        "count_strategy": "frequency_resampled",
    }
    decision = _decision(event)
    assert decision["probability_on_neural_maxima_in_draw_population"] == 0.5
    assert not decision["selected_is_neural_maximum_in_draw_population"]
    result = describe_events([event])
    assert result["groups"][0]["neural_count_bypass_decisions"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_probabilities", [0.8, 0.4]),
        ("final_probabilities", [-0.1, 1.1]),
        ("draw_candidate_indices", [2, 2]),
        ("selected_candidate_index", 1),
        ("final_probabilities", [0.0, 1.0]),
    ],
)
def test_invalid_captured_law_fails_closed(field, value):
    event = deepcopy(_event())
    event[field] = value
    with pytest.raises(TraceAnalysisError):
        _decision(event)


def test_deterministic_alternatives_are_separate_from_stochastic_draws():
    event = _event()
    event["final_probabilities"] = [1.0, 0.0]
    result = describe_events([event])
    assert result["groups"][0]["multiple_candidate_decisions"] == 1
    assert result["groups"][0]["stochastic_decisions"] == 0


def test_novelty_preservation_keeps_failed_attempts_in_denominator():
    metrics = {
        field: 1.0
        for field in (
            "valid_fraction",
            "exact_l1_yield_per_attempt",
            "effective_component_count",
            "unique_exact_l1_products_per_attempt",
            "mean_pairwise_ecfp4_distance",
            "unique_open_ended_exact_l1_products_per_attempt",
        )
    }

    def assessment(novel):
        return {
            "metrics": metrics,
            "all_attempt_metrics": {
                field: {"count": novel, "denominator": 10, "fraction": novel / 10}
                for field in (
                    "decomposed_products_with_any_novel_component",
                    "whole_product_novel_to_train",
                )
            },
        }

    result = preservation_metrics(assessment(8), assessment(6))
    for field in ("decomposed_products_with_any_novel_component", "whole_product_novel_to_train"):
        assert not result[f"{field}_all_attempt_incidence"]["no_observed_loss"]
