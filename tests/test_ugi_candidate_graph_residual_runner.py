from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.phase1.multireaction import ugi_candidate_graph_residual as runner
from forge.model.ugi_candidate_graph_residual import CandidateGraphResidual, candidate_class_loss


def _candidate(classes=(0, 1), positive=(True, False), nodes=3):
    adjacency = np.zeros((len(classes), nodes, nodes), dtype=np.bool_)
    for candidate in range(len(classes)):
        center = candidate % nodes
        for node in range(nodes):
            if node != center:
                adjacency[candidate, center, node] = True
                adjacency[candidate, node, center] = True
    enumerated = tuple(SimpleNamespace(offspring=(0,) * nodes, closures=()) for _ in classes)
    return SimpleNamespace(
        role=runner.ROLES[0],
        candidates=enumerated,
        enumerated_candidates=enumerated,
        enumerated_identity_sha256=str(
            runner.sha256_json([list(item.offspring) for item in enumerated])
        ),
        unavailable_reason=None,
        failed_candidate_index=None,
        graph_node_indices=tuple(range(nodes)),
        adjacency=adjacency,
        node_colors=np.eye(nodes, 2, dtype=np.float32),
        positive_mask=np.asarray(positive, dtype=np.bool_),
        graph_class_ids=classes,
        graph_class_count=len(set(classes)),
    )


def _unavailable_candidate():
    candidate = _candidate((), ())
    candidate.unavailable_reason = "baseline_root_alignment_failure"
    candidate.failed_candidate_index = 1
    candidate.enumerated_candidates = (
        SimpleNamespace(offspring=(1, 1, 0), closures=()),
        SimpleNamespace(offspring=(2, 0, 0), closures=((0, 2),)),
        SimpleNamespace(offspring=(1, 0, 0), closures=()),
    )
    candidate.enumerated_identity_sha256 = str(
        runner.sha256_json(
            [
                {
                    "offspring": list(item.offspring),
                    "closures": [list(edge) for edge in item.closures],
                }
                for item in candidate.enumerated_candidates
            ]
        )
    )
    return candidate


def _case(index=0, role_index=0, candidate=None):
    if candidate is None:
        candidate = _candidate()
    nodes = candidate.node_colors.shape[0]
    return {
        "cache_index": index,
        "role": runner.ROLES[role_index],
        "role_index": role_index,
        "draw_index": index,
        "flow_time": 0.5,
        "hidden": torch.arange(nodes * 4, dtype=torch.float32).reshape(nodes, 4) / 10,
        "base_scores": torch.zeros(len(candidate.candidates), dtype=torch.float64),
    }


def _config():
    return {
        "schema_version": "forge.ugi_candidate_graph_residual_config.v1",
        "target_program": "ugi_3cr_agile",
        "partition": {
            "component_domain": "forge.ugi_candidate_graph_component_partition.v1:2026090824:",
            "product_domain": "forge.ugi_candidate_graph_product_partition.v1:2026090824:",
            "bucket_count": 5,
            "evaluation_bucket": 0,
        },
        "flow_times": [0.2, 0.5, 0.8, 0.95],
        "draws": {
            "fit": 1024,
            "amine_disjoint": 256,
            "aldehyde_disjoint": 256,
            "repeated_component": 256,
        },
        "feature_batch_size": 64,
        "training_batch_size": 64,
        "optimizer_steps": 64,
        "candidate_chunk_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "width": 32,
        "cpu_threads": 2,
        "device": "cpu",
        "precision": "float32",
        "seeds": {
            "draws": 2026090825,
            "noise": 2026090826,
            "order": 2026090827,
            "initialization": 2026090828,
        },
        "policy": {
            "backbone_training": False,
            "molecular_generation": False,
            "remote_compute": False,
            "gate_changes": False,
            "final_checkpoint_only": True,
            "feature_records_train_only": True,
        },
    }


def test_metrics_use_graph_class_mass_instead_of_best_candidate():
    candidate = _candidate((0, 0, 1), (True, True, False))
    scores = torch.tensor([math.log(0.3), math.log(0.3), math.log(0.4)], dtype=torch.float64)
    assert int(scores.argmax()) == 2
    metrics = runner._case_metrics(scores, candidate)
    assert metrics["graph_argmax_correct"]
    assert metrics["target_probability"] == pytest.approx(0.6)
    assert metrics["target_nll"] == pytest.approx(-math.log(0.6))
    # Relabeling class IDs and reordering symmetric representatives does not change correctness.
    reordered = _candidate((0, 1, 1), (False, True, True))
    other = runner._case_metrics(scores[torch.tensor([2, 0, 1])], reordered)
    assert other == metrics


def test_metrics_keep_serialization_multiplicity_in_probability():
    candidate = _candidate((0, 0, 1), (True, True, False))
    assert runner._case_metrics(torch.zeros(3, dtype=torch.float64), candidate)[
        "target_probability"
    ] == pytest.approx(2 / 3)
    deduplicated = _candidate()
    assert runner._case_metrics(torch.zeros(2, dtype=torch.float64), deduplicated)[
        "target_probability"
    ] == pytest.approx(1 / 2)


def test_summary_keeps_empty_absent_allpositive_and_time_denominators():
    rows = [
        {
            **runner._case_metrics(torch.zeros(2, dtype=torch.float64), _candidate()),
            "flow_time": 0.2,
        },
        {
            **runner._case_metrics(torch.empty(0, dtype=torch.float64), _candidate((), ())),
            "flow_time": 0.2,
        },
        {
            **runner._case_metrics(
                torch.zeros(2, dtype=torch.float64), _candidate((0, 1), (False, False))
            ),
            "flow_time": 0.5,
        },
        {
            **runner._case_metrics(
                torch.zeros(2, dtype=torch.float64), _candidate((0, 0), (True, True))
            ),
            "flow_time": 0.5,
        },
    ]
    summary = runner._summary(rows)
    assert summary["cases"] == 4 and summary["empty"] == 1 and summary["target_absent"] == 2
    assert summary["mean_target_probability_all"] == pytest.approx(0.375)
    assert summary["informative_cases"] == 1
    assert summary["informative_mean_target_probability"] == pytest.approx(0.5)
    assert summary["informative_mean_nll"] == pytest.approx(math.log(2))
    assert runner._summary([row for row in rows if row["flow_time"] == 0.2])[
        "mean_target_probability_all"
    ] == pytest.approx(0.25)
    assert runner._summary([row for row in rows if row["flow_time"] == 0.5])[
        "mean_target_probability_all"
    ] == pytest.approx(0.5)
    empty = runner._summary([])
    assert empty["cases"] == 0 and empty["mean_target_probability_all"] is None
    assert empty["informative_mean_nll"] is None


def test_coverage_records_negative_preflight_without_numpy_scalar_assumption():
    candidates = {}
    selection = {"fit": {"draws": [{"cache_index": 0}, {"cache_index": 1}, {"cache_index": 1}]}}
    for role in runner.ROLES:
        candidates[0, role] = _candidate((), ())
        candidates[1, role] = _candidate((0, 0), (True, True))
    coverage = runner._coverage(candidates, selection)
    for role in runner.ROLES:
        assert coverage["fit"][role] == {
            "draws": 3,
            "empty": 1,
            "law_unavailable": 0,
            "target_absent": 1,
            "singleton_graph_class": 2,
            "reachable_competing_graph_classes": 0,
        }


def test_coverage_counts_draw_multiplicity_and_both_roles_separately():
    selection = {"fit": {"draws": [{"cache_index": 0}, {"cache_index": 0}]}}
    candidates = {
        (0, runner.ROLES[0]): _candidate(),
        (0, runner.ROLES[1]): _candidate((0, 1), (False, False)),
    }
    coverage = runner._coverage(candidates, selection)["fit"]
    assert coverage[runner.ROLES[0]]["reachable_competing_graph_classes"] == 2
    assert coverage[runner.ROLES[1]]["target_absent"] == 2
    assert coverage[runner.ROLES[1]]["reachable_competing_graph_classes"] == 0


def test_unavailable_receipt_preserves_whole_raw_law_and_failure_position():
    candidate = _unavailable_candidate()
    receipt = runner._candidate_receipt(candidate)
    assert receipt["candidate_count"] == receipt["positive_count"] == 0
    assert receipt["graph_class_count"] == 0 and not receipt["target_present"]
    assert receipt["candidates"] == []
    assert receipt["unavailable_reason"] == candidate.unavailable_reason
    assert receipt["failed_candidate_index"] == 1
    assert receipt["enumerated_identity_sha256"] == candidate.enumerated_identity_sha256
    assert receipt["enumerated_candidates"] == [
        {"offspring": [1, 1, 0], "closures": []},
        {"offspring": [2, 0, 0], "closures": [[0, 2]]},
        {"offspring": [1, 0, 0], "closures": []},
    ]
    assert len(candidate.enumerated_candidates) == 3 and len(candidate.candidates) == 0


def test_unavailable_coverage_retains_each_original_draw_and_other_role():
    selection = {"fit": {"draws": [{"cache_index": 0}, {"cache_index": 0}]}}
    candidates = {
        (0, runner.ROLES[0]): _unavailable_candidate(),
        (0, runner.ROLES[1]): _candidate(),
    }
    coverage = runner._coverage(candidates, selection)["fit"]
    assert coverage[runner.ROLES[0]]["draws"] == 2
    assert coverage[runner.ROLES[0]]["law_unavailable"] == 2
    assert coverage[runner.ROLES[0]]["empty"] == coverage[runner.ROLES[0]]["target_absent"] == 2
    assert coverage[runner.ROLES[1]]["law_unavailable"] == 0
    assert coverage[runner.ROLES[1]]["reachable_competing_graph_classes"] == 2


@pytest.mark.parametrize("use_adjacency", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 2, 128])
def test_chunked_two_pass_matches_full_autograd_for_all_parameters(use_adjacency, chunk_size):
    candidate = _candidate((0, 0, 1, 2, 2), (True, True, False, False, False), nodes=4)
    case = _case(candidate=candidate)
    case["base_scores"] = torch.tensor([1.0, -2.0, 0.5, 0.25, -0.75], dtype=torch.float64)
    head = CandidateGraphResidual(
        4, 2, width=8, initialization_seed=491, use_adjacency=use_adjacency
    )
    with torch.no_grad():
        head.output.weight.copy_(torch.linspace(-0.3, 0.4, 8)[None])
        head.output.bias.fill_(0.2)
    reference = copy.deepcopy(head)
    denominator = 8  # Includes other absent/empty role cases in the original batch.
    direct = (
        candidate_class_loss(
            runner._residual_scores(reference, case, candidate, chunk_size),
            torch.tensor([0, 5]),
            torch.tensor(candidate.positive_mask),
        ).per_row_loss[0]
        / denominator
    )
    direct.backward()
    loss = runner._backward_case(head, case, candidate, chunk_size, denominator)
    assert loss == pytest.approx(float(direct.detach()), rel=1e-12, abs=1e-12)
    for (name, value), (other_name, other) in zip(
        head.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        assert value.grad is not None and other.grad is not None
        torch.testing.assert_close(value.grad, other.grad, rtol=2e-5, atol=2e-8, msg=name)


@pytest.mark.parametrize(
    ("classes", "positive"), [((), ()), ((0, 1), (False, False)), ((0, 0), (True, True))]
)
def test_unlearnable_or_allpositive_cases_have_zero_gradient(classes, positive):
    candidate = _candidate(classes, positive)
    head = CandidateGraphResidual(4, 2, initialization_seed=5)
    result = runner._backward_case(head, _case(candidate=candidate), candidate, 2, 8)
    assert result == 0 and all(value.grad is None for value in head.parameters())


class _DegreeToy(torch.nn.Module):
    def __init__(self, hidden_dim, node_color_dim, *, width, initialization_seed, use_adjacency):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(()))
        self.use_adjacency = use_adjacency

    def forward(self, hidden, adjacency, node_colors, node_mask, flow_time, role_index):
        return self.theta * adjacency.sum(-1)[:, 0]


@pytest.mark.parametrize("unavailable", [False, True])
def test_fit_retains_original_batch_mass_order_and_exact_zero_identity(
    tmp_path, monkeypatch, unavailable
):
    specifications = [
        ((0, 1), (True, False)),
        ((), ()),
        ((0, 1), (False, False)),
        ((0, 0), (True, True)),
        ((0, 0, 1), (True, True, False)),
        ((0,), (True,)),
        ((0, 1, 1, 1), (True, False, False, False)),
        ((0, 1), (True, False)),
    ]
    candidates, cases = {}, []
    for slot, (classes, positive) in enumerate(specifications):
        exposure, role_index = divmod(slot, 2)
        candidate = _candidate(classes, positive)
        if unavailable and not classes:
            candidate = _unavailable_candidate()
        candidates[exposure, runner.ROLES[role_index]] = candidate
        cases.append(_case(exposure, role_index, candidate))
    config = {
        **_config(),
        "training_batch_size": 2,
        "optimizer_steps": 2,
        "candidate_chunk_size": 2,
        "learning_rate": 0.0,
        "gradient_clip_norm": 100.0,
    }
    recorded = []

    class Recorder:
        def __init__(self, parameters, **kwargs):
            self.parameters = list(parameters)

        def zero_grad(self, **kwargs):
            for value in self.parameters:
                value.grad = None

        def step(self):
            recorded.append(float(self.parameters[0].grad or 0.0))

        def state_dict(self):
            return {}

    monkeypatch.setattr(runner, "CandidateGraphResidual", _DegreeToy)
    monkeypatch.setattr(runner.torch.optim, "AdamW", Recorder)
    _, receipts = runner._fit({"fit": cases}, candidates, config, tmp_path)
    order = json.loads((tmp_path / "exposure_order.json").read_text())["product_time_indices"]
    assert sorted(order) == list(range(4))
    assert (
        order
        == torch.randperm(
            4, generator=torch.Generator().manual_seed(config["seeds"]["order"])
        ).tolist()
    )
    expected_loss, expected_gradient = [], []
    for start in (0, 2):
        losses, gradients = [], []
        for exposure in order[start : start + 2]:
            for role in runner.ROLES:
                candidate = candidates[exposure, role]
                n = len(candidate.candidates)
                p = int(candidate.positive_mask.sum())
                if p and p < n:
                    losses.append(math.log(n / p))
                    degree = candidate.adjacency.sum(-1)[:, 0]
                    gradients.append(float(degree.mean() - degree[candidate.positive_mask].mean()))
        expected_loss.append(sum(losses) / 4)
        expected_gradient.append(sum(gradients) / 4)
    assert recorded == pytest.approx(expected_gradient * 2)
    for arm in ("degree_control", "graph"):
        assert receipts[arm]["losses"] == pytest.approx(expected_loss)
        assert receipts[arm]["updates"] == 2
        assert receipts[arm]["zero_residual_equal_all_fit_cases"]
        assert receipts[arm]["exposure_counts"] == {
            "empty": 1,
            "absent": 1,
            "all_positive": 2,
            "informative": 4,
        }
    assert receipts["degree_control"]["state_sha256"] == receipts["graph"]["state_sha256"]


@pytest.mark.parametrize("mutation", ["orphan", "role_order", "draw_identity", "time_identity"])
def test_fit_rejects_incomplete_or_misaligned_exposures(tmp_path, mutation):
    cases = [_case(0, 0), _case(0, 1)]
    if mutation == "orphan":
        cases.append(_case(1, 0))
    elif mutation == "role_order":
        cases.reverse()
    elif mutation == "draw_identity":
        cases[1]["draw_index"] = 1
    else:
        cases[1]["flow_time"] = 0.8
    with pytest.raises(runner.CandidateGraphStudyError, match="paired"):
        runner._fit(
            {"fit": cases},
            {},
            {**_config(), "training_batch_size": 1, "optimizer_steps": 1},
            tmp_path,
        )


def test_prespecified_configuration_is_accepted():
    runner._validate(_config())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("flow_times", [0.95]),
        ("width", 64),
        ("candidate_chunk_size", 64),
        ("learning_rate", 0.01),
        ("optimizer_steps", 65),
        ("precision", "float64"),
        ("target_program", "other"),
    ],
)
def test_frozen_configuration_fields_cannot_drift(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(runner.CandidateGraphStudyError, match=key):
        runner._validate(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("component_domain", "new-seed:"),
        ("product_domain", "new-seed:"),
        ("bucket_count", 10),
        ("evaluation_bucket", 1),
    ],
)
def test_partition_cannot_be_reselected_after_results(key, value):
    config = _config()
    config["partition"][key] = value
    with pytest.raises(runner.CandidateGraphStudyError, match="partition"):
        runner._validate(config)


@pytest.mark.parametrize(
    "policy", ["gate_changes", "molecular_generation", "backbone_training", "remote_compute"]
)
def test_diagnostic_policy_cannot_expand(policy):
    config = _config()
    config["policy"][policy] = True
    with pytest.raises(runner.CandidateGraphStudyError, match="policy"):
        runner._validate(config)


def test_feature_extraction_uses_native_noisy_topology_and_preserves_role_pair_slices(
    tmp_path, monkeypatch
):
    selected = {"fit": {"draws": [{"cache_index": 10}, {"cache_index": 11}]}}
    candidates = {}
    for index in (10, 11):
        for role_index, role in enumerate(runner.ROLES):
            candidates[index, role] = SimpleNamespace(
                graph_node_indices=(role_index, 2),
                expected_index=index,
            )
    clean_calls, noisy_calls, model_calls, scoring_calls = [], [], [], []

    def collate(records, **kwargs):
        clean = {
            "nodes": torch.tensor(records)[:, None].expand(-1, 3),
            "parents": torch.tensor([[0, 0, 1]]).expand(len(records), -1).clone(),
            "parent_bonds": torch.zeros(len(records), 3, dtype=torch.long),
            "closure_left": torch.ones(len(records), 1, dtype=torch.long),
            "closure_right": torch.full((len(records), 1), 2, dtype=torch.long),
            "closure_bonds": torch.ones(len(records), 1, dtype=torch.long),
        }
        for key in (
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "component_instance_states",
            "role_morphology_states",
        ):
            clean[key] = torch.ones(len(records), 3, dtype=torch.long)
        clean_calls.append(clean)
        return clean

    def noise(clean, node_marginal, bond_marginal, t, generator):
        noisy = {
            name: clean[name].clone()
            for name in (
                "nodes",
                "parents",
                "parent_bonds",
                "closure_left",
                "closure_right",
                "closure_bonds",
            )
        }
        noisy["parents"].zero_()
        noisy["closure_left"].zero_()
        noisy["closure_right"].fill_(1)
        assert not torch.equal(noisy["parents"], clean["parents"])
        noisy_calls.append(noisy)
        return noisy

    class Model:
        def __call__(self, **kwargs):
            noisy = noisy_calls[-1]
            for field in ("parents", "closure_left", "closure_right"):
                assert kwargs[field] is noisy[field]
                assert not torch.equal(kwargs[field], clean_calls[-1][field])
            assert kwargs["return_hidden_state"] is True
            ids = kwargs["nodes"][:, 0].float()
            hidden = ids[:, None, None] + torch.arange(12).reshape(1, 3, 4).float() / 100
            model_calls.append(kwargs)
            return {"hidden_state": hidden, "nodes": kwargs["nodes"].float()}

    def score(candidate, one):
        assert one["hidden_state"].shape == (1, 3, 4)
        assert int(one["nodes"][0, 0]) == candidate.expected_index
        scoring_calls.append(candidate.expected_index)
        return torch.tensor([float(candidate.expected_index)], dtype=torch.float64)

    monkeypatch.setattr(runner, "collate_synthesis_program_training_batch", collate)
    monkeypatch.setattr(runner, "noise_synthesis_program_batch", noise)
    monkeypatch.setattr(runner, "score_candidates", score)
    cache = SimpleNamespace(records=lambda indices: indices, vocabulary=object())
    config = {**_config(), "flow_times": [0.2, 0.8], "feature_batch_size": 2}
    package = {
        "model_config": {"maximum_closures": 1},
        "node_marginal": [1.0],
        "bond_marginal": [1.0],
    }
    features, ledger = runner._extract_features(
        Model(), package, cache, selected, candidates, config, tmp_path
    )
    assert len(model_calls) == len(ledger) == 2
    assert scoring_calls == [10, 10, 11, 11, 10, 10, 11, 11]
    cases = features["fit"]
    assert [(case["cache_index"], case["role_index"], case["flow_time"]) for case in cases] == [
        (index, role, t) for t in (0.2, 0.8) for index in (10, 11) for role in (0, 1)
    ]
    for case in cases:
        assert case["base_scores"].tolist() == [float(case["cache_index"])]
        assert not case["hidden"].requires_grad
        assert case["hidden"].shape == (2, 4)
    assert all(row["target_topology_substituted"] is False for row in ledger)
    assert [row["noise_seed"] for row in ledger] == [
        config["seeds"]["noise"],
        config["seeds"]["noise"] + 1,
    ]
