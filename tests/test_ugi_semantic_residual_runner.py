"""Bounded runner checks using synthetic metadata and eight feature records only."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.phase1.multireaction.ugi_semantic_residual as runner
from experiments.phase1.multireaction.ugi_semantic_residual import (
    UgiSemanticResidualStudyError,
    _advancement,
    _atom_metrics,
    _bucket,
    _fit_heads,
    _placement_summary,
    _resume_saved_fit,
    _select_draws,
)
from forge.core.hashing import PinError
from forge.model.ugi_semantic_placement import rank_candidate_scores


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _selection_fixture() -> tuple[list[dict], dict]:
    config = {
        "partition": {"product_domain": "test-only-product-partition:"},
        "seeds": {"draws": 79},
        "draws": {"fit": 2000, "component_disjoint": 17, "repeated_component": 13},
    }
    ids = {"fit": [], "repeated": []}
    for number in range(100):
        product_id = f"test-product-{number}"
        name = (
            "repeated" if _bucket(product_id, config["partition"]["product_domain"]) == 0 else "fit"
        )
        ids[name].append(product_id)
    rows = [
        {
            "product_id": ids["fit"][0],
            "component_partition": "fit",
            "amine_smiles": "fixture-shared-head",
            "source_weight": 1.0,
            "applicable": True,
        },
        {
            "product_id": ids["fit"][1],
            "component_partition": "fit",
            "amine_smiles": "fixture-shared-head",
            "source_weight": 99.0,
            "applicable": True,
        },
        {
            "product_id": "fixture-disjoint-product",
            "component_partition": "evaluation",
            "amine_smiles": "fixture-disjoint-head",
            "source_weight": 3.0,
            "applicable": True,
            "target_present": False,
            "candidate_count": 0,
            "informative_ranking": False,
        },
        {
            "product_id": ids["repeated"][0],
            "component_partition": "fit",
            "amine_smiles": "fixture-shared-head",
            "source_weight": 2.0,
            "applicable": True,
            "target_present": True,
            "candidate_count": 1,
            "informative_ranking": False,
        },
        {
            "product_id": "fixture-coverage-exclusion",
            "component_partition": "evaluation",
            "amine_smiles": "fixture-excluded-head",
            "source_weight": 10000.0,
            "applicable": False,
            "exclusion_reason": "fixture-outside-residual-scope",
        },
    ]
    return rows, config


def test_draws_preserve_source_measure_counts_disjointness_and_coverage_input() -> None:
    rows, config = _selection_fixture()
    before = copy.deepcopy(rows)
    selected = _select_draws(rows, config)
    assert selected == _select_draws(rows, config)
    assert rows == before
    assert {name: len(value["draws"]) for name, value in selected.items()} == config["draws"]
    assert selected["fit"]["population_rows"] == 2
    assert selected["fit"]["source_weight_sum"] == 100
    # Two equally frequent rows have a 1:99 declared source measure. Raw row counts would
    # yield about 1,000 heavy-row draws, far outside this intentionally broad fixed-seed bound.
    heavy_id = rows[1]["product_id"]
    heavy_draws = sum(row["product_id"] == heavy_id for row in selected["fit"]["draws"])
    assert 1940 < heavy_draws < 2000
    fit_products = {row["product_id"] for row in selected["fit"]["draws"]}
    repeated_products = {row["product_id"] for row in selected["repeated_component"]["draws"]}
    fit_heads = {row["amine_smiles"] for row in selected["fit"]["draws"]}
    disjoint_heads = {row["amine_smiles"] for row in selected["component_disjoint"]["draws"]}
    assert not fit_products & repeated_products
    assert not fit_heads & disjoint_heads
    assert selected["component_disjoint"]["heads_absent_from_fit_population"] == [
        "fixture-disjoint-head"
    ]
    assert selected["repeated_component"]["heads_absent_from_fit_population"] == []
    assert all(
        row["product_id"] != "fixture-coverage-exclusion"
        for group in selected.values()
        for row in group["draws"]
    )
    assert rows[-1]["applicable"] is False
    assert rows[-1]["exclusion_reason"] == "fixture-outside-residual-scope"


def test_candidate_failures_and_singletons_are_not_filtered_from_applicable_draws() -> None:
    rows, config = _selection_fixture()
    selected = _select_draws(rows, config)
    assert len(selected["component_disjoint"]["draws"]) == 17
    assert all(row["target_present"] is False for row in selected["component_disjoint"]["draws"])
    assert all(row["candidate_count"] == 0 for row in selected["component_disjoint"]["draws"])
    assert len(selected["repeated_component"]["draws"]) == 13
    assert all(row["candidate_count"] == 1 for row in selected["repeated_component"]["draws"])


def test_component_overlap_and_empty_populations_fail_closed() -> None:
    rows, config = _selection_fixture()
    rows[2]["amine_smiles"] = rows[0]["amine_smiles"]
    with pytest.raises(UgiSemanticResidualStudyError, match="leaked"):
        _select_draws(rows, config)
    rows, config = _selection_fixture()
    rows[2]["applicable"] = False
    with pytest.raises(UgiSemanticResidualStudyError, match="empty"):
        _select_draws(rows, config)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_source_weight_fails_instead_of_dropping_row(weight: float) -> None:
    rows, config = _selection_fixture()
    rows[0]["source_weight"] = weight
    with pytest.raises(UgiSemanticResidualStudyError, match="source measure"):
        _select_draws(rows, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("component_partition", "heldout"),
        ("component_partition", "unknown"),
        ("applicable", "false"),
        ("applicable", 1),
    ],
)
def test_malformed_partition_or_applicability_is_not_silently_filtered(
    field: str, value: object
) -> None:
    rows, config = _selection_fixture()
    rows[-1][field] = value
    with pytest.raises(UgiSemanticResidualStudyError, match="malformed"):
        _select_draws(rows, config)


def test_atom_metrics_use_actual_corruption_and_exclude_padding() -> None:
    logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [100.0, -100.0]]] * 2)
    data = {
        "target": torch.tensor([[0, 1, 1], [1, 0, 1]]),
        "noisy": torch.tensor([[1, 1, 0], [0, 0, 0]]),
        "mask": torch.tensor([[True, True, False], [True, True, False]]),
    }
    metrics = _atom_metrics(logits, data)
    assert metrics["all_variable"]["coordinates"] == 4
    assert metrics["all_variable"]["correct"] == 2
    assert metrics["all_variable"]["accuracy"] == 0.5
    assert metrics["corrupted"]["coordinates"] == 2
    assert metrics["corrupted"]["correct"] == 1
    assert metrics["corrupted"]["accuracy"] == 0.5
    assert metrics["unchanged"]["coordinates"] == 2
    assert metrics["unchanged"]["correct"] == 1
    expected_nll = -logits.log_softmax(-1)[0, 0].mean().item()
    assert metrics["all_variable"]["nll"] == pytest.approx(expected_nll)
    assert metrics["corrupted"]["nll"] == pytest.approx(expected_nll)
    assert metrics["unchanged"]["nll"] == pytest.approx(expected_nll)
    data["noisy"] = data["target"].clone()
    assert _atom_metrics(logits, data)["corrupted"] == {
        "coordinates": 0,
        "correct": 0,
        "accuracy": None,
        "nll": None,
    }


def _features() -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 48,
        "logits": torch.zeros(8, 2, 2),
        "mask": torch.tensor([[True, False], [True, True]] * 4),
        "target": torch.tensor([[0, 1], [1, 0]] * 4),
        "noisy": torch.ones(8, 2, dtype=torch.long),
        "t": torch.tensor([0.2, 0.5, 0.8, 0.95] * 2),
        "semantics": torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]] * 4),
    }


def _fit_config() -> dict:
    return {
        "optimizer_steps": 2,
        "training_batch_size": 4,
        "semantic_scales": [10, 10, 4, 4],
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "seeds": {"order": 31, "initialization": 29},
    }


def test_synthetic_fit_has_identical_initialization_exposure_and_final_checkpoints(
    tmp_path: Path,
) -> None:
    data = _features()
    before = {name: value.clone() for name, value in data.items()}
    heads, receipts = _fit_heads(data, _fit_config(), tmp_path)
    order = json.loads((tmp_path / "exposure_order.json").read_text())["feature_indices"]
    assert sorted(order) == list(range(8))
    assert len(order) == len(set(order)) == 8
    assert set(heads) == set(receipts) == {"control", "semantic"}
    assert (
        receipts["control"]["initial_state_sha256"] == receipts["semantic"]["initial_state_sha256"]
    )
    assert (
        receipts["control"]["parameters"]["parameter_shapes"]
        == receipts["semantic"]["parameters"]["parameter_shapes"]
    )
    initial = torch.load(tmp_path / "initial_head.pt", weights_only=True)
    assert torch.count_nonzero(initial["output.weight"]) == 0
    assert torch.count_nonzero(initial["output.bias"]) == 0
    for arm, head in heads.items():
        assert receipts[arm]["updates"] == 2
        assert receipts[arm]["examples"] == 8
        assert len(receipts[arm]["losses"]) == 2
        assert receipts[arm]["parameters"]["use_semantics"] is (arm == "semantic")
        assert head.training is False
        checkpoint = torch.load(tmp_path / f"head_{arm}.pt", weights_only=True)
        assert checkpoint["arm"] == arm
        assert set(checkpoint) == {"arm", "head_state", "optimizer_state"}
        for name, value in head.state_dict().items():
            assert torch.equal(checkpoint["head_state"][name], value)
        assert torch.count_nonzero(checkpoint["head_state"]["output.weight"]) > 0
        assert {
            int(state["step"]) for state in checkpoint["optimizer_state"]["state"].values()
        } == {2}
        assert json.loads((tmp_path / f"training_{arm}.json").read_text()) == receipts[arm]
    assert all(torch.equal(before[name], value) for name, value in data.items())
    assert all(value.grad is None for value in data.values())


def test_fixed_exposure_rejects_incomplete_features_before_writing(tmp_path: Path) -> None:
    data = {name: value[:-1] for name, value in _features().items()}
    with pytest.raises(UgiSemanticResidualStudyError, match="fixed exposure"):
        _fit_heads(data, _fit_config(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_primary_placement_denominator_retains_absent_targets_but_excludes_empty_and_singleton() -> (
    None
):
    ranks = [
        rank_candidate_scores([], []),
        rank_candidate_scores([0.0], [0]),
        rank_candidate_scores([2.0, 1.0], [0]),
        rank_candidate_scores([2.0, 1.0], []),
    ]
    summary = _placement_summary(ranks)
    assert summary["attempts"] == 4
    assert summary["empty"] == 1
    assert summary["singleton"] == 1
    assert summary["target_absent"] == 2
    assert summary["selected_correct"] == 2
    assert summary["multi_candidate_attempts"] == 2
    assert summary["multi_candidate_selected_correct"] == 1
    assert summary["primary_accuracy"] == 0.5
    assert summary["informative_with_present_target"] == 1
    assert _placement_summary(ranks[:2])["primary_accuracy"] is None
    assert _placement_summary([])["primary_accuracy"] is None


def _passing_evaluation() -> dict:
    group = {
        arm: {
            "placement": {
                "primary_accuracy": placement,
                "informative_with_present_target": 3,
                "target_absent": 1,
            },
            "atoms": {"corrupted": {"accuracy": atom}},
        }
        for arm, placement, atom in (
            ("baseline", 0.25, 0.6),
            ("control", 0.5, 0.7),
            ("semantic", 0.75, 0.7),
        )
    }
    return {"component_disjoint": group}


def test_advancement_requires_strict_placement_gain_but_allows_equal_atom_accuracy() -> None:
    checks = _advancement(_passing_evaluation(), True, True)
    assert all(checks.values())
    assert checks["semantic_beats_baseline_placement"] is True
    assert checks["semantic_beats_control_placement"] is True
    assert checks["no_corrupted_atom_accuracy_loss_vs_control"] is True


@pytest.mark.parametrize(
    ("arm", "path", "value", "failed_check"),
    [
        ("semantic", ("placement", "primary_accuracy"), 0.25, "semantic_beats_baseline_placement"),
        ("semantic", ("placement", "primary_accuracy"), 0.5, "semantic_beats_control_placement"),
        (
            "semantic",
            ("atoms", "corrupted", "accuracy"),
            0.5,
            "no_corrupted_atom_accuracy_loss_vs_baseline",
        ),
        (
            "semantic",
            ("atoms", "corrupted", "accuracy"),
            0.65,
            "no_corrupted_atom_accuracy_loss_vs_control",
        ),
        ("baseline", ("placement", "primary_accuracy"), None, "semantic_beats_baseline_placement"),
        ("control", ("atoms", "corrupted", "accuracy"), None, "semantic_beats_control_placement"),
        (
            "semantic",
            ("placement", "informative_with_present_target"),
            0,
            "nonempty_informative_population",
        ),
        ("semantic", ("placement", "target_absent"), 2, "no_target_presence_loss"),
    ],
)
def test_each_outcome_gate_fails_closed(
    arm: str, path: tuple[str, ...], value: object, failed_check: str
) -> None:
    evaluation = _passing_evaluation()
    destination = evaluation["component_disjoint"][arm]
    for field in path[:-1]:
        destination = destination[field]
    destination[path[-1]] = value
    checks = _advancement(evaluation, True, True)
    assert checks[failed_check] is False
    assert not all(checks.values())


@pytest.mark.parametrize(
    ("same_universe", "unchanged_padding", "failed_check"),
    [(False, True, "same_candidate_symbol_universe"), (True, False, "unmodified_padding")],
)
def test_structural_preservation_gates_override_better_outcomes(
    same_universe: bool, unchanged_padding: bool, failed_check: str
) -> None:
    checks = _advancement(_passing_evaluation(), same_universe, unchanged_padding)
    assert checks[failed_check] is False
    assert checks["semantic_beats_baseline_placement"] is True
    assert checks["semantic_beats_control_placement"] is True
    assert not all(checks.values())


def test_paired_target_loss_blocks_despite_favorable_pooled_coverage_and_accuracy() -> None:
    evaluation = _passing_evaluation()
    evaluation["component_disjoint"]["semantic"]["placement"]["target_absent"] = 0
    checks = _advancement(evaluation, True, True, paired_target_losses=1)
    assert checks["no_target_presence_loss"] is True
    assert checks["semantic_beats_baseline_placement"] is True
    assert checks["semantic_beats_control_placement"] is True
    assert checks["no_paired_target_presence_loss_in_either_population"] is False
    assert not all(checks.values())


def _pin(path: Path, repo: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(repo).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_fixture_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def saved_fit(tmp_path: Path) -> dict:
    prior = tmp_path / "prior"
    prior.mkdir()
    data = _features()
    fit_config = _fit_config()
    heads, training = _fit_heads(data, fit_config, prior)
    selection = {
        group: {"draws": [{"product_id": f"fixture-{group}-{index}"} for index in range(2)]}
        for group in ("fit", "component_disjoint", "repeated_component")
    }
    for group in selection:
        np.savez_compressed(
            prior / f"features_{group}.npz", **{key: value.numpy() for key, value in data.items()}
        )
    _write_fixture_json(prior / "selection.json", selection)
    _write_fixture_json(
        prior / "failure.json",
        {"status": "failed_no_admitted_result", "error": "fixture evaluation failure"},
    )
    source_names = (
        "forge/model/ugi_semantic_placement.py",
        "experiments/phase1/multireaction/ugi_semantic_residual.py",
        "forge/model/ugi_semantic_residual.py",
    )
    sources, archived = {}, {}
    for index, name in enumerate(source_names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# test-only source receipt {index}\n")
        sources[name] = _pin(path, tmp_path)
        archive = tmp_path / "source_archive" / f"{index}.py"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        archived[name] = _pin(archive, tmp_path)
    support = tmp_path / "fixture_support.txt"
    support.write_text("test-only support receipt\n")
    original = {
        **fit_config,
        "flow_times": [0.2, 0.5, 0.8, 0.95],
        "sources": sources,
        "inputs": {"fixture_support": _pin(support, tmp_path)},
    }
    original_path = tmp_path / "original_config.json"
    _write_fixture_json(original_path, original)
    manifest = {
        "prior_config": _pin(original_path, tmp_path),
        "source_archive": archived,
        "artifacts": {path.name: _pin(path, tmp_path) for path in sorted(prior.iterdir())},
    }
    manifest_path = tmp_path / "recovery_manifest.json"
    _write_fixture_json(manifest_path, manifest)
    recovery = copy.deepcopy(original)
    # Both permitted repair sources change while their original source bytes remain archived.
    for name in source_names[:2]:
        path = tmp_path / name
        path.write_text("# test-only evaluation repair\n")
        recovery["sources"][name] = _pin(path, tmp_path)
    recovery["inputs"]["recovery_manifest"] = _pin(manifest_path, tmp_path)
    return {
        "repo": tmp_path,
        "prior": prior,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "config": recovery,
        "selection": selection,
        "data": data,
        "heads": heads,
        "training": training,
    }


def test_recovery_authenticates_original_final_heads_and_features_without_refitting(
    saved_fit: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("evaluation recovery must never refit")

    monkeypatch.setattr(runner, "_fit_heads", forbidden_fit)
    before = {path.name: path.read_bytes() for path in saved_fit["prior"].iterdir()}
    features, heads, training, manifest = _resume_saved_fit(
        saved_fit["manifest_path"], saved_fit["config"], saved_fit["selection"], saved_fit["repo"]
    )
    assert training == saved_fit["training"]
    assert manifest == saved_fit["manifest"]
    for group, values in features.items():
        assert group in saved_fit["selection"]
        for name, value in values.items():
            assert torch.equal(value, saved_fit["data"][name])
    for arm, head in heads.items():
        assert head.training is False
        assert head.use_semantics is (arm == "semantic")
        for name, value in head.state_dict().items():
            assert torch.equal(value, saved_fit["heads"][arm].state_dict()[name])
    assert {path.name: path.read_bytes() for path in saved_fit["prior"].iterdir()} == before


def test_recovery_rejects_changed_selection(saved_fit: dict) -> None:
    changed = copy.deepcopy(saved_fit["selection"])
    changed["fit"]["draws"].reverse()
    with pytest.raises(UgiSemanticResidualStudyError, match="selected examples"):
        _resume_saved_fit(
            saved_fit["manifest_path"], saved_fit["config"], changed, saved_fit["repo"]
        )


@pytest.mark.parametrize("change", ["learning_rate", "inputs", "nonrepair_source"])
def test_recovery_rejects_changes_outside_evaluation_repair(saved_fit: dict, change: str) -> None:
    changed = copy.deepcopy(saved_fit["config"])
    if change == "learning_rate":
        changed["learning_rate"] *= 2
    elif change == "inputs":
        changed["inputs"]["fixture_support"]["sha256"] = "0" * 64
    else:
        changed["sources"]["forge/model/ugi_semantic_residual.py"]["sha256"] = "0" * 64
    with pytest.raises(UgiSemanticResidualStudyError, match="recovery changes"):
        _resume_saved_fit(
            saved_fit["manifest_path"], changed, saved_fit["selection"], saved_fit["repo"]
        )


@pytest.mark.parametrize(
    "target", ["head_semantic.pt", "features_fit.npz", "source_archive", "prior_config"]
)
def test_recovery_rejects_original_receipt_byte_drift(saved_fit: dict, target: str) -> None:
    if target == "source_archive":
        pin = next(iter(saved_fit["manifest"]["source_archive"].values()))
    elif target == "prior_config":
        pin = saved_fit["manifest"]["prior_config"]
    else:
        pin = saved_fit["manifest"]["artifacts"][target]
    path = saved_fit["repo"] / pin["path"]
    path.write_bytes(path.read_bytes() + b"\nchanged fixture bytes")
    with pytest.raises(PinError, match="changed"):
        _resume_saved_fit(
            saved_fit["manifest_path"],
            saved_fit["config"],
            saved_fit["selection"],
            saved_fit["repo"],
        )


def test_rehashed_checkpoint_drift_still_fails_saved_training_digest(saved_fit: dict) -> None:
    path = saved_fit["prior"] / "head_semantic.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["head_state"]["output.bias"] += 1
    torch.save(checkpoint, path)
    manifest = copy.deepcopy(saved_fit["manifest"])
    manifest["artifacts"][path.name] = _pin(path, saved_fit["repo"])
    _write_fixture_json(saved_fit["manifest_path"], manifest)
    with pytest.raises(UgiSemanticResidualStudyError, match="original final checkpoint"):
        _resume_saved_fit(
            saved_fit["manifest_path"],
            saved_fit["config"],
            saved_fit["selection"],
            saved_fit["repo"],
        )


def test_recovery_rejects_completed_evaluation_receipt(saved_fit: dict) -> None:
    path = saved_fit["prior"] / "evaluation_component_disjoint.json"
    _write_fixture_json(path, {"status": "complete_test_fixture"})
    manifest = copy.deepcopy(saved_fit["manifest"])
    manifest["artifacts"][path.name] = _pin(path, saved_fit["repo"])
    _write_fixture_json(saved_fit["manifest_path"], manifest)
    with pytest.raises(UgiSemanticResidualStudyError, match="incomplete evaluation"):
        _resume_saved_fit(
            saved_fit["manifest_path"],
            saved_fit["config"],
            saved_fit["selection"],
            saved_fit["repo"],
        )
