from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from forge.data.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/model/phase1_ugi_decoration_coupling_production_fresh_census_v1.json"
COLLECTOR_PATH = (
    REPO / "scripts/phase1_collect_ugi_decoration_coupling_production_fresh_census_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "phase1_collect_ugi_decoration_coupling_production_fresh_census_v1",
    COLLECTOR_PATH,
)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_fresh_census_contract_is_frozen_and_independent() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    design = config["design"]

    assert config["status"] == "frozen_before_independent_production_census"
    assert design["attempted_draws"] == 4096
    assert design["shards"] == 4
    assert design["programs_per_shard"] == 1024
    assert design["terminal_decoder_mode"] == "stochastic"
    assert design["retry_or_repair"] is False

    program_seeds = set(design["program_seeds"])
    flow_seeds = set(design["flow_seeds"])
    terminal_seeds = set(design["terminal_seeds"])
    assert len(program_seeds) == len(flow_seeds) == len(terminal_seeds) == 4
    assert program_seeds.isdisjoint(flow_seeds)
    assert program_seeds.isdisjoint(terminal_seeds)
    assert flow_seeds.isdisjoint(terminal_seeds)


def test_fresh_census_programs_and_implementation_are_hash_pinned() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    for record in (*config["inputs"].values(), *config["implementation"].values()):
        path = REPO / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]

    for shard, seed in enumerate(config["design"]["program_seeds"]):
        record = config["inputs"][f"program_shard_{shard:02d}"]
        schedule = json.loads((REPO / record["path"]).read_text())
        assert schedule["seed"] == seed
        assert (
            schedule["input_prior"]["sha256"]
            == config["inputs"]["all_fold_program_prior"]["sha256"]
        )
        assert len(schedule["samples"]) == config["design"]["programs_per_shard"]


def test_fresh_census_gates_are_prespecified_and_nonselecting() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    policy = config["decision_policy"]

    assert policy["checkpoint_or_threshold_selection"] is False
    assert policy["fresh_census_outputs_reused_for_model_selection"] is False
    assert policy["minimum_valid_fraction"] == 0.95
    assert policy["required_exact_l1_fraction_of_valid"] == 1.0
    assert policy["minimum_unique_fraction_of_valid"] == 0.98
    assert policy["minimum_three_handle_fraction_of_reconstructed"] == 0.99
    assert policy["required_zero_adjacent_tail_branch_components"] is True
    assert policy["chemistry_and_branch_metrics_are_descriptive_not_posthoc_gates"] is True


def test_component_similarity_distinguishes_exact_and_interpolative_components() -> None:
    result = COLLECTOR._nearest_component_similarity(
        ["CCCC", "CCCCC", "c1ccccc1"],
        ["CCCC", "CCCCCC"],
    )
    assert result["generated_unique_components"] == 3
    assert result["reference_unique_components"] == 2
    assert result["fraction_exact_reference_identity"] == 1 / 3
    assert result["nearest_reference_tanimoto_quantiles"]["minimum"] < 0.4
    assert result["nearest_reference_tanimoto_quantiles"]["maximum"] == 1.0
