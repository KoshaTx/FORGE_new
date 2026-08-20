from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments/archive/producers/phase1_collect_ugi_production_refit_fresh_census.py"
SPEC = importlib.util.spec_from_file_location(
    "phase1_collect_ugi_production_refit_fresh_census", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_component_similarity_distinguishes_exact_and_interpolative_components() -> None:
    result = MODULE._nearest_component_similarity(
        ["CCCC", "CCCCC", "c1ccccc1"],
        ["CCCC", "CCCCCC"],
    )
    assert result["generated_unique_components"] == 3
    assert result["reference_unique_components"] == 2
    assert result["fraction_exact_reference_identity"] == 1 / 3
    assert result["nearest_reference_tanimoto_quantiles"]["minimum"] < 0.4
    assert result["nearest_reference_tanimoto_quantiles"]["maximum"] == 1.0


def test_fresh_census_contract_uses_independent_nonselection_seeds() -> None:
    config = json.loads(
        (REPO / "configs/model/phase1_ugi_production_refit_fresh_census_v1.json").read_text()
    )
    design = config["design"]
    assert design["attempted_draws"] == 4096
    assert len(set(design["program_seeds"])) == 4
    assert set(design["program_seeds"]).isdisjoint(design["flow_seeds"])
    assert set(design["program_seeds"]).isdisjoint(design["terminal_seeds"])
    assert set(design["flow_seeds"]).isdisjoint(design["terminal_seeds"])
    assert config["decision_policy"]["checkpoint_or_threshold_selection"] is False
    assert (
        config["decision_policy"]["fresh_census_outputs_reused_for_prospective_selection"] is False
    )
