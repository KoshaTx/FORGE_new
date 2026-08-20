import json
from pathlib import Path

from forge.design.schedule.ugi_grouped_smc_schedule_qualification import (
    build_grouped_smc_schedule_qualification,
)


def test_grouped_smc_schedule_has_nontrivial_within_program_ancestry() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = build_grouped_smc_schedule_qualification(
        repo,
        repo / "configs/model/phase1_ugi_grouped_smc_schedule_qualification_v1.json",
    )
    assert result["status"] == "grouped_smc_schedule_qualified_nonexecuting"
    assert len(result["seed_schedules"]) == 8
    all_program_hashes: set[str] = set()
    for seed_schedule in result["seed_schedules"]:
        assert len(seed_schedule["programs"]) == 16
        assert len(seed_schedule["particles"]) == 64
        seed_hashes = {item["morphology_program_sha256"] for item in seed_schedule["programs"]}
        assert len(seed_hashes) == 16
        assert not all_program_hashes & seed_hashes
        all_program_hashes.update(seed_hashes)
        for program_index in range(16):
            group = [
                item
                for item in seed_schedule["particles"]
                if item["program_index"] == program_index
            ]
            assert len(group) == 4
            assert len({item["morphology_program_sha256"] for item in group}) == 1
            assert len({item["stochastic_particle_seed"] for item in group}) == 4
    assert len(all_program_hashes) == 128
    assert result["design"]["morphology_program_overlap_across_seeds"] == 0
    assert result["interpretation"]["grouped_lambda_zero_bitwise_run_qualified"] is False
    assert result["scope"]["candidate_selection"] is False
    assert result["scope"]["sealed_holdout_accessed"] is False


def test_committed_grouped_smc_schedule_receipt_matches_builder() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads(
        (repo / "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json").read_text()
    )
    assert expected == build_grouped_smc_schedule_qualification(
        repo,
        repo / "configs/model/phase1_ugi_grouped_smc_schedule_qualification_v1.json",
    )
