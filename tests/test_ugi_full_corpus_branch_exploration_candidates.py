from __future__ import annotations

import subprocess
import sys

from forge.design.corpus.ugi_full_corpus_branch_exploration_candidates import (
    program_shard,
    realized_branch_class,
)


def test_program_shard_preserves_branch_schedule_coordinates() -> None:
    records = []
    for index, branch_class in enumerate(
        ("aldehyde_origin_branched", "isocyanide_origin_branched")
    ):
        records.append(
            {
                "draw_index": index,
                "branch_class": branch_class,
                "program_sha256": str(index) * 64,
                "program": {
                    "node_counts": [4, 16, 12],
                    "junction_budgets": [0, int(index == 0), int(index == 1)],
                    "cycle_ranks": [0, 0, 0],
                    "attachment_counts": [1, 1, 1],
                },
            }
        )
    schedule = {
        "schema_version": "forge.ugi_branch_exploration_schedule.v1",
        "status": "frozen_before_branch_exploration_generation",
        "schedule_sha256": "a" * 64,
        "design": {"seed": 7},
        "records": records,
    }

    shard = program_shard(schedule, shard_index=0, shard_draws=2)

    assert [row["product_id"] for row in shard["samples"]] == [
        "branch-exploration-v1-00000",
        "branch-exploration-v1-00001",
    ]
    assert shard["stratum_branch_counts"] == {
        "aldehyde_origin_branched": 1,
        "isocyanide_origin_branched": 1,
    }


def test_realized_branch_class_uses_carbon_topology_not_program_label() -> None:
    branch, descriptors = realized_branch_class(
        {
            "oxoester_aldehyde_body_tail": "CCCCCC(C)CCC(=O)OCCCCCC=O",
            "isocyanide_tail": "CCCCCCCCCCCC[N+]#[C-]",
        }
    )

    assert branch == "aldehyde_origin_branched"
    aldehyde = descriptors["oxoester_aldehyde_body_tail"]
    assert aldehyde["carbon_atoms"] == 16
    assert aldehyde["carbon_branch_points"] == 1
    assert aldehyde["adjacent_carbon_branch_edges"] == 0
    assert aldehyde["ester_carbonyls"] == 1


def test_generation_contract_does_not_import_analysis_only_sklearn() -> None:
    code = """
import importlib.abc
import sys

class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise RuntimeError('generation imported analysis-only sklearn')
        return None

sys.meta_path.insert(0, BlockSklearn())
import forge.design.corpus.ugi_full_corpus_branch_exploration_candidates
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
