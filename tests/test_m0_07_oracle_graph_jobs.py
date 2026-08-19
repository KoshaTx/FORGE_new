from __future__ import annotations

import csv
import gzip
from collections import Counter
from pathlib import Path

from forge.bio.oracle_graph_jobs import build_graph_job_rows

REPO = Path(__file__).resolve().parents[1]


def _assignments() -> list[dict[str, str]]:
    with gzip.open(
        REPO / "results/m0_07/oracle_split_assignments.csv.gz",
        "rt",
        newline="",
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_graph_job_matrix_is_complete_and_deterministic() -> None:
    arguments = {
        "architectures": (
            "whole_graph_dmpnn",
            "whole_graph_edge_gin",
            "ugi_component_role_aware_dmpnn",
        ),
        "endpoints": ("expt_Hela", "expt_Raw"),
        "seeds": (1729, 11729, 21729),
        "selection_schemes": {
            "lantern_scaffold_balanced",
            "held_head_5fold",
            "held_aldehyde_5fold",
            "held_isocyanide_5fold",
            "held_head_aldehyde_pair_5fold",
            "held_head_isocyanide_pair_5fold",
            "held_aldehyde_isocyanide_pair_5fold",
        },
        "fit_output_root": "results/m0_07/graph_fits",
    }
    first = build_graph_job_rows(_assignments(), **arguments)
    second = build_graph_job_rows(_assignments(), **arguments)

    assert first == second
    assert len(first) == 576
    assert len({row["job_id"] for row in first}) == 576
    assert len({row["ensemble_id"] for row in first}) == 192
    assert sum(row["selection_eligible"] == "true" for row in first) == 558
    assert Counter(row["architecture"] for row in first) == {
        "whole_graph_dmpnn": 192,
        "whole_graph_edge_gin": 192,
        "ugi_component_role_aware_dmpnn": 192,
    }
    assert Counter(row["endpoint"] for row in first) == {
        "expt_Hela": 288,
        "expt_Raw": 288,
    }


def test_graph_jobs_preserve_frozen_partition_sizes() -> None:
    rows = build_graph_job_rows(
        _assignments(),
        architectures=("whole_graph_dmpnn",),
        endpoints=("expt_Hela",),
        seeds=(1729,),
        selection_schemes={"lantern_scaffold_balanced"},
        fit_output_root="results/m0_07/graph_fits",
    )
    by_partition = {(row["scheme"], row["fold"]): row for row in rows}

    assert len(by_partition) == 32
    assert (
        by_partition[("lantern_scaffold_balanced", 0)]["train_rows"],
        by_partition[("lantern_scaffold_balanced", 0)]["calibration_rows"],
        by_partition[("lantern_scaffold_balanced", 0)]["test_rows"],
    ) == (880, 110, 110)
    assert (
        by_partition[("held_aldehyde_5fold", 0)]["train_rows"],
        by_partition[("held_aldehyde_5fold", 0)]["calibration_rows"],
        by_partition[("held_aldehyde_5fold", 0)]["test_rows"],
    ) == (700, 100, 300)
    assert (
        by_partition[("held_aldehyde_5fold", 1)]["train_rows"],
        by_partition[("held_aldehyde_5fold", 1)]["calibration_rows"],
        by_partition[("held_aldehyde_5fold", 1)]["test_rows"],
    ) == (788, 112, 200)
