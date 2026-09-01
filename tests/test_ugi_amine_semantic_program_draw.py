from __future__ import annotations

import copy

import pytest

from experiments.phase1.multireaction.ugi_amine_semantic_program_draw import (
    UgiAmineSemanticProgramDrawError,
    _programs,
)


def _document() -> dict:
    return {
        "samples": [
            {
                "sample_index": 0,
                "program": {
                    "node_counts": [6, 20, 12],
                    "junction_budgets": [1, 1, 0],
                    "cycle_ranks": [0, 0, 0],
                    "attachment_counts": [1, 1, 1],
                },
            }
        ]
    }


def test_semantic_draw_preserves_the_exact_coarse_program_order() -> None:
    programs = _programs(_document(), count=1)

    assert programs[0].node_counts == (6, 20, 12)
    assert programs[0].junction_budgets == (1, 1, 0)


def test_semantic_draw_rejects_reordered_indices() -> None:
    changed = copy.deepcopy(_document())
    changed["samples"][0]["sample_index"] = 4

    with pytest.raises(UgiAmineSemanticProgramDrawError, match="not ordered"):
        _programs(changed, count=1)
