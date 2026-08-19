from __future__ import annotations

from forge.product.ugi_branch_spacing_checkpoint_promotion import _branch_contract


def _sample(offspring: list[int]) -> dict:
    return {
        "samples": [
            {
                "program": {"attachment_counts": [1, 1, 1]},
                "offspring_by_role": {
                    "amine_head": offspring,
                    "oxoester_aldehyde_body_tail": offspring,
                    "isocyanide_tail": offspring,
                },
            }
        ]
    }


def test_branch_contract_uses_decoded_tree_adjacency() -> None:
    separated_in_preorder_but_adjacent_in_tree = [2, 1, 0, 2, 0, 0]

    observed = _branch_contract(_sample(separated_in_preorder_but_adjacent_in_tree), [2, 1, 1])

    assert observed["maximum_observed_by_role"] == {
        "amine_head": 2,
        "oxoester_aldehyde_body_tail": 2,
        "isocyanide_tail": 2,
    }
    assert observed["status"] == "fail"
    assert observed["violation_indices"] == [0]


def test_branch_contract_passes_path_like_tail_forests() -> None:
    observed = _branch_contract(_sample([2, 1, 0, 1, 0]), [2, 1, 1])

    assert observed["status"] == "pass"
    assert observed["violation_count"] == 0
