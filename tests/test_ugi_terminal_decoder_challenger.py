from __future__ import annotations

from experiments.phase1.product_l1.sampling.ugi_terminal_decoder_challenger import (
    summarize_terminal_decoder_arm,
)


def test_terminal_decoder_summary_keeps_semantic_failures_in_denominator() -> None:
    sample = {
        "statistics": {"valid_molecules": 2},
        "samples": [
            {
                "valid": True,
                "component_reconstruction_valid": True,
                "l1_forward_verification": {"exact_product_reconstructed": True},
                "component_smiles_by_role": {
                    "oxoester_aldehyde_body_tail": "CCCC=O",
                    "isocyanide_tail": "[C-]#[N+]CCCC",
                },
            },
            {
                "valid": True,
                "component_reconstruction_valid": False,
                "l1_forward_verification": None,
                "component_smiles_by_role": None,
            },
        ],
    }

    summary = summarize_terminal_decoder_arm(sample)

    assert summary["valid_molecules"] == 2
    assert summary["exact_l1_eligible_molecules"] == 1
    assert summary["exact_l1_fraction_of_valid"] == 0.5
    assert summary["semantic_failures_among_valid"] == {
        "component_reconstruction_failure": 1,
        "reconstructed_but_not_exact_l1": 0,
    }
    assert (
        summary["tail_chemotypes_exact_l1_eligible_only"]["oxoester_aldehyde_body_tail"][
            "component_occurrences"
        ]
        == 1
    )
