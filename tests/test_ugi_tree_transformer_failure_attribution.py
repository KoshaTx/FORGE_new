from __future__ import annotations

from typing import Any

import pytest

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_failure_attribution import (
    UgiTreeTransformerFailureAttributionError,
    classify_attempt,
    unsupported_edge_signatures,
)


def _sample(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "valid": True,
        "failure_type": None,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {"exact_product_reconstructed": True},
    }
    value.update(overrides)
    return value


def _common(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"valid": True, "exact_l1_program": True}
    value.update(overrides)
    return value


def _local(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "raw_exact_l1_program": True,
        "local_support_qualified_exact_l1": True,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("sample", "common", "local", "expected"),
    [
        (
            _sample(valid=False, failure_type="TerminalSupportFailure"),
            _common(valid=False, exact_l1_program=False),
            _local(raw_exact_l1_program=False, local_support_qualified_exact_l1=False),
            "invalid_terminal_support",
        ),
        (
            _sample(valid=False, failure_type="MoleculeSanitizationFailure"),
            _common(valid=False, exact_l1_program=False),
            _local(raw_exact_l1_program=False, local_support_qualified_exact_l1=False),
            "invalid_molecule_sanitization",
        ),
        (
            _sample(),
            _common(exact_l1_program=False),
            _local(raw_exact_l1_program=False, local_support_qualified_exact_l1=False),
            "valid_native_forward_exact_retro_abstention",
        ),
        (
            _sample(),
            _common(),
            _local(local_support_qualified_exact_l1=False),
            "exact_l1_unsupported_local_chemistry",
        ),
        (
            _sample(),
            _common(),
            _local(),
            "exact_l1_local_supported",
        ),
    ],
)
def test_primary_attribution_is_mutually_exclusive(
    sample: dict[str, Any],
    common: dict[str, Any],
    local: dict[str, Any],
    expected: str,
) -> None:
    assert classify_attempt(sample, common, local) == expected


def test_retro_abstention_requires_independent_native_forward_exactness() -> None:
    with pytest.raises(
        UgiTreeTransformerFailureAttributionError,
        match="not native-forward exact",
    ):
        classify_attempt(
            _sample(
                component_reconstruction_valid=False,
                l1_forward_verification=None,
            ),
            _common(exact_l1_program=False),
            _local(raw_exact_l1_program=False, local_support_qualified_exact_l1=False),
        )


def test_unsupported_edge_signatures_are_recomputed_from_policy() -> None:
    class CarbonOnlySupport:
        def allows_program_edge(
            self,
            program_id: str,
            left: str,
            bond_state: int,
            right: str,
        ) -> bool:
            assert program_id == "ugi_3cr_agile"
            return left == right == "C"

    assert unsupported_edge_signatures("CCN", CarbonOnlySupport()) == ["C-N:SINGLE"]  # type: ignore[arg-type]
