from __future__ import annotations

from forge.potency.ugi_bounded_structural_envelope import (
    FEATURES,
    MODE_BOUNDED,
    MODE_OPEN,
    BoundedStructuralEnvelopeController,
    _bounded_census_action,
    _sha256_payload,
    lipid_native_features,
)


def _envelope(*, maximum: int = 100) -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": "phase1_ugi_bounded_structural_envelope.v1",
        "bounds_by_view": {
            view: {
                feature: {"minimum_inclusive": 0, "maximum_inclusive": maximum}
                for feature in FEATURES
            }
            for view in ("product", "amine", "aldehyde", "isocyanide")
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def test_lipid_native_features_are_small_graph_only_integer_contract() -> None:
    features = lipid_native_features("CC(=O)OCC")

    assert tuple(features) == FEATURES
    assert features["heavy_atoms"] == 6
    assert features["carbon_atoms"] == 4
    assert features["oxygen_atoms"] == 2
    assert features["other_hetero_atoms"] == 0
    assert features["ester_or_ether_count"] == 1
    assert all(isinstance(value, int) and value >= 0 for value in features.values())


def test_open_mode_is_identity_even_without_a_terminal_graph() -> None:
    controller = BoundedStructuralEnvelopeController(_envelope())

    assessment = controller.assess_terminal(
        mode=MODE_OPEN,
        valid_terminal=False,
        exact_l1=False,
        structures=None,
    )

    assert assessment.action == "identity_noop"
    assert assessment.active is False
    assert assessment.identity_multiplier == 1.0
    assert assessment.incremental_potential == 0.0


def test_bounded_mode_requires_valid_exact_l1_terminal() -> None:
    controller = BoundedStructuralEnvelopeController(_envelope())

    assessment = controller.assess_terminal(
        mode=MODE_BOUNDED,
        valid_terminal=True,
        exact_l1=False,
        structures=None,
    )

    assert assessment.action == "abstain_invalid_or_nonexact_terminal"
    assert assessment.active is False


def test_all_three_new_remains_abstained_inside_envelope() -> None:
    action = _bounded_census_action(
        ("amine", "aldehyde", "isocyanide"),
        envelope_inside=True,
        current_policy_action="applicability_supported_no_score",
    )

    assert action == "abstain_all_three_new_unsupported"
