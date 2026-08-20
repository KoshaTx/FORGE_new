from __future__ import annotations

import inspect

import pytest

from forge.potency.endpoint import EndpointError, endpoint_ids, load_endpoint


def test_endpoint_registry_has_no_implicit_default() -> None:
    assert endpoint_ids() == (
        "im_functional_editing",
        "im_vaccination",
        "liver_functional_editing",
    )
    assert (
        inspect.signature(load_endpoint).parameters["endpoint_id"].default
        is inspect.Parameter.empty
    )


@pytest.mark.parametrize("endpoint_id", endpoint_ids())
def test_each_endpoint_has_complete_required_stages(endpoint_id: str) -> None:
    endpoint = load_endpoint(endpoint_id)
    specification = endpoint.specification
    specification.validate()
    required = [readout for readout in specification.readouts if readout.priority == "required"]
    assert {readout.stage for readout in required} == {"bridge", "in_vivo", "safety"}
    assert all(readout.prospective_only for readout in specification.readouts)
    assert endpoint.missing_required_readouts({}) == endpoint.required_readout_ids()


def test_vaccine_stub_does_not_treat_expression_as_immunogenicity() -> None:
    specification = load_endpoint("im_vaccination").specification
    disallowed = " ".join(specification.disallowed_inferences).lower()
    required_ids = set(load_endpoint("im_vaccination").required_readout_ids())
    assert "reporter expression alone" in disallowed
    assert "antigen_specific_response" in required_ids
    assert "second_adaptive_immunity_arm" not in required_ids


def test_liver_stub_requires_sequence_verified_function() -> None:
    endpoint = load_endpoint("liver_functional_editing")
    required_ids = set(endpoint.required_readout_ids())
    assert "sequence_verified_liver_editing" in required_ids
    assert "hepatocyte_expression_or_editing" in required_ids
    assert "reporter_delivery_and_biodistribution" in required_ids
    assert "dose_response" not in required_ids


def test_muscle_stub_separates_reporter_delivery_from_editing() -> None:
    endpoint = load_endpoint("im_functional_editing")
    required_ids = set(endpoint.required_readout_ids())
    disallowed = " ".join(endpoint.specification.disallowed_inferences).lower()
    assert "intramuscular_reporter_delivery" in required_ids
    assert "functional_muscle_editing" in required_ids
    assert "reporter expression alone" in disallowed


def test_unknown_endpoint_is_rejected() -> None:
    with pytest.raises(EndpointError, match="unknown endpoint"):
        load_endpoint("not_an_endpoint")  # type: ignore[arg-type]
