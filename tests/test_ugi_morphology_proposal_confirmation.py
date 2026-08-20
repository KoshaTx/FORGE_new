from __future__ import annotations

from forge.design.sampling.ugi_morphology_proposal_confirmation import (
    ConfirmationDesign,
    _keyed_seed,
)


def test_confirmation_design_has_48_shards() -> None:
    design = ConfirmationDesign(
        programs=3072,
        shard_programs=64,
        particle_seed_base=11,
        terminal_seed_base=12,
        device="cpu",
    )
    assert design.shard_count == 48


def test_confirmation_seed_is_coordinate_keyed() -> None:
    first = _keyed_seed(11, purpose="particle", population_index=4)
    assert first == _keyed_seed(11, purpose="particle", population_index=4)
    assert first != _keyed_seed(11, purpose="particle", population_index=5)
    assert first != _keyed_seed(11, purpose="terminal", population_index=4)
