from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.phase1.multireaction.mechanism_study import (
    TransformerMechanismStudyError,
    run_transformer_mechanism_study,
)
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results/phase1/shared_synthesis_program_mixed_cache_v1/cache.npz"
CONFIG = REPO / "configs/multireaction/shared_mixed_training_smoke_v1.json"


def test_expanded_cache_has_exact_folds_and_equal_family_training_measure() -> None:
    with SynthesisProgramProductionCache(CACHE) as cache:
        assert cache.fold_counts() == {
            "ugi_3cr_agile": {"train": 66_464, "calibration": 15_800, "heldout": 30_122},
            "bl_2023_repeated_aza_michael": {
                "train": 30_000,
                "calibration": 8_000,
                "heldout": 8_000,
            },
            "lx_2024_repeated_reductive_amination": {
                "train": 30_000,
                "calibration": 8_000,
                "heldout": 8_000,
            },
        }
        programs = cache.vocabulary.program_states[1:]
        measure = cache.training_measure({program: 1.0 / 3.0 for program in programs})
        assert np.isclose(measure.sum(), 1.0)
        for program in programs:
            assert np.isclose(measure[cache.indices(program_id=program)].sum(), 1.0 / 3.0)
        node_sources, bond_sources = cache.program_role_source_marginals(
            measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=4,
            probability_floor=1e-5,
            backoff_strength=1.0,
        )
        assert node_sources.shape == (
            len(cache.vocabulary.program_states),
            len(cache.vocabulary.role_states),
            len(cache.atom_vocabulary),
        )
        assert bond_sources.shape == (
            len(cache.vocabulary.program_states),
            len(cache.vocabulary.role_states),
            4,
        )
        assert np.all(node_sources > 0) and np.all(bond_sources > 0)
        assert np.allclose(node_sources.sum(axis=-1), 1.0)
        assert np.allclose(bond_sources.sum(axis=-1), 1.0)


def test_paid_full_profile_remains_authorization_blocked(tmp_path: Path) -> None:
    with pytest.raises(
        TransformerMechanismStudyError,
        match="profile is not authorized: full",
    ):
        run_transformer_mechanism_study(
            CONFIG,
            REPO,
            tmp_path / "output",
            work_dir=tmp_path / "work",
            profile="full",
            replicate=0,
            allocated_device="cpu",
            resume=False,
        )
