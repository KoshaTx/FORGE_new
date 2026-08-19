from __future__ import annotations

from pathlib import Path

import numpy as np

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_morphology_corpus import (
    balanced_product_weights,
    component_marginal_errors,
    component_program_ledger,
    expanded_component_census,
    family_balanced_component_weights,
    load_expanded_ugi_morphology_corpus,
    load_ugi_morphology_corpus,
    sample_family_balanced_records,
    source_stratified_family_weights,
)

REPO = Path(__file__).resolve().parents[1]


def test_balanced_weights_make_each_role_component_marginal_uniform() -> None:
    rows = []
    for amine in ("a0", "a1"):
        for aldehyde in ("b0", "b1", "b2"):
            for isocyanide in ("c0", "c1"):
                repeats = 4 if (amine, aldehyde, isocyanide) == ("a0", "b0", "c0") else 1
                rows.extend(
                    {
                        "amine_head_smiles": amine,
                        "oxoester_aldehyde_body_tail_smiles": aldehyde,
                        "isocyanide_tail_smiles": isocyanide,
                    }
                    for _ in range(repeats)
                )
    weights = balanced_product_weights(rows)
    errors = component_marginal_errors(rows, weights)
    uniform_errors = component_marginal_errors(rows, np.ones(len(rows)))

    assert np.isclose(weights.sum(), 1.0)
    assert all(errors[role] < uniform_errors[role] for role in ROLE_NAMES)
    assert weights.max() / weights.min() < 12.0


def test_source_stratified_family_weights_freeze_source_mass() -> None:
    rows = []
    for source, repeats in (("realism", 5), ("expanded", 2)):
        for family_index in range(2):
            rows.extend(
                {
                    "source_stratum": source,
                    "amine_head_family_id": f"a{family_index}",
                    "oxoester_aldehyde_body_tail_family_id": f"b{family_index}",
                    "isocyanide_tail_family_id": f"c{family_index}",
                }
                for _ in range(repeats if family_index == 0 else 1)
            )
    weights = source_stratified_family_weights(
        rows,
        source_mass={"realism": 0.6, "expanded": 0.4},
    )

    assert np.isclose(weights.sum(), 1.0)
    for source, expected in (("realism", 0.6), ("expanded", 0.4)):
        indices = [index for index, row in enumerate(rows) if row["source_stratum"] == source]
        assert np.isclose(weights[indices].sum(), expected)
        for role in ROLE_NAMES:
            mass = {}
            for index in indices:
                family = rows[index][f"{role}_family_id"]
                mass[family] = mass.get(family, 0.0) + float(weights[index]) / expected
            assert max(mass.values()) - min(mass.values()) < 0.34


def test_full_ugi_corpus_materializes_unique_components_and_strict_folds() -> None:
    corpus = load_ugi_morphology_corpus(
        REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz",
        REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_products.csv.gz",
        REPO / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_atoms.csv.gz",
        REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )

    assert {fold: len(records) for fold, records in corpus.records_by_fold.items()} == {
        "train": 4362,
        "calibration": 3350,
        "heldout": 4674,
    }
    assert {
        role: sum(key[0] == role for key in corpus.unique_components) for role in ROLE_NAMES
    } == {
        "amine_head": 24,
        "oxoester_aldehyde_body_tail": 62,
        "isocyanide_tail": 9,
    }
    ledger = component_program_ledger(corpus)
    assert len(ledger) == 95
    for role in ROLE_NAMES:
        assert np.isclose(
            sum(float(row["component_weight"]) for row in ledger if row["role"] == role),
            1.0,
        )


def test_expanded_corpus_caches_components_and_samples_frozen_families() -> None:
    corpus = load_expanded_ugi_morphology_corpus(
        REPO / "results/phase1/ugi_expanded_exemplars/component_exemplar_ledger.csv.gz",
        REPO / "results/phase1/ugi_expanded_exemplars/semantic_products.csv.gz",
        REPO / "results/phase1/ugi_expanded_exemplars/semantic_atoms.csv.gz",
        REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )
    census = expanded_component_census(corpus)

    assert census["unique_components"] == 424
    assert census["component_counts"] == {
        "amine_head": 264,
        "oxoester_aldehyde_body_tail": 107,
        "isocyanide_tail": 53,
    }
    assert census["semantic_exemplar_products"] == 421
    assert any(component.attachment_count == 2 for component in corpus.unique_components.values())
    weights = family_balanced_component_weights(corpus, fold="train")
    for role in ROLE_NAMES:
        assert np.isclose(
            sum(value for (key_role, _), value in weights.items() if key_role == role), 1
        )
    sampled = sample_family_balanced_records(
        corpus,
        fold="train",
        count=32,
        rng=np.random.default_rng(13),
    )
    for record in sampled:
        for component in record.components:
            metadata = corpus.component_metadata[(component.role, component.component_key)]
            assert metadata["family_fold"] == "train"
