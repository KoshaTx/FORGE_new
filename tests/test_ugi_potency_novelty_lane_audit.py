from forge.bio.ugi_potency_novelty_lane_audit import audit_pair_split_novelty


def test_pair_holdout_does_not_imply_component_identity_holdout() -> None:
    rows = [
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B1C1",
            "stage": "test",
        },
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B1C2",
            "stage": "train",
        },
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B2C1",
            "stage": "calibration",
        },
    ]
    result = audit_pair_split_novelty(rows, folds=(0,))
    assert result["all_test_pairs_unseen"] is True
    assert result["all_test_component_identities_seen_individually"] is True
    assert result["total_identity_novelty_patterns"] == {"aldehyde_seen__isocyanide_seen": 1}


def test_pair_audit_detects_exact_new_component() -> None:
    rows = [
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B3C1",
            "stage": "test",
        },
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B1C1",
            "stage": "train",
        },
        {
            "scheme": "held_aldehyde_isocyanide_pair_5fold",
            "fold": "0",
            "label": "A1B2C2",
            "stage": "calibration",
        },
    ]
    result = audit_pair_split_novelty(rows, folds=(0,))
    assert result["all_test_component_identities_seen_individually"] is False
    assert result["total_identity_novelty_patterns"] == {"aldehyde_exact_new__isocyanide_seen": 1}
