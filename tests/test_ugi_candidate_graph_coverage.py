"""TRAIN boundaries, source measure, and independent two-component partitioning."""

from __future__ import annotations

import copy

import pytest

import experiments.phase1.multireaction.ugi_candidate_graph_coverage as coverage


def fixture_data():
    config = {
        "draws": {
            "fit": 1024,
            "amine_disjoint": 256,
            "aldehyde_disjoint": 256,
            "repeated_component": 256,
        },
        "seeds": {"draws": 2026090825},
        "partition": {
            "component_domain": "forge.ugi_candidate_graph_component_partition.v1:2026090824:",
            "product_domain": "forge.ugi_candidate_graph_product_partition.v1:2026090824:",
            "bucket_count": 5,
            "evaluation_bucket": 0,
        },
    }
    # Fixed known hash buckets: product-7 is repeated; CC=O is withheld, CCC=O is not.
    identities = [
        ("product-0", "CN", "CCC=O", 1.0, True),
        ("product-1", "CN", "CCC=O", 99.0, True),
        ("product-2", "NCCO", "CCC=O", 2.0, True),
        ("product-3", "CN", "CC=O", 3.0, True),
        ("product-4", "NCCO", "CC=O", 4.0, True),
        ("product-7", "CN", "CCC=O", 5.0, True),
        ("product-8", "CN", "CCC=O", 1000.0, False),
    ]
    rows, assignments = [], {}
    for index, (product, amine, aldehyde, weight, applicable) in enumerate(identities):
        rows.append(
            {
                "product_id": product,
                "cache_index": index,
                "amine_smiles": amine,
                "applicable": applicable,
                "source_weight": weight,
                "source_probability": weight / 1114.0,
                "source_stratum": "synthetic_fixture",
                "partition": "historical_partition",
                "reasons": [] if applicable else ["outside_frozen_support"],
                "candidate_count": 0,
                "target_present": False,
            }
        )
        assignments[product] = {
            "product_id": product,
            "primary_product_fold": "train",
            "component_smiles": dict(zip(coverage.ROLES, (amine, aldehyde), strict=True)),
        }
    return rows, assignments, config


def test_split_preserves_source_measure_absences_exclusions_and_overlap():
    rows, assignments, config = fixture_data()
    original = copy.deepcopy((rows, assignments, config))
    result = coverage.build_selection(rows, assignments, config)
    assert (rows, assignments, config) == original
    assert result == coverage.build_selection(list(reversed(rows)), assignments, config)
    groups, receipt = result["groups"], result["coverage"]
    assert receipt["all_train"]["rows"] == 7
    assert receipt["applicable"]["rows"] == 6
    assert receipt["excluded"]["rows"] == 1
    assert receipt["excluded"]["source_weight_sum"] == 1000.0
    assert receipt["exclusion_reasons"] == {"outside_frozen_support": 1}
    assert receipt["component_disjoint_evaluation_overlap"]["product_ids"] == ["product-4"]
    for name, expected in {
        "fit": 2,
        "amine_disjoint": 2,
        "aldehyde_disjoint": 2,
        "repeated_component": 1,
    }.items():
        assert groups[name]["population"]["rows"] == expected
        assert len(groups[name]["draws"]) == config["draws"][name]
        assert all(
            row["candidate_count"] == 0 and not row["target_present"]
            for row in groups[name]["draws"]
        )
        assert all(row["partition"] == "historical_partition" for row in groups[name]["draws"])
    fit = groups["fit"]["draws"]
    assert sum(row["product_id"] == "product-1" for row in fit) > 990
    assert {row["candidate_graph_draw_probability"] for row in fit} == {0.01, 0.99}
    for role, group in zip(coverage.ROLES, ("amine_disjoint", "aldehyde_disjoint"), strict=True):
        assert not {row["component_smiles"][role] for row in fit} & {
            row["component_smiles"][role] for row in groups[group]["draws"]
        }
        assert groups[group]["draws_with_component_seen_in_actual_fit_draws"][role] == 0


def test_canonical_identity_controls_split_not_source_string():
    rows, assignments, config = fixture_data()
    expected = coverage.build_selection(rows, assignments, config)
    assignments["product-0"]["component_smiles"]["amine_head"] = "NC"
    rows[0]["amine_smiles"] = "NC"
    result = coverage.build_selection(rows, assignments, config)
    assert result["coverage_rows"][0]["component_smiles"]["amine_head"] == "CN"
    assert (
        result["coverage_rows"][0]["component_withheld"]
        == expected["coverage_rows"][0]["component_withheld"]
    )


class MaskOnly(dict):
    def __getitem__(self, key):
        if key != "primary_product_fold":
            raise AssertionError("non-TRAIN identity or structure was accessed")
        return "sealed"


def test_loader_masks_before_structure_and_identity_access(monkeypatch, tmp_path):
    train = {
        "primary_product_fold": "train",
        "product_id": "fixture",
        "amine_head_smiles": "N[C@@H](C)CO",
        "oxoester_aldehyde_body_tail_smiles": "O=CCC",
    }
    monkeypatch.setattr(coverage, "iter_csv", lambda path: iter([MaskOnly(), train]))
    result = coverage.load_train_assignments(tmp_path / "fixture.csv")
    assert set(result) == {"fixture"}
    assert result["fixture"]["component_smiles"] == {
        "amine_head": "CC(N)CO",
        "oxoester_aldehyde_body_tail": "CCC=O",
    }
    monkeypatch.setattr(coverage, "iter_csv", lambda path: iter([train, train]))
    with pytest.raises(coverage.UgiCandidateGraphCoverageError, match="duplicate TRAIN"):
        coverage.load_train_assignments(tmp_path / "fixture.csv")


def test_builder_masks_supplied_nontrain_assignment_before_structure():
    rows, assignments, config = fixture_data()
    assignments[rows[0]["product_id"]] = MaskOnly()
    with pytest.raises(coverage.UgiCandidateGraphCoverageError, match="explicit TRAIN"):
        coverage.build_selection(rows, assignments, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_weight", 0),
        ("source_weight", float("nan")),
        ("source_probability", -1),
        ("applicable", "true"),
    ],
)
def test_bad_coverage_measure_and_applicability_fail(field, value):
    rows, assignments, config = fixture_data()
    rows[0][field] = value
    with pytest.raises(coverage.UgiCandidateGraphCoverageError):
        coverage.build_selection(rows, assignments, config)


def test_duplicate_ids_missing_assignments_and_empty_partitions_fail():
    rows, assignments, config = fixture_data()
    with pytest.raises(coverage.UgiCandidateGraphCoverageError, match="duplicate coverage"):
        coverage.build_selection(rows + [rows[0]], assignments, config)
    missing = dict(assignments)
    del missing[rows[-1]["product_id"]]
    with pytest.raises(coverage.UgiCandidateGraphCoverageError, match="explicit TRAIN"):
        coverage.build_selection(rows, missing, config)
    rows[2]["applicable"] = rows[4]["applicable"] = False
    with pytest.raises(coverage.UgiCandidateGraphCoverageError, match="empty population"):
        coverage.build_selection(rows, assignments, config)
