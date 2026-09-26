"""Small independent witnesses protect full-product closure and exact set deduplication."""

import pytest
from residual_debt import closes_with_listings, minimal_sets, product_frontier


def test_duplicate_alternatives_do_not_multiply_products():
    assert minimal_sets([frozenset("a"), frozenset("a"), frozenset("ab")]) == [frozenset("a")]


def test_shared_leaf_closes_both_required_branches():
    assert product_frontier([[frozenset("a")], [frozenset("a")]]) == [frozenset("a")]


def test_all_other_branches_required():
    assert product_frontier([[frozenset("a")], [frozenset("b")]]) == [frozenset("ab")]


def test_branch_without_supported_path_blocks_listing_completion():
    assert product_frontier([[frozenset("a")], []]) == []


def test_alternative_paths_retain_both_minimal_solutions():
    assert set(product_frontier([[frozenset("a"), frozenset("b")], [frozenset("a")]])) == {
        frozenset("a")
    }


def test_cap_fails_without_truncation():
    with pytest.raises(ValueError, match="no truncated result"):
        minimal_sets([frozenset("a"), frozenset("b")], cap=1)


def test_direct_boolean_rejects_invalid_path_and_nonexact_product():
    row = {"exact_L1": True, "branches": [{"identity": "root", "makeable": False}]}
    components = {
        "root": {
            "paths": [{"path_supported": False, "leaves": [{"identity": "a", "listed": False}]}]
        }
    }
    assert not closes_with_listings(row, components, {"a"})
    components["root"]["paths"][0]["path_supported"] = True
    assert closes_with_listings(row, components, {"a"})
    assert not closes_with_listings(row, components, set())
    row["exact_L1"] = False
    assert not closes_with_listings(row, components, {"a"})
