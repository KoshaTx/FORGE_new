"""Adversarial checks for the bounded primary-article audit, with no source mutation."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).with_name("audit.py")
    spec = importlib.util.spec_from_file_location("primary_article_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "defect, message",
    [
        ("article_hash", "User receipt and primary article disagree"),
        ("missing_procedure", "Incomplete source execution/procedure chain"),
        ("unreviewed_article", "chemistry pages were not visually reviewed"),
        ("wrong_reactant", "Reviewed reactant identity or role disagrees"),
        ("wrong_yield", "Reported isolated yield conflicts"),
    ],
)
def test_source_defects_fail_before_admission(audit, monkeypatch, defect, message):
    original_read = audit.read

    def read_with_defect(path):
        payload = original_read(path)
        if path.name != "review.json":
            return payload
        if defect == "article_hash":
            payload["source_assets"]["jo000463n.pdf"]["sha256"] = "0" * 64
        elif defect == "missing_procedure":
            payload["reported_reaction"]["reference_chain"][
                "required_preparation_link_unavailable"
            ] = True
        elif defect == "unreviewed_article":
            payload["source_reviews"]["primary"]["source_asset"]["review_status"] = "pending"
        elif defect == "wrong_reactant":
            payload["reported_reaction"]["reactant"]["canonical_smiles"] = "CCO"
        elif defect == "wrong_yield":
            payload["reported_reaction"]["isolated_yield_percent"] = 82
        return payload

    monkeypatch.setattr(audit, "read", read_with_defect)
    with pytest.raises(ValueError, match=message):
        audit.describe()
