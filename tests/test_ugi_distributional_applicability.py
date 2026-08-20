from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pytest

from experiments.archive.producers.phase1_audit_ugi_distributional_applicability import _write_once
from forge.potency.applicability.ugi_distributional_applicability import (
    ChemicalReference,
    DistancePair,
    _bins,
    _csv_bytes,
    _performance,
    _view_bin,
)


def test_exact_new_identity_can_be_chemically_interpolative() -> None:
    reference = ChemicalReference(("CCCC", "CCCCCC"))
    exact = reference.distance("CCCC")
    intermediate = reference.distance("CCCCC")

    assert exact.exact_identity_seen is True
    assert intermediate.exact_identity_seen is False
    assert intermediate.fingerprint >= 0.0
    assert intermediate.descriptor >= 0.0


def test_multiview_bin_uses_the_most_extrapolative_view() -> None:
    thresholds = {
        view: {
            "fingerprint": {"interpolative_max": 0.2, "boundary_max": 0.5},
            "descriptor": {"interpolative_max": 0.2, "boundary_max": 0.5},
        }
        for view in ("product", "amine", "aldehyde", "isocyanide")
    }
    distances = {
        "product": DistancePair(0.1, 0.1, False),
        "amine": DistancePair(0.1, 0.1, True),
        "aldehyde": DistancePair(0.3, 0.3, False),
        "isocyanide": DistancePair(0.7, 0.1, False),
    }
    views, overall = _bins(distances, thresholds)

    assert views == {
        "product": "interpolative",
        "amine": "interpolative",
        "aldehyde": "boundary",
        "isocyanide": "extrapolative",
    }
    assert overall == "extrapolative"


def test_view_bin_requires_both_fixed_views_to_pass() -> None:
    thresholds = {
        "fingerprint": {"interpolative_max": 0.2, "boundary_max": 0.5},
        "descriptor": {"interpolative_max": 0.2, "boundary_max": 0.5},
    }
    assert _view_bin(DistancePair(0.1, 0.3, False), thresholds) == "boundary"
    assert _view_bin(DistancePair(0.6, 0.1, False), thresholds) == "extrapolative"


def test_performance_reports_prediction_and_coverage_metrics() -> None:
    rows = [
        {"y_true": 0.0, "y_pred": 0.0, "covered90": True},
        {"y_true": 1.0, "y_pred": 1.0, "covered90": True},
        {"y_true": 2.0, "y_pred": 1.0, "covered90": False},
    ]
    result = _performance(rows)

    assert result["records"] == 3
    assert result["r2"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert result["coverage90"] == pytest.approx(2.0 / 3.0)


def test_csv_output_is_deterministic_and_write_once(tmp_path: Path) -> None:
    fields = ("a", "b")
    rows = ({"a": 1, "b": "x"}, {"a": 2, "b": "y"})
    first = _csv_bytes(rows, fields)
    second = _csv_bytes(rows, fields)
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        decoded = list(csv.DictReader(io.StringIO(handle.read().decode())))
    assert decoded == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    path = tmp_path / "result.json"
    _write_once(path, b"first")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_once(path, b"second")
    assert path.read_bytes() == b"first"
