from __future__ import annotations

import json
from pathlib import Path

from forge.maintenance import build_test_baseline_report


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_report_distinguishes_known_blockers_from_regressions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# reviewed\ntests/test_one.py::test_known\n")
    missing = tmp_path / "missing.txt"
    missing.write_text("# inputs\nresults/missing.json\nresults/present.json\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/present.json").write_text("{}")
    collected = _write(
        tmp_path / "cache/nodeids",
        ["tests/test_one.py::test_known", "tests/test_two.py::test_new"],
    )
    lastfailed = _write(
        tmp_path / "cache/lastfailed",
        {
            "tests/test_one.py::test_known": True,
            "tests/test_two.py::test_new": True,
            "tests/test_old.py::test_removed": True,
        },
    )

    report = build_test_baseline_report(
        tmp_path,
        baseline_path=baseline,
        missing_inputs_path=missing,
        lastfailed_path=lastfailed,
        collected_path=collected,
        observed_output=tmp_path / "observed.json",
    )

    assert report["status"] == "regression"
    assert report["new_failures"] == ["tests/test_two.py::test_new"]
    assert report["stale_cache_nodes"] == ["tests/test_old.py::test_removed"]
    assert report["documented_missing_inputs"]["missing"] == ["results/missing.json"]


def test_report_passes_the_no_new_failure_gate_with_known_failures(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("tests/test_one.py::test_known\n")
    missing = tmp_path / "missing.txt"
    missing.write_text("")
    collected = _write(tmp_path / "cache/nodeids", ["tests/test_one.py::test_known"])
    lastfailed = _write(
        tmp_path / "cache/lastfailed", {"tests/test_one.py::test_known": True}
    )
    output = tmp_path / "report.json"

    report = build_test_baseline_report(
        tmp_path,
        baseline_path=baseline,
        missing_inputs_path=missing,
        lastfailed_path=lastfailed,
        collected_path=collected,
        observed_output=tmp_path / "observed.json",
        output=output,
    )

    assert report["status"] == "blocked_known_failures"
    assert report["no_new_failures"] is True
    assert report["ready"] is False
    assert json.loads(output.read_text()) == report
