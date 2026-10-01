from pathlib import Path
from types import SimpleNamespace

import conftest
import pytest


def _item(path: Path, *, strict: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        iter_markers=lambda _: [SimpleNamespace(args=(str(path),))],
        config=SimpleNamespace(getoption=lambda _: strict),
    )


def test_missing_external_artifact_is_reported_not_passed(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="External historical artifacts unavailable"):
        conftest.pytest_runtest_setup(_item(tmp_path / "missing.json"))


def test_strict_reproduction_fails_when_the_same_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="missing.json"):
        conftest.pytest_runtest_setup(_item(tmp_path / "missing.json", strict=True))


def test_present_artifact_runs_the_test_even_if_its_contents_are_wrong(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("corrupt bytes; downstream integrity assertions must still execute")
    conftest.pytest_runtest_setup(_item(path))
