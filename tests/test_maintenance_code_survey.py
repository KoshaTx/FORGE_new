"""The single-contract survey must not label unreferenced code safe to delete."""

from __future__ import annotations

from pathlib import Path

import pytest
from forge_maintenance.__main__ import build_parser
from forge_maintenance.code_survey import _classify_reachability, survey_supported_studies


def test_unreached_code_requires_manual_study_review() -> None:
    assert (
        _classify_reachability(in_cli=False, in_paper=False, historical_pins_archived=True)
        == "review_unreached"
    )
    assert (
        _classify_reachability(in_cli=False, in_paper=False, historical_pins_archived=False)
        == "blocked_historical_pin"
    )


def _write(repo: Path, path: str, text: str = "") -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def _rows(report):
    return {row["path"]: row for row in report["classifications"]}


def test_combined_survey_keeps_both_studies_and_relative_reexports(tmp_path: Path) -> None:
    _write(tmp_path, "forge/shared/__init__.py", "from . import first\n")
    _write(tmp_path, "forge/shared/first.py", "from .second import value\n")
    _write(tmp_path, "forge/shared/second.py", "value = 1\n")
    _write(tmp_path, "forge/compose.py")
    _write(tmp_path, "experiments/phase1/three/stages.py", "from forge import shared\n")
    _write(tmp_path, "experiments/phase1/twentytwo/stages.py", "import forge.compose\n")
    report = survey_supported_studies(tmp_path)
    rows = _rows(report)
    assert all(rows[p]["classification"] == "keep_active" for p in rows if p.startswith("forge/"))
    assert report["safe_for_automated_deletion"] is False
    assert report == survey_supported_studies(tmp_path)


def test_combined_survey_resolves_aliases_and_full_script_paths(tmp_path: Path) -> None:
    _write(tmp_path, "forge/one.py")
    _write(tmp_path, "forge/two.py")
    _write(tmp_path, "forge/a/load.py")
    _write(tmp_path, "forge/b/load.py")
    _write(
        tmp_path,
        "cli/__init__.py",
        """from importlib import import_module as load
from importlib.util import spec_from_file_location as file_spec
load("forge.one")
file_spec("two", "forge/two.py")
""",
    )
    _write(tmp_path, "Makefile", "check:\n\tpython forge/a/load.py\n")
    _write(tmp_path, "configs/entry.json", '{"command": "python forge/b/load.py"}')
    rows = _rows(survey_supported_studies(tmp_path))
    for path in ("forge/one.py", "forge/two.py", "forge/a/load.py", "forge/b/load.py"):
        assert rows[path]["classification"] == "keep_active"


def test_reference_and_historical_consumers_are_distinct(tmp_path: Path) -> None:
    _write(tmp_path, "forge/oracle.py")
    _write(tmp_path, "forge/historical.py")
    _write(tmp_path, "forge/unreached.py")
    _write(tmp_path, "tests/test_oracle.py", "import forge.oracle\n")
    _write(tmp_path, "experiments/archive/producers/old.py", "import forge.historical\n")
    rows = _rows(survey_supported_studies(tmp_path))
    assert rows["forge/oracle.py"]["classification"] == "keep_reference"
    assert rows["forge/historical.py"]["classification"] == "keep_historical"
    assert rows["forge/unreached.py"]["classification"] == "review_unreached"


@pytest.mark.parametrize(
    "source", ["import importlib\nimportlib.import_module(name)", "def broken("]
)
def test_unbounded_dynamic_use_or_parse_failure_blocks_unreached(
    tmp_path: Path, source: str
) -> None:
    _write(tmp_path, "forge/unused.py")
    _write(tmp_path, "cli/__init__.py", source)
    report = survey_supported_studies(tmp_path)
    assert report["unresolved_references"]
    assert _rows(report)["forge/unused.py"]["classification"] == "blocked"


def test_moved_historical_path_preserves_dependency(tmp_path: Path) -> None:
    _write(tmp_path, "forge/moved.py")
    _write(
        tmp_path, "docs/artifact_path_moves.json", '{"moves":{"scripts/old.py":"forge/moved.py"}}'
    )
    _write(tmp_path, "paper/old/README.md", "python scripts/old.py")
    rows = _rows(survey_supported_studies(tmp_path))
    assert "historical" in rows["forge/moved.py"]["roles"]
    assert any(
        edge["original_target"] == "scripts/old.py"
        for edge in rows["forge/moved.py"]["incoming_references"]
    )


@pytest.mark.parametrize("path", ["forge/item.py", "configs/study.json", "README.md"])
def test_inventory_fingerprint_detects_stale_inputs(tmp_path: Path, path: str) -> None:
    _write(tmp_path, path, "{}" if path.endswith(".json") else "# old\n")
    before = survey_supported_studies(tmp_path)["fingerprint"]
    _write(tmp_path, path, '{"changed":true}' if path.endswith(".json") else "# new\n")
    assert survey_supported_studies(tmp_path)["fingerprint"] != before


def test_scope_and_historical_contract_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["survey", "--scope", "supported-studies", "--contract", "old.json"]
        )


def test_invalid_config_is_reported_instead_of_admitting_retirement(tmp_path: Path) -> None:
    _write(tmp_path, "forge/unreached.py")
    _write(tmp_path, "configs/broken.json", "{")
    report = survey_supported_studies(tmp_path)
    assert any("parse-json" in issue for issue in report["unresolved_references"])
    assert _rows(report)["forge/unreached.py"]["classification"] == "blocked"


def test_stage_registration_import_is_retained_without_execution(tmp_path: Path) -> None:
    _write(tmp_path, "forge/registry.py", 'raise RuntimeError("must never execute")\n')
    _write(
        tmp_path,
        "experiments/phase1/stages.py",
        'from forge.registry import stage\n@stage("study.v1")\ndef run(): pass\n',
    )
    report = survey_supported_studies(tmp_path)
    assert _rows(report)["forge/registry.py"]["classification"] == "keep_active"


def test_pin_changes_invalidate_fingerprint_and_block_missing_identity(tmp_path: Path) -> None:
    import hashlib
    import json

    _write(tmp_path, "forge/old.py", "# source\n")
    digest = hashlib.sha256(b"# source\n").hexdigest()
    _write(
        tmp_path, "results/old/result.json", json.dumps({"path": "forge/old.py", "sha256": digest})
    )
    before = survey_supported_studies(tmp_path)
    assert _rows(before)["forge/old.py"]["classification"] == "keep_historical"
    _write(
        tmp_path,
        "results/old/result.json",
        json.dumps({"path": "forge/old.py", "sha256": "0" * 64}),
    )
    after = survey_supported_studies(tmp_path)
    assert before["fingerprint"] != after["fingerprint"]
    assert _rows(after)["forge/old.py"]["classification"] == "blocked"


def test_model_eval_is_not_python_dynamic_execution(tmp_path: Path) -> None:
    _write(tmp_path, "forge/unused.py")
    _write(tmp_path, "cli/__init__.py", "model.eval()\n")
    report = survey_supported_studies(tmp_path)
    assert report["unresolved_references"] == []


def test_executable_result_audit_is_a_historical_consumer(tmp_path: Path) -> None:
    _write(tmp_path, "forge/helper.py", "def historical_helper(): pass\n")
    _write(tmp_path, "results/old/audit.py", "from forge.helper import historical_helper\n")
    report = survey_supported_studies(tmp_path)
    assert _rows(report)["forge/helper.py"]["classification"] == "keep_historical"
