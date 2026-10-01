"""CI must consume the recorded environment without rewriting its evidence identity."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_ci_uv_commands_preserve_the_lockfile() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("run", "").startswith("uv ")
    ]
    assert commands
    for command in commands:
        tokens = shlex.split(command)
        assert tokens[1] in {"sync", "run"}
        assert tokens[2] == "--frozen", command


def test_make_test_commands_preserve_the_lockfile() -> None:
    env = {key: value for key, value in os.environ.items() if key not in {"UV_RUN", "MAKEFLAGS"}}
    result = subprocess.run(
        ["make", "--dry-run", "test-one", "TEST=tests/test_reviewer_generation.py"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "uv run --frozen python -m pytest" in result.stdout


def test_ci_runs_code_checks_independently_of_historical_availability() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["core"]["steps"]
    for target in ("typecheck", "lint-core", "test-core"):
        step = next(step for step in steps if step.get("run") == f"uv run --frozen make {target}")
        assert "!cancelled()" in step["if"]
        assert "steps.install.outcome == 'success'" in step["if"]
    result = subprocess.run(
        ["make", "--dry-run", "test-core"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "provenance verify" not in result.stdout
    assert "pytest" in result.stdout
