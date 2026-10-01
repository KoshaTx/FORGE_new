"""CI must consume the recorded environment without rewriting its evidence identity."""

from __future__ import annotations

import hashlib
import json
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
        ["make", "--dry-run", "test-one", "TEST=tests/test_review_submission.py"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "uv run --frozen python -m pytest" in result.stdout


def test_all_committed_historical_archive_blobs_match_their_digest() -> None:
    manifest = json.loads((ROOT / "provenance/frozen-code/manifest.json").read_text())
    for entry in manifest["entries"]:
        payload = (ROOT / entry["blob_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["original_path"]
