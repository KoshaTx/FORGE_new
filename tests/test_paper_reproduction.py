from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from forge_paper import PaperContract, PaperContractError, diagnose_paper, verify_paper
from forge_paper.build import build_overleaf_bundle

from forge_cli import build_parser

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "configs/reproduction/iclr2027.json"


def test_authoritative_paper_contract_has_complete_direct_references() -> None:
    contract = PaperContract.load(CONTRACT)
    assert contract.paper_id == "forge-iclr2027"
    assert len(contract.evidence_roots) == 12
    assert len(contract.generated_outputs) == 3
    assert len(contract.figure_outputs) == 10

    diagnosis = diagnose_paper(REPO, CONTRACT)
    assert diagnosis["artifact_replay_ready"] is True
    assert all(not values for values in diagnosis["references"].values())


def test_artifact_replay_is_not_mislabeled_as_full_recomputation() -> None:
    replay = verify_paper(REPO, CONTRACT)
    strict = verify_paper(REPO, CONTRACT, strict=True)
    assert replay["ok"] is True
    assert replay["mode"] == "artifact-replay"
    assert replay["unresolved"] == []
    assert replay["unresolved_count"] > 0
    assert strict["mode"] == "full-recompute"
    assert strict["ok"] is False
    assert strict["unresolved"]
    assert strict["unresolved_count"] == len(strict["unresolved"])


def test_contract_rejects_an_unpinned_source(tmp_path: Path) -> None:
    document = json.loads(CONTRACT.read_text())
    document["source"]["sha256"] = "not-a-digest"
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(document))
    with pytest.raises(PaperContractError, match="SHA-256"):
        PaperContract.load(path)


def test_overleaf_bundle_is_byte_reproducible_and_minimal(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = build_overleaf_bundle(REPO, CONTRACT, first, verify_compile=False)
    two = build_overleaf_bundle(REPO, CONTRACT, second, verify_compile=False)
    assert one["sha256"] == two["sha256"]

    contract = PaperContract.load(CONTRACT)
    with zipfile.ZipFile(first) as archive:
        members = set(archive.namelist())
    assert "FORGE_ICLR2027_paper.tex" in members
    assert {
        Path(pin.path).relative_to("paper").as_posix() for pin in contract.figure_outputs
    } <= members
    assert not any(member.endswith((".aux", ".log", ".pdf")) for member in members)


def test_cli_exposes_supported_data_provenance_and_paper_commands() -> None:
    parser = build_parser()
    for argv in (
        ["data", "verify"],
        ["provenance", "verify"],
        ["paper", "verify"],
        ["paper", "reproduce"],
        ["paper", "build"],
        ["paper", "bundle"],
    ):
        assert parser.parse_args(argv).function is not None


def test_cli_does_not_expose_repository_maintenance() -> None:
    """Repository bookkeeping is not part of the installed package's command surface.

    `survey` classifies this repository's own files and `test-report` reads its pytest cache;
    neither means anything to someone who installed the package. They live in
    `tools/forge_maintenance`, outside `src/`, and run through `make code-survey` and
    `make test-baseline-report`.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(["maintenance", "survey"])
