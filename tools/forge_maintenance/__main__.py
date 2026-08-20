"""Command line for the repository maintenance tools.

This wiring used to live in `forge_cli` as `forge maintenance ...`. It moved here with the code so
the installed package's command surface describes the science rather than this repository's own
bookkeeping, and so nothing in `forge` has to import a tool that only makes sense inside the
working tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge_maintenance.code_survey import survey_code
from forge_maintenance.test_baseline import build_test_baseline_report

DEFAULT_CONTRACT = Path("configs/reproduction/iclr2027.json")


def _repo() -> Path:
    """Locate the repository root the same way `forge_cli` does."""
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "forge").is_dir():
            return candidate.resolve()
    raise RuntimeError("could not locate the FORGE repository root")


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _under(repo: Path, value: str) -> Path:
    """Resolve a path argument against the repository root."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo / candidate


def _optional(repo: Path, value: str | None) -> Path | None:
    """Resolve an argument that is genuinely optional, unlike the ones argparse defaults."""
    return _under(repo, value) if value else None


def _command_survey(args: argparse.Namespace) -> int:
    repo = _repo()
    contract = _under(repo, args.contract) if args.contract else repo / DEFAULT_CONTRACT
    if not contract.is_file():
        raise FileNotFoundError(f"paper reproduction contract not found: {contract}")
    result = survey_code(repo, contract, output=_optional(repo, args.output))
    _print(result)
    return 0 if result["safe_for_automated_deletion"] else 2


def _command_test_report(args: argparse.Namespace) -> int:
    repo = _repo()
    output = _optional(repo, args.output)
    observed_output = _optional(repo, args.observed_output) or (
        output.with_name("observed_failures.json")
        if output
        else repo / "build/observed_failures.json"
    )
    result = build_test_baseline_report(
        repo,
        baseline_path=_under(repo, args.baseline),
        missing_inputs_path=_under(repo, args.missing_inputs),
        lastfailed_path=_under(repo, args.lastfailed),
        collected_path=_under(repo, args.collected),
        observed_output=observed_output,
        output=output,
    )
    _print(result)
    return 0 if result["no_new_failures"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_maintenance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    survey = commands.add_parser("survey", help="classify code against the paper and CLI roots")
    survey.add_argument("--contract")
    survey.add_argument("--output")
    survey.set_defaults(function=_command_survey)

    report = commands.add_parser("test-report", help="compare a pytest run with known blockers")
    report.add_argument("--baseline", default="tests/baseline_failures.txt")
    report.add_argument("--missing-inputs", default="docs/missing_test_inputs.txt")
    report.add_argument("--lastfailed", default=".pytest_cache/v/cache/lastfailed")
    report.add_argument("--collected", default=".pytest_cache/v/cache/nodeids")
    report.add_argument("--observed-output")
    report.add_argument("--output")
    report.set_defaults(function=_command_test_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"forge_maintenance: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
