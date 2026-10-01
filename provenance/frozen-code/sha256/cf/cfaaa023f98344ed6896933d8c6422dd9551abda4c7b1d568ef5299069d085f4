"""Command line for the repository maintenance tools.

This wiring used to live in `cli` as `forge maintenance ...`. It moved here with the code so
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

DEFAULT_BENCHMARK_SEEDS = (20260825, 20260826, 20260827)

DEFAULT_CONTRACT = Path("configs/reproduction/iclr2027.json")


def _repo() -> Path:
    """Locate the repository root the same way `cli` does."""
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "forge").is_dir():
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


def _command_assessment_benchmark(args: argparse.Namespace) -> int:
    from forge_maintenance.assessment_benchmark import run_benchmark

    repo = _repo()
    seeds = [int(value) for value in args.seed] if args.seed else list(DEFAULT_BENCHMARK_SEEDS)
    ledger_dir = _optional(repo, args.ledger_dir) or repo / "build/assessment_benchmark"
    result = run_benchmark(repo, seeds=seeds, ledger_dir=ledger_dir)
    output = _optional(repo, args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _print(result)
    return 0


def _command_evaluation_profile(args: argparse.Namespace) -> int:
    from forge_maintenance.evaluation_profile import profile_production_evaluation

    repo = _repo()
    arm = repo / "results/phase1" / args.arm
    output = _optional(repo, args.output)
    result = profile_production_evaluation(
        repo,
        config_path=_under(repo, args.config),
        cache_path=_under(repo, args.cache),
        checkpoint_archive_path=arm / "checkpoints.tar",
        training_result_path=arm / "training_result.json",
        output_dir=_optional(repo, args.work_dir) or repo / "build/evaluation_profile",
        calibration_samples=int(args.calibration_samples),
        heldout_samples=int(args.heldout_samples),
        component_disjoint_record_limit=int(args.component_disjoint_record_limit),
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _print(result)
    return 0


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

    benchmark = commands.add_parser(
        "assessment-benchmark",
        help="time the CPU Ugi assessor suite and digest every emitted row and metric",
    )
    benchmark.add_argument("--seed", action="append")
    benchmark.add_argument("--ledger-dir")
    benchmark.add_argument("--output")
    benchmark.set_defaults(function=_command_assessment_benchmark)

    evaluation = commands.add_parser(
        "evaluation-profile",
        help="attribute CPU wall time across the real production evaluation stage",
    )
    evaluation.add_argument("--arm", default="shared_bias_parallel_program_role_seed0_v2")
    evaluation.add_argument(
        "--config",
        default="configs/multireaction/shared_bias_parallel_program_role_seed0_evaluation_v2.json",
    )
    evaluation.add_argument(
        "--cache", default="results/phase1/shared_synthesis_program_mixed_cache_v1/cache.npz"
    )
    evaluation.add_argument("--calibration-samples", default=32)
    evaluation.add_argument("--heldout-samples", default=192)
    evaluation.add_argument("--component-disjoint-record-limit", default=64)
    evaluation.add_argument("--work-dir")
    evaluation.add_argument("--output")
    evaluation.set_defaults(function=_command_evaluation_profile)
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
