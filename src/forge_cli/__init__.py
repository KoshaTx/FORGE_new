"""The single command-line interface for FORGE experiments and provenance checks.

Outside `forge` because it sits on top of everything rather than inside anything: it composes the
science in `forge`, and the DAG runner in `forge_experiment`. Nothing in either package imports it, so the dependency runs one way and the
library stays usable without the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forge_experiment import ExperimentRunner, diagnose_experiment, verify_run_directory
from forge_experiment.errors import ExperimentError


def _repo() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "forge").is_dir():
            return candidate.resolve()
    raise RuntimeError("could not locate the FORGE repository root")


def _spec_path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix == ".json" or candidate.parent != Path("."):
        path = candidate if candidate.is_absolute() else repo / candidate
    else:
        path = repo / "configs" / "experiments" / f"{value}.json"
    if not path.is_file():
        raise FileNotFoundError(f"experiment specification not found: {path}")
    return path.resolve()


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _runner(repo: Path, _: str) -> ExperimentRunner:
    # Modal is added as a separate backend without changing scientific stages.  Until that backend
    # lands, refusing it here is safer than silently running a full job locally.
    return ExperimentRunner(repo)


def _experiment_specs(repo: Path) -> list[Path]:
    root = repo / "configs" / "experiments"
    return sorted(root.glob("*.json")) if root.is_dir() else []


def _command_doctor(args: argparse.Namespace) -> int:
    repo = _repo()
    specs = [_spec_path(repo, args.experiment)] if args.experiment else _experiment_specs(repo)
    diagnoses = [diagnose_experiment(repo, path) for path in specs]
    _print(
        {
            "experiments": diagnoses,
            "ready": all(item["ready"] for item in diagnoses),
            "repository": str(repo),
            "schema_version": "forge.doctor.v1",
        }
    )
    return 0 if all(item["ready"] for item in diagnoses) else 2


def _command_artifacts_verify(_: argparse.Namespace) -> int:
    from forge_provenance.pins import main as verify_main

    return verify_main([])


def _paper_contract(repo: Path, value: str | None = None) -> Path:
    candidate = Path(value) if value else Path("configs/reproduction/iclr2027.json")
    path = candidate if candidate.is_absolute() else repo / candidate
    if not path.is_file():
        raise FileNotFoundError(f"paper reproduction contract not found: {path}")
    return path.resolve()


def _command_data_vendor(args: argparse.Namespace) -> int:
    from forge.data.vendor import main as vendor_main

    arguments = []
    if args.allow_partial:
        arguments.append("--allow-partial")
    if args.refresh_provenance:
        arguments.append("--refresh-provenance")
    return vendor_main(arguments)


def _command_data_verify(args: argparse.Namespace) -> int:
    from forge.data.vendor import main as vendor_main

    arguments = ["--verify"]
    if args.allow_partial:
        arguments.append("--allow-partial")
    return vendor_main(arguments)


def _command_provenance_verify(args: argparse.Namespace) -> int:
    from forge_provenance.pins import main as verify_main

    arguments: list[str] = []
    if args.code:
        arguments.extend(("--root", "results", "--root", "docs/provenance", "--root", "configs"))
    if args.strict:
        arguments.append("--strict")
    if args.expect_verified is not None:
        arguments.extend(("--expect-verified", str(args.expect_verified)))
    if args.allow_drift:
        arguments.extend(("--allow-drift", str(args.allow_drift)))
    if args.json:
        arguments.append("--json")
    return verify_main(arguments)


def _command_provenance_archive(_: argparse.Namespace) -> int:
    from forge_provenance.archive import main as archive_main

    return archive_main([])


def _command_paper_doctor(args: argparse.Namespace) -> int:
    from forge_paper import diagnose_paper

    repo = _repo()
    result = diagnose_paper(repo, _paper_contract(repo, args.contract))
    _print(result)
    ready = result["full_recompute_ready"] if args.strict else result["artifact_replay_ready"]
    return 0 if ready else 2


def _command_paper_verify(args: argparse.Namespace) -> int:
    from forge_paper import verify_paper

    repo = _repo()
    result = verify_paper(repo, _paper_contract(repo, args.contract), strict=args.strict)
    _print(result)
    return 0 if result["ok"] else 2


def _command_paper_render(args: argparse.Namespace) -> int:
    from forge_paper.verification import render_publication_outputs

    repo = _repo()
    _print(render_publication_outputs(repo, _paper_contract(repo, args.contract)))
    return 0


def _command_paper_reproduce(args: argparse.Namespace) -> int:
    from forge_paper.verification import reproduce_paper_artifacts

    repo = _repo()
    _print(reproduce_paper_artifacts(repo, _paper_contract(repo, args.contract)))
    return 0


def _command_paper_build(args: argparse.Namespace) -> int:
    from forge_paper.build import build_pdf

    repo = _repo()
    output = Path(args.output) if args.output else Path("build/paper/FORGE_ICLR2027_paper.pdf")
    output = output if output.is_absolute() else repo / output
    _print(build_pdf(repo, _paper_contract(repo, args.contract), output))
    return 0


def _command_paper_bundle(args: argparse.Namespace) -> int:
    from forge_paper.build import build_overleaf_bundle

    repo = _repo()
    output = (
        Path(args.output) if args.output else Path("build/paper/FORGE_ICLR2027_paper_overleaf.zip")
    )
    output = output if output.is_absolute() else repo / output
    _print(
        build_overleaf_bundle(
            repo,
            _paper_contract(repo, args.contract),
            output,
            verify_compile=not args.no_compile_check,
        )
    )
    return 0


def _command_experiment_list(_: argparse.Namespace) -> int:
    repo = _repo()
    rows = []
    from forge_experiment.spec import ExperimentSpec

    for path in _experiment_specs(repo):
        spec = ExperimentSpec.load(path)
        rows.append(
            {
                "description": spec.description,
                "experiment_id": spec.experiment_id,
                "path": str(path.relative_to(repo)),
                "profiles": list(spec.profiles),
                "replicates": dict(spec.replicates),
                "stages": len(spec.stages),
            }
        )
    _print({"experiments": rows})
    return 0


def _command_experiment_plan(args: argparse.Namespace) -> int:
    repo = _repo()
    spec_path = _spec_path(repo, args.experiment)
    if args.backend == "modal":
        from forge_experiment.modal import modal_request_plan

        _print(
            modal_request_plan(
                repo,
                spec_path,
                profile=args.profile,
                replicate=args.replicate,
                device=args.device,
            )
        )
        return 0
    plan = _runner(repo, args.backend).plan(
        spec_path,
        profile=args.profile,
        device=args.device,
        replicate=args.replicate,
    )
    _print(plan.to_mapping())
    return 0


def _command_experiment_run(args: argparse.Namespace) -> int:
    repo = _repo()
    spec_path = _spec_path(repo, args.experiment)
    if args.backend == "modal":
        from forge_experiment.modal import launch_modal

        return launch_modal(
            repo,
            spec_path,
            profile=args.profile,
            replicate=args.replicate,
            device=args.device,
            resume=args.resume,
        )
    result = _runner(repo, args.backend).run(
        spec_path,
        profile=args.profile,
        device=args.device,
        resume=args.resume,
        replicate=args.replicate,
    )
    _print(
        {
            "experiment_id": result.plan.experiment_id,
            "run_dir": str(result.plan.run_dir),
            "run_id": result.plan.run_id,
            "stages": list(result.stage_manifests),
            "status": "complete",
        }
    )
    return 0


def _command_experiment_verify(args: argparse.Namespace) -> int:
    repo = _repo()
    if args.backend != "local":
        raise ExperimentError(
            "verify a downloaded Modal run with `forge experiment verify-run RUN_ID`"
        )
    result = _runner(repo, args.backend).verify(
        _spec_path(repo, args.experiment),
        profile=args.profile,
        device=args.device,
        replicate=args.replicate,
    )
    _print(
        {
            "experiment_id": result.plan.experiment_id,
            "run_id": result.plan.run_id,
            "stages": list(result.stage_manifests),
            "status": "verified",
        }
    )
    return 0


def _command_experiment_reproduce(args: argparse.Namespace) -> int:
    repo = _repo()
    if args.backend != "local":
        raise ExperimentError("independent reproduction currently runs locally only")
    result = _runner(repo, args.backend).reproduce(
        _spec_path(repo, args.experiment),
        profile=args.profile,
        device=args.device,
        replicate=args.replicate,
    )
    _print(result)
    return 0


def _command_experiment_status(args: argparse.Namespace) -> int:
    repo = _repo()
    matches = list((repo / "runs").glob(f"*/{args.run_id}/run.json"))
    if not matches:
        raise FileNotFoundError(f"run id not found: {args.run_id}")
    if len(matches) > 1:
        raise RuntimeError(f"run id is ambiguous: {args.run_id}")
    _print(json.loads(matches[0].read_text()))
    return 0


def _command_experiment_verify_run(args: argparse.Namespace) -> int:
    repo = _repo()
    matches = list((repo / "runs").glob(f"*/{args.run_id}"))
    matches = [path for path in matches if (path / "run.json").is_file()]
    if not matches:
        raise FileNotFoundError(f"run id not found: {args.run_id}")
    if len(matches) > 1:
        raise RuntimeError(f"run id is ambiguous: {args.run_id}")
    _print(verify_run_directory(matches[0]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="diagnose experiment inputs and environment")
    doctor.add_argument("experiment", nargs="?")
    doctor.set_defaults(function=_command_doctor)

    artifacts = subcommands.add_parser("artifacts", help="verify frozen artifact provenance")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)
    verify_artifacts = artifact_commands.add_parser("verify")
    verify_artifacts.set_defaults(function=_command_artifacts_verify)

    data = subcommands.add_parser("data", help="install or verify hash-pinned source data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    vendor = data_commands.add_parser("vendor")
    vendor.add_argument("--allow-partial", action="store_true")
    vendor.add_argument("--refresh-provenance", action="store_true")
    vendor.set_defaults(function=_command_data_vendor)
    verify_data = data_commands.add_parser("verify")
    verify_data.add_argument("--allow-partial", action="store_true")
    verify_data.set_defaults(function=_command_data_verify)

    provenance = subcommands.add_parser("provenance", help="verify or archive historical pins")
    provenance_commands = provenance.add_subparsers(dest="provenance_command", required=True)
    verify_provenance = provenance_commands.add_parser("verify")
    verify_provenance.add_argument("--code", action="store_true", help="also scan frozen configs")
    verify_provenance.add_argument("--strict", action="store_true")
    verify_provenance.add_argument("--expect-verified", type=int)
    verify_provenance.add_argument("--allow-drift", type=int, default=0)
    verify_provenance.add_argument("--json", action="store_true")
    verify_provenance.set_defaults(function=_command_provenance_verify)
    archive_provenance = provenance_commands.add_parser("archive")
    archive_provenance.set_defaults(function=_command_provenance_archive)

    paper = subcommands.add_parser("paper", help="verify, render, and build the frozen manuscript")
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)
    for name, function in (
        ("doctor", _command_paper_doctor),
        ("verify", _command_paper_verify),
        ("render", _command_paper_render),
        ("reproduce", _command_paper_reproduce),
        ("build", _command_paper_build),
        ("bundle", _command_paper_bundle),
    ):
        command = paper_commands.add_parser(name)
        command.add_argument("--contract")
        if name in {"doctor", "verify"}:
            command.add_argument("--strict", action="store_true")
        if name in {"build", "bundle"}:
            command.add_argument("--output")
        if name == "bundle":
            command.add_argument("--no-compile-check", action="store_true")
        command.set_defaults(function=function)

    # Repository maintenance is deliberately absent from this CLI. `survey` and `test-report`
    # classify this repository's own files, which is not part of the installed package's surface;
    # they live in `tools/forge_maintenance` and run via `make code-survey` / `make
    # test-baseline-report`.

    experiment = subcommands.add_parser("experiment", help="plan and execute experiment DAGs")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    list_command = experiment_commands.add_parser("list")
    list_command.set_defaults(function=_command_experiment_list)

    for name, function in (
        ("plan", _command_experiment_plan),
        ("run", _command_experiment_run),
        ("verify", _command_experiment_verify),
        ("reproduce", _command_experiment_reproduce),
    ):
        command = experiment_commands.add_parser(name)
        command.add_argument("experiment")
        command.add_argument("--profile", choices=("smoke", "full"), required=True)
        command.add_argument("--backend", choices=("local", "modal"), default="local")
        command.add_argument("--device", choices=("cpu", "mps", "cuda"))
        command.add_argument("--replicate", type=int, default=0)
        if name == "run":
            command.add_argument("--resume", action="store_true")
        command.set_defaults(function=function)

    status = experiment_commands.add_parser("status")
    status.add_argument("run_id")
    status.set_defaults(function=_command_experiment_status)

    verify_run = experiment_commands.add_parser(
        "verify-run", help="verify a run from its downloaded self-contained manifests"
    )
    verify_run.add_argument("run_id")
    verify_run.set_defaults(function=_command_experiment_verify_run)
    return parser


# Every FORGE domain error derives from RuntimeError or ValueError, so catching those two reaches
# all of them -- but it also reaches subclasses that mean "this code is broken" rather than "this
# input is invalid". RecursionError and NotImplementedError are RuntimeErrors; the UnicodeError
# family are ValueErrors. Those keep their traceback: reporting a runaway recursion as a tidy
# `forge: maximum recursion depth exceeded` makes a bug indistinguishable from a validation failure
# and throws away the stack that would explain it. A bare UnicodeError here means it escaped a
# reader that should have named the offending file, which is also a defect worth seeing whole.
_DEFECTS_NOT_DIAGNOSTICS = (RecursionError, NotImplementedError, UnicodeError)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except _DEFECTS_NOT_DIAGNOSTICS:
        raise
    except (ExperimentError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"forge: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
