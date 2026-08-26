"""The single command-line interface for FORGE experiments and provenance checks.

Outside `forge` because it sits on top of everything rather than inside anything: it composes the
science in `forge`, and the DAG runner in `experiments._runtime`. Nothing in either package imports it, so the dependency runs one way and the
library stays usable without the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments import load_catalog, resolve_specification, specification_paths
from experiments._runtime import ExperimentRunner, diagnose_experiment, verify_run_directory
from experiments._runtime.errors import ExperimentError


def _repo() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[1]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "forge").is_dir():
            return candidate.resolve()
    raise RuntimeError("could not locate the FORGE repository root")


def _spec_path(repo: Path, value: str) -> Path:
    return resolve_specification(repo, value)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _runner(repo: Path, _: str) -> ExperimentRunner:
    # Modal is added as a separate backend without changing scientific stages.  Until that backend
    # lands, refusing it here is safer than silently running a full job locally.
    load_catalog()
    return ExperimentRunner(repo)


def _experiment_specs(repo: Path) -> list[Path]:
    return list(specification_paths(repo))


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

    return int(verify_main([]))


def _paper_contract(repo: Path, value: str | None = None) -> Path:
    candidate = Path(value) if value else Path("configs/reproduction/iclr2027.json")
    path = candidate if candidate.is_absolute() else repo / candidate
    if not path.is_file():
        raise FileNotFoundError(f"paper reproduction contract not found: {path}")
    return path.resolve()


def _paper_experiment_matrix(repo: Path, value: str | None = None) -> Path:
    candidate = (
        Path(value) if value else Path("configs/reproduction/natbiotech_v1_experiments.json")
    )
    path = candidate if candidate.is_absolute() else repo / candidate
    if not path.is_file():
        raise FileNotFoundError(f"paper experiment matrix not found: {path}")
    return path.resolve()


def _command_data_vendor(args: argparse.Namespace) -> int:
    from forge_data.vendor import main as vendor_main

    arguments = []
    if args.allow_partial:
        arguments.append("--allow-partial")
    if args.refresh_provenance:
        arguments.append("--refresh-provenance")
    return int(vendor_main(arguments))


def _command_data_verify(args: argparse.Namespace) -> int:
    from forge_data.vendor import main as vendor_main

    arguments = ["--verify"]
    if args.allow_partial:
        arguments.append("--allow-partial")
    return int(vendor_main(arguments))


def _command_provenance_verify(args: argparse.Namespace) -> int:
    from forge_provenance.pins import main as verify_main

    arguments: list[str] = []
    if args.code:
        arguments.extend(
            (
                "--root",
                "results",
                "--root",
                "docs/provenance",
                "--root",
                "configs",
                "--root",
                "experiments",
            )
        )
    if args.strict:
        arguments.append("--strict")
    if args.expect_verified is not None:
        arguments.extend(("--expect-verified", str(args.expect_verified)))
    if args.allow_drift:
        arguments.extend(("--allow-drift", str(args.allow_drift)))
    if args.json:
        arguments.append("--json")
    return int(verify_main(arguments))


def _command_provenance_archive(_: argparse.Namespace) -> int:
    from forge_provenance.archive import main as archive_main

    return int(archive_main([]))


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


def _command_paper_experiments(args: argparse.Namespace) -> int:
    from forge_paper import diagnose_experiment_matrix

    repo = _repo()
    result = diagnose_experiment_matrix(repo, _paper_experiment_matrix(repo, args.matrix))
    _print(result)
    if args.strict and not result["setup_complete_for_all_retained_rows"]:
        return 2
    return 0


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


def _command_paper_render_results_v1(args: argparse.Namespace) -> int:
    from forge_paper import render_v1_results

    repo = _repo()
    rows = Path(args.rows) if Path(args.rows).is_absolute() else repo / args.rows
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(render_v1_results(config, repo, rows, output, strict=not args.allow_partial))
    return 0


def _command_paper_render_completed_evidence_v1(args: argparse.Namespace) -> int:
    from forge_paper import render_completed_evidence_v1

    repo = _repo()
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    result = Path(args.result) if Path(args.result).is_absolute() else repo / args.result
    _print(render_completed_evidence_v1(config, repo, output, result_path=result))
    return 0


def _command_paper_render_forge_samples(args: argparse.Namespace) -> int:
    from forge_paper import render_forge_generated_sample_figure

    repo = _repo()
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    result = Path(args.result) if Path(args.result).is_absolute() else repo / args.result
    _print(render_forge_generated_sample_figure(config, repo, output, result_path=result))
    return 0


def _command_paper_render_forge_sample_atlas(args: argparse.Namespace) -> int:
    from forge_paper import render_forge_generated_sample_atlas

    repo = _repo()
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    result = Path(args.result) if Path(args.result).is_absolute() else repo / args.result
    _print(render_forge_generated_sample_atlas(config, repo, output, result_path=result))
    return 0


def _command_paper_collect_results_v1(args: argparse.Namespace) -> int:
    from forge_paper.results_v1 import (
        common_assessment_seed_row,
        held_family_seed_rows,
        mechanism_seed_rows,
        write_v1_result_rows,
    )

    repo = _repo()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    if args.common and args.route_union is None:
        raise ValueError("--route-union is required when collecting common method rows")
    route_union = resolve(args.route_union) if args.route_union is not None else None
    rows = [
        common_assessment_seed_row(resolve(value), repo, route_union_result=route_union)
        for value in args.common
    ]
    for value in args.mechanism_evaluation:
        rows.extend(mechanism_seed_rows(resolve(value), repo))
    for value in args.held_family_evaluation:
        rows.extend(held_family_seed_rows(resolve(value), repo))
    output = resolve(args.output)
    write_v1_result_rows(output, rows)
    _print({"schema_version": "forge.natbiotech_v1_result_collection.v1", "rows": len(rows)})
    return 0


def _command_experiment_list(_: argparse.Namespace) -> int:
    repo = _repo()
    rows = []
    from experiments._runtime.spec import ExperimentSpec

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
        from experiments._runtime.modal import modal_request_plan

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
        from experiments._runtime.modal import launch_modal

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


def _command_experiment_select_accelerator(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.accelerator_selection import (
        adjudicate_accelerator_benchmarks,
    )

    run_dirs = [Path(value) if Path(value).is_absolute() else repo / value for value in args.runs]
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(adjudicate_accelerator_benchmarks(run_dirs, repo, output))
    return 0


def _command_experiment_adjudicate_production(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.production_adjudication import (
        adjudicate_production_runs,
    )

    run_dirs = [Path(value) if Path(value).is_absolute() else repo / value for value in args.runs]
    catalogue_run_dirs = [
        Path(value) if Path(value).is_absolute() else repo / value for value in args.catalogue_runs
    ]
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(
        adjudicate_production_runs(
            run_dirs,
            repo,
            output,
            catalogue_run_dirs=catalogue_run_dirs,
        )
    )
    return 0


def _command_experiment_adjudicate_final_production(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.final_production_adjudication import (
        adjudicate_final_production_runs,
    )

    def resolve(values: list[str]) -> list[Path]:
        return [Path(value) if Path(value).is_absolute() else repo / value for value in values]

    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(
        adjudicate_final_production_runs(
            resolve(args.final_runs),
            resolve(args.control_runs),
            resolve(args.catalogue_runs),
            repo,
            output,
        )
    )
    return 0


def _command_experiment_adjudicate_local_chemistry_resampling(
    args: argparse.Namespace,
) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.local_chemistry_resampling_adjudication import (
        adjudicate_local_chemistry_resampling,
    )

    run_dirs = [Path(value) if Path(value).is_absolute() else repo / value for value in args.runs]
    original = Path(args.original) if Path(args.original).is_absolute() else repo / args.original
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(adjudicate_local_chemistry_resampling(run_dirs, original, repo, output))
    return 0


def _command_experiment_adjudicate_local_morphology_seed0(
    args: argparse.Namespace,
) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.local_morphology_resampling_adjudication import (
        adjudicate_local_morphology_seed0,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        adjudicate_local_morphology_seed0(
            resolve(args.run),
            resolve(args.config),
            repo,
            resolve(args.output),
        )
    )
    return 0


def _command_experiment_diagnose_production(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.production_failure_diagnosis import (
        diagnose_production_failure,
    )

    aggregate = (
        Path(args.aggregate) if Path(args.aggregate).is_absolute() else repo / args.aggregate
    )
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(diagnose_production_failure(aggregate, repo, output))
    return 0


def _command_experiment_freeze_transformer_qualification(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.transformer_qualification import (
        freeze_transformer_qualification,
    )

    run_dir = Path(args.run) if Path(args.run).is_absolute() else repo / args.run
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(freeze_transformer_qualification(run_dir, repo, output))
    return 0


def _command_experiment_assess_common_ugi(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.common_assessment import run_common_ugi_assessment

    attempts = Path(args.attempts) if Path(args.attempts).is_absolute() else repo / args.attempts
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(
        run_common_ugi_assessment(
            config,
            repo,
            attempts,
            output,
            method_id=args.method,
            seed=args.seed,
            expected_attempts=args.expected_attempts,
        )
    )
    return 0


def _command_experiment_assess_ugi_v0_transformer(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
        run_ugi_v0_transformer_assessment,
    )

    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(run_ugi_v0_transformer_assessment(config, repo, output))
    return 0


def _command_experiment_assess_lipid_realism(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.lipid_realism_assessment import (
        run_lipid_realism_assessment,
    )

    attempts = Path(args.attempts) if Path(args.attempts).is_absolute() else repo / args.attempts
    config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(
        run_lipid_realism_assessment(
            config,
            repo,
            attempts,
            output,
            method_id=args.method,
            seed=args.seed,
            expected_attempts=args.expected_attempts,
        )
    )
    return 0


def _command_experiment_aggregate_lipid_realism(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.lipid_realism_aggregation import (
        aggregate_lipid_realism,
    )

    results = [Path(value) if Path(value).is_absolute() else repo / value for value in args.results]
    output = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    _print(
        aggregate_lipid_realism(
            results,
            repo,
            output,
            expected_seeds=args.expected_seeds,
        )
    )
    return 0


def _command_experiment_import_external_ugi(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.external_ugi_contract import (
        import_external_attempts,
        load_external_baseline_manifest,
        load_visible_component_ids,
        write_external_attempt_ledger,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    manifest = load_external_baseline_manifest(resolve(args.manifest))
    if args.method not in manifest:
        raise ValueError(f"external method is not declared: {args.method}")
    method = manifest[args.method]
    visible = (
        load_visible_component_ids(resolve(args.train_components))
        if args.train_components is not None
        else ()
    )
    attempts = import_external_attempts(
        method,
        resolve(args.receipt),
        resolve(args.samples),
        expected_seed=args.seed,
        expected_attempts=args.expected_attempts,
        method_visible_component_ids=visible,
    )
    output = resolve(args.output)
    artifact = write_external_attempt_ledger(output, attempts)
    _print(
        {
            "schema_version": "forge.external_ugi_import.v1",
            "method_id": args.method,
            "seed": args.seed,
            "attempts": len(attempts),
            "artifact": artifact,
        }
    )
    return 0


def _command_experiment_supersede_genmol_labels(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.external_ugi_contract import (
        load_external_baseline_manifest,
        supersede_genmol_empty_labels,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    methods = load_external_baseline_manifest(resolve(args.manifest))
    _print(
        supersede_genmol_empty_labels(
            methods["genmol_safe"],
            resolve(args.receipt),
            resolve(args.samples),
            resolve(args.output),
            expected_seed=args.seed,
            expected_attempts=args.expected_attempts,
        )
    )
    return 0


def _command_experiment_prepare_external_ugi(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.native_baseline_ports import (
        prepare_native_baseline_run,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        prepare_native_baseline_run(
            resolve(args.manifest),
            resolve(args.common_export),
            resolve(args.checkout),
            resolve(args.output),
            method_id=args.method,
            seed=args.seed,
            attempts=args.attempts,
            profile=args.profile,
        )
    )
    return 0


def _command_experiment_run_external_ugi(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.native_baseline_runtime import run_native_baseline

    def resolve(value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        run_native_baseline(
            resolve(args.request),  # type: ignore[arg-type]
            resolve(args.checkout),  # type: ignore[arg-type]
            resolve(args.output),  # type: ignore[arg-type]
            manifest_path=resolve(args.manifest),  # type: ignore[arg-type]
            tokenizer_snapshot=resolve(args.tokenizer_snapshot),
        )
    )
    return 0


def _command_experiment_freeze_common_route_union(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.method_blind_route_union import (
        freeze_method_blind_route_union,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        freeze_method_blind_route_union(
            [resolve(value) for value in args.assessed], resolve(args.output)
        )
    )
    return 0


def _command_experiment_adjudicate_common_route_union(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.method_blind_route_union import (
        adjudicate_method_blind_route_union,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        adjudicate_method_blind_route_union(
            resolve(args.union),
            resolve(args.evidence),
            [resolve(value) for value in args.assessed],
            resolve(args.output),
        )
    )
    return 0


def _command_experiment_build_common_route_evidence(args: argparse.Namespace) -> int:
    repo = _repo()
    from experiments.phase1.multireaction.method_blind_route_union import (
        build_method_blind_route_evidence,
    )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo / path

    _print(
        build_method_blind_route_evidence(
            resolve(args.config),
            repo,
            resolve(args.output),
        )
    )
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
    verify_provenance.add_argument(
        "--code",
        action="store_true",
        help="also scan frozen configs and active experiment specifications",
    )
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

    paper_experiments = paper_commands.add_parser(
        "experiments",
        help="diagnose every experiment and baseline row named by the v1 manuscript",
    )
    paper_experiments.add_argument("--matrix")
    paper_experiments.add_argument(
        "--strict",
        action="store_true",
        help="fail until every retained manuscript row has a runnable setup",
    )
    paper_experiments.set_defaults(function=_command_paper_experiments)

    render_results = paper_commands.add_parser(
        "render-results-v1",
        help="aggregate verified seed rows and render v1 paper table/figure inputs",
    )
    render_results.add_argument("rows")
    render_results.add_argument(
        "--config", default="configs/reproduction/natbiotech_v1_render_v1.json"
    )
    render_results.add_argument("--output", default="build/paper/v1/generated")
    render_results.add_argument(
        "--allow-partial",
        action="store_true",
        help="write a clearly marked non-publication preview without every retained row",
    )
    render_results.set_defaults(function=_command_paper_render_results_v1)

    render_completed = paper_commands.add_parser(
        "render-completed-evidence-v1",
        help="render every completed hash-pinned v1 result without filling missing methods",
    )
    render_completed.add_argument(
        "--config",
        default="configs/reproduction/natbiotech_v1_completed_evidence_v1.json",
    )
    render_completed.add_argument("--output", default="paper/v1/generated")
    render_completed.add_argument(
        "--result", default="results/phase1/natbiotech_v1_completed_evidence_render_v1/result.json"
    )
    render_completed.set_defaults(function=_command_paper_render_completed_evidence_v1)

    render_samples = paper_commands.add_parser(
        "render-forge-samples",
        help="render the deterministic display-only FORGE generated-lipid figure",
    )
    render_samples.add_argument(
        "--config",
        default="configs/reproduction/natbiotech_v1_forge_sample_figure_v2.json",
    )
    render_samples.add_argument("--output", default="paper/v1/figures/forge_generated_samples_v2")
    render_samples.add_argument(
        "--result",
        default="results/phase1/forge_generated_sample_figure_v2/result.json",
    )
    render_samples.set_defaults(function=_command_paper_render_forge_samples)

    render_atlas = paper_commands.add_parser(
        "render-forge-sample-atlas",
        help="render the deterministic appendix atlas of generated FORGE lipids",
    )
    render_atlas.add_argument(
        "--config",
        default="configs/reproduction/natbiotech_v1_forge_sample_atlas_v1.json",
    )
    render_atlas.add_argument(
        "--output",
        default="paper/v1/figures/forge_generated_sample_atlas_v1",
    )
    render_atlas.add_argument(
        "--result",
        default="results/phase1/forge_generated_sample_atlas_v1/result.json",
    )
    render_atlas.set_defaults(function=_command_paper_render_forge_sample_atlas)

    collect_results = paper_commands.add_parser(
        "collect-results-v1",
        help="extract normalized hash-pinned rows from common and mechanism result artifacts",
    )
    collect_results.add_argument("--common", action="append", default=[])
    collect_results.add_argument(
        "--route-union",
        help="passing method-blind union adjudication used for every common route metric",
    )
    collect_results.add_argument("--mechanism-evaluation", action="append", default=[])
    collect_results.add_argument("--held-family-evaluation", action="append", default=[])
    collect_results.add_argument("--output", required=True)
    collect_results.set_defaults(function=_command_paper_collect_results_v1)

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

    select_accelerator = experiment_commands.add_parser(
        "select-accelerator",
        help="adjudicate verified L4/A100 benchmark runs and write a pinned selection receipt",
    )
    select_accelerator.add_argument("runs", nargs="+")
    select_accelerator.add_argument(
        "--output",
        default="results/phase1/shared_synthesis_program_accelerator_selection_v1/result.json",
    )
    select_accelerator.set_defaults(function=_command_experiment_select_accelerator)

    adjudicate_production = experiment_commands.add_parser(
        "adjudicate-production",
        help="adjudicate three verified full production runs under the frozen paired-seed rule",
    )
    adjudicate_production.add_argument("runs", nargs="+")
    adjudicate_production.add_argument(
        "--catalogue-runs",
        nargs="*",
        default=(),
        help="three matched full finite-component-catalogue runs",
    )
    adjudicate_production.add_argument(
        "--output",
        default="results/phase1/shared_synthesis_program_production_adjudication_v1/result.json",
    )
    adjudicate_production.set_defaults(function=_command_experiment_adjudicate_production)

    adjudicate_final_production = experiment_commands.add_parser(
        "adjudicate-final-production",
        help="aggregate the promoted FORGE arm with frozen controls and catalogue runs",
    )
    adjudicate_final_production.add_argument("final_runs", nargs=3)
    adjudicate_final_production.add_argument("--control-runs", nargs=3, required=True)
    adjudicate_final_production.add_argument("--catalogue-runs", nargs=3, required=True)
    adjudicate_final_production.add_argument(
        "--output",
        default="results/phase1/final_bl_core_production_adjudication_v1/result.json",
    )
    adjudicate_final_production.set_defaults(
        function=_command_experiment_adjudicate_final_production
    )

    adjudicate_local_chemistry = experiment_commands.add_parser(
        "adjudicate-local-chemistry-resampling",
        help="aggregate and audit three verified constrained frozen-checkpoint resampling runs",
    )
    adjudicate_local_chemistry.add_argument("runs", nargs=3)
    adjudicate_local_chemistry.add_argument(
        "--original",
        default="results/phase1/final_bl_core_production_adjudication_v1/result.json",
    )
    adjudicate_local_chemistry.add_argument(
        "--output",
        default="results/phase1/local_chemistry_resampling_adjudication_v1/result.json",
    )
    adjudicate_local_chemistry.set_defaults(
        function=_command_experiment_adjudicate_local_chemistry_resampling
    )

    adjudicate_local_morphology = experiment_commands.add_parser(
        "adjudicate-local-morphology-seed0",
        help="adjudicate the prespecified seed-0 role-local ring-morphology diagnostic",
    )
    adjudicate_local_morphology.add_argument("run")
    adjudicate_local_morphology.add_argument(
        "--config",
        default="configs/multireaction/local_morphology_seed0_adjudication_v1.json",
    )
    adjudicate_local_morphology.add_argument(
        "--output",
        default="results/phase1/local_morphology_seed0_adjudication_v1/result.json",
    )
    adjudicate_local_morphology.set_defaults(
        function=_command_experiment_adjudicate_local_morphology_seed0
    )

    diagnose_production = experiment_commands.add_parser(
        "diagnose-production",
        help="diagnose the frozen production failure from training/calibration traces only",
    )
    diagnose_production.add_argument(
        "--aggregate",
        default="results/phase1/shared_synthesis_program_production_adjudication_v1/result.json",
    )
    diagnose_production.add_argument(
        "--output",
        default="results/phase1/shared_synthesis_program_production_failure_diagnosis_v1/result.json",
    )
    diagnose_production.set_defaults(function=_command_experiment_diagnose_production)

    freeze_transformer = experiment_commands.add_parser(
        "freeze-transformer-qualification",
        help="freeze one verified Transformer overfit run as a result receipt",
    )
    freeze_transformer.add_argument("run")
    freeze_transformer.add_argument(
        "--output",
        default="results/phase1/reaction_program_transformer_overfit_v1/result.json",
    )
    freeze_transformer.set_defaults(function=_command_experiment_freeze_transformer_qualification)

    assess_common = experiment_commands.add_parser(
        "assess-common-ugi",
        help="apply the common exact-L1 and frozen route-evidence assessment to an attempt ledger",
    )
    assess_common.add_argument("attempts")
    assess_common.add_argument("--method", required=True)
    assess_common.add_argument("--seed", required=True, type=int)
    assess_common.add_argument("--expected-attempts", type=int)
    assess_common.add_argument(
        "--config", default="configs/multireaction/common_ugi_assessment_v1.json"
    )
    assess_common.add_argument("--output", required=True)
    assess_common.set_defaults(function=_command_experiment_assess_common_ugi)

    assess_v0_transformer = experiment_commands.add_parser(
        "assess-ugi-v0-transformer",
        help="run the matched v0-versus-Ugi-Transformer assessment",
    )
    assess_v0_transformer.add_argument(
        "--config",
        default="configs/model/phase1_ugi_v0_production_transformer_assessment_v1.json",
    )
    assess_v0_transformer.add_argument(
        "--output",
        default="results/phase1/ugi_v0_production_transformer_assessment_v1",
    )
    assess_v0_transformer.set_defaults(function=_command_experiment_assess_ugi_v0_transformer)

    assess_realism = experiment_commands.add_parser(
        "assess-lipid-realism",
        help="apply the method-blind observed-lipid structural-realism assessment to a ledger",
    )
    assess_realism.add_argument("attempts")
    assess_realism.add_argument("--method", required=True)
    assess_realism.add_argument("--seed", required=True, type=int)
    assess_realism.add_argument("--expected-attempts", type=int)
    assess_realism.add_argument(
        "--config", default="configs/multireaction/common_lipid_realism_v1.json"
    )
    assess_realism.add_argument("--output", required=True)
    assess_realism.set_defaults(function=_command_experiment_assess_lipid_realism)

    aggregate_realism = experiment_commands.add_parser(
        "aggregate-lipid-realism",
        help="aggregate complete method seed sets from the common lipid-realism assessment",
    )
    aggregate_realism.add_argument("results", nargs="+")
    aggregate_realism.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=(20260825, 20260826, 20260827),
    )
    aggregate_realism.add_argument("--output", required=True)
    aggregate_realism.set_defaults(function=_command_experiment_aggregate_lipid_realism)

    import_external = experiment_commands.add_parser(
        "import-external-ugi",
        help="validate one pinned upstream native receipt and write its canonical attempt ledger",
    )
    import_external.add_argument("--method", required=True)
    import_external.add_argument("--receipt", required=True)
    import_external.add_argument("--samples", required=True)
    import_external.add_argument("--seed", required=True, type=int)
    import_external.add_argument("--expected-attempts", required=True, type=int)
    import_external.add_argument("--train-components")
    import_external.add_argument("--manifest", default="configs/baselines/external_ugi_v1.json")
    import_external.add_argument("--output", required=True)
    import_external.set_defaults(function=_command_experiment_import_external_ugi)

    supersede_genmol = experiment_commands.add_parser(
        "supersede-genmol-empty-labels",
        help="hash-pin the label-only correction for GenMol attempts with empty outputs",
    )
    supersede_genmol.add_argument("--receipt", required=True)
    supersede_genmol.add_argument("--samples", required=True)
    supersede_genmol.add_argument("--seed", required=True, type=int)
    supersede_genmol.add_argument("--expected-attempts", required=True, type=int)
    supersede_genmol.add_argument("--manifest", default="configs/baselines/external_ugi_v1.json")
    supersede_genmol.add_argument("--output", required=True)
    supersede_genmol.set_defaults(function=_command_experiment_supersede_genmol_labels)

    prepare_external = experiment_commands.add_parser(
        "prepare-external-ugi",
        help="freeze a native RGFN, DeFoG or GenMol request from the common Ugi export",
    )
    prepare_external.add_argument("--method", required=True)
    prepare_external.add_argument("--common-export", required=True)
    prepare_external.add_argument("--checkout", required=True)
    prepare_external.add_argument("--seed", required=True, type=int)
    prepare_external.add_argument("--attempts", required=True, type=int)
    prepare_external.add_argument("--profile", required=True, choices=("smoke", "full"))
    prepare_external.add_argument("--manifest", default="configs/baselines/external_ugi_v1.json")
    prepare_external.add_argument("--output", required=True)
    prepare_external.set_defaults(function=_command_experiment_prepare_external_ugi)

    run_external = experiment_commands.add_parser(
        "run-external-ugi",
        help="execute one prepared native baseline request in its pinned upstream environment",
    )
    run_external.add_argument("--request", required=True)
    run_external.add_argument("--checkout", required=True)
    run_external.add_argument("--manifest", default="configs/baselines/external_ugi_v1.json")
    run_external.add_argument("--tokenizer-snapshot")
    run_external.add_argument("--output", required=True)
    run_external.set_defaults(function=_command_experiment_run_external_ugi)

    freeze_route_union = experiment_commands.add_parser(
        "freeze-common-route-union",
        help="freeze a method-blind exact-L1 component union from common assessed ledgers",
    )
    freeze_route_union.add_argument("assessed", nargs="+")
    freeze_route_union.add_argument("--output", required=True)
    freeze_route_union.set_defaults(function=_command_experiment_freeze_common_route_union)

    build_route_evidence = experiment_commands.add_parser(
        "build-common-route-evidence",
        help="disposition a blind component union under the frozen exact-source policy",
    )
    build_route_evidence.add_argument("--config", required=True)
    build_route_evidence.add_argument("--output", required=True)
    build_route_evidence.set_defaults(function=_command_experiment_build_common_route_evidence)

    adjudicate_route_union = experiment_commands.add_parser(
        "adjudicate-common-route-union",
        help="score all common methods against one complete method-blind route-evidence union",
    )
    adjudicate_route_union.add_argument("assessed", nargs="+")
    adjudicate_route_union.add_argument("--union", required=True)
    adjudicate_route_union.add_argument("--evidence", required=True)
    adjudicate_route_union.add_argument("--output", required=True)
    adjudicate_route_union.set_defaults(function=_command_experiment_adjudicate_common_route_union)
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
