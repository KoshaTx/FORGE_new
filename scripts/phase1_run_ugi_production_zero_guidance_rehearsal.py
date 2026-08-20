#!/usr/bin/env python3
"""Run the real, nonselecting Ugi production rehearsal at zero guidance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.schedule.ugi_production_terminal_route_evaluator import (
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.design.guidance.ugi_production_zero_guidance_config import (
    load_production_zero_guidance_rehearsal_plan,
)
from forge.design.guidance.ugi_production_zero_guidance_result import (
    ProductionTerminalSupportAuditCollector,
    build_production_zero_guidance_execution_result,
)
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.design.sampling.ugi_selected_generator_implementation import (
    require_selected_generator_implementation_unchanged,
)
from forge.design.sampling.ugi_selected_restartable_generator import (
    build_selected_step1000_restartable_generator_lane,
)
from forge.design.guidance.ugi_zero_guidance_rehearsal import (
    QualifiedRoutePlannerFactoryAdapter,
    ZeroGuidanceRehearsalContract,
    current_terminal_route_assessment_source_sha256,
    run_zero_guidance_route_rehearsal,
)
from forge.route.engine.planner_cache import FilePlannerCache
from forge.route.engine.planner_cache_snapshot import (
    build_file_planner_cache_snapshot_manifest,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_rehearsal_v1.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_production_zero_guidance_rehearsal_v1"
EXECUTION_MANIFEST_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_execution.v1"
EXECUTION_ORCHESTRATION_SCHEMA_VERSION = (
    "phase1_ugi_production_zero_guidance_orchestration_implementation.v1"
)
EXECUTION_ORCHESTRATION_SOURCE_PATHS = (
    "scripts/phase1_run_ugi_production_zero_guidance_rehearsal.py",
    "src/forge/product/ugi_matched_budget_orchestration.py",
    "src/forge/product/ugi_matched_planner_cache_binding.py",
    "src/forge/product/ugi_production_terminal_route_evaluator.py",
    "src/forge/product/ugi_production_zero_guidance_config.py",
    "src/forge/product/ugi_production_zero_guidance_result.py",
    "src/forge/product/ugi_zero_guidance_rehearsal.py",
    "src/forge/route/planner_cache.py",
    "src/forge/route/planner_cache_snapshot.py",
    "src/forge/route/ugi3_production_planner_qualification.py",
)


class ProductionZeroGuidanceExecutionError(RuntimeError):
    """Raised when the production rehearsal cannot preserve a clean ownership boundary."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bytes(value: Any) -> bytes:
    return (_stable_json(value) + "\n").encode()


def _build_orchestration_implementation(repo: Path) -> dict[str, Any]:
    sources = []
    for relative_path in EXECUTION_ORCHESTRATION_SOURCE_PATHS:
        path = repo / relative_path
        if not path.is_file() or path.is_symlink():
            raise ProductionZeroGuidanceExecutionError(
                f"orchestration source is missing or not a real file: {relative_path}"
            )
        sources.append({"path": relative_path, "sha256": sha256_file(path)})
    content = {
        "schema_version": EXECUTION_ORCHESTRATION_SCHEMA_VERSION,
        "sources": sources,
    }
    return {**content, "implementation_sha256": _sha256_payload(content)}


def _require_orchestration_implementation_unchanged(
    repo: Path,
    expected: dict[str, Any],
) -> None:
    if _build_orchestration_implementation(repo) != expected:
        raise ProductionZeroGuidanceExecutionError(
            "production orchestration implementation changed during execution"
        )


def _require_clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ProductionZeroGuidanceExecutionError(
                "output path must be a real directory or not yet exist"
            )
        if any(output_dir.iterdir()):
            raise ProductionZeroGuidanceExecutionError(
                "production output directory must be absent or empty; refusing warm-cache reuse"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def run_production_zero_guidance_rehearsal(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute generation and routing once, with no scalar, guidance or selection."""

    repo = repo_root.resolve()
    config = config_path.resolve()
    output = output_dir.resolve()
    try:
        config_relative_path = config.relative_to(repo)
    except ValueError as error:
        raise ProductionZeroGuidanceExecutionError(
            "production config must be located inside the authenticated repository"
        ) from error
    orchestration_implementation = _build_orchestration_implementation(repo)
    _require_clean_output(output)

    lane = build_selected_step1000_restartable_generator_lane(repo)
    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc="2026-08-03T04:00:00Z",
        expected_cumulative_source_inputs_sha256=(
            "adfc6767d44d9aa56cfb46bba2253e485ec35f5dace0af0f5a11f9cd72bcd3c6"
        ),
        selected_generator_checkpoint_sha256=(lane.bindings.generator_checkpoint_sha256),
        graph_support=lane.graph_support,
        l1_reverifier=lane.callback.l1_reverifier,
        candidate_record_resolver=native_completion_record_from_locked_terminal,
    )
    plan = load_production_zero_guidance_rehearsal_plan(
        repo,
        config,
        planner_context=factory.planner_context,
    )
    if lane.adapter.identity != plan.identity:
        raise ProductionZeroGuidanceExecutionError(
            "selected generator lane and authenticated production plan differ"
        )
    if factory.assessment_as_of_utc != plan.assessment_as_of_utc:
        raise ProductionZeroGuidanceExecutionError(
            "planner factory and authenticated production plan use different assessment times"
        )
    if factory.source_inputs_sha256 != plan.cumulative_source_inputs_sha256:
        raise ProductionZeroGuidanceExecutionError(
            "planner factory and authenticated production plan use different source inputs"
        )

    cache_root = output / "cache"
    base_cache = FilePlannerCache(cache_root / "base")
    guided_cache = FilePlannerCache(cache_root / "guided")
    post_hoc_cache = FilePlannerCache(cache_root / "post_hoc")
    base_snapshot = build_file_planner_cache_snapshot_manifest(
        base_cache,
        factory.planner_context,
        assessment_at_utc=plan.assessment_as_of_utc,
    )
    if base_snapshot.entry_count != 0:
        raise ProductionZeroGuidanceExecutionError(
            "new production rehearsal must begin from an empty immutable base cache"
        )

    collector = ProductionTerminalSupportAuditCollector(factory.build_planner)
    contract = ZeroGuidanceRehearsalContract(
        base_seed=plan.seed,
        guidance_strength=0.0,
        assessment_at_utc=plan.assessment_as_of_utc,
        budget_limits=plan.budget_limits,
        generator_identity=plan.identity,
        planner_context_sha256=plan.planner_context_sha256,
        expected_cache_snapshot_sha256=base_snapshot.snapshot_sha256,
        terminal_route_assessment_source_sha256=(current_terminal_route_assessment_source_sha256()),
        route_qualification_sha256=factory.qualification_sha256,
    )
    composer_result = run_zero_guidance_route_rehearsal(
        plan.schedule,
        contract=contract,
        generator=lane.adapter,
        planner_context=factory.planner_context,
        base_cache=base_cache,
        guided_overlay=guided_cache,
        post_hoc_overlay=post_hoc_cache,
        l1_reverifier=lane.callback.l1_reverifier,
        planner_factory=QualifiedRoutePlannerFactoryAdapter(
            qualification_sha256=factory.qualification_sha256,
            build_planner=collector.build_planner,
        ),
    )
    execution_result = build_production_zero_guidance_execution_result(
        composer_result,
        collector.snapshots,
    )
    require_selected_generator_implementation_unchanged(
        repo,
        lane.implementation_qualification,
    )
    _require_orchestration_implementation_unchanged(repo, orchestration_implementation)
    if sha256_file(config) != plan.config_sha256:
        raise ProductionZeroGuidanceExecutionError("production config changed during execution")

    plan_value = plan.to_dict()
    planner_qualification_value = {
        **factory.planner_context_qualification.to_dict(),
        "qualification_sha256": (factory.planner_context_qualification.qualification_sha256),
    }
    internal_roles_value = {
        **factory.internal_role_manifest.to_dict(),
        "manifest_sha256": factory.internal_role_manifest.manifest_sha256,
    }
    generator_implementation_value = lane.implementation_qualification.to_dict()
    output_values = {
        "plan.json": plan_value,
        "planner_context_qualification.json": planner_qualification_value,
        "internal_role_manifest.json": internal_roles_value,
        "generator_implementation_qualification.json": generator_implementation_value,
        "orchestration_implementation_qualification.json": orchestration_implementation,
        "composer_result.json": composer_result.to_dict(),
        "result.json": execution_result.to_dict(),
    }
    output_bytes = {name: _canonical_bytes(value) for name, value in output_values.items()}
    for name, payload in output_bytes.items():
        _atomic_write(output / name, payload)

    manifest_content = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "status": "production_zero_guidance_rehearsal_complete",
        "inputs": {
            "config": {
                "path": str(config_relative_path),
                "sha256": plan.config_sha256,
            },
            "script": {
                "path": str(Path(__file__).resolve().relative_to(repo)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "outputs": {
            name: {"sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(output_bytes.items())
        },
        "planner_context_sha256": factory.planner_context_sha256,
        "planner_context_qualification_sha256": (
            factory.planner_context_qualification.qualification_sha256
        ),
        "planner_factory_qualification_sha256": factory.qualification_sha256,
        "internal_role_manifest_sha256": factory.internal_role_manifest.manifest_sha256,
        "generator_implementation_sha256": (
            lane.implementation_qualification.implementation_sha256
        ),
        "orchestration_implementation_sha256": orchestration_implementation[
            "implementation_sha256"
        ],
        "execution_result_sha256": execution_result.result_sha256,
        "paired_support_count": execution_result.paired_support_count,
        "scope": {
            "guidance_strength": 0,
            "synthesis_scalar_defined": False,
            "nonzero_synthesis_guidance": False,
            "biological_guidance": False,
            "candidate_selection": False,
            "private_holdout_accessed": False,
        },
    }
    manifest = {
        **manifest_content,
        "manifest_sha256": _sha256_payload(manifest_content),
    }
    _atomic_write(output / "execution_manifest.json", _canonical_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = run_production_zero_guidance_rehearsal(
        repo_root=REPO,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
