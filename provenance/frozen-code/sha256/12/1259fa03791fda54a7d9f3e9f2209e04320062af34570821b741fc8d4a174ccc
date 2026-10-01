"""Attribute wall time across the real synthesis-program production evaluation.

The production evaluation stage runs on a paid accelerator, so nobody had measured where its
wall time actually goes.  This tool runs the *real* entry point --
`run_synthesis_program_production_evaluation` -- against the real pinned inputs on CPU, at a
reduced but structurally identical budget, and reports exclusive wall time per phase.

Nothing here changes the evaluated computation.  Every phase boundary is installed by wrapping
the function the evaluation already calls, and the wrappers are removed afterwards.  The budget
is reduced through the config's own `h100_preflight` execution scope, which exists precisely so
a reduced full-device budget stays bound to an explicit, non-production scope.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "forge.synthesis_program_production_evaluation_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_production_evaluation_profile.v1"


class EvaluationProfileError(ValueError):
    """The profiled evaluation cannot be set up from the requested pins."""


class PhaseTimer:
    """Exclusive wall time per named phase, with nesting handled by subtraction.

    A phase entered inside another phase is charged only to itself; the enclosing phase keeps
    the time it spent outside its children.  That makes the emitted seconds sum to the total,
    which is the property a percentage table needs.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._children: list[float] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        self._children.append(0.0)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            child = self._children.pop()
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - child
            self.calls[name] = self.calls.get(name, 0) + 1
            if self._children:
                self._children[-1] += elapsed

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.phase(name):
                return function(*args, **kwargs)

        wrapped.__name__ = getattr(function, "__name__", name)
        wrapped.__wrapped__ = function  # type: ignore[attr-defined]
        return wrapped


class _Patches:
    """Reversible attribute patches, so a profiled process leaves no changed module behind."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any]] = []

    def set(self, owner: Any, name: str, value: Any) -> None:
        self._undo.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def wrap(self, timer: PhaseTimer, owner: Any, name: str, label: str) -> None:
        self.set(owner, name, timer.wrap(label, getattr(owner, name)))

    def restore(self) -> None:
        while self._undo:
            owner, name, original = self._undo.pop()
            setattr(owner, name, original)


def _reduced_config(
    source: Mapping[str, Any],
    *,
    calibration_samples: int,
    heldout_samples: int,
    component_disjoint_record_limit: int,
    checkpoint_steps: list[int] | None,
) -> dict[str, Any]:
    """Reduce only the per-cell sample counts, and bind the reduction to a non-production scope."""

    if source.get("schema_version") != CONFIG_SCHEMA:
        raise EvaluationProfileError("profiled config is not a production evaluation config")
    config = json.loads(json.dumps(source))
    full = config["full"]
    full["device"] = "cpu"
    full["calibration_samples"] = int(calibration_samples)
    full["heldout_samples"] = int(heldout_samples)
    full["component_disjoint_record_limit"] = int(component_disjoint_record_limit)
    if checkpoint_steps is not None:
        full["checkpoint_steps"] = [int(value) for value in checkpoint_steps]
    # The frozen production scope requires the frozen budget; a reduced budget is exactly what
    # `h100_preflight` exists to carry, and the evaluation refuses to call such a run production.
    config["execution_scope"] = "h100_preflight"
    return config


def _install(timer: PhaseTimer, patches: _Patches) -> None:
    """Wrap every function the evaluation calls for a distinguishable unit of work."""

    from experiments.phase1.multireaction import production_evaluation as pe
    from forge.model import reaction_program_evaluation as rpe
    from forge.model import synthesis_program_layout as spl
    from forge.model import synthesis_program_sampling as sps

    # Entry-point phases, wrapped where the evaluation module resolves them.
    patches.wrap(timer, pe, "sha256_file", "pin.sha256_file")
    patches.wrap(timer, pe, "resolve_pin", "pin.resolve_and_verify")
    patches.wrap(timer, pe, "artifact_record", "pin.artifact_record")
    patches.wrap(timer, pe, "pin_record", "pin.pin_record")
    patches.wrap(timer, pe, "read_json_object", "io.read_json_object")
    patches.wrap(timer, pe, "write_json", "io.write_result_json")
    patches.wrap(timer, pe, "load_reaction_program_specifications", "setup.program_specs")
    patches.wrap(timer, pe, "SynthesisProgramProductionCache", "setup.production_cache_open")
    patches.wrap(timer, pe, "SynthesisProgramLayoutPrior", "setup.layout_prior_construction")
    patches.wrap(
        timer, pe, "load_reaction_program_training_references", "setup.training_references"
    )
    patches.wrap(timer, pe, "_validate_archive_members", "checkpoint.archive_member_validation")
    patches.wrap(timer, pe, "adjudicate_reaction_program_rows", "assess.exact_l1_decomposition")
    patches.wrap(timer, pe, "evaluate_reaction_program_samples", "assess.sample_metrics")
    patches.wrap(timer, pe, "_component_disjoint_reconstruction", "assess.component_disjoint")
    patches.wrap(timer, pe, "write_attempt_ledger", "report.common_ugi_attempt_ledger")
    patches.wrap(timer, pe, "load_ugi_identity_references", "report.ugi_identity_references")
    patches.wrap(timer, pe, "cross_role_fidelity_to_heldout", "report.cross_role_fidelity")
    patches.wrap(timer, pe, "_write_molecule_report", "report.molecule_report_render")
    patches.wrap(timer, pe, "_write_samples", "report.samples_jsonl_gz")

    # Adapter construction is two class methods on two different classes.
    patches.wrap(
        timer, pe.RegistryRepeatedReactionProgram, "from_registry", "setup.adapter_construction"
    )
    patches.wrap(timer, pe.Ugi3AssemblyAdapter, "from_registry", "setup.adapter_construction")

    # Layout sampling, wrapped on the class so the instance the evaluation builds is covered.
    patches.wrap(timer, spl.SynthesisProgramLayoutPrior, "sample", "sample.layout_prior_draw")

    # Sampling internals.  `_load_checkpoint` also wraps the returned model's forward, so the
    # 32 denoising steps and the terminal pass are attributed separately from everything else.
    patches.wrap(timer, sps, "collate_synthesis_program_layouts", "sample.layout_collation")
    patches.wrap(timer, sps, "decode_synthesis_program_strict_argmax", "sample.terminal_decode")
    patches.wrap(timer, sps, "_terminal_smiles", "sample.smiles_construction")
    patches.wrap(timer, sps, "rstar_step", "sample.rstar_update")
    patches.wrap(timer, sps, "pointer_rstar_step", "sample.rstar_update")
    patches.wrap(timer, sps, "_fixed_state_exact_tensor", "sample.fixed_state_audit")
    patches.wrap(timer, sps, "_fixed_state_exact_records", "sample.fixed_state_audit")
    patches.wrap(timer, sps, "_initial_state", "sample.initial_state")
    patches.wrap(timer, sps, "_restore_fixed_states_in_place", "sample.fixed_state_restore")
    patches.wrap(timer, sps, "_program_conditioning", "sample.program_conditioning")

    # Metric internals.
    patches.wrap(timer, rpe, "_mean_pairwise_distance", "assess.pairwise_ecfp4_diversity")

    original_sample = pe.sample_synthesis_program_products
    original_load = pe._load_checkpoint

    def sample_products(*args: Any, **kwargs: Any) -> Any:
        with timer.phase("sample.other"):
            return original_sample(*args, **kwargs)

    def load_checkpoint(*args: Any, **kwargs: Any) -> Any:
        with timer.phase("checkpoint.unpack_and_build_model"):
            model, package = original_load(*args, **kwargs)
        patched = timer.wrap("sample.model_forward", model.forward)
        model.forward = patched  # type: ignore[method-assign]
        return model, package

    patches.set(pe, "sample_synthesis_program_products", sample_products)
    patches.set(pe, "_load_checkpoint", load_checkpoint)


def profile_production_evaluation(
    repo: Path,
    *,
    config_path: Path,
    cache_path: Path,
    checkpoint_archive_path: Path,
    training_result_path: Path,
    output_dir: Path,
    calibration_samples: int,
    heldout_samples: int,
    component_disjoint_record_limit: int,
    checkpoint_steps: list[int] | None = None,
    replicate: int = 0,
) -> dict[str, Any]:
    """Run the real evaluation on CPU under a reduced budget and attribute its wall time."""

    from experiments.phase1.multireaction.production_evaluation import (
        run_synthesis_program_production_evaluation,
    )
    from forge.core.io import read_json_object, write_json

    source = read_json_object(config_path, error=EvaluationProfileError, label="evaluation config")
    config = _reduced_config(
        source,
        calibration_samples=calibration_samples,
        heldout_samples=heldout_samples,
        component_disjoint_record_limit=component_disjoint_record_limit,
        checkpoint_steps=checkpoint_steps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profiled_config = output_dir / "profiled_evaluation_config.json"
    write_json(profiled_config, config)

    timer = PhaseTimer()
    patches = _Patches()
    _install(timer, patches)
    try:
        started = time.perf_counter()
        result = run_synthesis_program_production_evaluation(
            profiled_config,
            repo,
            cache_path,
            checkpoint_archive_path,
            training_result_path,
            output_dir,
            profile="full",
            replicate=replicate,
            allocated_device="cpu",
        )
        total = time.perf_counter() - started
    finally:
        patches.restore()

    attributed = sum(timer.seconds.values())
    phases = [
        {
            "phase": name,
            "seconds": round(timer.seconds[name], 4),
            "calls": timer.calls[name],
            "percent_of_total": round(100.0 * timer.seconds[name] / total, 3),
        }
        for name in sorted(timer.seconds, key=lambda key: -timer.seconds[key])
    ]
    phases.append(
        {
            "phase": "unattributed_evaluation_body",
            "seconds": round(total - attributed, 4),
            "calls": 1,
            "percent_of_total": round(100.0 * (total - attributed) / total, 3),
        }
    )
    import torch

    return {
        "schema_version": RESULT_SCHEMA,
        "budget": {
            "calibration_samples": int(calibration_samples),
            "heldout_samples": int(heldout_samples),
            "component_disjoint_record_limit": int(component_disjoint_record_limit),
            "checkpoint_steps": config["full"]["checkpoint_steps"],
            "sample_steps": int(config["full"]["sample_steps"]),
            "batch_size": int(config["full"]["batch_size"]),
            "terminal_decode_policy": str(config["full"]["terminal_decode_policy"]),
            "execution_scope": "h100_preflight",
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch_threads": int(torch.get_num_threads()),
        },
        "total_seconds": round(total, 4),
        "sample_rows": int(result["sample_rows"]),
        "status": str(result["status"]),
        "phases": phases,
    }


__all__ = [
    "EvaluationProfileError",
    "PhaseTimer",
    "RESULT_SCHEMA",
    "profile_production_evaluation",
]
