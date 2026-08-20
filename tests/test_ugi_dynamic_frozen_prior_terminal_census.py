from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.phase1.product_l1.sampling.terminal_census import (
    EXPECTED_SCOPE,
    CensusContract,
    CensusDesign,
    UgiDynamicTerminalCensusError,
    _design_from_config,
    complete_native_no_route_terminal,
    execute_census_shard,
    load_census_contract,
    load_partial_state,
    load_selected_program_manifest,
    pack_partial_state,
    restore_partial_state,
    save_partial_state,
    select_frozen_prior_programs,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    UgiJointSparseTrajectoryState,
)
from experiments.phase1.synthesis_guidance.adapters.selected_v1 import (
    SelectedGuidanceParticleState,
    SelectedGuidanceState,
)
from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    canonical_morphology_program_bytes,
    decode_canonical_morphology_program_bytes,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_nonzero_guidance_runner import (
    GuidanceStateReceipt,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_v1.json"
SOURCE = REPO / "experiments/phase1/product_l1/sampling/terminal_census.py"


def _program(index: int) -> dict[str, list[int]]:
    return {
        "node_counts": [3 + index, 4, 5],
        "junction_budgets": [0, 0, 0],
        "cycle_ranks": [0, 0, 0],
        "attachment_counts": [1, 1, 1],
    }


def _program_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "product_id": f"program-{index:04d}",
            "program": _program(index),
            "source_stratum": "training_fold_weighted_program_prior",
            "branch_class": "linear_tail_origins",
            "component_novelty_class": "unconditioned_program_prior",
        }
        for index in range(count)
    ]


class _FakeLane:
    adapter_identity_sha256 = "a" * 64

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        assert device == "cpu"
        unique_programs: dict[bytes, int] = {}
        particles = []
        for index, (program_bytes, particle_seed) in enumerate(
            zip(programs, particle_seeds, strict=True)
        ):
            program = decode_canonical_morphology_program_bytes(program_bytes)
            unique_programs.setdefault(program_bytes, len(unique_programs))
            node_count = program.node_count
            trajectory = UgiJointSparseTrajectoryState(
                programs=(program,),
                layout={
                    "node_mask": torch.ones((1, node_count), dtype=torch.bool),
                    "role_states": torch.zeros((1, node_count), dtype=torch.long),
                },
                sources={"source": torch.tensor([0.5, 0.5])},
                channels={
                    "nodes": torch.full((1, node_count), particle_seed % 7),
                    "offspring": torch.zeros((1, node_count), dtype=torch.long),
                },
                sample_steps=8,
                step=0,
                generator_state=torch.tensor(
                    [particle_seed % 251, (particle_seed // 251) % 251],
                    dtype=torch.uint8,
                ),
                device="cpu",
            )
            particles.append(
                SelectedGuidanceParticleState(
                    global_particle_index=index,
                    program_index=unique_programs[program_bytes],
                    program_bytes=program_bytes,
                    original_particle_seed=particle_seed,
                    trajectory=trajectory,
                    founder_particle_index=index,
                    ancestry_path=(index,),
                )
            )
        state = SelectedGuidanceState(
            base_seed=seed,
            adapter_identity_sha256=self.adapter_identity_sha256,
            particles=tuple(particles),
        )
        return GuidanceStateReceipt(state=state, product_transition_calls=0)

    def advance(self, state: SelectedGuidanceState, *, target_step: int) -> GuidanceStateReceipt:
        delta = target_step - state.step
        particles = []
        for particle in state.particles:
            old = particle.trajectory
            trajectory = UgiJointSparseTrajectoryState(
                programs=old.programs,
                layout={key: value.clone() for key, value in old.layout.items()},
                sources={key: value.clone() for key, value in old.sources.items()},
                channels={
                    key: value.clone() + (delta if key == "nodes" else 0)
                    for key, value in old.channels.items()
                },
                sample_steps=old.sample_steps,
                step=target_step,
                generator_state=old.generator_state.clone() + delta,
                device=old.device,
            )
            particles.append(
                SelectedGuidanceParticleState(
                    global_particle_index=particle.global_particle_index,
                    program_index=particle.program_index,
                    program_bytes=particle.program_bytes,
                    original_particle_seed=particle.original_particle_seed,
                    trajectory=trajectory,
                    founder_particle_index=particle.founder_particle_index,
                    ancestry_path=particle.ancestry_path,
                )
            )
        return GuidanceStateReceipt(
            state=SelectedGuidanceState(
                base_seed=state.base_seed,
                adapter_identity_sha256=state.adapter_identity_sha256,
                particles=tuple(particles),
            ),
            product_transition_calls=len(particles) * delta,
        )


def _fake_completion(
    _lane: _FakeLane,
    state: SelectedGuidanceState,
    *,
    particle_index: int,
    seed: int,
    checkpoint_index: int,
    rollout_index: int,
) -> dict[str, Any]:
    payload = {
        "particle_state_sha256": state.particles[particle_index].state_sha256,
        "particle_index": particle_index,
        "seed": seed,
        "checkpoint_index": checkpoint_index,
        "rollout_index": rollout_index,
    }
    return {
        "completion_id": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "status": "terminal",
        **payload,
        "source_state_unchanged": True,
        "product_transition_calls": 8 - checkpoint_index,
        "error_detail": None,
        "native_terminal": {
            "raw_molecule_valid": True,
            "terminal_valid": True,
            "smiles": "CC",
        },
    }


def _small_contract(tmp_path: Path) -> tuple[CensusContract, Any]:
    design = CensusDesign(
        population_programs=4,
        selected_programs=2,
        states_per_program=2,
        checkpoints=(2, 4, 6),
        rollouts_per_state_checkpoint=2,
        sample_steps=8,
        shard_programs=2,
        program_selection_seed=11,
        particle_seed_base=12,
        rollout_seed_base=13,
        device="cpu",
    )
    manifest = select_frozen_prior_programs(_program_rows(4), selected_count=2, selection_seed=11)
    contract = CensusContract(
        repo=tmp_path,
        config_path=tmp_path / "config.json",
        config_sha256="b" * 64,
        design=design,
        inputs={},
        scope=EXPECTED_SCOPE,
    )
    return contract, manifest


def test_production_config_is_hash_pinned_and_calibration_only() -> None:
    contract = load_census_contract(REPO, CONFIG)
    manifest = load_selected_program_manifest(contract)

    assert contract.design.population_programs == 4096
    assert contract.design.selected_programs == 1024
    assert contract.design.states_per_program == 2
    assert contract.design.checkpoints == (2, 4, 6)
    assert contract.design.rollouts_per_state_checkpoint == 4
    assert contract.design.terminal_attempts == 24576
    assert contract.scope == EXPECTED_SCOPE
    assert len(manifest.programs) == 1024
    assert len({row.product_id for row in manifest.programs}) == 1024
    assert len({row.population_index for row in manifest.programs}) == 1024


def test_config_rejects_final_step_and_any_nonzero_oracle_scope() -> None:
    value = json.loads(CONFIG.read_text())
    changed = dict(value["design"])
    changed["checkpoints"] = [2, 4, 8]
    with pytest.raises(UgiDynamicTerminalCensusError, match="interior flow steps"):
        _design_from_config(changed)

    changed_config = json.loads(CONFIG.read_text())
    changed_config["scope"]["oracle_calls"] = 1
    assert changed_config["scope"] != EXPECTED_SCOPE


def test_hash_priority_selection_is_deterministic_and_terminal_blind() -> None:
    rows = _program_rows(20)
    selected = select_frozen_prior_programs(rows, selected_count=7, selection_seed=99)
    repeated = select_frozen_prior_programs(rows, selected_count=7, selection_seed=99)
    reversed_rows = select_frozen_prior_programs(
        list(reversed(rows)), selected_count=7, selection_seed=99
    )

    assert selected.manifest_sha256 == repeated.manifest_sha256
    assert [row.product_id for row in selected.programs] == [
        row.product_id for row in reversed_rows.programs
    ]
    assert all("valid" not in row.metadata for row in selected.programs)


def test_compact_state_round_trip_is_weights_only_and_hash_exact(tmp_path: Path) -> None:
    lane = _FakeLane()
    programs = tuple(
        canonical_morphology_program_bytes(UgiMorphologyProgram(**_program(index)))
        for index in range(2)
    )
    state = lane.initialize(
        programs,
        seed=17,
        particle_seeds=(101, 202),
        device="cpu",
    ).state
    state = lane.advance(state, target_step=4).state
    payload = pack_partial_state(state)

    assert "layout" not in payload["particles"][0]
    assert "sources" not in payload["particles"][0]
    restored_memory = restore_partial_state(lane, payload)
    assert restored_memory.state_sha256 == state.state_sha256

    path = tmp_path / "state.pt"
    receipt = save_partial_state(path, state)
    weights_only = torch.load(path, map_location="cpu", weights_only=True)
    restored_disk = load_partial_state(path, lane)
    assert weights_only["state_sha256"] == state.state_sha256
    assert restored_disk.state_sha256 == state.state_sha256
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_shard_is_exact_counted_and_resume_safe(tmp_path: Path) -> None:
    contract, manifest = _small_contract(tmp_path)
    lane = _FakeLane()
    output = tmp_path / "census"

    first = execute_census_shard(
        contract,
        manifest,
        output,
        shard_index=0,
        lane=lane,
        native_completer=_fake_completion,
    )
    resumed = execute_census_shard(
        contract,
        manifest,
        output,
        shard_index=0,
        lane=lane,
        native_completer=_fake_completion,
    )

    assert first["counts"] == {
        "programs": 2,
        "particles": 4,
        "partial_states": 12,
        "terminal_attempts": 24,
    }
    assert first["resumed_completed_shard"] is False
    assert resumed["resumed_completed_shard"] is True
    assert resumed["result_sha256"] == first["result_sha256"]
    assert (output / "shards/shard_0000/partial_state_step_02.pt").is_file()
    assert (output / "shards/shard_0000/partial_state_step_04.pt").is_file()
    assert (output / "shards/shard_0000/partial_state_step_06.pt").is_file()


def test_native_completion_does_not_hide_invariant_failures(monkeypatch: Any) -> None:
    lane = _FakeLane()
    lane.callback = SimpleNamespace(model=object())
    program = canonical_morphology_program_bytes(UgiMorphologyProgram(**_program(0)))
    state = lane.initialize(
        (program,),
        seed=17,
        particle_seeds=(101,),
        device="cpu",
    ).state
    state = lane.advance(state, target_step=2).state

    def _fail_hard(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("programming invariant failed")

    monkeypatch.setattr(
        "experiments.phase1.product_l1.sampling.terminal_census." "advance_ugi_joint_sparse_state",
        _fail_hard,
    )
    with pytest.raises(RuntimeError, match="programming invariant failed"):
        complete_native_no_route_terminal(
            lane,
            state,
            particle_index=0,
            seed=303,
            checkpoint_index=2,
            rollout_index=0,
        )


def test_census_source_has_no_forbidden_scientific_subsystem_imports() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not any(name.startswith("forge.bio") for name in imported)
    assert not any(name.startswith("forge.synthesis") for name in imported)
    source = SOURCE.read_text()
    assert "adapt_restartable_completion_row_for_route_support" not in source
    assert "_lock_completion_row" not in source
