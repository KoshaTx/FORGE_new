from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import forge.product.ugi_dynamic_frozen_prior_terminal_census as census_module
from forge.product.ugi_dynamic_frozen_prior_terminal_census import (
    CensusContract,
    CensusDesign,
    _jsonl_gzip_bytes,
    _sha256_payload,
    aggregate_completed_census,
    execute_census_shard,
    select_frozen_prior_programs,
)
from forge.product.ugi_dynamic_frozen_prior_terminal_census_validator import (
    EXPECTED_SCOPE,
    UgiDynamicTerminalCensusValidationError,
    load_validation_contract,
    validate_completed_census,
    write_validation_receipt,
)
from forge.product.ugi_joint_sparse_sampling import UgiJointSparseTrajectoryState
from forge.product.ugi_nonzero_guidance_runner import GuidanceStateReceipt
from forge.product.ugi_restartable_terminal_support_adapter import (
    decode_canonical_morphology_program_bytes,
)
from forge.product.ugi_selected_guidance_adapter import (
    SelectedGuidanceParticleState,
    SelectedGuidanceState,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_validator_v1.json"
SOURCE = REPO / "src/forge/product/ugi_dynamic_frozen_prior_terminal_census_validator.py"
RUNNER = REPO / "scripts/phase1_validate_ugi_dynamic_frozen_prior_terminal_census.py"


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
        return GuidanceStateReceipt(
            state=SelectedGuidanceState(
                base_seed=seed,
                adapter_identity_sha256=self.adapter_identity_sha256,
                particles=tuple(particles),
            ),
            product_transition_calls=0,
        )

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


class _NondeterministicSourceLane(_FakeLane):
    def __init__(self) -> None:
        self.initializations = 0

    def initialize(self, *args: Any, **kwargs: Any) -> GuidanceStateReceipt:
        receipt = super().initialize(*args, **kwargs)
        self.initializations += 1
        for particle in receipt.state.particles:
            particle.trajectory.sources["source"].fill_(float(self.initializations))
        return receipt


def _fake_completion(
    _lane: _FakeLane,
    state: SelectedGuidanceState,
    *,
    particle_index: int,
    seed: int,
    checkpoint_index: int,
    rollout_index: int,
) -> dict[str, Any]:
    identity = {
        "adapter_identity_sha256": state.adapter_identity_sha256,
        "particle_state_sha256": state.particles[particle_index].state_sha256,
        "particle_index": particle_index,
        "checkpoint_index": checkpoint_index,
        "rollout_index": rollout_index,
        "seed": seed,
    }
    return {
        "completion_id": _sha256_payload(identity),
        "status": "terminal",
        **identity,
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
        population_programs=6,
        selected_programs=4,
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
    manifest = select_frozen_prior_programs(_program_rows(6), selected_count=4, selection_seed=11)
    contract = CensusContract(
        repo=tmp_path,
        config_path=tmp_path / "config.json",
        config_sha256="b" * 64,
        design=design,
        inputs={},
        scope=census_module.EXPECTED_SCOPE,
    )
    return contract, manifest


def _build_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CensusContract, Any, Path, _FakeLane]:
    contract, manifest = _small_contract(tmp_path)
    lane = _FakeLane()
    output = tmp_path / "census"
    for shard_index in range(contract.design.shard_count):
        execute_census_shard(
            contract,
            manifest,
            output,
            shard_index=shard_index,
            lane=lane,
            native_completer=_fake_completion,
        )
    monkeypatch.setattr(census_module, "load_census_contract", lambda *_args: contract)
    monkeypatch.setattr(census_module, "load_selected_program_manifest", lambda *_args: manifest)
    aggregate_completed_census(tmp_path, contract.config_path, output)
    return contract, manifest, output, lane


def _tree_sha256s(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def _reseal_shard_terminals(
    output: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
    *,
    update_logical_hash: bool,
) -> None:
    shard_dir = output / "shards" / f"shard_{shard_index:04d}"
    terminal_path = shard_dir / "terminals.jsonl.gz"
    terminal_path.write_bytes(_jsonl_gzip_bytes(rows))
    receipt_path = shard_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["terminals"]["sha256"] = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    if update_logical_hash:
        receipt["artifacts"]["terminals"]["logical_sha256"] = _sha256_payload(rows)
    content = {key: value for key, value in receipt.items() if key != "result_sha256"}
    receipt["result_sha256"] = _sha256_payload(content)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_validator_accepts_complete_evidence_without_mutating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, manifest, output, lane = _build_evidence(tmp_path, monkeypatch)
    before = _tree_sha256s(output)

    result = validate_completed_census(contract, manifest, output, lane=lane)

    assert result["status"] == "validated_complete_dynamic_terminal_census_read_only"
    assert result["counts"]["terminal_attempts"] == 48
    assert result["global_identities"] == {
        "particle_seeds": 8,
        "unique_particle_seeds": 8,
        "rollout_seeds": 48,
        "unique_rollout_seeds": 48,
        "completion_ids": 48,
        "unique_completion_ids": 48,
        "coordinates": 48,
        "unique_coordinates": 48,
    }
    assert _tree_sha256s(output) == before

    receipt_path = tmp_path / "validation" / "receipt.json"
    write_validation_receipt(receipt_path, result, census_dir=output)
    assert json.loads(receipt_path.read_text())["result_sha256"] == result["result_sha256"]
    with pytest.raises(UgiDynamicTerminalCensusValidationError, match="outside"):
        write_validation_receipt(output / "validator.json", result, census_dir=output)


def test_validator_recomputes_each_shard_terminal_logical_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, manifest, output, lane = _build_evidence(tmp_path, monkeypatch)
    terminal_path = output / "shards/shard_0000/terminals.jsonl.gz"
    rows = _read_rows(terminal_path)
    rows[0]["native_terminal"]["smiles"] = "CCC"
    _reseal_shard_terminals(output, 0, rows, update_logical_hash=False)

    with pytest.raises(UgiDynamicTerminalCensusValidationError, match="logical hash"):
        validate_completed_census(contract, manifest, output, lane=lane)


def test_validator_rejects_resealed_coordinate_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, manifest, output, lane = _build_evidence(tmp_path, monkeypatch)
    terminal_path = output / "shards/shard_0000/terminals.jsonl.gz"
    rows = _read_rows(terminal_path)
    for key in (
        "population_index",
        "state_replicate",
        "checkpoint_index",
        "rollout_index",
    ):
        rows[0][key] = rows[1][key]
    _reseal_shard_terminals(output, 0, rows, update_logical_hash=True)

    with pytest.raises(UgiDynamicTerminalCensusValidationError, match="repeats"):
        validate_completed_census(contract, manifest, output, lane=lane)


def test_validator_detects_omitted_layout_or_source_nondeterminism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, manifest, output, _lane = _build_evidence(tmp_path, monkeypatch)

    with pytest.raises(UgiDynamicTerminalCensusValidationError, match="nondeterministic"):
        validate_completed_census(
            contract,
            manifest,
            output,
            lane=_NondeterministicSourceLane(),
        )


def test_validator_rejects_changed_aggregate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, manifest, output, lane = _build_evidence(tmp_path, monkeypatch)
    aggregate_terminal = output / "terminal_ledger.jsonl.gz"
    aggregate_terminal.write_bytes(aggregate_terminal.read_bytes() + b"changed")

    with pytest.raises(UgiDynamicTerminalCensusValidationError, match="artifact changed"):
        validate_completed_census(contract, manifest, output, lane=lane)


def test_production_validator_config_is_pinned_and_read_only() -> None:
    contract = load_validation_contract(REPO, CONFIG)

    assert contract.scope == EXPECTED_SCOPE
    assert contract.census_contract.design.terminal_attempts == 24_576
    assert set(contract.inputs) == {
        "census_config",
        "census_source",
        "validator_source",
        "validator_runner",
        "validator_tests",
    }


def test_validator_and_runner_have_no_scientific_or_execution_side_effects() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not any(name.startswith("forge.bio") for name in imported)
    assert not any(name.startswith("forge.route") for name in imported)
    assert "run_census_shard" not in RUNNER.read_text()
    assert "aggregate_completed_census" not in RUNNER.read_text()
