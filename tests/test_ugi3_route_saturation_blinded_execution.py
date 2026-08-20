from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from rdkit import rdBase

from experiments.archive.phase1.synthesis_audits.ugi3_route_saturation_blinded_execution import (
    CONFIG_SCHEMA_VERSION,
    EXECUTABLE_STATUS,
    REQUIRED_DECISION,
    REQUIRED_POLICY,
    RUNTIME_DEPENDENCY_MANIFEST_SCHEMA_VERSION,
    RUNTIME_DEPENDENCY_MODULES,
    Ugi3BlindedHoldoutExecutionError,
    _rename_directory_noreplace,
    clopper_pearson_interval,
    evaluate_registry_pair,
    prepare_execution,
    run_one_shot_blinded_holdout,
    validate_completed_blinded_holdout,
)
from forge.corpus.r1_prime_audit import sha256_file
from forge.synthesis.assessment.ugi3_route_registry_pair_contract import (
    TARGET_ROLE,
    TARGET_SMILES,
    component_key_sha256,
)

REPO = Path(__file__).resolve().parents[1]
MODULE = (
    REPO / "experiments/archive/phase1/synthesis_audits/ugi3_route_saturation_blinded_execution.py"
)
CLI = REPO / "experiments/archive/producers/phase1_run_ugi3_route_saturation_blinded_holdout.py"
HEADLESS = REPO / "experiments/phase1/product_l1/sampling/ugi_blinded_headless_sampling.py"
JOINT_FLOW = REPO / "forge/model/ugi_joint_sparse_flow.py"
CONTRACT = REPO / "forge/synthesis/assessment/ugi3_route_registry_pair_contract.py"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _spec(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _record(role: str, smiles: str, complete: bool) -> dict[str, Any]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "component_key_sha256": component_key_sha256(role, smiles),
        "route_complete": complete,
        "source_class": "synthetic_exact_test",
        "value": {"evidence_support": "exact_identity" if complete else "not_applicable"},
    }


def _eligible_row(aldehyde: str) -> dict[str, Any]:
    return {
        "valid": True,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {"exact_product_reconstructed": True},
        "smiles": "PRIVATE_PRODUCT_IDENTITY",
        "component_smiles_by_role": {
            "amine_head": "HEAD",
            "oxoester_aldehyde_body_tail": aldehyde,
            "isocyanide_tail": "TAIL",
        },
    }


def _registries() -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
    shared = {
        ("amine_head", "HEAD"): _record("amine_head", "HEAD", True),
        ("isocyanide_tail", "TAIL"): _record("isocyanide_tail", "TAIL", True),
        ("oxoester_aldehyde_body_tail", "OTHER"): _record(
            "oxoester_aldehyde_body_tail", "OTHER", True
        ),
    }
    r0 = {
        **shared,
        (TARGET_ROLE, TARGET_SMILES): _record(TARGET_ROLE, TARGET_SMILES, False),
    }
    r1 = {
        **shared,
        (TARGET_ROLE, TARGET_SMILES): _record(TARGET_ROLE, TARGET_SMILES, True),
    }
    return r0, r1


def _fixture(tmp_path: Path, *, rows: int = 3) -> tuple[Path, Path, Path, Path]:
    joint = tmp_path / "joint.pt"
    closure = tmp_path / "closure.pt"
    prepared_cache = tmp_path / "ugi_training_cache.pt"
    dependency_manifest = tmp_path / "headless_dependencies.json"
    reactions = tmp_path / "reactions.json"
    seal = tmp_path / "holdout_seal.json"
    programs = tmp_path / "programs.json"
    for path, payload in (
        (joint, b"joint"),
        (closure, b"closure"),
        (prepared_cache, b"prepared-cache"),
        (reactions, b"{}\n"),
        (seal, b"{}\n"),
        (programs, b'{"samples":[]}\n'),
    ):
        path.write_bytes(payload)
    checkpoint_specs = {
        "joint_checkpoint": _spec(joint),
        "closure_checkpoint": _spec(closure),
        "qualified_reactions": _spec(reactions),
    }
    _write_json(
        dependency_manifest,
        {
            "schema_version": RUNTIME_DEPENDENCY_MANIFEST_SCHEMA_VERSION,
            "status": "frozen_renderer_free_runtime_dependency_closure",
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "pyyaml": yaml.__version__,
            },
            "modules": {
                module_name: _spec(Path(importlib.import_module(module_name).__file__))
                for module_name in RUNTIME_DEPENDENCY_MODULES
            },
        },
    )
    final = tmp_path / "sample"
    contract = tmp_path / "holdout_contract.json"
    _write_json(
        contract,
        {
            "schema_version": "phase1_ugi3_route_saturation_holdout_config.v1",
            "inputs": checkpoint_specs,
            "holdout_sampling": {"sealed_sample_output": str(final)},
        },
    )
    protocol = tmp_path / "protocol.json"
    anchor = {
        "contract": _spec(contract),
        "seal": _spec(seal),
        "program_draw": {**_spec(programs), "rows": rows},
        "seeds": {"program": 101, "flow": 102, "terminal": 103},
        "sampling": {
            "sample_steps": 8,
            "batch_size": 2,
            "maximum_adjacent_branch_runs": [2, 1, 1],
            "terminal_decoder_mode": "bond_stochastic",
            "terminal_temperature": 1.0,
        },
        "sealed_sample_output": str(final),
    }
    _write_json(protocol, {"sealed_holdout_anchor": anchor})
    binding = tmp_path / "binding.json"
    _write_json(binding, {"protocol": _spec(protocol)})
    config = tmp_path / "execution.json"
    sampling = {
        "program_rows": rows,
        "program_seed": 101,
        "flow_seed": 102,
        "closure_seed": 103,
        "terminal_seed": 103,
        "sample_steps": 8,
        "batch_size": 2,
        "maximum_adjacent_branch_runs": [2, 1, 1],
        "terminal_decoder_mode": "bond_stochastic",
        "terminal_temperature": 1.0,
        "device": "cpu",
        "evaluate_exact_l1_terminal_admission": True,
        "render_molecules": False,
        "biological_guidance_enabled": False,
    }
    _write_json(
        config,
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "status": EXECUTABLE_STATUS,
            "experiment_id": "synthetic_blinded_test",
            "inputs": {
                "protocol": _spec(protocol),
                "binding": _spec(binding),
                "holdout_contract": _spec(contract),
                "holdout_seal": _spec(seal),
                "program_draw": _spec(programs),
                **checkpoint_specs,
                "prepared_training_cache": _spec(prepared_cache),
            },
            "implementation": {
                "execution_module": _spec(MODULE),
                "execution_cli": _spec(CLI),
                "headless_sampling_module": _spec(HEADLESS),
                "headless_dependency_manifest": _spec(dependency_manifest),
                "joint_flow_module": _spec(JOINT_FLOW),
                "registry_pair_contract_module": _spec(CONTRACT),
            },
            "sampling": sampling,
            "outputs": {
                "final_directory": str(final),
                "attempt_directory": str(tmp_path / ".sample.one_shot_attempt"),
                "one_shot_claim": str(tmp_path / "sample.one_shot.claim.json"),
            },
            "policy": REQUIRED_POLICY,
            "decision": REQUIRED_DECISION,
        },
    )
    return config, binding, final, tmp_path / "sample.one_shot.claim.json"


def _binding_result(binding: Path) -> dict[str, Any]:
    return {
        "registry_pair_bound": True,
        "holdout_reveal_authorized": True,
        "readiness": "immutable_registry_pair_valid_holdout_reveal_authorized",
        "authorized_delta_count": 1,
        "binding_sha256": sha256_file(binding),
    }


def _preflight(prepared: Any) -> Any:
    return SimpleNamespace(
        preflight_summary={
            "status": "runtime_dependencies_preflighted_without_sampling",
            "joint_checkpoint_schema": "phase1_ugi_joint_sparse_checkpoint.v1",
            "closure_checkpoint_schema": "phase1_ugi_sparse_closure_checkpoint.v2",
            "prepared_cache_sha256": prepared.input_hashes["prepared_training_cache"],
            "prepared_cache_schema": "phase1_ugi_training_cache.v1",
            "atom_vocabulary_size": 8,
            "program_rows": prepared.program_rows,
            "qualified_reaction_loaded": True,
            "models_constructed_and_state_loaded": True,
            "molecular_sampling_performed": False,
            "rdkit_molecular_renderer_imported": False,
        }
    )


def _sample_payload(prepared: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase1_ugi3_route_saturation_private_sample.v1",
        "status": "sampled_once_headless",
        "visibility": "private_identity_bearing_hash_pinned",
        "render": None,
        "rendered_molecules": 0,
        "rendering_disabled": True,
        "samples": rows,
        "statistics": {},
        "sampling": {},
        "pinned_inputs": {
            name: {
                "path": str(prepared.input_paths[name]),
                "sha256": prepared.input_hashes[name],
            }
            for name in (
                "program_draw",
                "joint_checkpoint",
                "closure_checkpoint",
                "prepared_training_cache",
                "qualified_reactions",
            )
        },
    }


def _sealed_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    tuple[dict[tuple[str, str], dict[str, Any]], ...],
]:
    config_path, binding, final, _ = _fixture(tmp_path)
    rows = [_eligible_row(TARGET_SMILES), _eligible_row("OTHER")]
    rows.append(
        {
            "valid": False,
            "component_reconstruction_valid": False,
            "l1_forward_verification": {"exact_product_reconstructed": False},
        }
    )
    registries_pair = _registries()

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def sampler(prepared: Any, *_: Any) -> dict[str, Any]:
        return _sample_payload(prepared, rows)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    run_one_shot_blinded_holdout(
        REPO,
        config_path,
        _binding_validator=validator,
        _preflight_fn=_preflight,
        _sample_fn=sampler,
        _registry_loader=registries,
    )
    return config_path, binding, final, registries_pair


def _reseal_public(final: Path, public: dict[str, Any]) -> None:
    public_path = final / "public_aggregate.json"
    _write_json(public_path, public)
    public_hash = sha256_file(public_path)
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["public_aggregate.json"]["sha256"] = public_hash
    _write_json(manifest_path, manifest)
    seal_path = final / "seal.json"
    seal = json.loads(seal_path.read_text())
    seal["manifest"]["sha256"] = sha256_file(manifest_path)
    seal["public_aggregate"]["sha256"] = public_hash
    _write_json(seal_path, seal)


def test_clopper_pearson_interval_matches_exact_edge_cases() -> None:
    lower_zero, upper_zero = clopper_pearson_interval(0, 10)
    lower_all, upper_all = clopper_pearson_interval(10, 10)

    assert lower_zero == 0.0
    assert upper_zero == pytest.approx(0.3084971078, abs=1e-9)
    assert lower_all == pytest.approx(0.6915028922, abs=1e-9)
    assert upper_all == 1.0
    assert clopper_pearson_interval(0, 0) == (0.0, 1.0)


def test_paired_evaluator_uses_one_exact_l1_denominator_and_hashed_ledgers() -> None:
    rows = [
        _eligible_row(TARGET_SMILES),
        _eligible_row("OTHER"),
        {
            "valid": False,
            "component_reconstruction_valid": False,
            "l1_forward_verification": {"exact_product_reconstructed": False},
            "smiles": "PRIVATE_INVALID_IDENTITY",
        },
    ]
    r0, r1 = _registries()
    evaluation = evaluate_registry_pair(
        rows,
        r0,
        r1,
        experiment_id="synthetic",
        sampling={
            "program_seed": 1,
            "flow_seed": 2,
            "closure_seed": 3,
            "terminal_seed": 3,
        },
        sealed_program_rows=3,
    )

    public = evaluation.public_aggregate
    assert public["sampled_products"] == 3
    assert public["exact_l1_eligible_products"] == 2
    assert public["eligible_products_complete_in_r0"] == 1
    assert public["eligible_products_complete_in_r1"] == 2
    assert public["marginal_completion_count"] == 1
    assert public["marginal_completion_fraction"] == 0.5
    assert public["target_component_occurrences"] == 1
    assert public["target_component_occurrences_that_flip_product_completion"] == 1
    assert len(evaluation.component_rows) == 6
    assert all("smiles" not in json.dumps(row).lower() for row in evaluation.eligibility_rows)
    assert all("smiles" not in json.dumps(row).lower() for row in evaluation.component_rows)
    assert all("smiles" not in json.dumps(row).lower() for row in evaluation.product_rows)
    assert evaluation.product_rows[-1]["transition"] == "ineligible_exact_l1"


def test_prepare_rejects_rendering_or_input_hash_changes_before_claim(tmp_path: Path) -> None:
    config_path, _, _, claim = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["sampling"]["render_molecules"] = True
    _write_json(config_path, config)

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="sampling settings"):
        prepare_execution(REPO, config_path)
    assert not claim.exists()


def test_runner_requires_validator_authorization_before_irreversible_claim(
    tmp_path: Path,
) -> None:
    config_path, _, _, claim = _fixture(tmp_path)

    def reject(*_: Any) -> dict[str, Any]:
        return {
            "registry_pair_bound": True,
            "holdout_reveal_authorized": False,
            "readiness": "not_authorized",
            "authorized_delta_count": 1,
            "binding_sha256": "0" * 64,
        }

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="did not authorize"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=reject,
            _preflight_fn=_preflight,
        )
    assert not claim.exists()


def test_renderer_free_runner_import_does_not_load_rdkit_draw() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "import experiments.archive.phase1.synthesis_audits.ugi3_route_saturation_blinded_execution as execution; "
                "assert 'rdkit.Chem.Draw' not in sys.modules; "
                "assert 'rdkit.Chem.Draw.rdMolDraw2D' not in sys.modules; "
                f"root=Path({str(REPO)!r}).resolve(); "
                "loaded={name for name,module in sys.modules.items() "
                "if name.startswith(('forge.','experiments.')) "
                "and getattr(module,'__file__',None) "
                "and Path(module.__file__).resolve().is_relative_to(root)}; "
                "assert loaded == set(execution.RUNTIME_DEPENDENCY_MODULES), "
                "(sorted(loaded-set(execution.RUNTIME_DEPENDENCY_MODULES)), "
                "sorted(set(execution.RUNTIME_DEPENDENCY_MODULES)-loaded))"
            ),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preflight_failure_occurs_before_claim_and_sampler(tmp_path: Path) -> None:
    config_path, binding, _, claim = _fixture(tmp_path)
    r0, r1 = _registries()
    sampler_called = False

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    def preflight(*_: Any) -> Any:
        raise RuntimeError("synthetic preflight failure")

    def sampler(*_: Any) -> dict[str, Any]:
        nonlocal sampler_called
        sampler_called = True
        return {}

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="preflight failed before claim"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _registry_loader=registries,
            _preflight_fn=preflight,
            _sample_fn=sampler,
        )
    assert not claim.exists()
    assert sampler_called is False


def test_malformed_registry_is_rejected_before_preflight_or_claim(tmp_path: Path) -> None:
    config_path, binding, final, claim = _fixture(tmp_path)
    attempt = tmp_path / ".sample.one_shot_attempt"
    r0, r1 = _registries()
    r0[("amine_head", "HEAD")]["route_complete"] = "yes"
    preflight_called = False

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    def preflight(prepared: Any) -> Any:
        nonlocal preflight_called
        preflight_called = True
        return _preflight(prepared)

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="record shape"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _registry_loader=registries,
            _preflight_fn=preflight,
        )
    assert preflight_called is False
    assert not os.path.lexists(claim)
    assert not os.path.lexists(attempt)
    assert not os.path.lexists(final)


def test_preflight_to_claim_rehash_detects_input_mutation(tmp_path: Path) -> None:
    config_path, binding, _, claim = _fixture(tmp_path)
    r0, r1 = _registries()

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    def mutate(prepared: Any) -> Any:
        prepared.input_paths["qualified_reactions"].write_text('{"changed":true}\n')
        return _preflight(prepared)

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="changed during preflight"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _registry_loader=registries,
            _preflight_fn=mutate,
        )
    assert not claim.exists()


def test_dangling_final_symlink_blocks_preflight_and_claim(tmp_path: Path) -> None:
    config_path, binding, final, claim = _fixture(tmp_path)
    final.symlink_to(tmp_path / "missing-output")
    r0, r1 = _registries()
    preflight_called = False

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    def preflight(prepared: Any) -> Any:
        nonlocal preflight_called
        preflight_called = True
        return _preflight(prepared)

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="already exists"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _registry_loader=registries,
            _preflight_fn=preflight,
        )
    assert preflight_called is False
    assert not claim.exists()


def test_atomic_publication_never_replaces_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "attempt"
    destination = tmp_path / "sample"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source\n")
    (destination / "owner.txt").write_text("owner\n")

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="appeared|no-replace"):
        _rename_directory_noreplace(source, destination)
    assert (source / "source.txt").read_text() == "source\n"
    assert (destination / "owner.txt").read_text() == "owner\n"


def test_runner_seals_once_without_rendering_and_refuses_second_attempt(
    tmp_path: Path,
) -> None:
    config_path, binding, final, claim = _fixture(tmp_path)
    rows = [_eligible_row(TARGET_SMILES), _eligible_row("OTHER")]
    rows.append(
        {
            "valid": False,
            "component_reconstruction_valid": False,
            "l1_forward_verification": {"exact_product_reconstructed": False},
        }
    )
    r0, r1 = _registries()

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def sampler(prepared: Any, *_: Any) -> dict[str, Any]:
        return _sample_payload(prepared, rows)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    result = run_one_shot_blinded_holdout(
        REPO,
        config_path,
        _binding_validator=validator,
        _preflight_fn=_preflight,
        _sample_fn=sampler,
        _registry_loader=registries,
    )

    assert result["public_aggregate"]["marginal_completion_count"] == 1
    assert final.is_dir()
    assert claim.is_file()
    assert not list(final.glob("*.png"))
    assert (final / "private_sample.json.gz").is_file()
    assert (final / "private_exact_l1_eligibility.json.gz").is_file()
    assert (final / "private_component_transitions.json.gz").is_file()
    assert (final / "private_product_transitions.json.gz").is_file()
    validated = validate_completed_blinded_holdout(
        REPO,
        final,
        config_path=config_path,
        _registry_loader=registries,
    )
    assert validated["status"] == "completed_blinded_holdout_valid"

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="already exists|already claimed"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _preflight_fn=_preflight,
            _sample_fn=sampler,
            _registry_loader=registries,
        )


def test_failed_attempt_is_left_claimed_and_cannot_be_retried(tmp_path: Path) -> None:
    config_path, binding, _, claim = _fixture(tmp_path)
    attempt = tmp_path / ".sample.one_shot_attempt"
    r0, r1 = _registries()

    def validator(*_: Any) -> dict[str, Any]:
        return _binding_result(binding)

    def fail(*_: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic sampler failure")

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return r0, r1

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="retry is prohibited"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _preflight_fn=_preflight,
            _sample_fn=fail,
            _registry_loader=registries,
        )
    assert claim.is_file()
    assert (attempt / "failure.json").is_file()

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="prior one-shot attempt"):
        run_one_shot_blinded_holdout(
            REPO,
            config_path,
            _binding_validator=validator,
            _preflight_fn=_preflight,
            _sample_fn=fail,
            _registry_loader=registries,
        )


def test_completed_validator_rejects_extra_files(tmp_path: Path) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    (final / "unexpected.txt").write_text("unexpected\n")

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="file set changed"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )


def test_completed_validator_rejects_missing_required_file(tmp_path: Path) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    (final / "private_sample.json.gz").unlink()

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="file set changed"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )


def test_completed_validator_rejects_symlinked_artifact(tmp_path: Path) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    manifest = final / "manifest.json"
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(external_manifest)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="symbolic link"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )


def test_completed_validator_rejects_nested_identity_key_even_if_resealed(
    tmp_path: Path,
) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    public_path = final / "public_aggregate.json"
    public = json.loads(public_path.read_text())
    r0_families = public["aggregate_counts_by_evidence_family"]["r0"]
    original_name = next(iter(r0_families))
    r0_families["private_product_identity"] = r0_families.pop(original_name)
    _reseal_public(final, public)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="unsafe free-form key"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )


def test_completed_validator_recomputes_exact_confidence_interval(tmp_path: Path) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    public_path = final / "public_aggregate.json"
    public = json.loads(public_path.read_text())
    public["confidence_interval_95pct"] = [0.0, 1.0]
    _reseal_public(final, public)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="recomputed exactly"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )


def test_completed_validator_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    config_path, _, final, registries_pair = _sealed_fixture(tmp_path)
    seal_path = final / "seal.json"
    seal = json.loads(seal_path.read_text())
    seal["manifest"]["path"] = "../manifest.json"
    _write_json(seal_path, seal)

    def registries(*_: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], ...]:
        return registries_pair

    with pytest.raises(Ugi3BlindedHoldoutExecutionError, match="manifest reference"):
        validate_completed_blinded_holdout(
            REPO,
            final,
            config_path=config_path,
            _registry_loader=registries,
        )
