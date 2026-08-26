from __future__ import annotations

from pathlib import Path

import pytest

from experiments.phase1.multireaction.external_ugi_contract import (
    ExternalBaselineError,
    import_external_attempts,
    load_external_baseline_manifest,
    supersede_genmol_empty_labels,
)
from experiments.phase1.multireaction.native_baseline_ports import NativeBaselinePortError
from experiments.phase1.multireaction.native_baseline_runtime import _genmol_sample_row
from forge.core.hashing import sha256_file
from forge.core.io import write_csv, write_json
from forge.model.common_ugi_benchmark import load_attempt_ledger

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/baselines/external_ugi_v1.json"


def _native_files(tmp_path, method: dict, *, seed: int = 11):
    samples = tmp_path / "samples.csv"
    write_csv(
        samples,
        [
            {"attempt_index": 0, "status": "generated", "product_smiles": "CC"},
            {"attempt_index": 1, "status": "failed", "product_smiles": ""},
        ],
        ["attempt_index", "status", "product_smiles"],
    )
    receipt = tmp_path / "receipt.json"
    write_json(
        receipt,
        {
            "schema_version": "forge.external_ugi_native_run_receipt.v1",
            "method_id": method["method_id"],
            "upstream_commit": method["commit"],
            "seed": seed,
            "requested_attempts": 2,
            "repairs_or_retries": False,
            "generator_calls": 2,
            "reaction_calls": 0,
            "route_calls": 0,
            "oracle_calls": 0,
            "wall_seconds": 1.0,
            "samples_sha256": str(sha256_file(samples)),
        },
    )
    return receipt, samples


def test_external_importer_preserves_failed_attempts_and_call_totals(tmp_path) -> None:
    methods = load_external_baseline_manifest(MANIFEST)
    method = methods["defog_unconditional"]
    receipt, samples = _native_files(tmp_path, method)
    attempts = import_external_attempts(
        method, receipt, samples, expected_seed=11, expected_attempts=2
    )
    assert [attempt.status for attempt in attempts] == ["generated", "failed"]
    assert sum(attempt.generator_calls for attempt in attempts) == 2


def test_finite_external_method_must_disclose_visible_components(tmp_path) -> None:
    methods = load_external_baseline_manifest(MANIFEST)
    method = methods["rgfn"]
    receipt, samples = _native_files(tmp_path, method)
    with pytest.raises(ExternalBaselineError, match="disclose"):
        import_external_attempts(method, receipt, samples, expected_seed=11, expected_attempts=2)


def test_genmol_runtime_classifies_empty_decode_as_invalid() -> None:
    assert _genmol_sample_row(3, "") == {
        "attempt_index": 3,
        "status": "invalid",
        "product_smiles": "",
    }
    assert _genmol_sample_row(4, None)["status"] == "invalid"
    assert _genmol_sample_row(5, "CC")["status"] == "generated"


def test_genmol_empty_label_supersession_is_explicit_and_payload_preserving(tmp_path) -> None:
    method = load_external_baseline_manifest(MANIFEST)["genmol_safe"]
    native = tmp_path / "native"
    native.mkdir()
    samples = native / "samples.csv"
    write_csv(
        samples,
        [
            {"attempt_index": 0, "status": "generated", "product_smiles": "CC"},
            {"attempt_index": 1, "status": "generated", "product_smiles": ""},
            {"attempt_index": 2, "status": "invalid", "product_smiles": ""},
        ],
        ["attempt_index", "status", "product_smiles"],
    )
    receipt = native / "receipt.json"
    write_json(
        receipt,
        {
            "schema_version": "forge.external_ugi_native_run_receipt.v1",
            "method_id": "genmol_safe",
            "upstream_commit": method["commit"],
            "seed": 11,
            "requested_attempts": 3,
            "repairs_or_retries": False,
            "generator_calls": 3,
            "reaction_calls": 0,
            "route_calls": 0,
            "oracle_calls": 0,
            "wall_seconds": 1.5,
            "samples_sha256": str(sha256_file(samples)),
        },
    )
    with pytest.raises(ValueError, match="generated attempt must contain product_smiles"):
        import_external_attempts(method, receipt, samples, expected_seed=11, expected_attempts=3)

    result = supersede_genmol_empty_labels(
        method,
        receipt,
        samples,
        tmp_path / "corrected",
        expected_seed=11,
        expected_attempts=3,
    )
    attempts = load_attempt_ledger(
        tmp_path / "corrected/attempts.jsonl.gz",
        expected_method="genmol_safe",
        expected_seed=11,
        expected_attempts=3,
    )
    assert [attempt.status for attempt in attempts] == ["generated", "invalid", "invalid"]
    assert [attempt.product_smiles for attempt in attempts] == ["CC", None, None]
    assert result["corrected_attempts"] == 1
    assert result["gates"]["molecular_payloads_preserved"] is True
    assert result["gates"]["generation_not_rerun"] is True
    assert result["gates"]["model_not_retrained"] is True
    assert result["source"]["samples"]["sha256"] == str(sha256_file(samples))


def test_manifest_separates_runnable_ports_from_scientific_exclusions() -> None:
    methods = load_external_baseline_manifest(MANIFEST)
    assert {
        method_id
        for method_id, method in methods.items()
        if method["integration_status"] == "native_port_ready"
    } == {"rgfn", "defog_unconditional", "genmol_safe"}
    assert methods["ou_dag_chem"]["integration_status"] == "excluded_no_author_implementation"
    assert methods["synflownet"]["integration_status"] == "excluded_incompatible_reaction_arity"
    assert methods["syncogen"]["integration_status"] == "excluded_unlicensed"


def test_excluded_method_cannot_be_prepared_as_if_it_were_runnable(tmp_path) -> None:
    from experiments.phase1.multireaction.native_baseline_ports import (
        prepare_native_baseline_run,
    )

    with pytest.raises(NativeBaselinePortError, match="not admitted"):
        prepare_native_baseline_run(
            MANIFEST,
            tmp_path / "missing-export",
            tmp_path / "missing-checkout",
            tmp_path / "out",
            method_id="synflownet",
            seed=11,
            attempts=2,
            profile="smoke",
        )


def test_modal_native_image_mounts_complete_experiment_package_and_binds_source() -> None:
    source = (REPO / "experiments/phase1/multireaction/modal_external_ugi_app.py").read_text()
    assert 'LOCAL_REPO / "experiments" / "catalog.py"' in source
    assert 'LOCAL_REPO / "experiments" / "_runtime"' in source
    assert 'LOCAL_REPO / "experiments" / "installation_smoke"' in source
    assert '"source_sha256": source_sha256' in source
    assert "observed_source_sha256 != expected_source_sha256" in source
    assert 'uv_pip_install("setuptools==80.9.0")' in source
    assert "function.spawn(job_key, source_sha256)" in source
    assert "modal.FunctionCall.from_id" in source
    assert '"schema_version": "forge.external_ugi_modal_call.v1"' in source
    assert "resume it instead of " in source
    assert "launching a duplicate" in source


def test_reviewed_native_compatibility_edits_are_fail_closed() -> None:
    source = (REPO / "experiments/phase1/multireaction/native_baseline_runtime.py").read_text()
    assert '_replace_once(main_path, "import graph_tool\\n", "")' in source
    assert '"persistent_workers=True)"' in source
    assert "expected=2" in source
    assert 'defog_env["PYTHONPATH"]' in source
    assert 'defog_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"' in source
    assert '"dataset=guacamol"' in source
    assert "self.processed_paths[self.file_idx], weights_only=False" in source
    assert 'dataset_infos.ref_metrics = {"val": {}, "test": {}}' in source
    assert 'max_steps=cfg.train.get("max_steps", -1)' in source
    assert 'limit_train_batches=cfg.train.get("limit_train_batches", 1.0)' in source
    assert 'limit_val_batches=cfg.train.get("limit_val_batches", 1.0)' in source
    assert 'limit_test_batches=cfg.train.get("limit_test_batches", 1.0)' in source
    assert 'if "training_batches" in parameters' in source
    assert "bs = self.cfg.general.sampling_batch_size" in source
    assert "to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)" in source
    assert "class NoOpLogger(LoggerBase)" in source
    assert "Trainer.logger = @forge_native.NoOpLogger()" in source
    assert "device=self.device" in source
    assert 'output_dir / "checkpoint.pt"' in source
    assert source.count('output_dir / "checkpoint.ckpt"') == 2
    assert 'for scratch_name in ("work", "upstream")' in source
