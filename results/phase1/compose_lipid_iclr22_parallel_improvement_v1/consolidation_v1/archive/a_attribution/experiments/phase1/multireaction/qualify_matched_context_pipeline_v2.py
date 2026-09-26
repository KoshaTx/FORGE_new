"""Replay the existing conditioned pool through the parameterized null-evaluation selector."""

import argparse
import json
import resource
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch

from rdkit import rdBase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_context_evaluation_v5 as fixture  # noqa: E402
import matched_context_pipeline_v2 as pipeline  # noqa: E402

ROOT = fixture.ROOT
OUT = fixture.OUT / "pipeline_qualification_v2"
RESEARCH = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/quality"


def freeze():
    fixture.ensure_source()
    OUT.mkdir(exist_ok=False)
    reference = ROOT / "results/phase1/compose_lipid_quality_confirmation_v1"
    quality = ROOT / "results/phase1/compose_lipid_quality_selection_v2"
    evaluation = ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation"
    files = {
        "producer": Path(__file__),
        "preserved_v1_failure": fixture.OUT / "pipeline_qualification_v1/failure.json",
        "pipeline": Path(pipeline.__file__),
        "fixture": fixture.OUT / "protocol.json",
        "fixture_result": fixture.OUT / "fixture_v1/result.json",
        "pool": reference / "compact_attempts.json",
        "original_candidate_diagnostics": reference / "assessment/d1_candidate_diagnostics.json",
        "original_supported_selection": reference / "assessment/d1_support_aware_selected.json",
        "expected_selection": RESEARCH / "all22_context_preserving/result.json",
        "expected_gates": RESEARCH / "all22_original_reselection/gates.json",
        "gate_policy": evaluation / "gate_policy_v3.json",
        "gates": evaluation / "gates.py",
        "selector": evaluation / "design_selector.py",
        "graph_replay": evaluation / "replay.py",
        "candidate_preparation": quality / "run.py",
        "reference_loader": reference / "select.py",
        "reference_policy": quality / "aema_domain/protocol.json",
        "support_reference": quality / "reference_support.pkl",
        "support_receipt": quality / "reference_support.json",
        "support_policy": quality / "policy.json",
        "recount": Path(__file__).with_name("verify_saved_pool_controls.py"),
    }
    fixture.write(
        OUT / "protocol.json",
        {
            "inputs": {key: fixture.pin(path) for key, path in files.items()},
            "execution_source": "immutable qualified null source; gate logical forge paths map to those actual imported bytes and require original exact SHA",
            "source_sha256": "1c77ed6e9c7b9251c9423666aadafa9967083db8ad540d369186577ec9980ae1",
            "maximum_CPU_seconds": 450,
            "maximum_wall_seconds": 900,
            "threads": 1,
            "requests": 1408,
            "candidate_entries": 11331,
            "expected": "Exact original candidate identities, saved support baseline, all gate assessments and context-preserving selection; source metadata retained separately",
            "quality_basis": "Original all-candidate gate policy; selected-only15 transported assessments excluded uniformly",
            "new_model_or_proposal_or_TEST_calls": 0,
            "quality_promotion": False,
        },
    )
    print(json.dumps({"protocol": fixture.pin(OUT / "protocol.json")}))


def run():
    started = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (450, 451))

    def deadline(*_):
        raise TimeoutError("Bounded selector qualification deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 900)
    protocol = fixture.read(OUT / "protocol.json")
    for value in protocol["inputs"].values():
        fixture.authenticate(value)
    p, payload = fixture.inputs()
    fixture.torch.set_num_threads(1)
    run_dir = OUT / "run_v1"
    run_dir.mkdir(exist_ok=False)
    rows = fixture.read(protocol["inputs"]["pool"]["path"])
    fixture.write(run_dir / "compact_attempts.json", rows)
    hashes = fixture.frozen_construction.load(
        fixture.frozen_construction.MIRROR
        / "results/phase1/compose_lipid_posttraining_v1/adjudicate.py",
        "matched_pipeline_hash_cache",
    ).VerifiedHashes()
    try:
        with patch("forge.core.hashing.sha256_file", hashes), rdBase.BlockLogs():
            pipeline.selection(
                ROOT, run_dir, payload, rows, fixture.pin, fixture.write, lambda: None
            )
        hashes.validate()
        result = fixture.read(run_dir / "selection/result.json")
        expected = fixture.read(protocol["inputs"]["expected_selection"]["path"])["by_family"]
        actual_diagnostics = fixture.read(run_dir / "selection/candidate_diagnostics.json")[
            "candidates"
        ]
        original_diagnostics = fixture.read(
            protocol["inputs"]["original_candidate_diagnostics"]["path"]
        )["candidates"]
        assert actual_diagnostics == original_diagnostics, "Candidate preparation differs"
        actual_gates = fixture.read(run_dir / "selection/gates.json")["attempts"]
        expected_gates = fixture.read(protocol["inputs"]["expected_gates"]["path"])["attempts"]
        old = {(r["index"], r["ordinal"]): r["assessment"] for r in expected_gates}
        assert {
            (r["index"], r["ordinal"]): r["assessment"] for r in actual_gates
        } == old, "Gate assessment differs"
        original_supported = {
            r["index"]: r
            for r in fixture.read(protocol["inputs"]["original_supported_selection"]["path"])[
                "attempts"
            ]
        }
        for family, value in result["by_family"].items():
            assert value["selected"] == expected[family]["selections"], family
            for c in value["support_baseline"]:
                old_support = original_supported[c["request"]]
                assert c["ordinal"] == old_support["selected_ordinal"]
                assert c["smiles"] == old_support["selected_smiles"]
        selected = [c for value in result["by_family"].values() for c in value["selected"]]
        output = {
            "passed": True,
            "complete": True,
            "protocol": fixture.pin(OUT / "protocol.json"),
            "result": fixture.pin(run_dir / "selection/result.json"),
            "requests": len(selected),
            "families": len(result["by_family"]),
            "candidate_diagnostics_identical": len(actual_diagnostics),
            "gate_assessments_identical": len(actual_gates),
            "full_selected_candidate_dictionaries_identical": True,
            "exact": sum(c["exact"] for c in selected),
            "design": sum(v["summary"]["design"] for v in result["by_family"].values()),
            "CPU_seconds": time.process_time() - started,
            "new_model_or_proposal_or_TEST_calls": 0,
            "quality_promotion": False,
        }
        fixture.write(OUT / "result.json", output)
        print(json.dumps(output), flush=True)
    except BaseException as error:
        fixture.write(
            OUT / "failure.json",
            {
                "complete": False,
                "error": repr(error),
                "CPU_seconds": time.process_time() - started,
                "protocol": fixture.pin(OUT / "protocol.json"),
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
