"""Publish qualified inference readiness without admitting a null quality result."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
OUT = (
    ROOT
    / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution/matched_context_null_v5"
)


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    files = {
        "producer": Path(__file__),
        "input_recipe": OUT / "launch_inputs.json",
        "sampling_fixture": OUT / "fixture_v1/result.json",
        "pipeline_protocol": OUT / "pipeline_qualification_v2/protocol.json",
        "pipeline_qualification": OUT / "pipeline_qualification_v2/result.json",
        "merge_protocol": OUT / "merge_qualification_v1/protocol.json",
        "merge_qualification": OUT / "merge_qualification_v1/result.json",
        "execution_tests": OUT / "execution_tests_v2.xml",
        "boundary_tests": OUT.parent / "matched_context_null_v2/boundary_tests.xml",
        "preserved_pipeline_failure": OUT / "pipeline_qualification_v1/failure.json",
    }
    files.update(
        {
            f"preserved_fixture_failure_v{i}": OUT.parent
            / f"matched_context_null_v{i}/failure.json"
            for i in range(1, 5)
        }
    )
    sampling, pipeline, merge = (
        read(files[k])
        for k in ("sampling_fixture", "pipeline_qualification", "merge_qualification")
    )
    assert sampling["passed"] and pipeline["passed"] and merge["passed"]
    assert pipeline["requests"] == merge["requests"] == 1408
    tests = {}
    for key in ("execution_tests", "boundary_tests"):
        suites = list(ET.parse(files[key]).getroot().iter("testsuite"))
        assert suites and not any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        )
        tests[key] = sum(int(s.get("tests", 0)) for s in suites)
    failures_charge = sum(
        read(files[f"preserved_fixture_failure_v{i}"])["conservative_CPU_charge_seconds"]
        for i in range(1, 5)
    )
    value = {
        "passed": True,
        "ready_for_checkpoint_binding": True,
        "full_null_evaluation_complete": False,
        "inputs": {k: pin(v) for k, v in files.items()},
        "sampling": {
            "fixture_batches": len(sampling["evidence"]),
            "requests": sum(8 for _ in sampling["evidence"]),
            "all14_conditioned_heads_bit_exact": True,
            "all_fixture_constructor_JSON_bytes_exact": True,
            "null_zero9_context_initial_noise_and_repeat_determinism": True,
            "CPU_seconds": sampling["CPU_seconds"],
            "fixture_weights_only22_updates_no_quality_claim": True,
        },
        "pipeline": {
            k: pipeline[k]
            for k in (
                "requests",
                "families",
                "candidate_diagnostics_identical",
                "gate_assessments_identical",
                "full_selected_candidate_dictionaries_identical",
                "exact",
                "design",
                "CPU_seconds",
            )
        },
        "merge": {
            k: merge[k]
            for k in (
                "requests",
                "constructor_shards",
                "generation_shards",
                "merged_complete_dictionaries_identical",
                "all_full_and_compact_hashes_verified",
                "CPU_seconds",
            )
        },
        "tests": tests,
        "preserved_failure_costs": {
            "fixture_conservative_CPU_charge": failures_charge,
            "selector_failure_measured_CPU_seconds": read(files["preserved_pipeline_failure"])[
                "CPU_seconds"
            ],
            "not_end_to_end_total": "Test/import/preparation CPU is not included in this component ledger; all work remains bounded by the authorized preflight budget.",
        },
        "remaining_requirements": [
            "Authenticated successful full2794 null checkpoint with original seed2026092401, config, exact22-family exposure and root training admission.",
            "Bind checkpoint/config/admission to a fresh1408-request output directory.",
            "Root inference review binds exact request/recipe and sampling5400CPU plus pipeline1800CPU caps before execution.",
            "Independent completed-output recount and chemistry/tree-basis adjudication before any quality or manuscript promotion.",
        ],
        "metric_scope": "Same externally supplied layout, masks and initial noise; joint null training mask/prior/conditioning contrast, not isolated embedding or model-versus-rules attribution.",
        "quality_scope": "Matched original all-candidate gate rubric reproduces1387 exact/1312 design. Later15 identity-bound transported-tree assessments and1324 current quality total are separate evidence, never inherited by new null candidates.",
        "heldout_or_independent_seed_evidence": False,
        "new_paid_calls": 0,
        "manuscript_or_production_changes": False,
    }
    with (OUT / "readiness.json").open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"readiness": pin(OUT / "readiness.json")}))


if __name__ == "__main__":
    main()
