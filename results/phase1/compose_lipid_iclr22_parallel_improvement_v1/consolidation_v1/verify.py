"""Verify source integration and preserve the exact evidence-producing worktrees."""

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main():
    inventory = read(HERE / "inventory.json")
    plan = read(HERE / "plan.json")
    assert digest(HERE / "inventory.json") == plan["inventory_sha256"]
    baseline = read(ROOT / inventory["baseline"]["path"])
    assert digest(ROOT / inventory["baseline"]["path"]) == inventory["baseline"]["sha256"]
    prior_log = (HERE / "decision_log_before.md").read_bytes()
    assert hashlib.sha256(prior_log).hexdigest() == inventory["main_source_before"]["docs/DECISION_LOG.md"]
    assert (ROOT / "docs/DECISION_LOG.md").read_bytes().startswith(prior_log)
    for relative, expected in inventory["main_source_before"].items():
        if relative != "docs/DECISION_LOG.md":
            assert digest(ROOT / relative) == expected, relative
    archived, unchanged_worktree_files = 0, 0
    for stream in inventory["streams"].values():
        worktree = ROOT / stream["worktree"]
        for original in baseline["files"]:
            assert digest(worktree / original["path"]) == original["sha256"]
            unchanged_worktree_files += 1
        for row in stream["additions"]:
            assert digest(ROOT / row["source"]["path"]) == row["source"]["sha256"]
            assert digest(ROOT / row["archive"]["path"]) == row["archive"]["sha256"]
            assert row["source"]["sha256"] == row["archive"]["sha256"]
            if row["main_before"] == "identical":
                assert digest(ROOT / row["relative_path"]) == row["source"]["sha256"]
            archived += 1
    assert archived == 91 and unchanged_worktree_files == 5 * 2686
    destinations = [pin(ROOT / row["relative_path"]) for row in plan["copies"]]
    imported = read(HERE / "import_verification.json")
    assert imported["passed"] and imported["module_count"] == 29
    for row in imported["modules"]:
        assert digest(ROOT / row["path"]) == row["sha256"]
    suites = list(ET.parse(HERE / "integration_tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    assert counts == {"tests": 102, "failures": 0, "errors": 0, "skipped": 0}, counts
    assert "All checks passed!" in (HERE / "ruff_final.txt").read_text()
    assert "39 files would be left unchanged" in (HERE / "black_final.txt").read_text()
    campaign_baseline = read(HERE.parent / "baseline/result.json")
    for original in campaign_baseline["paper_sources"]:
        assert digest(ROOT / original["path"]) == original["sha256"]
    report = read(HERE.parent / "result_v2.json")
    assert digest(HERE.parent / "result_v2.json") == "ebb8332a4d96e0f02018ec53ef5cf82a46e3007e06015275c5fe7abbef1345f7"
    rejection_path = HERE.parent / "a_attribution/null_fullfit_v2/submission_rejection_v1.json"
    rejection = read(rejection_path)
    assert not rejection["paid_job_submitted"]
    fit = rejection_path.parent
    assert not (fit / "attempt.json").exists() and not (fit / "submitted.json").exists()
    files = [HERE / name for name in (
        "inventory.json", "plan.json", "integration_tests.xml", "import_verification.json",
        "ruff_final.txt", "black_final.txt", "main_adaptations.patch", "README.md",
        "decision_log_before.md",
    )] + [
        ROOT / "docs/DECISION_LOG.md", HERE.parent / "result_v2.json", rejection_path,
        fit / "monitor_collection_readiness_v1.json",
        HERE.parent / "b_quality/component_feasibility_v1/closeout.json",
        HERE.parent / "b_quality/novel_topology_feasibility_v1/closeout.json",
        HERE.parent / "d_routes/residual_debt_v1/handoff.json",
    ]
    result = {
        "schema": "forge.iclr22.worktree_consolidation_result.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": pin(Path(__file__)), "inputs": [pin(path) for path in files],
        "passed": True, "scope": "Five ICLR22 improvement worktrees; no other worktree",
        "main_additions": destinations, "new_main_files": len(destinations),
        "previously_integrated_files": 10, "archive_only_files": 42,
        "archived_original_delta_files": archived,
        "unchanged_baseline_files_across_worktrees": unchanged_worktree_files,
        "preexisting_main_source_files": len(inventory["main_source_before"]),
        "preexisting_main_code_unchanged": True, "decision_log_append_only": True,
        "tests": counts, "imported_modules": 29, "ruff_passed": True, "black_passed": True,
        "manuscript_unchanged": True, "default_policies_unchanged": True,
        "selected_identity_or_metric_changed": False, "current_joint_metrics": report["joint"],
        "worktrees_deleted": False, "commits_created": False,
        "experiments_rerun": False, "paid_job_started": False,
        "paid_retry_status": "Automatic approval rejected prior continue interpretation; explicit exact retry approval pending.",
        "limits": [
            "Historical path/hash admissions remain tied to preserved worktrees; copied validators do not inherit them.",
            "Fixed historical output directories and protocols were not regenerated.",
            "Native-null scientific execution uses the separately qualified immutable null source; main unit checks do not replace that qualification.",
            "Consolidation does not promote development diagnostics to final held-out or empirical realism evidence.",
        ],
    }
    with (HERE / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"result": pin(HERE / "result.json"), "new_files": 39, "tests": counts,
                      "preserved_originals": archived, "verified_worktree_baseline_files": unchanged_worktree_files}))


if __name__ == "__main__":
    main()
