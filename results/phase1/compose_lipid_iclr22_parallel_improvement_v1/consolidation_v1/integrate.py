"""Copy reviewed dependency closures, preserving immutable worktree evidence."""

import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
MULTI = "experiments/phase1/multireaction/"
ROUTE = "experiments/phase1/route_improvement/"


def paths(prefix, names):
    return [prefix + name + ".py" for name in names.split()]


FILES = {
    "a_attribution": paths(MULTI, """
        saved_pool_selector saved_pool_attribution saved_pool_attribution_v3
        augment_saved_pool_gates verify_saved_pool_controls report_saved_pool_attribution_v3
        matched_context_null matched_context_construction_v2 matched_context_evaluation_v5
        matched_context_pipeline_v2 qualify_matched_context_pipeline_v2
        qualify_matched_context_merge run_matched_context_null close_matched_context_readiness
    """) + paths("tests/", """
        test_saved_pool_attribution test_saved_pool_gate_join
        test_matched_context_null test_matched_context_execution
    """),
    "b_quality": paths(MULTI, """
        quality_lineage_diagnostic quality_closure_diagnostic verify_quality_closure
        handoff_quality_closure component_feasibility_census_v2 novel_topology_feasibility_v2
    """) + paths("tests/", "test_component_feasibility_census_v2 test_novel_topology_feasibility_v2"),
    "c_generalization": [],
    "d_routes": paths(ROUTE, """
        audit_new_leaves candidate_routes new_clock_routes residual_debt
        test_candidate_routes test_residual_debt
    """) + paths("tests/", "test_new_clock_routes"),
    "e_diversity": paths(MULTI, """
        component_concentration_experiment verify_component_concentration
        handoff_component_concentration joint_component_selection_v2 verify_joint_component_selection
    """) + paths("tests/", "test_joint_component_selection_v2"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory_path = HERE / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    planned, classified = [], []
    for stream, value in inventory["streams"].items():
        wanted = set(FILES[stream])
        for row in value["additions"]:
            relative = row["relative_path"]
            classification = (
                "already_in_main" if row["main_before"] == "identical" else
                "integrate_completed_dependency" if relative in wanted else
                "preserve_historical_or_superseded_producer"
            )
            classified.append(dict(row, stream=stream, disposition=classification))
            if relative not in wanted:
                continue
            source, archived, target = ROOT / row["source"]["path"], ROOT / row["archive"]["path"], ROOT / relative
            assert sha(source) == sha(archived) == row["source"]["sha256"]
            assert not target.exists(), relative
            original = source.read_text()
            adapted = original.replace(
                "ROOT = WORKTREE.parent.parent", "ROOT = (WORKTREE / \"results\").resolve().parent"
            )
            if relative == MULTI + "quality_lineage_diagnostic.py":
                old = 'ROOT = Path("/Users/rahulmaganti/Kosha/forge_new")'
                assert original.count(old) == 1
                adapted = adapted.replace(old, 'ROOT = (WORKTREE / "results").resolve().parent')
            if relative == ROUTE + "audit_new_leaves.py":
                assert original.count("MAIN = ROOT.parents[1]") == 1
                adapted = adapted.replace("MAIN = ROOT.parents[1]", 'MAIN = (ROOT / "results").resolve().parent')
            if relative == ROUTE + "residual_debt.py":
                old = "MAIN = Path(__file__).resolve().parents[5]"
                assert original.count(old) == 1
                adapted = adapted.replace(old, 'MAIN = (Path(__file__).resolve().parents[3] / "results").resolve().parent')
            planned.append({
                "stream": stream, "relative_path": relative,
                "source": row["source"], "archive": row["archive"],
                "destination_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                "location_adapted": original != adapted,
                "patch": "".join(difflib.unified_diff(
                    original.splitlines(keepends=True), adapted.splitlines(keepends=True),
                    fromfile=row["source"]["path"], tofile=relative,
                )),
                "content": adapted,
            })
        assert wanted <= {row["relative_path"] for row in value["additions"]}
    assert len(planned) == 39
    plan = {
        "schema": "forge.iclr22.consolidation_plan.v1",
        "inventory_sha256": sha(inventory_path), "producer_sha256": sha(Path(__file__)),
        "copies": [{k: v for k, v in row.items() if k != "content"} for row in planned],
        "all_delta_dispositions": classified,
        "no_existing_main_files_overwritten": True,
        "original_worktrees_preserved": True,
        "default_policies_and_manuscript_unchanged": True,
        "historical_result_admissions_not_rebound": True,
    }
    with (HERE / "plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    for row in planned:
        target = ROOT / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(row["content"])
        assert sha(target) == row["destination_sha256"]
    print(json.dumps({
        "copied": len(planned), "location_adaptations": sum(row["location_adapted"] for row in planned),
        "already_in_main": sum(row["disposition"] == "already_in_main" for row in classified),
        "archive_only": sum(row["disposition"] == "preserve_historical_or_superseded_producer" for row in classified),
        "plan_sha256": sha(HERE / "plan.json"),
    }))


if __name__ == "__main__":
    main()
