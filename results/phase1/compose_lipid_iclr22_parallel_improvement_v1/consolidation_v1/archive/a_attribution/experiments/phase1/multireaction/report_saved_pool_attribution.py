"""Summarize frozen ordering controls and identify the actual score boundary."""

import csv
import difflib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from experiments.phase1.multireaction.saved_pool_attribution import (
    OUTPUT,
    RUN,
    WORKTREE,
    digest,
    pin,
    read,
    write,
)


def main():
    result = read(RUN / "result.json")
    protocol = read(RUN / "protocol.json")
    assert result["complete"] and result["protocol"] == pin(RUN / "protocol.json")
    for value in protocol["inputs"].values():
        assert digest(Path(value["path"])) == value["sha256"], value["path"]
    snapshots = RUN / "source_snapshot"
    snapshots.mkdir(exist_ok=False)
    for key in ("producer", "candidate_selector", "baseline_selector"):
        shutil.copyfile(protocol["inputs"][key]["path"], snapshots / (key + ".py"))
    diagnostics = read(Path(protocol["inputs"]["candidate_ledger"]["path"]))["candidates"]
    assessed = [r for r in diagnostics if r["status"] == "assessed"]
    assert len(assessed) == 11331
    keys = {tuple(sorted(r["candidate"])) for r in assessed}
    assert len(keys) == 1
    assert not any("score" in k or "logit" in k or "prob" in k for k in next(iter(keys)))
    selected = read(Path(protocol["inputs"]["current_selections"]["path"]))["by_family"]
    index = {(r["index"], r["ordinal"]): r for r in assessed}
    kinds, selected_kinds = defaultdict(Counter), defaultdict(Counter)
    for row in assessed:
        kinds[row["family"]][row["kind"]] += 1
    for family, values in selected.items():
        for c in values["selections"]:
            selected_kinds[family][index[c["request"], c["ordinal"]]["kind"]] += 1
    cost_rows = read(Path(protocol["inputs"]["compact_ledger"]["path"]))
    assert len(cost_rows) == 1408
    costs = defaultdict(Counter)
    for row in cost_rows:
        costs[row["family"]].update(row["branches"]["d1"]["costs"])
    audit = {
        "producer": pin(Path(__file__)),
        "inputs": protocol["inputs"],
        "candidate_fields": next(iter(keys)),
        "candidate_entries": len(assessed),
        "numeric_model_score_in_final_selector": False,
        "ordinal_origin": "enumeration of raw followed by ordered constructed proposals; canonical duplicates dropped without changing original ordinal",
        "final_objective": ["qualified_design", "supported_distinct", "supported_count", "ordinal"],
        "upstream_model_dependencies": {
            "component_constraints": "Node-logit sum ranks donor copies after predicate deficit; node/bond logit changes rank bounded mutations.",
            "scaffold_construction": "Registered scaffold fixes structure, then unchanged saved model predictions supply remaining readout choices.",
            "symmetric_arms": "Saved parent logits choose allowed attachment; constrained readout still consumes all prediction heads.",
        },
        "candidate_kinds_by_family": {k: dict(v) for k, v in kinds.items()},
        "current_selected_kinds_by_family": {k: dict(v) for k, v in selected_kinds.items()},
        "unchanged_sunk_D1_costs_by_family": {k: dict(v) for k, v in costs.items()},
        "all_controls_share_candidates_and_sunk_budget": True,
        "rules_only_control_available_from_saved_pool": False,
        "reason": "Every saved proposal derives from learned states and/or learned prediction rankings; deleting a nonexistent selector score cannot remove these dependencies.",
        "next_discriminating_control": "Reconstruct proposals from the same layouts and caps with explicit TRAIN count-prior scores, independently qualifying unchanged-score replay first; report constructor-only score intervention separately from a genuinely untrained generator.",
        "new_model_or_chemistry_calls": 0,
    }
    write(RUN / "score_boundary_audit.json", audit)
    rows, globals = [], {}
    for arm, arm_result in result["arms"].items():
        globals[arm] = {k: v for k, v in arm_result["summary"].items() if isinstance(v, int)}
        globals[arm]["changed_products_vs_ordinal"] = sum(
            len(v["paired_vs_ordinal_control"]["changed_product_requests"])
            for v in arm_result["by_family"].values()
        )
        globals[arm]["CPU_seconds"] = arm_result["CPU_seconds"]
        for family, entry in arm_result["by_family"].items():
            assert entry["all_floors_pass"]
            row = {
                "arm": arm,
                "family": family,
                **{k: v for k, v in entry["summary"].items() if isinstance(v, int)},
            }
            row["changed_products_vs_ordinal"] = len(
                entry["paired_vs_ordinal_control"]["changed_product_requests"]
            )
            for key, value in entry["paired_vs_ordinal_control"]["paired"].items():
                row[key + "_gains"] = len(value["gains"])
                row[key + "_losses"] = len(value["losses"])
            rows.append(row)
    with (RUN / "per_family.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "producer": pin(Path(__file__)),
        "inputs": {
            "result": pin(RUN / "result.json"),
            "protocol": pin(RUN / "protocol.json"),
            "qualification": pin(RUN / "qualification.json"),
            "score_audit": pin(RUN / "score_boundary_audit.json"),
        },
        "global": globals,
        "per_family": pin(RUN / "per_family.csv"),
        "all_132_family_controls_optimal": all(
            f["report"]["status"] == "optimal_independently_verified"
            for a in result["arms"].values()
            for f in a["by_family"].values()
        ),
        "candidate_quality_promotion": False,
        "method_improvement_claim": "No better learned generator established; frozen ordering dependence measured.",
        "successful_CPU_seconds": result["CPU_seconds"]
        + read(RUN / "qualification.json")["CPU_seconds"],
        "earlier_failure_CPU_conservative_charge": 5,
        "accounting_excludes": "setup, tests, freeze and report aggregation; no large compute outside this record",
    }
    write(RUN / "summary.json", summary)
    table = [
        "| Ordering | Exact | Frozen design | Context flags | Design, no context | Changed products |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in protocol["arms"]:
        d = globals[arm]
        table.append(
            f"| {arm} | {d['exact']} | {d['design']} | {d['context']} | {d['design_no_context']} | {d['changed_products_vs_ordinal']} |"
        )
    lines = [
        "# Saved-pool ordering controls",
        "",
        "All six arms retain all 1,408 requests and the same 11,331 assessed candidate entries. Eligibility masks and current-panel floors are identical across arms. No model, chemistry, TEST or network calls were made.",
        "",
        *table,
        "",
        "The frozen historical selector exactly reproduced all 1,408 saved identities. The candidate implementation's default path matched every result and solver report. All 132 family/arm solves were optimal, with exact-status, product/role diversity, novelty and context-subset checks passing. Six focused tests and Black/Ruff checks passed before protocol freeze.",
        "",
        "There is no explicit model score in the final selector. The learned model enters upstream donor, mutation, scaffold and symmetric-arm construction. These ordering controls therefore establish neither a rules-only baseline nor the effect of removing the model. The same candidate pool, proposal budgets, source checks and sunk generation expense apply to every arm.",
        "",
        "Changing the final tie-break can remove context flags, but context flags are not a validated scalar realism score. The controls alter hundreds of molecules. This remains a development selection diagnostic; no arm is selected as best or promoted. The all-candidate frozen design rubric predates later selected-only ring receipts: 1,312 is the correct design denominator here, and it must not be replaced with the newer 1,324 for unchanged baseline molecules. A complete current-assessor, independent-reference and route recount is required for any candidate panel before promotion.",
        "",
        "Costs: qualification/profile 9.764224 CPU seconds; six-arm controls 18.990845 CPU seconds, single solver thread. Earlier v1 comparison stopped before interventions because Python dataclass tuples were compared directly with JSON arrays; its source, protocol and failure are preserved with a conservative five-CPU-second charge. Version2 only normalizes the serialized comparison and changes the output path.",
        "",
        "Commands (run from the isolated A worktree with PYTHONPATH set to that worktree and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1):",
        "",
        "```sh",
        ".venv/bin/python -m pytest -q tests/test_saved_pool_attribution.py",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution freeze",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution qualify",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution run",
        ".venv/bin/python -m experiments.phase1.multireaction.report_saved_pool_attribution",
        "```",
        "",
        "Existing output paths are immutable. See submission.json for the actual detached command/PID, protocol.json for all source/input pins, result.json for every selection and paired request identity, per_family.csv for compact family results, and score_boundary_audit.json for the upstream dependency trace. Root owns admission, manuscript changes and the decision log.",
    ]
    (RUN / "README.md").write_text("\n".join(lines) + "\n")
    patch = []
    for rel in [
        "experiments/phase1/multireaction/saved_pool_selector.py",
        "experiments/phase1/multireaction/saved_pool_attribution.py",
        "experiments/phase1/multireaction/report_saved_pool_attribution.py",
        "tests/test_saved_pool_attribution.py",
    ]:
        patch.extend(
            difflib.unified_diff(
                [],
                (WORKTREE / rel).read_text().splitlines(True),
                fromfile="/dev/null",
                tofile="b/" + rel,
            )
        )
    (OUTPUT / "candidate.patch").write_text("".join(patch))
    print(
        json.dumps(
            {
                "summary": pin(RUN / "summary.json"),
                "score_audit": pin(RUN / "score_boundary_audit.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
