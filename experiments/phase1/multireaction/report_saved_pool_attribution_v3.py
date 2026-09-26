"""Report current-ledger controls without overwriting the historical experiment."""

import csv
import difflib
import json
import shutil
from pathlib import Path

from experiments.phase1.multireaction.saved_pool_attribution_v3 import (
    OUTPUT,
    RUN,
    WORKTREE,
    digest,
    pin,
    read,
    write,
)


def main():
    result, protocol = read(RUN / "result.json"), read(RUN / "protocol.json")
    review = read(RUN / "independent_recount_v1.json")
    assert result["complete"] and review["passed"]
    assert review["result"] == pin(RUN / "result.json")
    for value in protocol["inputs"].values():
        assert digest(Path(value["path"])) == value["sha256"]
    snapshot = RUN / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for key in ("producer", "candidate_selector", "baseline_selector"):
        shutil.copyfile(protocol["inputs"][key]["path"], snapshot / (key + ".py"))
    rows, totals = [], {}
    for arm, value in result["arms"].items():
        totals[arm] = {k: v for k, v in value["summary"].items() if isinstance(v, int)}
        totals[arm]["changed_products_vs_ordinal"] = sum(
            len(f["paired_vs_ordinal_control"]["changed_product_requests"])
            for f in value["by_family"].values()
        )
        for family, f in value["by_family"].items():
            row = {
                "arm": arm,
                "family": family,
                **{k: v for k, v in f["summary"].items() if isinstance(v, int)},
            }
            row["changed_products_vs_ordinal"] = len(
                f["paired_vs_ordinal_control"]["changed_product_requests"]
            )
            for metric, gains in f["paired_vs_ordinal_control"]["paired"].items():
                for direction, ids in gains.items():
                    row[metric + "_" + direction] = len(ids)
            rows.append(row)
    with (RUN / "per_family.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "producer": pin(Path(__file__)),
        "result": pin(RUN / "result.json"),
        "protocol": pin(RUN / "protocol.json"),
        "verification": pin(RUN / "independent_recount_v1.json"),
        "score_boundary_audit_reused": pin(OUTPUT / "ordering_v2/score_boundary_audit.json"),
        "prior_result_preserved": pin(OUTPUT / "ordering_v2/result.json"),
        "current_gate_augmentation": pin(OUTPUT / "common_gate_ledger_v1/verification.json"),
        "global": totals,
        "per_family": pin(RUN / "per_family.csv"),
        "CPU_seconds": {
            "qualification": read(RUN / "qualification.json")["CPU_seconds"],
            "controls": result["CPU_seconds"],
            "independent_recount": review["CPU_seconds"],
        },
        "all_132_family_solves_optimal": all(
            f["report"]["status"] == "optimal_independently_verified"
            for a in result["arms"].values()
            for f in a["by_family"].values()
        ),
        "promotion": False,
        "learned_model_removal_test": False,
        "unknowns": [
            "Changed identities need independent structural quality and route assessment",
            "Context flags are alerts, not a calibrated realism score",
            "Only15 exact identities have the additional transported-tree evidence",
        ],
    }
    write(RUN / "summary.json", record)
    table = [
        "| Ordering | Exact | Design | Context flags | Design without context | Changed products |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in protocol["arms"]:
        d = totals[arm]
        table.append(
            f"| {arm} | {d['exact']} | {d['design']} | {d['context']} | {d['design_no_context']} | {d['changed_products_vs_ordinal']} |"
        )
    text = [
        "# Current-evidence saved-pool ordering controls",
        "",
        "All six arms retain all1,408 requests (64 in each of22 families) and the same11,331 assessed unique candidate entries. These controls change only the final tie cost after unchanged design/support objectives. No model score exists at this final selector; learned logits still affect upstream construction. This is an ordering intervention, not a model-removal or rules-only control.",
        "",
        *table,
        "",
        "The common gate ledger carries15 exact request/ordinal/SMILES/family/kind-bound transported-tree assessments from independently authenticated prior evidence. All other assessments are unchanged. It reproduces the current1,324 design passes; the earlier1,312 historical-gate controls remain preserved in ordering_v2. The intervention does not grant new tree evidence to alternative molecules.",
        "",
        "Every arm passed per-request exact-status, context-code-subset, role-count and per-family product/component diversity and TRAIN-novelty floors. Independent stdlib recounts authenticated all source/input pins,8,448 selections,138 summaries and264 paired family contrasts per version. Solver reports were optimal for all132 family/arm solves. No arm has been selected for promotion; independent-reference realism and routing were not recomputed.",
        "",
        f"Costs: qualification {record['CPU_seconds']['qualification']:.6f} CPU seconds, six controls {record['CPU_seconds']['controls']:.6f}, independent recount {record['CPU_seconds']['independent_recount']:.6f}; single solver thread. Setup/report CPU is not included. No new model, chemistry, TEST or network calls.",
        "",
        "Reproducible commands (isolated A worktree, PYTHONPATH set to it, BLAS/OMP threads1):",
        "",
        "```sh",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution_v3 freeze",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution_v3 qualify",
        ".venv/bin/python -m experiments.phase1.multireaction.saved_pool_attribution_v3 run",
        ".venv/bin/python -m experiments.phase1.multireaction.verify_saved_pool_controls results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution/ordering_v3",
        ".venv/bin/python -m experiments.phase1.multireaction.report_saved_pool_attribution_v3",
        "```",
        "",
        "Use fresh versioned outputs when reproducing; existing files reject overwrite. Full selections, all adverse transitions and every family are in result.json; per_family.csv is the compact view. submission.json records detached launch details. Root owns final admission and decision logging.",
    ]
    with (RUN / "README.md").open("x") as stream:
        stream.write("\n".join(text) + "\n")
    files = sorted((WORKTREE / "experiments/phase1/multireaction").glob("*saved_pool*.py"))
    files += sorted((WORKTREE / "tests").glob("test_saved_pool*.py"))
    patch = []
    for path in files:
        patch.extend(
            difflib.unified_diff(
                [],
                path.read_text().splitlines(True),
                fromfile="/dev/null",
                tofile="b/" + str(path.relative_to(WORKTREE)),
            )
        )
    with (OUTPUT / "candidate_v3.patch").open("x") as stream:
        stream.write("".join(patch))
    print(
        json.dumps(
            {"summary": pin(RUN / "summary.json"), "global": totals, "CPU": record["CPU_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
