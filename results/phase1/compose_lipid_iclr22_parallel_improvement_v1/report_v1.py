"""Consolidate frozen development results, preserving populations and evidence scopes."""

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def context(gate):
    return any(f["tier"] == "context_required" for f in gate["chemical"]["flags"])


def metrics(selections, gates, routes):
    rows = {r["index"]: r for r in routes}
    assert len(rows) == len(selections)
    for candidate in selections:
        assert candidate["exact"] == rows[candidate["request"]]["exact_L1"]
    components = Counter(p["smiles"] for c in selections for p in c["components"])
    component_roles = Counter((p["role"], p["smiles"]) for c in selections for p in c["components"])
    count = sum(components.values())
    return {
        "requests": len(selections),
        "exact_L1": sum(c["exact"] for c in selections),
        "limited_design": sum(gates[c["request"]]["qualified_design_pass"] for c in selections),
        "context_flagged": sum(context(gates[c["request"]]) for c in selections),
        "design_without_context": sum(gates[c["request"]]["qualified_design_pass"] and not context(gates[c["request"]]) for c in selections),
        "computational_makeability": sum(r["combined_primary"] for r in routes),
        "L2_ready": sum(r["L2_ready"] for r in routes),
        "L3_direct_only": sum(r["L3_direct_only"] for r in routes),
        "strict_dossiers": sum(r["strict_secondary"] for r in routes),
        "makeable_and_design": sum(r["combined_primary"] and gates[r["index"]]["qualified_design_pass"] for r in routes),
        "makeable_design_without_context": sum(r["combined_primary"] and gates[r["index"]]["qualified_design_pass"] and not context(gates[r["index"]]) for r in routes),
        "required_component_observations": count,
        "unique_components_role_agnostic": len(components),
        "component_role_sum_squares": sum(v * v for v in component_roles.values()),
        "component_role_agnostic_sum_squares": sum(v * v for v in components.values()),
        "component_role_agnostic_inverse_Simpson": count * count / sum(v * v for v in components.values()),
        "component_role_agnostic_Shannon_effective": math.exp(math.log(count) - sum(v * math.log(v) for v in components.values()) / count),
        "unique_products": len({c["smiles"] for c in selections}),
        "TRAIN_novel_product_occurrences": sum(c["product_train_novel"] for c in selections),
    }


def main():
    current_path = ROOT / "results/phase1/compose_lipid_iclr22_research_v1/quality/all22_context_preserving/result.json"
    current = read(current_path)
    original = sorted([c for family in current["by_family"].values() for c in family["selections"]], key=lambda c: c["request"])
    gate_path = HERE / "a_attribution/common_gate_ledger_v1/gates.json"
    gate_ledger = {(r["index"], r["ordinal"]): r["assessment"] for r in read(gate_path)["attempts"]}
    original_gates = {c["request"]: gate_ledger[c["request"], c["ordinal"]] for c in original}
    joint_path = HERE / "integration/joint_selection_v1/run_v2/result.json"
    joint = read(joint_path)
    selected = joint["selection"]
    joint_gates = {r["request"]: r["design_assessment"] for r in joint["selected_evidence"]}
    initial_path = HERE / "d_routes/full_pair_v2/control/products.json"
    initial_routes = read(initial_path)["products"]
    destination = HERE / "d_routes/new_clock_recount_v1"
    completion = read(destination / "completion.json")
    assert completion["all_completed"]
    route_arms, route_validations = {}, {}
    inputs = [current_path, gate_path, joint_path, initial_path, destination / "completion.json"]
    for arm in ("original", "B", "joint"):
        directory = destination / arm
        validation = read(directory / "validation.json")
        assert validation["passed"] and validation["control_exactly_reproduces_saved_records"]
        assert validation["full_denominator"] == 1408 and validation["branches"] == 3723
        assert not any(r["losses"] for r in validation["transitions"].values())
        product_path = directory / "products.json"
        route_arms[arm] = read(product_path)["products"]
        route_validations[arm] = validation
        assert [r["index"] for r in route_arms[arm]] == list(range(1408))
        inputs.extend([product_path, directory / "result.json", directory / "validation.json", directory / "protocol.json"])
    assert [c["request"] for c in selected] == [c["request"] for c in original] == list(range(1408))
    joint_requests = read(destination / "joint_requests.json")["requests"]
    for c, request, verdict in zip(selected, joint_requests, route_arms["joint"], strict=True):
        assert c["request"] == request["index"] == verdict["index"]
        assert (request["canonical_product"] if c["exact"] else request["selected_smiles"]) == c["smiles"]
    baseline = metrics(original, original_gates, initial_routes)
    routes_only = metrics(original, original_gates, route_arms["original"])
    combined = metrics(selected, joint_gates, route_arms["joint"])
    assert baseline["computational_makeability"] == 616 and combined["computational_makeability"] == 624
    assert baseline["limited_design"] == 1324 and combined["limited_design"] == 1331
    by_family = []
    for family in sorted(current["by_family"]):
        ids = {r["index"] for r in initial_routes if r["family"] == family}
        before = metrics([c for c in original if c["request"] in ids], original_gates, [r for r in initial_routes if r["index"] in ids])
        after = metrics([c for c in selected if c["request"] in ids], joint_gates, [r for r in route_arms["joint"] if r["index"] in ids])
        assert before["requests"] == after["requests"] == 64
        by_family.append({"family": family, "before": before, "joint": after})
    assert len(by_family) == 22
    frozen = read(HERE / "baseline/result.json")
    for value in frozen["paper_sources"]:
        assert pin(ROOT / value["path"]) == value
    extras = [
        "baseline/result.json", "integration/joint_selection_v1/root_review.json",
        "integration/joint_selection_v1/run_v2/audit_v1/result.json",
        "integration/joint_selection_v1/run_v2/interpretation_v1/result.json",
        "integration/joint_selection_v1/closeout_v1.json", "d_routes/closeout_v1/result.json",
        "b_quality/result.json", "c_generalization/result.json", "e_diversity/closeout_v2.json",
        "e_diversity/isocyanide_diagnosis_v1/run_v4/result.json",
        "a_attribution/ordering_v3/summary.json", "a_attribution/null_fullfit_v1/failure_admission.json",
        "a_attribution/null_fullfit_v2/readiness_review.json", "a_attribution/matched_context_null_v5/readiness.json",
        "integration/source_integration_v1.json", "integration/source_integration_e_v1.json",
        "integration/source_integration_joint_v1.json", "integration/bc_tests.xml",
        "integration/e_tests.xml", "integration/joint_policy_tests.xml",
    ]
    inputs.extend(HERE / name for name in extras)
    result = {
        "schema": "forge.iclr22.parallel_improvement_development_snapshot.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "producer": pin(Path(__file__)),
        "inputs": [pin(p) for p in inputs], "population": 1408, "families": 22,
        "evidence_class": "Single-fit, TRAIN-derived request development cohort; adaptive method development, not independent final held-out evidence.",
        "route_evidence_clock": route_validations["joint"]["common_as_of_utc"],
        "baseline": baseline, "routes_only": routes_only, "joint": combined,
        "by_family": by_family,
        "route_comparisons": {k: {f: v[f] for f in ("before", "after", "transitions", "control_exactly_reproduces_saved_records")} for k, v in route_validations.items()},
        "synthetic_CAL_distribution": read(HERE / "integration/joint_selection_v1/run_v2/audit_v1/result.json")["macro22"],
        "generalization": read(HERE / "c_generalization/result.json"),
        "independent_empirical_lipid_realism_established": False,
        "default_generation_or_selection_replaced": False, "manuscript_changed": False,
        "unchanged_manuscript_source_files": len(frozen["paper_sources"]), "sealed_TEST_accessed": False,
        "trained_null_fit_status": "First run failed before update1; corrected exact retry ready, fresh user approval pending.",
        "trained_null_comparison_complete": False,
        "additional_work_in_progress": "Bounded all128 A3/ketone-Ugi4 complete-component compatibility census; no result admitted in this snapshot.",
        "research_goal_complete": False,
        "limitations": [
            "Limited design and exact assembly do not certify broad molecular realism, pKa, delivery or experimental execution.",
            "TRAIN support and synthetic-CAL diagnostics are not independent quality references; some family-level distribution metrics regress.",
            "Directory listings do not establish stock; strict dossiers stay a separately unchanged metric.",
            "B-only and E-only tradeoffs are preserved; component changes never inherit prior identity-specific routes.",
            "Independent fit seeds and prespecified final held-out comparisons remain open."
        ],
    }
    with (HERE / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"result": pin(HERE / "result.json"), "baseline": baseline, "joint": combined}))


if __name__ == "__main__":
    main()
