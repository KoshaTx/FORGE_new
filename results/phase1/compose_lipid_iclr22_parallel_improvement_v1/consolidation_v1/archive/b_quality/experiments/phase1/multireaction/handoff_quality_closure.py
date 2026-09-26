"""Publish identity-bound route handoff and close the bounded B experiment."""

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.phase1.multireaction.quality_closure_diagnostic import HERE
from experiments.phase1.multireaction.quality_lineage_diagnostic import (
    OUT,
    PATHS,
    ROOT,
    WORKTREE,
    pin,
    read,
    write,
)
from forge.model.reaction_program_flow import derive_role_morphology_states


def main():
    start = time.process_time()
    result = read(HERE / "result.json")
    verified = read(HERE / "verification.json")
    source = (
        ROOT
        / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1/routes/recount_closeout_v3/result.json"
    )
    original = read(source)
    cohort = {r["index"]: dict(r) for r in original["rows"]}
    layouts = torch.load(PATHS["layouts"], weights_only=False, map_location="cpu")["layouts"]

    def morphology(idx, state):
        record = layouts[idx].record
        graph = SimpleNamespace(**{k: np.asarray(v) for k, v in state.items()})
        return derive_role_morphology_states(
            SimpleNamespace(
                node_count=record.node_count,
                role_states=record.role_states,
                core_position_states=record.core_position_states,
                graph=graph,
            )
        )

    changed = []
    checked = 0
    selected = {
        (c["request"], c["ordinal"]): c
        for f in result["families"].values()
        for c in f["selections"]
    }
    for receipt in result["parent_receipts"]:
        row = read(ROOT / receipt["path"])
        idx = row["request"]
        for p in row["assessed_proposals"]:
            assert np.array_equal(
                morphology(idx, row["parent"]["tree_state"]),
                morphology(idx, p["graph"]["tree_state"]),
            )
            checked += 1
            c = p["candidate"]
            if (idx, c["ordinal"]) not in selected:
                continue
            assert selected[idx, c["ordinal"]] == c and p["eligible"]
            old = cohort[idx]
            changed.append(
                {
                    "index": idx,
                    "family": row["family"],
                    "before_smiles": old["smiles"],
                    "smiles": c["smiles"],
                    "constitution_id": hashlib.sha256(c["smiles"].encode()).hexdigest(),
                    "components": c["components"],
                    "source_assessment": p["source_check"],
                    "design_assessment": p["assessment"],
                    "parent_artifact": receipt,
                    "candidate_ordinal": c["ordinal"],
                    "inherited_route_verdict": None,
                    "inherited_model_likelihood": None,
                }
            )
            cohort[idx] = {
                **old,
                "ordinal": c["ordinal"],
                "smiles": c["smiles"],
                "components": c["components"],
                "exact_L1": c["exact"],
                "limited_design_pass": p["assessment"]["qualified_design_pass"],
                "context_codes": sorted(
                    {
                        f["code"]
                        for f in p["assessment"]["chemical"]["flags"]
                        if f["tier"] == "context_required"
                    }
                ),
                "failed_axes": [
                    k for k, v in p["assessment"]["status_by_axis"].items() if v == "fail"
                ],
                "unknown_axes": [
                    k for k, v in p["assessment"]["status_by_axis"].items() if v == "abstain"
                ],
                "L2_ready": None,
                "computational_makeability": None,
                "route_evidence_status": "new_identity_requires_exact_component_route_evaluation",
                "source_assessment_pin": receipt,
            }
    assert len(changed) == 19 and len(cohort) == 1408
    assert (
        sum(r["limited_design_pass"] for r in cohort.values()) == result["full_cohort_design_after"]
    )
    inputs = {
        str(p.relative_to(ROOT)): pin(p)
        for p in (
            Path(__file__),
            source,
            HERE / "result.json",
            HERE / "verification.json",
            PATHS["layouts"],
        )
    }
    write(
        HERE / "changed_products.json",
        {
            "schema": "forge.changed_products.route_handoff.v1",
            "inputs": inputs,
            "rows": changed,
            "changed_count": 19,
            "full_denominator": 1408,
            "inherited_routes_for_changed_products": False,
            "official_cohort_replaced": False,
        },
    )
    write(
        HERE / "candidate_cohort_1408.json",
        {
            "schema": "forge.quality.candidate_cohort.v1",
            "inputs": inputs,
            "rows": [cohort[i] for i in sorted(cohort)],
            "full_denominator": 1408,
            "changed_indices": [r["index"] for r in changed],
            "unchanged_rows": 1389,
            "new_identity_route_status": "19 null verdicts, explicit reassessment required",
            "candidate_only": True,
        },
    )
    modules = {
        k: pin(Path(m.__file__))
        for k, m in sys.modules.items()
        if k.startswith("forge.") and getattr(m, "__file__", None)
    }
    assert all((ROOT / p["path"]).is_relative_to(WORKTREE) for p in modules.values())
    write(
        HERE / "handoff.json",
        {
            "complete": True,
            "result": pin(HERE / "result.json"),
            "verification": pin(HERE / "verification.json"),
            "changed_products": pin(HERE / "changed_products.json"),
            "cohort": pin(HERE / "candidate_cohort_1408.json"),
            "morphology_four_coordinates_exact_in_all_proposals": checked,
            "loaded_forge_modules_from_isolated_worktree": modules,
            "full_request_denominators_preserved": True,
            "all19_before_after_images_visually_inspected": verified["images"],
            "visual_interpretation": "Macrocycle endpoints shrink to requested allowed sizes, but large many-arm A3 architectures and complex oxygenated ketone structures remain. Images establish connectivity only, not independent chemical realism. All19 pairs included, no favourable selection.",
            "distributional_tradeoffs": "A3 syntheticCAL FP membership36→35/64; aldehydeUgi4 54→55/64 and descriptor57→59/64; ketone1/64 and aryl2/63 unchanged. Fingerprint diversity slightly decreases forA3 and increases for bothUgi4. Context counts unchanged. No overall quality promotion.",
            "head_scope": "Existing aldehydeUgi4 predicate only; five selected failures remain. Other families remain outside scope. No pKa or efficacy inference.",
            "CPU_seconds": time.process_time() - start,
            "model_TEST_network_GPU_calls": 0,
        },
    )
    write(
        OUT / "result.json",
        {
            "complete": True,
            "scope": "Isolated bounded stream B, unmerged candidate; no official cohort or manuscript change.",
            "lineage": pin(OUT / "lineage_result.json"),
            "intervention": pin(HERE / "result.json"),
            "verification": pin(HERE / "verification.json"),
            "handoff": pin(HERE / "handoff.json"),
            "new_model_training_calls": 0,
            "matched_full_population": 1408,
            "exact_L1_before_after": 1387,
            "limited_design_before": 1324,
            "limited_design_after": 1343,
            "overall_realism_promotion": False,
            "negative_results_preserved": [
                "No context-flag reduction",
                "A3 fingerprint membership falls by one",
                "ArylRA unchanged",
                "39 fused/ambiguous constructed-basis proposal abstentions",
                "Four role-cycle mismatch parents outside endpoint-only scope",
                "Five aldehydeUgi4 head failures remain",
            ],
            "costs_CPU_seconds": {
                "lineage": read(OUT / "lineage_result.json")["CPU_seconds"],
                "construction_selection": result["CPU_seconds"],
                "verification_and_metrics": verified["CPU_seconds"],
                "handoff": time.process_time() - start,
            },
        },
    )
    print(
        json.dumps(
            {
                "result": pin(OUT / "result.json"),
                "handoff": pin(HERE / "handoff.json"),
                "changed_products": pin(HERE / "changed_products.json"),
                "cohort": pin(HERE / "candidate_cohort_1408.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
