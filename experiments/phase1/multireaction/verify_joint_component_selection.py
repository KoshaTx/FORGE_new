"""Independent saved-row verification, frozen quality diagnostic and all-change gallery."""

from __future__ import annotations

import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Draw

from experiments.phase1.multireaction.verify_component_concentration import pin, read, write
from forge.model.common_lipid_realism import fit_robust_descriptor_scale
from results.phase1.compose_lipid_iclr22_research_v1.calibration_distribution.run import vectors
from results.phase1.compose_lipid_iclr22_table_completion_v1.quality.analyze import PATHS
from results.phase1.compose_lipid_iclr22_table_completion_v1.readout_quality.analyze import (
    arm_metrics,
)

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
OUT = (
    ROOT
    / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/integration/joint_selection_v1/run_v2"
)
AUDIT = OUT / "audit_v1"
ROUTES = ("L2_ready", "combined_primary", "L3_direct_only", "strict_secondary")


def contexts(g):
    return {f["code"] for f in g["chemical"]["flags"] if f["tier"] == "context_required"}


def floor_counts(rows):
    groups = defaultdict(Counter)
    novel = defaultdict(Counter)
    for c in rows:
        if c["connected"]:
            groups["product"][c["smiles"]] += 1
            if c["product_train_novel"]:
                novel["product"][c["smiles"]] += 1
        for part in c["components"]:
            group = "component:" + part["role"]
            groups[group][part["smiles"]] += 1
            groups["all_components"][part["smiles"]] += 1
            for flag in ("global_train_novel", "role_train_novel"):
                if part[flag]:
                    novel[group + ":" + flag][part["smiles"]] += 1
                    novel["all_components:" + flag][part["smiles"]] += 1
    return groups, novel


def floors(before, after):
    left, ln = floor_counts(before)
    right, rn = floor_counts(after)
    assert set(left) == set(right)
    for k, a in left.items():
        b = right[k]
        assert sum(a.values()) == sum(b.values())
        assert len(b) >= len(a)
        assert sum(v * v for v in b.values()) <= sum(v * v for v in a.values())
        assert math.prod(v**v for v in b.values()) <= math.prod(v**v for v in a.values())
    for k, a in ln.items():
        assert len(rn[k]) >= len(a) and sum(rn[k].values()) >= sum(a.values())
    assert sum(c["local_features_observed"] for c in after) >= sum(
        c["local_features_observed"] for c in before
    )
    assert len({c["smiles"] for c in after if c["local_features_observed"]}) >= len(
        {c["smiles"] for c in before if c["local_features_observed"]}
    )
    return {
        "all_groups_passed": True,
        "global_components_before": len(left["all_components"]),
        "global_components_after": len(right["all_components"]),
        "component_sum_squares_before": sum(v * v for v in left["all_components"].values()),
        "component_sum_squares_after": sum(v * v for v in right["all_components"].values()),
    }


def main():
    start = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    AUDIT.mkdir(exist_ok=False)
    p = read(OUT / "protocol.json")
    r = read(OUT / "result.json")
    assert r["complete"] and not r["whole_panel_fallback"]
    assert r["protocol"] == pin(OUT / "protocol.json")
    for value in p["inputs"].values():
        assert pin(Path(value["path"]))["sha256"] == value["sha256"]
    sources = {
        k: read(Path(v["path"]))
        for k, v in p["inputs"].items()
        if Path(v["path"]).suffix == ".json"
    }
    paths = {
        "producer": Path(__file__),
        "protocol": OUT / "protocol.json",
        "result": OUT / "result.json",
        "prior_quality": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/result.json",
        "estimators": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/analyze.py",
        "descriptor_estimator": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/quality/analyze.py",
        **{
            k: PATHS[k]
            for k in (
                "CAL",
                "TRAIN_reference",
                "metrics",
                "descriptors",
                "regions",
                "vector_helper",
                "distribution_protocol",
            )
        },
    }
    write(
        AUDIT / "protocol.json",
        {
            "inputs": {k: pin(v) for k, v in paths.items()},
            "requests": 1408,
            "families": 22,
            "per_family": 64,
            "changed": 36,
            "scope": "Independent saved-row/source/gate/route recount, original quality metric exact reproduction then unchanged1821syntheticCAL estimator, all36before-after pairs rendered. No retuning, selection, model/source/route calls.",
            "CPU_seconds_cap": 60,
            "threads": 1,
        },
    )
    baseline = {
        x["request"]: x for f in sources["current"]["by_family"].values() for x in f["selections"]
    }
    assert set(baseline) == set(range(1408))
    selected = {x["request"]: x for x in r["selection"]}
    assert set(selected) == set(baseline)
    gates = {(x["index"], x["ordinal"]): x["assessment"] for x in sources["gates"]["attempts"]}
    e_selected = {x["request"]: x for x in sources["E_selection"]["selections"]}
    b_selected = {
        x["request"]: x
        for f in sources["B_selection"]["families"].values()
        for x in f["selections"]
    }
    bw = {x["index"]: x for x in sources["B_witness"]["rows"]}
    ew = {x["index"]: x for x in sources["E_witness"]["records"]}
    products = {
        a: {x["index"]: x for x in sources[a + "_products"]["products"]} for a in ("D", "B", "E")
    }
    requests = {
        a: {x["index"]: x for x in sources[a + "_requests"]["requests"]} for a in ("D", "B", "E")
    }
    evidence = {x["request"]: x for x in r["selected_evidence"]}
    changes = []
    totals = Counter()
    familyfloors = {}
    for i, before in baseline.items():
        after = selected[i]
        ev = evidence[i]
        arm = ev["source_arm"]
        expected = before if arm == "D" else (b_selected if arm == "B" else e_selected)[i]
        assert after == expected
        verdict = products[arm][i]
        request = requests[arm][i]
        assert ev["route_verdict"] == verdict and ev["route_request"] == request
        oldgate = gates[i, before["ordinal"]]
        newgate = bw[i]["design_assessment"] if arm == "B" else gates[i, after["ordinal"]]
        assert ev["design_assessment"] == newgate
        assert before["exact"] == after["exact"] == verdict["exact_L1"]
        assert not oldgate["qualified_design_pass"] or newgate["qualified_design_pass"]
        assert contexts(newgate) <= contexts(oldgate)
        for k in ROUTES:
            assert not products["D"][i][k] or verdict[k]
        assert request["selected_smiles"] == after["smiles"]
        if arm != "D":
            witness = bw[i] if arm == "B" else ew[i]
            check = witness["source_assessment"] if arm == "B" else witness["proposal"]["check"]
            assert check["exact"] is True
            component_sets = {
                tuple(sorted(c.items()))
                for q in check["checks"]
                for c in q.get("accepted_components", [])
            }
            assert len(component_sets) == 1
            accepted = dict(next(iter(component_sets)))

            def canon(smiles):
                return Chem.MolToSmiles(
                    Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False
                )

            assert {k: canon(v) for k, v in accepted.items()} == {
                p["role"]: p["smiles"] for p in after["components"]
            }
            assert request["canonical_product"] == after["smiles"]
            assert before["smiles"] != after["smiles"]
            changes.append(
                {
                    "index": i,
                    "family": request["family"],
                    "source_arm": arm,
                    "before": before["smiles"],
                    "after": after["smiles"],
                    "before_context": sorted(contexts(oldgate)),
                    "after_context": sorted(contexts(newgate)),
                }
            )
        totals.update(
            {
                "requests": 1,
                "exact": after["exact"],
                "design": newgate["qualified_design_pass"],
                "context": bool(contexts(newgate)),
                **{k: verdict[k] for k in ROUTES},
                "makeable_design": verdict["combined_primary"] and newgate["qualified_design_pass"],
                "makeable_design_no_context": verdict["combined_primary"]
                and newgate["qualified_design_pass"]
                and not contexts(newgate),
            }
        )
    assert len(changes) == 36 and Counter(x["source_arm"] for x in changes) == {"B": 7, "E": 29}
    assert (
        totals["design"] == 1331
        and totals["combined_primary"] == 623
        and totals["L3_direct_only"] == 121
        and totals["L2_ready"] == 1189
        and totals["strict_secondary"] == 27
    )
    pooled = floors(list(baseline.values()), list(selected.values()))
    quality = {}
    cal = defaultdict(list)
    for x in read(paths["CAL"])["rows"]:
        cal[x["family"]].append(x["canonical_smiles"])
    assert sum(map(len, cal.values())) == 1821
    train = read(paths["TRAIN_reference"])["product_controls"]
    prior = {x["family"]: x["arms"]["selected"] for x in read(paths["prior_quality"])["families"]}
    with rdBase.BlockLogs():
        for family, old in sorted(sources["current"]["by_family"].items()):
            before = old["selections"]
            after = [selected[c["request"]] for c in before]
            assert len(before) == len(after) == 64
            familyfloors[family] = floors(before, after)
            ref, rm, _ = vectors(cal[family])
            _, tm, _ = vectors(train[family][:64])
            scale = fit_robust_descriptor_scale(tm)
            left = arm_metrics([c["smiles"] for c in before], ref, rm, scale)
            right = arm_metrics([c["smiles"] for c in after], ref, rm, scale)
            for key in (
                "counts",
                "fingerprints",
                "descriptors",
                "descriptor_neighborhood",
                "descriptor_statistics",
            ):
                assert left[key] == prior[family][key], (family, key)
            quality[family] = {"current": left, "joint": right}
    images = []
    changes.sort(key=lambda c: c["index"])
    for first in range(0, len(changes), 6):
        page = changes[first : first + 6]
        mols = []
        legends = []
        for c in page:
            for arm in ("before", "after"):
                mols.append(Chem.MolFromSmiles(c[arm]))
                legends.append(f"{c['index']} {c['family']} {arm} ({c['source_arm']})")
        image = AUDIT / f"changes-{first//6+1:02d}.png"
        Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(640, 360), legends=legends).save(image)
        images.append(pin(image))
    macro = {
        a: {
            "FP_precision": sum(
                v[a]["fingerprints"]["fingerprint_precision_among_unique"] for v in quality.values()
            )
            / 22,
            "FP_coverage": sum(
                v[a]["fingerprints"]["fingerprint_coverage"] for v in quality.values()
            )
            / 22,
            "FP_internal_diversity": sum(
                v[a]["fingerprints"]["mean_pairwise_tanimoto_distance"] for v in quality.values()
            )
            / 22,
            "FP_members": sum(
                v[a]["fingerprints"]["unique_member_count"] for v in quality.values()
            ),
            "descriptor_precision": sum(
                v[a]["descriptor_neighborhood"]["precision"] for v in quality.values()
            )
            / 22,
            "descriptor_coverage": sum(
                v[a]["descriptor_neighborhood"]["coverage"] for v in quality.values()
            )
            / 22,
            "descriptor_members": sum(
                v[a]["descriptor_neighborhood"]["member_observations"] for v in quality.values()
            ),
        }
        for a in ("current", "joint")
    }
    write(
        AUDIT / "result.json",
        {
            "passed": True,
            "complete": True,
            "protocol": pin(AUDIT / "protocol.json"),
            "joint_result": pin(OUT / "result.json"),
            "totals": dict(totals),
            "all36_source_identity_and_current_gate_bindings": True,
            "all1408_typed_route_verdict_bindings": True,
            "per_request_positive_routes_preserved": True,
            "family_floors": familyfloors,
            "pooled_floors": pooled,
            "all_changes": changes,
            "images": images,
            "synthetic_CAL_quality": quality,
            "macro22": macro,
            "baseline_quality_exact_reproduction": True,
            "CPU_seconds": time.process_time() - start,
            "new_solver_model_source_route_classifier_TEST_network_GPU_calls": 0,
            "official_cohort_promotion": False,
            "visual_scope": "All36 pairs rendered, no rendering failure; human visual inspection recorded separately.",
        },
    )
    print(
        json.dumps(
            {
                "result": pin(AUDIT / "result.json"),
                "totals": dict(totals),
                "macro22": macro,
                "CPU_seconds": time.process_time() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
