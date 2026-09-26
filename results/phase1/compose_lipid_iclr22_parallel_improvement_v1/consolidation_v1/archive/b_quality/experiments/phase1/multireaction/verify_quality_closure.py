"""Recount frozen repair outputs, independently replay graphs/gates, and show all changes."""

from __future__ import annotations

import hashlib
import json
import math
import signal
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Draw, rdMolDescriptors

from experiments.phase1.multireaction.quality_closure_diagnostic import HERE, REFERENCE, contexts
from experiments.phase1.multireaction.quality_lineage_diagnostic import (
    FAMILIES,
    PATHS,
    ROOT,
    pin,
    read,
    write,
)
from forge.model.common_lipid_realism import fit_robust_descriptor_scale
from forge.model.compose_lipid_quality import source_role_mapping
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from results.phase1.compose_lipid_component_decoder_v1.contracts import assess, load_all, matching
from results.phase1.compose_lipid_iclr22_research_v1.calibration_distribution.run import vectors
from results.phase1.compose_lipid_iclr22_table_completion_v1.quality.analyze import PATHS as QPATHS
from results.phase1.compose_lipid_iclr22_table_completion_v1.readout_quality.analyze import (
    arm_metrics,
)
from results.phase1.compose_lipid_structure_repair_v1.evaluation.gates import (
    assess_design,
    load_gate_context,
)


def grouped_counts(rows):
    counts, novel = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        if r["connected"]:
            counts["product"][r["smiles"]] += 1
            if r["product_train_novel"]:
                novel["product"][r["smiles"]] += 1
        for c in r["components"]:
            k = "component:" + c["role"]
            counts[k][c["smiles"]] += 1
            for scope in ("global", "role"):
                if c[scope + "_train_novel"]:
                    novel[k + ":" + scope][c["smiles"]] += 1
    return counts, novel


def floor_report(before, after):
    b, bn = grouped_counts(before)
    a, an = grouped_counts(after)
    assert set(a) == set(b)
    assert {r["request"]: r["exact"] for r in before} == {r["request"]: r["exact"] for r in after}
    rows = {}
    for group, values in b.items():
        x = a[group]
        assert sum(x.values()) == sum(values.values())
        assert len(x) >= len(values)
        assert sum(v * v for v in x.values()) <= sum(v * v for v in values.values())
        assert math.prod(v**v for v in x.values()) <= math.prod(v**v for v in values.values())
        rows[group] = {
            "observations": sum(x.values()),
            "unique_before": len(values),
            "unique_after": len(x),
            "squared_counts_before": sum(v * v for v in values.values()),
            "squared_counts_after": sum(v * v for v in x.values()),
            "integer_Shannon_noninferiority": True,
        }
    for group, values in bn.items():
        x = an[group]
        assert len(x) >= len(values) and sum(x.values()) >= sum(values.values())
        rows["novel:" + group] = {
            "observations_before": sum(values.values()),
            "observations_after": sum(x.values()),
            "unique_before": len(values),
            "unique_after": len(x),
        }
    return rows


def main():
    start = time.process_time()
    torch.set_num_threads(1)
    extra = [
        Path(__file__),
        HERE / "result.json",
        HERE / "protocol.json",
        QPATHS["CAL"],
        REFERENCE,
        ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/result.json",
        ROOT / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/analyze.py",
        ROOT / "results/phase1/compose_lipid_iclr22_table_completion_v1/quality/analyze.py",
    ]
    write(
        HERE / "verification_protocol.json",
        {
            "inputs": [pin(p) for p in extra],
            "CPU_seconds_cap": 120,
            "seed": None,
            "scope": "Saved-state/source/gate replay, exact integer diversity and novelty verification, all19 selected changes depicted. Descriptive syntheticCAL metrics use unchanged historical reference/estimators, not a selection objective or independent realism claim. No proposal construction or model calls.",
        },
    )

    def timeout(*_):
        raise TimeoutError("Verification CPU cap reached")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, 115)
    result = read(HERE / "result.json")
    protocol = read(HERE / "protocol.json")
    for p in [*protocol["inputs"].values(), *protocol["source_files"], *result["parent_receipts"]]:
        assert pin(ROOT / p["path"]) == p
    payload = torch.load(PATHS["layouts"], weights_only=False, map_location="cpu")
    layouts, atoms = payload["layouts"], payload["atoms"]
    baseline = read(PATHS["selected"])["by_family"]
    census = {r["request"]: r for r in read(PATHS["census"])["all1408_rows"]}
    assert (
        len(census) == 1408
        and sum(r["source_exact"] for r in census.values())
        == result["full_cohort_exact_before_after"]
    )
    assert (
        sum(r["qualified_design_pass"] for r in census.values())
        == result["full_cohort_design_before"]
    )
    by_candidate = {}
    controls = []
    counts = Counter()
    source_checks = 0
    with rdBase.BlockLogs():
        executors, _, source_controls = load_all()
        context = load_gate_context(ROOT, PATHS["gate_policy"])
        assert source_controls == result["source_controls"]
        for receipt in result["parent_receipts"]:
            row = read(ROOT / receipt["path"])
            idx = row["request"]
            parent = row["parent"]
            layout = layouts[idx]
            n, e = np.asarray(parent["nodes"]), np.asarray(parent["edges"])
            assert graph_smiles(n, e, atoms) == parent["smiles"] and fixed_graph_preserved(
                n, e, layout.record
            )
            check = assess(executors, layout, parent["smiles"])
            source_checks += 1
            assert check == row["baseline_source_check"]
            if parent["control"]:
                assert check["exact"] and not row["assessed_proposals"]
                controls.append(idx)
            for item in row["assessed_proposals"]:
                p = item["graph"]
                tn, te = state_graph(p["tree_state"])
                assert tn.tolist() == p["nodes"] and te.tolist() == p["edges"]
                assert fixed_graph_preserved(tn, te, layout.record)
                assert graph_smiles(tn, te, atoms) == p["smiles"]
                for k in ("nodes", "parents", "parent_bonds", "closure_bonds"):
                    assert p["tree_state"][k] == parent["tree_state"][k]
                for block in layout.record.component_blocks:
                    assert Counter(tn[block.start : block.stop]) == Counter(
                        n[block.start : block.stop]
                    )
                assert rdMolDescriptors.CalcMolFormula(
                    Chem.MolFromSmiles(p["smiles"])
                ) == rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(parent["smiles"]))
                check = assess(executors, layout, p["smiles"])
                source_checks += 1
                assert check == item["source_check"]
                audit = assess_design(
                    layout,
                    tn,
                    te,
                    p["tree_state"],
                    p["tree_basis"],
                    p["smiles"],
                    check["exact"],
                    context,
                )
                assert json.loads(json.dumps(audit)) == item["assessment"]
                assert (
                    p["route_verdict"]
                    is p["model_likelihood"]
                    is p["trajectory_log_probability"]
                    is None
                )
                c = item["candidate"]
                by_candidate[idx, c["ordinal"]] = item
                counts["proposals"] += 1
                counts["source_exact"] += check["exact"]
                counts["design"] += audit["qualified_design_pass"]
                counts["ring_abstain"] += audit["status_by_axis"]["ring"] == "abstain"
                counts["head_fail"] += audit["status_by_axis"]["head"] == "fail"
    reference = read(REFERENCE)
    all_train = set(reference["component_smiles_by_identity"].values())
    db = sqlite3.connect(f"file:{ROOT/reference['inputs']['membership']['path']}?mode=ro", uri=True)
    cal = defaultdict(list)
    for r in read(QPATHS["CAL"])["rows"]:
        cal[r["family"]].append(r["canonical_smiles"])
    old_metrics = {
        r["family"]: r["arms"]["selected"]
        for r in read(
            ROOT
            / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/result.json"
        )["families"]
    }
    floor_records, metrics, changes = {}, {}, []
    for family in FAMILIES:
        before = baseline[family]["selections"]
        after = result["families"][family]["selections"]
        by_id = {r["request"]: r for r in before}
        assert len(after) == len(before) == 64
        floor_records[family] = floor_report(before, after)
        design = 0
        for c in after:
            i = c["request"]
            old = by_id[i]
            if c == old:
                design += census[i]["qualified_design_pass"]
                continue
            item = by_candidate[i, c["ordinal"]]
            assert item["eligible"] and c == item["candidate"]
            audit = item["assessment"]
            assert audit["qualified_design_pass"] and contexts(audit) <= set(
                census[i]["context_codes"]
            )
            assert not old["local_features_observed"] or c["local_features_observed"]
            assert c["product_train_novel"] == (
                db.execute(
                    "select 1 from sources where constitution_id=?",
                    (hashlib.sha256(c["smiles"].encode()).hexdigest(),),
                ).fetchone()
                is None
            )
            mapping = source_role_mapping(
                {p["role"]: p["smiles"] for p in c["components"]},
                item["source_check"],
                matching(executors, layouts[i]),
            )
            for part in c["components"]:
                assert part["global_train_novel"] == (part["smiles"] not in all_train)
                role_train = {
                    reference["component_smiles_by_identity"][identity]
                    for identity in reference["component_ids_by_family_role"][family][
                        mapping[part["role"]]
                    ]
                }
                assert part["role_train_novel"] == (part["smiles"] not in role_train)
            design += 1
            changes.append(
                {
                    "request": i,
                    "family": family,
                    "before": old["smiles"],
                    "after": c["smiles"],
                    "before_context_codes": census[i]["context_codes"],
                    "after_context_codes": sorted(contexts(audit)),
                }
            )
        assert design == result["families"][family]["design_after"]
        ref, rm, _ = vectors(cal[family])
        _, tm, _ = vectors(reference["product_controls"][family][:64])
        scale = fit_robust_descriptor_scale(tm)
        with rdBase.BlockLogs():
            left = arm_metrics([c["smiles"] for c in before], ref, rm, scale)
            right = arm_metrics([c["smiles"] for c in after], ref, rm, scale)
        for key in (
            "counts",
            "fingerprints",
            "descriptors",
            "descriptor_neighborhood",
            "descriptor_statistics",
        ):
            assert left[key] == old_metrics[family][key]
        metrics[family] = {"baseline": left, "closure_candidate": right}
    assert len(changes) == result["full_cohort_design_after"] - result["full_cohort_design_before"]
    images = []
    for first in range(0, len(changes), 5):
        page = changes[first : first + 5]
        mols = []
        legends = []
        for r in page:
            for arm in ("before", "after"):
                mols.append(Chem.MolFromSmiles(r[arm]))
                legends.append(f"Request {r['request']} {r['family']} {arm}")
        path = HERE / f"all_changes_{first//5+1:02d}.png"
        Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(620, 360), legends=legends).save(path)
        images.append(pin(path))
    write(
        HERE / "verification.json",
        {
            "complete": True,
            "passed": True,
            "protocol": pin(HERE / "verification_protocol.json"),
            "result": pin(HERE / "result.json"),
            "source_gate_replay_calls": source_checks,
            "counts": dict(counts),
            "positive_noop_controls": controls,
            "floors": floor_records,
            "all_selected_changes": changes,
            "images": images,
            "descriptive_synthetic_CAL_metrics": metrics,
            "CPU_seconds": time.process_time() - start,
            "new_constructor_model_TEST_network_GPU_calls": 0,
            "scientific_limits": "Development decoder construction. Unknown TRAIN support is not invalid chemistry. No independent realism, route reuse, likelihood inheritance, holdout/seed uncertainty or paper promotion.",
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "changes": len(changes),
                "counts": dict(counts),
                "CPU_seconds": time.process_time() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
