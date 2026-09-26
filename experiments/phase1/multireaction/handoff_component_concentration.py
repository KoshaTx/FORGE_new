"""Bind saved exact L1 witnesses, global component counts, and separate route joins."""

from __future__ import annotations

import difflib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy
import scipy

from experiments.phase1.multireaction.verify_component_concentration import pin, read, write
from forge.model.compose_lipid_component_diversity import accepted_components
from forge.model.compose_lipid_quality import canonical_molecule

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
N = WORKTREE / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1"
OUT = N / "e_diversity"
RUN = OUT / "concentration_v1"


def component_totals(rows):
    counts = Counter(c["smiles"] for r in rows for c in r["components"])
    novel = [c["smiles"] for r in rows for c in r["components"] if c["global_train_novel"]]
    n = sum(counts.values())
    return {
        "observations": n,
        "unique": len(counts),
        "sum_squared": sum(v * v for v in counts.values()),
        "shannon_effective": math.exp(
            math.log(n) - sum(v * math.log(v) for v in counts.values()) / n
        ),
        "simpson_effective": n * n / sum(v * v for v in counts.values()),
        "global_TRAIN_novel_observations": len(novel),
        "global_TRAIN_novel_unique": len(set(novel)),
        "identities": dict(counts),
    }


def component_floor(old, new):
    assert old["observations"] == new["observations"]
    assert new["unique"] >= old["unique"]
    assert new["sum_squared"] <= old["sum_squared"]
    assert math.prod(v**v for v in new["identities"].values()) <= math.prod(
        v**v for v in old["identities"].values()
    )
    assert new["global_TRAIN_novel_observations"] >= old["global_TRAIN_novel_observations"]
    assert new["global_TRAIN_novel_unique"] >= old["global_TRAIN_novel_unique"]


def main():
    start = time.process_time()
    protocol = read(RUN / "protocol.json")
    compact = (
        WORKTREE / "results/phase1/compose_lipid_quality_confirmation_v1/compact_attempts.json"
    )
    original_protocol = (
        WORKTREE / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/protocol.json"
    )
    recorded = read(original_protocol)["inputs"]["ledger"]
    assert pin(compact)["sha256"] == recorded["sha256"]
    attempts = {r["index"]: r for r in read(compact)}
    changes = read(RUN / "changed_products.json")["records"]
    witnesses = []
    for rec in changes:
        c = rec["candidate"]
        r = attempts[c["request"]]
        branch = r["branches"]["d1"]
        proposal = [branch["raw"], *branch.get("proposals", [])][c["ordinal"]]
        assert r["family"] == rec["family"]
        assert canonical_molecule(proposal["smiles"])[0] == c["smiles"]
        parts = accepted_components(proposal["check"])
        assert {k: canonical_molecule(v)[0] for k, v in parts.items()} == {
            c["role"]: c["smiles"] for c in c["components"]
        }
        witnesses.append(
            {
                "index": c["request"],
                "family": r["family"],
                "ordinal": c["ordinal"],
                "canonical_smiles": c["smiles"],
                "source_locator": f"row[index={c['request']}].branches.d1.[raw,*proposals][{c['ordinal']}]",
                "proposal": proposal,
                "accepted_components": parts,
                "canonical_components": {p["role"]: p["smiles"] for p in c["components"]},
                "current_assessment": rec["assessment"],
            }
        )
    write(
        RUN / "source_witnesses.json",
        {
            "schema": "forge.saved_exact_source_witness_join.v1",
            "requests": 245,
            "inputs": {
                "compact_attempts": pin(compact),
                "original_assessment_protocol": pin(original_protocol),
                "changed_products": pin(RUN / "changed_products.json"),
                "extractor": pin(Path(__file__)),
                "canonicalizer": pin(WORKTREE / "forge/model/compose_lipid_quality.py"),
                "accepted_component_reader": pin(
                    WORKTREE / "forge/model/compose_lipid_component_diversity.py"
                ),
            },
            "records": witnesses,
            "scope": "Exact saved source assertions reauthenticated by identity; no source executor/route search rerun.",
        },
    )
    original = read(Path(protocol["inputs"]["current"]["path"]))["by_family"]
    result = read(RUN / "result.json")
    whole = {"current": [], "ordinal_control": [], "component_concentration": []}
    families = {}
    for family, p in result["family_results"].items():
        f = read(Path(p["path"]))
        rows = {
            "current": original[family]["selections"],
            **{
                a: f["arms"][a]["selection"] for a in ("ordinal_control", "component_concentration")
            },
        }
        c = {a: component_totals(v) for a, v in rows.items()}
        component_floor(c["current"], c["component_concentration"])
        families[family] = c
        for a, v in rows.items():
            whole[a].extend(v)
    totals = {a: component_totals(v) for a, v in whole.items()}
    component_floor(totals["current"], totals["component_concentration"])
    write(
        RUN / "global_component_verification.json",
        {
            "passed": True,
            "inputs": {
                "producer": pin(Path(__file__)),
                "protocol": pin(RUN / "protocol.json"),
                "result": pin(RUN / "result.json"),
            },
            "definition": "All accepted role observations pooled by canonical component identity irrespective of role; 3723 observations. Distinct from sum of separate role S2 objective. Novelty is exact existing full-TRAIN membership flag.",
            "pooled": totals,
            "by_family": families,
            "all_concentration_component_floors_passed": True,
        },
    )
    # Separate joins on the unchanged D cohort and B+D cohort. These do not assess E routing.
    quality_b = read(N / "b_quality/closure_v1/candidate_cohort_1408.json")
    bg = {r["index"]: r for r in quality_b["rows"]}
    gates = {
        (r["index"], r["ordinal"]): r["assessment"]
        for r in read(Path(protocol["inputs"]["common_gates"]["path"]))["attempts"]
    }
    original_quality = {
        r["request"]: gates[r["request"], r["ordinal"]]
        for v in original.values()
        for r in v["selections"]
    }
    joint = {}
    for name, path in [
        ("D_only", N / "d_routes/full_pair_v2/treatment/products.json"),
        ("B_plus_D", N / "d_routes/joint_b_d_v1/assessment/products.json"),
    ]:
        if not path.exists():
            continue
        products = read(path)["products"]
        assert len(products) == 1408
        counts = Counter()
        by_family = {}
        for r in products:
            i = r["index"]
            q = original_quality[i]
            design = (
                q["qualified_design_pass"] if name == "D_only" else bg[i]["limited_design_pass"]
            )
            flags = (
                [x["code"] for x in q["chemical"]["flags"] if x["tier"] == "context_required"]
                if name == "D_only"
                else bg[i]["context_codes"]
            )
            record = {
                "requests": 1,
                "makeable": bool(r["combined_primary"]),
                "design": bool(design),
                "makeable_and_design": bool(r["combined_primary"] and design),
                "makeable_design_no_context": bool(r["combined_primary"] and design and not flags),
                "L2_ready": bool(r["L2_ready"]),
                "strict": bool(r["strict_secondary"]),
            }
            counts.update(record)
            by_family.setdefault(r["family"], Counter()).update(record)
        joint[name] = {
            "route_products": pin(path),
            "totals": dict(counts),
            "by_family": {k: dict(v) for k, v in by_family.items()},
        }
    write(
        RUN / "separate_D_quality_joins.json",
        {
            "inputs": {
                "source": pin(Path(__file__)),
                "original_quality": protocol["inputs"]["common_gates"],
                "original_selection": protocol["inputs"]["current"],
                "B_quality_cohort": pin(N / "b_quality/closure_v1/candidate_cohort_1408.json"),
            },
            "cohorts": joint,
            "scope": "Exact original request joins; E routing not inferred; no merged B+E result.",
        },
    )
    source_files = [
        WORKTREE / "experiments/phase1/multireaction" / n
        for n in [
            "component_concentration_selector.py",
            "component_concentration_helpers.py",
            "component_concentration_experiment.py",
            "verify_component_concentration.py",
            "handoff_component_concentration.py",
        ]
    ] + [WORKTREE / "tests/test_component_concentration_selector.py"]
    patch = []
    for path in source_files:
        patch.extend(
            difflib.unified_diff(
                [],
                path.read_text().splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="b/" + str(path.relative_to(WORKTREE)),
            )
        )
    (OUT / "candidate.patch").write_text("".join(patch))
    write(
        OUT / "candidate_files.json",
        {
            "files": [pin(p) for p in source_files],
            "patch": pin(OUT / "candidate.patch"),
            "scope": "Six new files in isolated E worktree. No root source edits or production behavior change.",
        },
    )
    write(
        OUT / "result.json",
        {
            "complete": True,
            "qualified_candidate": True,
            "official_cohort_promotion": False,
            "experiment": pin(RUN / "result.json"),
            "independent_verification": pin(RUN / "independent_verification.json"),
            "global_component_verification": pin(RUN / "global_component_verification.json"),
            "source_witnesses": pin(RUN / "source_witnesses.json"),
            "changed_products": pin(RUN / "changed_products.json"),
            "candidate_cohort": pin(RUN / "candidate_cohort_1408.json"),
            "separate_D_joins": pin(RUN / "separate_D_quality_joins.json"),
            "candidate_files": pin(OUT / "candidate_files.json"),
            "tests": pin(OUT / "tests_v1.xml"),
            "environment": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
            },
            "new_model_source_executor_route_search_TEST_network_GPU_calls": 0,
            "handoff_cpu_seconds": time.process_time() - start,
            "remaining": "Independent root review; E routing and distributional quality unknown; no automatic B+E composition.",
        },
    )
    print(
        json.dumps(
            {
                "result": pin(OUT / "result.json"),
                "source_witnesses": pin(RUN / "source_witnesses.json"),
                "joint": {k: v["totals"] for k, v in joint.items()},
                "cpu_seconds": time.process_time() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
