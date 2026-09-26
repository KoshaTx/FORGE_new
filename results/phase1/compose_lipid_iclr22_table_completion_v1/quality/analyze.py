"""Complete descriptive quality statistics without sampling, selection, or new chemistry."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import rdBase
from sklearn.neighbors import NearestNeighbors

from forge.core.hashing import sha256_file
from forge.model.common_lipid_realism import DESCRIPTOR_NAMES, fit_robust_descriptor_scale
from forge.model.compose_lipid_quality import (
    canonical_molecule,
    descriptor_comparison,
    ring_systems,
)
from results.phase1.compose_lipid_iclr22_research_v1.calibration_distribution.run import vectors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R = ROOT / "results/phase1/compose_lipid_iclr22_research_v1"
PATHS = {
    "selected": R / "quality/all22_context_preserving/result.json",
    "current_cohort": R / "parallel_completion_v1/routes/recount_closeout_v3/result.json",
    "support": ROOT
    / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/d1_candidate_diagnostics.json",
    "gates": R / "quality/all22_original_reselection/gates.json",
    "ring_recovery": R / "evidence_completion_v1/quality/v2/result.json",
    "ring_verification": R / "evidence_completion_v1/quality/v2/verification.json",
    "distribution_protocol": R / "qualified_CAL_sensitivity/protocol.json",
    "distribution_result": R / "qualified_CAL_sensitivity/result.json",
    "CAL": R / "routes/cal_l1_qualification/exact_source_reference.json",
    "baseline": ROOT
    / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/d1_support_aware_selected.json",
    "TRAIN_reference": ROOT / "results/phase1/compose_lipid_quality_v1/reference.json",
    "metrics": ROOT / "forge/model/compose_lipid_quality.py",
    "descriptors": ROOT / "forge/model/common_lipid_realism.py",
    "regions": ROOT / "forge/model/lipid_context.py",
    "vector_helper": R / "calibration_distribution/run.py",
    "ring_metric": ROOT / "forge/model/compose_lipid_structural_audit.py",
    "gate_metric": ROOT / "results/phase1/compose_lipid_structure_repair_v1/evaluation/gates.py",
    "producer": Path(__file__),
}


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def write(path, value):
    if path.exists():
        raise FileExistsError(f"Preserve completed artifact: {path}")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def verify_pin(expected):
    assert pin(ROOT / expected["path"]) == expected, expected["path"]


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected nonempty finite one-dimensional observations")
    q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return dict(
        n=len(a),
        mean=float(a.mean()),
        sample_SD=float(a.std(ddof=1)) if len(a) > 1 else None,
        minimum=float(a.min()),
        maximum=float(a.max()),
        p05=float(q[0]),
        q25=float(q[1]),
        median=float(q[2]),
        q75=float(q[3]),
        p95=float(q[4]),
        IQR=float(q[3] - q[1]),
    )


def ring_status(ring):
    allocation = "fail" if ring["cycle_allocation_mismatch"] else "pass"
    cross = "fail" if ring["unexpected_cross_origin_edges"] else "pass"
    if ring["variable_cycle_size_not_applicable"]:
        size = "not_applicable"
    elif (
        not ring["comparison_to_sampled_sizes_qualified"]
        or ring["fundamental_size_mismatch"] is None
    ):
        size = "unassessed"
    else:
        size = "fail" if ring["fundamental_size_mismatch"] else "pass"
    combined = (
        "fail"
        if allocation == "fail" or cross == "fail"
        else "pass" if size == "not_applicable" else "abstain" if size == "unassessed" else size
    )
    return dict(
        role_cycle_allocation=allocation,
        cross_origin_edges=cross,
        ring_size=size,
        combined=combined,
        tree_basis=ring["tree_basis"],
    )


def descriptor_manifold(generated, reference, scale, *, neighbors=5):
    """Historical Euclidean reference-ball estimator on explicitly selected matrices.

    This is only the geometric estimator, not independent-reference admission or
    the old 96-atom support policy. All current declared-support observations remain.
    """
    if len(reference) <= neighbors:
        return {"status": "unassessed_insufficient_reference", "reference_n": len(reference)}
    r, g = scale.transform(reference), scale.transform(generated)
    radii = (
        NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
        .fit(r)
        .kneighbors(r)[0][:, -1]
    )
    distance = np.linalg.norm(g[:, None, :] - r[None, :, :], axis=2)
    contained = distance <= radii[None, :] + 1e-12
    members, covered = contained.any(axis=1), contained.any(axis=0)
    return dict(
        status="assessed",
        generated_n=len(g),
        reference_n=len(r),
        member_observations=int(members.sum()),
        precision=float(members.mean()),
        covered_reference=int(covered.sum()),
        coverage=float(covered.mean()),
        members=members.tolist(),
        reference_covered=covered.tolist(),
        reference_radii=radii.tolist(),
        nearest_distance=distance.min(axis=1).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        protocol = dict(
            schema="forge.iclr22.table_quality_completion_protocol.v1",
            inputs={k: pin(p) for k, p in PATHS.items()},
            seed=0,
            CPU_seconds_cap=90,
            population="Unchanged 1408 selected requests; same saved and TRAIN-control arms; exact 1821 synthetic CAL reference.",
            atom_coverage="Per-product mean of fraction of atoms with TRAIN-observed complete labeled radius-one environment; pooled atom fraction separately.",
            ring_support="All connected ring-bond systems present in family TRAIN support. Ring-free products reported separately, never as applicable passes.",
            conditions="Saved actual role-cycle and qualified fundamental-size predicates, with certified15 basis replacements; no new chemistry rules.",
            descriptor_statistics="Connected request observations, duplicates retained; SD ddof1; NumPy linear quantiles; 5/25/50/75/95th percentiles. Dispersion not uncertainty across training seeds.",
            descriptor_neighborhood="Historical Euclidean 5-neighbor CAL reference balls, +1e-12 tolerance; per-family scale fitted to the same first64 TRAIN controls using median/IQR, then populationSD, then1. All same connected observations; no historical96atom truncation.",
            descriptor_neighborhood_limits="New explicitly synthetic-CAL development calculation. Reference is reaction-enumerated and predominantly TRAIN-component-shared. Neither independent empirical realism nor a chemical-quality probability. No grouped C2ST or final-heldout admission.",
            reuse_validation="Recompute all 22x3x24 existing means, reference scales and Wasserstein values; require exact equality.",
            model_calls=0,
            chemistry_executor_calls=0,
            TEST_structures_read=0,
            source_policies_changed=False,
        )
        write(HERE / "protocol.json", protocol)
        print("Prepared pinned protocol")
        return
    protocol = read(HERE / "protocol.json")
    for expected in protocol["inputs"].values():
        verify_pin(expected)
    start = time.process_time()
    selected = read(PATHS["selected"])["by_family"]
    latest = {r["index"]: r for r in read(PATHS["current_cohort"])["rows"]}
    support_doc = read(PATHS["support"])
    for key in ["reference", "quality_module"]:
        verify_pin(support_doc["inputs"][key])
    support = {(r["index"], r["ordinal"]): r for r in support_doc["candidates"]}
    gates = {(r["index"], r["ordinal"]): r["assessment"] for r in read(PATHS["gates"])["attempts"]}
    recovery = read(PATHS["ring_recovery"])
    assert recovery["admitted"] == 15 and recovery["unresolved"] == 0
    for target in recovery["targets"]:
        assert target["admitted"] and not target["ambiguity"] and not target["errors"]
        gates[target["index"], target["ordinal"]] = target["recovered"][0]["assessment"]
    rows = []
    for family, data in sorted(selected.items()):
        for s in data["selections"]:
            key = (s["request"], s["ordinal"])
            d = support[key]
            a = gates[key]
            current = latest[s["request"]]
            assert (
                d["candidate"]["smiles"]
                == a["chemical"]["canonical_smiles"]
                == current["smiles"]
                == s["smiles"]
            )
            assert current["ordinal"] == s["ordinal"] and current["family"] == family
            ps = d["diagnostic"]["product_support"]
            assert ps["status"] == "assessed"
            canonical, mol = canonical_molecule(s["smiles"])
            assert canonical == s["smiles"]
            systems = ring_systems(mol)
            unknown = ps["unknown_ring_systems"]
            assert len(unknown) <= len(systems) and all(x in systems for x in unknown)
            atom_n = mol.GetNumAtoms()
            unknown_atoms = len(ps["unknown_atom_environments"])
            assert ps["atom_environment_coverage"] == 1 - unknown_atoms / atom_n
            rs = ring_status(a["ring"])
            assert rs["combined"] == a["status_by_axis"]["ring"]
            assert a["qualified_design_pass"] == current["limited_design_pass"]
            rows.append(
                dict(
                    family=family,
                    request=s["request"],
                    ordinal=s["ordinal"],
                    smiles=canonical,
                    atoms=atom_n,
                    unknown_atoms=unknown_atoms,
                    atom_environment_coverage=ps["atom_environment_coverage"],
                    ring_systems=len(systems),
                    unknown_ring_systems=unknown,
                    ring_support=(
                        "not_applicable" if not systems else "unknown" if unknown else "supported"
                    ),
                    reference_unique_constitutions=ps["reference_unique_constitutions"],
                    conditions=rs,
                )
            )
    assert len(rows) == len(latest) == 1408 and len({r["request"] for r in rows}) == 1408

    def summary(items):
        return dict(
            requests=len(items),
            atom_environment_coverage=statistics([r["atom_environment_coverage"] for r in items]),
            atom_observations=sum(r["atoms"] for r in items),
            unknown_atom_observations=sum(r["unknown_atoms"] for r in items),
            pooled_atom_coverage=1
            - sum(r["unknown_atoms"] for r in items) / sum(r["atoms"] for r in items),
            ring_system_observations=sum(r["ring_systems"] for r in items),
            unknown_ring_system_observations=sum(len(r["unknown_ring_systems"]) for r in items),
            ring_TRAIN_support=dict(Counter(r["ring_support"] for r in items)),
            conditions={
                k: dict(Counter(r["conditions"][k] for r in items)) for k in rows[0]["conditions"]
            },
        )

    quality = dict(
        totals=summary(rows),
        by_family={f: summary([r for r in rows if r["family"] == f]) for f in sorted(selected)},
    )
    assert quality["totals"]["conditions"]["combined"] == {"pass": 1339, "fail": 69}
    write(HERE / "quality_rows.json", rows)
    write(
        HERE / "quality_result.json",
        dict(
            complete=True,
            protocol=pin(HERE / "protocol.json"),
            rows=pin(HERE / "quality_rows.json"),
            **quality,
            CPU_seconds=time.process_time() - start,
        ),
    )
    prior_protocol = read(PATHS["distribution_protocol"])
    for name in [
        "CAL",
        "selected",
        "baseline",
        "TRAIN_reference",
        "metrics",
        "descriptors",
        "regions",
        "vector_helper",
    ]:
        verify_pin(prior_protocol["inputs"][name])
    old = {r["family"]: r for r in read(PATHS["distribution_result"])["rows"]}
    cal = defaultdict(list)
    saved = defaultdict(list)
    for r in read(PATHS["CAL"])["rows"]:
        assert (
            r["full_source_L1_assessed"] and r["source_L1_status"] == "exact_original_source_tuple"
        )
        cal[r["family"]].append(r["canonical_smiles"])
    for r in read(PATHS["baseline"])["attempts"]:
        saved[r["family"]].append(r["selected_smiles"])
    train = read(PATHS["TRAIN_reference"])["product_controls"]
    descriptor_rows = []
    for family in sorted(selected):
        if time.process_time() - start > protocol["CPU_seconds_cap"]:
            raise TimeoutError("Preserve partial diagnostics; no complete result")
        panels = dict(
            reference=cal[family],
            saved=saved[family],
            context_preserving=[r["smiles"] for r in selected[family]["selections"]],
            TRAIN_control=train[family][:64],
        )
        matrices = {k: vectors(s) for k, s in panels.items()}
        ref = matrices["reference"][1]
        scale = fit_robust_descriptor_scale(matrices["TRAIN_control"][1])
        arms = {}
        for arm, (smiles, matrix, counts) in matrices.items():
            assert counts["connected"] == counts["requests"] == len(matrix)
            arms[arm] = dict(
                counts=counts,
                descriptors={
                    name: statistics(matrix[:, j]) for j, name in enumerate(DESCRIPTOR_NAMES)
                },
            )
            if arm != "reference":
                comparison = descriptor_comparison(matrix, ref, DESCRIPTOR_NAMES)
                assert comparison == old[family]["arms"][arm]["descriptors"], (family, arm)
                manifold = descriptor_manifold(matrix, ref, scale)
                if manifold["status"] == "assessed":
                    unique_members = {
                        s: member for s, member in zip(smiles, manifold["members"], strict=True)
                    }
                    manifold["unique_n"] = len(unique_members)
                    manifold["unique_members"] = sum(unique_members.values())
                    manifold["precision_among_unique"] = sum(unique_members.values()) / len(
                        unique_members
                    )
                arms[arm]["descriptor_neighborhood"] = manifold
        descriptor_rows.append(
            dict(
                family=family,
                arms=arms,
                scaler=dict(
                    center=scale.center.tolist(),
                    scale=scale.scale.tolist(),
                    population="first64 frozen TRAIN controls of same family",
                ),
            )
        )
    assert sum(r["arms"]["reference"]["counts"]["requests"] for r in descriptor_rows) == 1821
    result = dict(
        schema="forge.iclr22.table_quality_completion_result.v1",
        complete=True,
        protocol=pin(HERE / "protocol.json"),
        quality=quality,
        quality_rows=pin(HERE / "quality_rows.json"),
        descriptors=descriptor_rows,
        existing_descriptor_comparisons_exactly_reproduced=22 * 3 * 24,
        CPU_seconds=time.process_time() - start,
        rdkit_version=rdBase.rdkitVersion,
        numpy_version=np.__version__,
        final_evidence_admitted=False,
        unmeasured=[
            "Independent source-disjoint empirical realism",
            "Source-grouped independent C2ST",
            "Across-model-seed uncertainty",
            "Biological efficacy",
        ],
    )
    write(HERE / "result.json", result)
    print(
        json.dumps(
            dict(complete=True, quality=quality["totals"], CPU_seconds=result["CPU_seconds"])
        )
    )


if __name__ == "__main__":
    main()
