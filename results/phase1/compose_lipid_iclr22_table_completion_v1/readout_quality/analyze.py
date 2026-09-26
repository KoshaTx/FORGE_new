"""Frozen-estimator descriptive comparison of every saved first-draw readout."""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import rdBase

from forge.model.common_lipid_realism import DESCRIPTOR_NAMES, fit_robust_descriptor_scale
from forge.model.compose_lipid_quality import descriptor_comparison, fingerprint_distribution
from results.phase1.compose_lipid_iclr22_research_v1.calibration_distribution.run import vectors
from results.phase1.compose_lipid_iclr22_table_completion_v1.quality.analyze import (
    PATHS as QUALITY_PATHS,
)
from results.phase1.compose_lipid_iclr22_table_completion_v1.quality.analyze import (
    descriptor_manifold,
    pin,
    read,
    statistics,
    verify_pin,
    write,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = HERE.parent / "assembly/readout_recovery_v4"
ARMS = (
    "conditioned_true_endpoint",
    "conditioned_terminal_argmax",
    "cyclic_true_endpoint",
    "cyclic_terminal_argmax",
    "selected",
)


def arm_metrics(smiles, reference, reference_matrix, scale):
    canonical, matrix, counts = vectors(smiles)
    fp = fingerprint_distribution(canonical, reference, requests=len(smiles), neighbors=5)
    desc = descriptor_comparison(matrix, reference_matrix, DESCRIPTOR_NAMES)
    if len(matrix):
        neighborhood = descriptor_manifold(matrix, reference_matrix, scale)
        if neighborhood["status"] == "assessed":
            neighborhood["member_observations_per_request"] = neighborhood[
                "member_observations"
            ] / len(smiles)
            unique = dict(zip(canonical, neighborhood["members"], strict=True))
            neighborhood.update(
                unique_n=len(unique),
                unique_members=sum(unique.values()),
                precision_among_unique=sum(unique.values()) / len(unique),
            )
    else:
        neighborhood = dict(
            status="unassessed_no_connected_outputs",
            generated_n=0,
            reference_n=len(reference_matrix),
            member_observations=0,
            member_observations_per_request=0.0,
        )
    if fp["status"] == "assessed_development_distribution_only":
        members = fp["fingerprint_precision_among_unique"] * fp["unique_generated"]
        assert np.isclose(members, round(members), atol=1e-10, rtol=0)
        fp["unique_member_count"] = round(members)
    return dict(
        counts=counts,
        fingerprints=fp,
        descriptors=desc,
        descriptor_neighborhood=neighborhood,
        descriptor_statistics=(
            {name: statistics(matrix[:, j]) for j, name in enumerate(DESCRIPTOR_NAMES)}
            if len(matrix)
            else {}
        ),
    )


def prepare():
    paths = {
        k: QUALITY_PATHS[k]
        for k in [
            "CAL",
            "selected",
            "current_cohort",
            "TRAIN_reference",
            "metrics",
            "descriptors",
            "regions",
            "vector_helper",
            "distribution_protocol",
            "distribution_result",
        ]
    }
    paths.update(
        producer=Path(__file__),
        reused_descriptor_producer=HERE.parent / "quality/analyze.py",
        reused_descriptor_result=HERE.parent / "quality/result.json",
        replay_protocol=REPLAY / "protocol.json",
        draw0_receipt=REPLAY / "draw0_checkpoint.json",
        conditioned_admission=REPLAY / "conditioned_root_admission.json",
        cyclic_admission=REPLAY / "cyclic_root_admission.json",
    )
    shards = {}
    for arm in ["conditioned", "cyclic"]:
        files = sorted((REPLAY / f"{arm}_v1/shards").glob("draw-0-*.json"))
        assert len(files) == 176
        shards[arm] = [pin(p) for p in files]
    write(
        HERE / "protocol.json",
        dict(
            schema="forge.iclr22.saved_first_draw_quality_protocol.v1",
            inputs={k: pin(v) for k, v in paths.items()},
            shards=shards,
            arms=list(ARMS),
            population="Identical1408 request IDs; 64 per22families; draw0 only; selected arm uses unchanged production selections over its original5draws/rules.",
            synthetic_CAL_reference_n=1821,
            CPU_seconds_cap=180,
            seed=0,
            fingerprint="Existing ECFP4 radius2 2048bit five-nearest-reference Tanimoto balls; unique connected constitutions; conditional precision and all-request distinct membership both retained.",
            descriptor_W1="Existing per-family CAL IQR, fallback max(CAL populationSD,1), all connected observations with duplicates retained.",
            descriptor_neighborhood="Existing Euclidean five-nearest-reference balls (+1e-12), exact frozen first64TRAIN controls median/IQR then populationSD then1 scale.",
            invalids="Every attempt retained in ledger and1408/64 yields; invalids contribute no manifold members. Conditional metrics explicitly report connected/unique denominators.",
            verification="Selected fingerprints and all24descriptor comparisons equal frozen qualified_CAL_sensitivity; selected descriptor-neighborhood output equals table_completion quality result.",
            context_flags="No new source-layout chemistry predicates or quality acceptance policy. This run measures saved validity/exactL1 plus distributional geometry only.",
            scientific_limits=[
                "Synthetic reaction-enumerated CAL predominantly shares TRAIN components.",
                "No empirical realism, independent heldout, seed uncertainty, efficacy or chemical-quality probability.",
                "Selected uses additional draws/rules/selection; contrast is descriptive, not an equal-budget causal model gain.",
                "Cyclic inference permutes program context; this is not a separately trained cyclic model.",
            ],
            model_calls=0,
            chemistry_executor_calls=0,
            TEST_structures_read=0,
            new_selections=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    protocol = read(HERE / "protocol.json")
    start = time.process_time()

    def timeout(*_):
        raise TimeoutError("No complete comparison admitted beyond frozen180CPU seconds")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, protocol["CPU_seconds_cap"])
    for expected in protocol["inputs"].values():
        verify_pin(expected)
    current = {r["index"]: r for r in read(QUALITY_PATHS["current_cohort"])["rows"]}
    assert len(current) == 1408
    observations = defaultdict(dict)
    for branch, shards in protocol["shards"].items():
        seen = set()
        for expected in shards:
            verify_pin(expected)
            shard = read(ROOT / expected["path"])
            assert shard["schedule"]["draw"] == 0 and len(shard["rows"]) == 8
            for r in shard["rows"]:
                i = r["index"]
                assert i not in seen and r["draw"] == 0 and r["family"] == current[i]["family"]
                seen.add(i)
                assert set(r["arms"]) == {f"{branch}_true_endpoint", f"{branch}_terminal_argmax"}
                for arm, value in r["arms"].items():
                    observations[i][arm] = dict(
                        smiles=value["smiles"],
                        exact_L1=value["check"]["exact"],
                        reason=value["reason"],
                    )
        assert seen == set(current)
    for i, r in current.items():
        observations[i]["selected"] = dict(smiles=r["smiles"], exact_L1=r["exact_L1"], reason=None)
    ledger = [
        dict(index=i, family=current[i]["family"], draw=0, arms=observations[i])
        for i in sorted(current)
    ]
    families = sorted({r["family"] for r in ledger})
    assert len(families) == 22 and Counter(r["family"] for r in ledger) == Counter(
        {f: 64 for f in families}
    )
    cal = defaultdict(list)
    for r in read(QUALITY_PATHS["CAL"])["rows"]:
        assert (
            r["full_source_L1_assessed"] and r["source_L1_status"] == "exact_original_source_tuple"
        )
        cal[r["family"]].append(r["canonical_smiles"])
    assert sum(map(len, cal.values())) == 1821
    train = read(QUALITY_PATHS["TRAIN_reference"])["product_controls"]
    old_fp = {r["family"]: r for r in read(QUALITY_PATHS["distribution_result"])["rows"]}
    old_desc = {r["family"]: r for r in read(HERE.parent / "quality/result.json")["descriptors"]}
    rows = []
    for family in families:
        subset = [r for r in ledger if r["family"] == family]
        reference, rm, rc = vectors(cal[family])
        _, tm, _ = vectors(train[family][:64])
        scale = fit_robust_descriptor_scale(tm)
        arms = {}
        for arm in ARMS:
            panel = [r["arms"][arm]["smiles"] for r in subset]
            a = arm_metrics(panel, reference, rm, scale)
            a["exact_L1"] = sum(r["arms"][arm]["exact_L1"] for r in subset)
            arms[arm] = a
        selected = arms["selected"]
        expected = old_fp[family]["arms"]["context_preserving"]
        assert {
            k: v for k, v in selected["fingerprints"].items() if k != "unique_member_count"
        } == expected["fingerprints"]
        assert selected["descriptors"] == expected["descriptors"]
        assert {
            k: v
            for k, v in selected["descriptor_neighborhood"].items()
            if k != "member_observations_per_request"
        } == old_desc[family]["arms"]["context_preserving"]["descriptor_neighborhood"]
        rows.append(
            dict(
                family=family,
                reference_counts=rc,
                arms=arms,
                scale=dict(center=scale.center.tolist(), scale=scale.scale.tolist()),
            )
        )
    summaries = {}
    for arm in ARMS:
        aa = [r["arms"][arm] for r in rows]
        summary = dict(
            requests=1408,
            valid=sum(a["counts"]["valid"] for a in aa),
            connected=sum(a["counts"]["connected"] for a in aa),
            unique_connected_within_family_sum=sum(a["counts"]["unique_connected"] for a in aa),
            exact_L1=sum(a["exact_L1"] for a in aa),
            fingerprint_unique_members=sum(
                a["fingerprints"].get("unique_member_count", 0) for a in aa
            ),
            descriptor_member_observations=sum(
                a["descriptor_neighborhood"]["member_observations"] for a in aa
            ),
        )
        for prefix, key, fields in [
            (
                "fingerprint",
                "fingerprints",
                (
                    "fingerprint_precision_among_unique",
                    "fingerprint_coverage",
                    "distinct_in_reference_manifold_per_request",
                    "nearest_reference_tanimoto_mean",
                    "mean_pairwise_tanimoto_distance",
                ),
            ),
            (
                "descriptor",
                "descriptor_neighborhood",
                ("precision", "coverage", "member_observations_per_request"),
            ),
        ]:
            for metric in fields:
                values = [a[key][metric] for a in aa if a[key].get(metric) is not None]
                summary[f"macro_{prefix}_{metric}"] = dict(
                    mean=float(np.mean(values)) if values else None,
                    assessed_families=len(values),
                    total_families=22,
                )
        summary["macro_descriptor_W1"] = {
            name: dict(
                mean=float(np.mean(vals)) if vals else None,
                assessed_families=len(vals),
                total_families=22,
            )
            for name in DESCRIPTOR_NAMES
            for vals in [
                [
                    a["descriptors"]["descriptors"][name]["normalized_wasserstein"]
                    for a in aa
                    if "descriptors" in a["descriptors"]
                ]
            ]
        }
        summaries[arm] = summary
    write(HERE / "observations.json", ledger)
    result = dict(
        schema="forge.iclr22.saved_first_draw_quality_result.v1",
        complete=True,
        protocol=pin(HERE / "protocol.json"),
        observations=pin(HERE / "observations.json"),
        families=rows,
        summaries=summaries,
        CPU_seconds=time.process_time() - start,
        selected_frozen_estimator_reproduction_exact=True,
        rdkit_version=rdBase.rdkitVersion,
        numpy_version=np.__version__,
        independent_realism_admitted=False,
        final_heldout_admitted=False,
    )
    write(HERE / "result.json", result)
    signal.setitimer(signal.ITIMER_PROF, 0)
    print(json.dumps(dict(complete=True, CPU_seconds=result["CPU_seconds"], summaries=summaries)))


if __name__ == "__main__":
    main()
