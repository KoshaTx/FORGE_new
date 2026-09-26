"""Compare unchanged generated and TRAIN-control panels with guarded synthetic CAL graphs."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from forge.model.common_lipid_realism import DESCRIPTOR_NAMES, _descriptor_vector
from forge.model.compose_lipid_quality import (
    canonical_molecule,
    descriptor_comparison,
    fingerprint_distribution,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def pin(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


def write(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def vectors(smiles):
    accepted = []
    matrix = []
    counts = {"requests": len(smiles), "valid": 0, "connected": 0}
    for s in smiles:
        parsed = canonical_molecule(s)
        if parsed is None:
            continue
        counts["valid"] += 1
        if len(Chem.GetMolFrags(parsed[1])) != 1:
            continue
        counts["connected"] += 1
        accepted.append(parsed[0])
        matrix.append(_descriptor_vector(parsed[1]))
    counts["unique_connected"] = len(set(accepted))
    return accepted, np.asarray(matrix, dtype=np.float64), counts


def main():
    paths = {
        "CAL": HERE.parent / "routes/cal_reference.json",
        "selected": ROOT
        / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/d1_support_aware_selected.json",
        "TRAIN_reference": ROOT / "results/phase1/compose_lipid_quality_v1/reference.json",
        "policy": ROOT / "results/phase1/compose_lipid_quality_v1/policy.json",
        "metrics": ROOT / "forge/model/compose_lipid_quality.py",
        "descriptors": ROOT / "forge/model/common_lipid_realism.py",
        "region_heuristics": ROOT / "forge/model/lipid_context.py",
        "implementation": Path(__file__),
    }
    pins = {key: pin(path) for key, path in paths.items()}
    protocol = {
        "schema_version": "forge.synthetic_CAL_distribution_protocol.v1",
        "inputs": pins,
        "population": "All2059guard-admitted product-disjoint synthetic CAL rows; all1408unchanged fresh generated requests; first64frozen TRAIN controls per family.",
        "claim": "Development structural distribution comparison, not empirical realism or source/component-disjoint generalization.",
        "hypothesis": "Generated-vs-reference differences can be separated from mismatched broad empirical references using the same family-matched CAL comparison for generated and TRAIN controls.",
        "controls": "Same CAL reference and estimator for both panels; no new model calls, proposal generation, selection or threshold tuning.",
        "fingerprint": {
            "radius": 2,
            "bits": 2048,
            "neighbors": 5,
            "identity_policy": "unique constitutional molecules",
        },
        "descriptors": list(DESCRIPTOR_NAMES),
        "descriptor_identity_policy": "All connected observations, retaining duplicates; normalize against CAL IQR with existing SD/min1fallback.",
        "seed": 0,
        "threads": 1,
        "maximum_wall_seconds": 180,
        "training_seed_count": 1,
        "uncertainty": "No source-group or independent-model interval claimed; point diagnostics only.",
        "missingness": "Keep all requests in yield denominators; report invalid/disconnected and insufficient reference support explicitly.",
        "generated_and_control_request_count_each": 1408,
        "read_TEST_molecules": False,
        "new_policy_or_model_selection": False,
    }
    protocol_path = HERE / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise FileExistsError("Frozen protocol changed")
    write(protocol_path, protocol)
    start = time.monotonic()
    cal = json.loads(paths["CAL"].read_text())
    assert cal["TEST_molecules_parsed"] == 0
    reference = json.loads(paths["TRAIN_reference"].read_text())
    selected = json.loads(paths["selected"].read_text())
    by_cal = defaultdict(list)
    by_gen = defaultdict(list)
    for row in cal["rows"]:
        assert (
            row["product_disjoint_reference_eligible"]
            and not row["graph_exclusions"]
            and not row["empirical_execution_claim"]
        )
        by_cal[row["family"]].append(row["canonical_smiles"])
    for row in selected["attempts"]:
        by_gen[row["family"]].append(row["selected_smiles"])
    assert set(by_gen) == set(by_cal) == set(reference["product_controls"]) and len(by_gen) == 22
    rows = []
    for family in sorted(by_gen):
        if time.monotonic() - start > protocol["maximum_wall_seconds"]:
            raise TimeoutError("Incomplete CAL comparison; no partial population result admitted")
        generated, gv, gcounts = vectors(by_gen[family])
        train, tv, tcounts = vectors(reference["product_controls"][family][:64])
        ref, rv, rcounts = vectors(by_cal[family])
        assert gcounts["requests"] == tcounts["requests"] == 64
        assert rcounts["connected"] == len(by_cal[family])
        arms = {}
        for name, smiles, features, counts in [
            ("generated", generated, gv, gcounts),
            ("TRAIN_control", train, tv, tcounts),
        ]:
            arms[name] = {
                "population": counts,
                "fingerprint": fingerprint_distribution(
                    smiles, ref, requests=counts["requests"], neighbors=5
                ),
                "descriptors": descriptor_comparison(features, rv, DESCRIPTOR_NAMES),
            }
        gfp = arms["generated"]["fingerprint"]
        tfp = arms["TRAIN_control"]["fingerprint"]
        difference = {
            key: gfp[key] - tfp[key]
            for key in (
                "fingerprint_precision_among_unique",
                "fingerprint_coverage",
                "nearest_reference_tanimoto_mean",
            )
            if key in gfp and key in tfp
        }
        descriptor_delta = {
            name: arms["generated"]["descriptors"]["descriptors"][name]["normalized_wasserstein"]
            - arms["TRAIN_control"]["descriptors"]["descriptors"][name]["normalized_wasserstein"]
            for name in DESCRIPTOR_NAMES
        }
        rows.append(
            {
                "family": family,
                "CAL_population": rcounts,
                "arms": arms,
                "generated_minus_TRAIN": difference,
                "normalized_wasserstein_generated_minus_TRAIN": descriptor_delta,
            }
        )
    result = {
        "schema_version": "forge.synthetic_CAL_distribution_result.v1",
        "protocol": pin(protocol_path),
        "inputs": pins,
        "complete": True,
        "families": rows,
        "counts": {
            "families": 22,
            "CAL_requests": sum(len(v) for v in by_cal.values()),
            "generated_requests": 1408,
            "TRAIN_control_requests": 1408,
        },
        "rdkit_version": rdBase.rdkitVersion,
        "seconds": time.monotonic() - start,
        "new_model_calls": 0,
        "new_selections": 0,
        "empirical_realism_claim": False,
        "final_paper_admission": False,
        "limits": [
            "CAL source-enumerated structures are not an empirical lipid-quality oracle and full source-L1 qualification is unassessed.",
            "CAL is constitution-disjoint from TRAIN, but most CAL products share TRAIN components.",
            "Family reference sizes differ; kNN precision/coverage depends on sample size. Both methods use the same reference within each comparison.",
            "Positive controls are not exact-layout matched to generated requests.",
            "One checkpoint with adaptive support-based selection and previously disclosed layout overlap; no independent training-seed uncertainty.",
        ],
    }
    write(HERE / "result.json", result)
    lines = [
        "# Guarded CAL distribution diagnostic",
        "",
        result["protocol"]["path"],
        "",
        "All generated outputs and TRAIN controls are unchanged. References are synthetic CAL structures, not experimentally validated lipids. No new overall quality score or acceptance cutoff is introduced.",
        "",
        "| Family | CAL n | Generated / TRAIN precision | Generated / TRAIN nearest similarity |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        g = row["arms"]["generated"]["fingerprint"]
        t = row["arms"]["TRAIN_control"]["fingerprint"]
        lines.append(
            f"| {row['family']} | {row['CAL_population']['requests']} | {g['fingerprint_precision_among_unique']:.3f} / {t['fingerprint_precision_among_unique']:.3f} | {g['nearest_reference_tanimoto_mean']:.3f} / {t['nearest_reference_tanimoto_mean']:.3f} |"
        )
    lines += [
        "",
        "All24descriptor distributions and their normalized distances are in result.json. Positive controls and generated outputs use the identical family CAL reference.",
        "",
        *["- " + value for value in result["limits"]],
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "seconds": result["seconds"], **result["counts"]}))


if __name__ == "__main__":
    main()
