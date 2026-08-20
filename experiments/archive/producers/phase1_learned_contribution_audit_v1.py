#!/usr/bin/env python3
"""What does the trained flow contribute that the constrained sampler does not?

The constraint-only control answered the attribution question, and the answer was not the one
the Stage 2 headline assumed. Under identical programs, masks, closure model, schedule, sources,
admission predicate and seeds, a denoiser that emits nothing but the role-specific empirical
source marginals reaches **higher** admission than the trained flow. Admission is therefore a
property of the constrained sampler and cannot be offered as evidence of learned competence.

That does not make the trained flow redundant, and this script measures what it does supply.
A legal Ugi product is not the same object as a plausible ionizable lipid: admission requires
only that the generated components react to give the generated product under the qualified
transform. Whether the result looks like the chemistry the corpus is made of is a separate
question, and it is the one the learned model should answer.

Three descriptive comparisons against a fixed random sample of the train fold, all on admitted
products only, all with the same seeds across arms:

  nearest-neighbour ECFP4 Tanimoto to the train corpus, which says whether generated chemistry
  sits near real Ugi lipids at all;
  cLogP, which for this family is the sharpest single lipid-likeness signal;
  molecular weight.

None of these is a fidelity metric in the generative-modelling sense and none is presented as
one. They are three descriptors chosen before the numbers were seen, sufficient to separate
"legal" from "lipid-like".
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, DataStructs, rdBase  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator  # noqa: E402

ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
REFERENCE_SAMPLE = 3000
ARM_SAMPLE = 400
SEED = 20260816


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p10": ordered[len(ordered) // 10],
        "p90": ordered[(9 * len(ordered)) // 10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "PATH"),
                        required=True, help="repeatable: a label and a sample artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    with gzip.open(REPO / ASSIGN, "rt", newline="") as handle:
        train = [row["canonical_product_smiles"] for row in csv.DictReader(handle)
                 if row["primary_product_fold"] == "train"]
    rng = random.Random(SEED)
    reference = rng.sample(train, REFERENCE_SAMPLE)
    with rdBase.BlockLogs():
        reference_fps = [generator.GetFingerprint(m) for s in reference
                         if (m := Chem.MolFromSmiles(s)) is not None]
        corpus_mw, corpus_logp = [], []
        for smiles in reference[:ARM_SAMPLE]:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            corpus_mw.append(Descriptors.MolWt(molecule))
            corpus_logp.append(Crippen.MolLogP(molecule))
    print(f"reference: {len(reference_fps)} train-fold products fingerprinted")

    results: dict[str, dict] = {
        "train_corpus_reference": {
            "products": len(corpus_mw),
            "molecular_weight": quantiles(corpus_mw),
            "clogp": quantiles(corpus_logp),
            "nearest_train_tanimoto": None,
        }
    }
    for label, path in args.arm:
        payload = json.loads(Path(path).read_text())
        admitted = [r for r in payload["samples"] if r.get("valid")]
        local = random.Random(SEED)
        chosen = local.sample(admitted, min(ARM_SAMPLE, len(admitted)))
        weights, logps, nearest = [], [], []
        with rdBase.BlockLogs():
            for row in chosen:
                molecule = Chem.MolFromSmiles(row["smiles"])
                if molecule is None:
                    continue
                weights.append(Descriptors.MolWt(molecule))
                logps.append(Crippen.MolLogP(molecule))
                nearest.append(max(DataStructs.BulkTanimotoSimilarity(
                    generator.GetFingerprint(molecule), reference_fps)))
        results[label] = {
            "admitted_available": len(admitted),
            "products_scored": len(weights),
            "molecular_weight": quantiles(weights),
            "clogp": quantiles(logps),
            "nearest_train_tanimoto": quantiles(nearest),
            "source": {"path": path, "sha256": sha256_file(Path(path))},
        }

    print(f"\n{'arm':18s} {'n':>5s} {'MW med':>8s} {'cLogP med':>10s} "
          f"{'NN-Tanimoto med':>16s} {'p10':>7s}")
    for label, values in results.items():
        nn = values["nearest_train_tanimoto"]
        print(f"{label:18s} {values.get('products_scored', values.get('products')):5d} "
              f"{values['molecular_weight']['median']:8.1f} {values['clogp']['median']:10.2f} "
              f"{(f'{nn[chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)]:.3f}' if nn else '—'):>16s} "
              f"{(f'{nn[chr(112)+chr(49)+chr(48)]:.3f}' if nn else '—'):>7s}")

    payload = {
        "schema_version": "phase1_forge_learned_contribution_audit.v1",
        "status": "complete_descriptive_comparison",
        "question": "what does the trained flow supply that the constrained sampler does not, "
                    "given that a marginal denoiser reaches higher admission than the trained "
                    "flow under identical constraints",
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "protocol": {"reference_sample": REFERENCE_SAMPLE, "per_arm_sample": ARM_SAMPLE,
                     "seed": SEED, "population": "admitted products only",
                     "fingerprint": "ECFP4, 2048 bits"},
        "arms": results,
        "nonclaims": [
            "These are three descriptors, not a generative fidelity metric. They separate legal "
            "from lipid-like; they do not establish distributional match.",
            "Nearest-neighbour Tanimoto to the train fold is a similarity statistic on binary "
            "folded fingerprints, which cannot distinguish alkyl homologues of different chain "
            "length. It is used here only to separate arms that differ by a wide margin.",
            "Admission is excluded as evidence of learned competence: the marginal null reaches "
            "a higher admission rate than the trained flow under identical constraints.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
