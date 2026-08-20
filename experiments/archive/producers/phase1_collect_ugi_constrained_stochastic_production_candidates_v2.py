#!/usr/bin/env python3
"""Seal the fresh matched stochastic Ugi production candidate ledger."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.corpus.r1_prime_audit import sha256_file
from experiments.phase1.product_l1.training.ugi_constrained_stochastic_production_candidates import (
    ARMS,
    CONFIG_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    UgiConstrainedStochasticProductionCandidatesError,
    exact_terminal_admission,
    gzip_jsonl,
    load_json,
    program_shard,
    stable_sha256,
)
from forge.corpus.ugi_held_component_gate import _reaction_contract

REPO = Path(__file__).resolve().parents[3]
RAW_ROOT = "results/phase1/ugi_constrained_stochastic_production_candidates_v2/raw"


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiConstrainedStochasticProductionCandidatesError(f"malformed input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiConstrainedStochasticProductionCandidatesError(
            f"input pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiConstrainedStochasticProductionCandidatesError(f"input pin changed: {label}")
    return path


def _canonical(smiles: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiConstrainedStochasticProductionCandidatesError(
            f"invalid admitted molecule: {smiles}"
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _seed(config: dict[str, Any], arm: str, shard: int, *, terminal: bool) -> int:
    arm_index = tuple(config["design"]["arms"]).index(arm)
    stream = arm_index * int(config["design"]["shards_per_arm"]) + shard
    key = "terminal_seed_base" if terminal else "flow_seed_base"
    return int(config["design"][key]) + stream


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def collect(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Verify every raw draw and build the immutable admitted-candidate ledger."""

    repo = repo.resolve()
    config = load_json(config_path.resolve(), label="production-candidate config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != "frozen_before_fresh_matched_terminal_generation"
        or tuple(config.get("design", {}).get("arms", ())) != ARMS
    ):
        raise UgiConstrainedStochasticProductionCandidatesError(
            "unsupported production-candidate config"
        )
    inputs = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    for label, record in config.get("implementation", {}).items():
        _pin(repo, record, label=f"implementation.{label}")
    schedule = load_json(inputs["schedule"], label="production schedule")
    schedule_result = load_json(inputs["schedule_result"], label="schedule result")
    if schedule_result.get(
        "status"
    ) != "matched_two_arm_production_candidate_schedule_frozen" or schedule_result.get(
        "artifacts", {}
    ).get(
        "schedule.json", {}
    ).get(
        "schedule_sha256"
    ) != schedule.get(
        "schedule_sha256"
    ):
        raise UgiConstrainedStochasticProductionCandidatesError(
            "production schedule receipt changed"
        )
    reaction = _reaction_contract(inputs["qualified_reactions"])
    reference_products = set()
    with gzip.open(inputs["assignments"], "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            reference_products.add(str(row["canonical_product_smiles"]))
    if len(reference_products) != 112386:
        raise UgiConstrainedStochasticProductionCandidatesError(
            "all-fold reference product population changed"
        )

    shard_draws = int(config["design"]["shard_draws"])
    shard_count = int(config["design"]["shards_per_arm"])
    records: list[dict[str, Any]] = []
    receipts = []
    for arm in ARMS:
        for shard in range(shard_count):
            expected_programs = program_shard(
                schedule,
                arm=arm,
                shard_index=shard,
                shard_draws=shard_draws,
            )
            result_path = repo / RAW_ROOT / arm / f"shard_{shard:02d}/result.json"
            result = load_json(result_path, label=f"raw result {arm}/{shard}")
            sampling = result.get("sampling", {})
            decoder = sampling.get("terminal_decoder", {})
            rows = result.get("samples")
            if (
                int(result.get("seed", -1)) != _seed(config, arm, shard, terminal=False)
                or decoder.get("mode") != "stochastic"
                or int(decoder.get("seed", -1)) != _seed(config, arm, shard, terminal=True)
                or float(decoder.get("temperature", -1.0)) != 1.0
                or int(sampling.get("sample_steps", -1)) != int(config["design"]["sample_steps"])
                or sampling.get("maximum_adjacent_branch_runs")
                != config["design"]["maximum_adjacent_branch_runs"]
                or sampling.get("terminal_tree_repairs") != 0
                or not isinstance(rows, list)
                or len(rows) != shard_draws
            ):
                raise UgiConstrainedStochasticProductionCandidatesError(
                    f"raw sampling contract changed: {arm}/{shard}"
                )
            start = shard * shard_draws
            scheduled = schedule["records"][start : start + shard_draws]
            for local_index, (terminal, expected, coordinate) in enumerate(
                zip(rows, expected_programs["samples"], scheduled, strict=True)
            ):
                arm_coordinate = coordinate["arms"][arm]
                draw_index = int(coordinate["draw_index"])
                if (
                    terminal.get("program") != expected["program"]
                    or terminal.get("product_id") != expected["product_id"]
                    or terminal.get("program") != arm_coordinate["program"]
                    or draw_index != start + local_index
                ):
                    raise UgiConstrainedStochasticProductionCandidatesError(
                        f"raw program alignment changed: {arm}/{shard}/{local_index}"
                    )
                admission = exact_terminal_admission(terminal, reaction)
                canonical = _canonical(str(terminal["smiles"])) if admission["admitted"] else None
                records.append(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "arm_id": arm,
                        "draw_index": draw_index,
                        "shard_index": shard,
                        "shard_local_index": local_index,
                        "common_uniform": float(coordinate["common_uniform"]),
                        "support_index": int(arm_coordinate["support_index"]),
                        "program_sha256": str(arm_coordinate["program_sha256"]),
                        "program": dict(arm_coordinate["program"]),
                        "selected_probability": float(arm_coordinate["selected_probability"]),
                        "broad_prior_probability": float(arm_coordinate["broad_prior_probability"]),
                        "support_proposal_probability": float(
                            arm_coordinate["support_proposal_probability"]
                        ),
                        "importance_ratio_broad_over_arm": float(
                            arm_coordinate["importance_ratio_broad_over_arm"]
                        ),
                        "support_score": float(arm_coordinate["support_score"]),
                        "flow_seed": _seed(config, arm, shard, terminal=False),
                        "terminal_seed": _seed(config, arm, shard, terminal=True),
                        "native_terminal": terminal,
                        "terminal_chemical_admission": admission,
                        "canonical_admitted_product": canonical,
                        "exact_refit_corpus_product": bool(
                            canonical is not None and canonical in reference_products
                        ),
                    }
                )
            receipts.append(
                {
                    "arm": arm,
                    "shard": shard,
                    "result": {
                        "path": str(result_path.relative_to(repo)),
                        "sha256": sha256_file(result_path),
                    },
                    "attempted": len(rows),
                    "valid": sum(row.get("valid") is True for row in rows),
                }
            )
    expected_total = int(config["design"]["draws_per_arm"]) * len(ARMS)
    if len(records) != expected_total:
        raise UgiConstrainedStochasticProductionCandidatesError(
            "combined candidate ledger has the wrong denominator"
        )

    arm_summaries = {}
    for arm in ARMS:
        subset = [row for row in records if row["arm_id"] == arm]
        admitted = [row for row in subset if row["terminal_chemical_admission"]["admitted"]]
        arm_summaries[arm] = {
            "raw_draws": len(subset),
            "raw_valid": sum(
                row["terminal_chemical_admission"]["raw_molecule_valid"] for row in subset
            ),
            "exact_l1": sum(row["terminal_chemical_admission"]["exact_l1"] for row in subset),
            "terminal_chemical_admitted": len(admitted),
            "terminal_chemical_admitted_fraction": len(admitted) / len(subset),
            "unique_admitted_products": len(
                {row["canonical_admitted_product"] for row in admitted}
            ),
            "corpus_absent_admitted": sum(
                not row["exact_refit_corpus_product"] for row in admitted
            ),
            "admission_reasons": dict(
                sorted(
                    Counter(row["terminal_chemical_admission"]["reason"] for row in subset).items()
                )
            ),
        }
    ledger = gzip_jsonl(records)
    ledger_hash = hashlib.sha256(ledger).hexdigest()
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_constrained_stochastic_production_candidate_generation",
        "generator_identity": (
            "constrained stochastic graph-flow generator for complete "
            "Ugi-compatible ionizable lipids"
        ),
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(inputs.items())
        },
        "design": dict(config["design"]),
        "arms": arm_summaries,
        "counts": {
            "raw_terminal_attempts": len(records),
            "draws_per_arm": int(config["design"]["draws_per_arm"]),
            "shards": len(receipts),
        },
        "raw_shard_receipts": receipts,
        "artifacts": {
            "terminal_ledger.jsonl.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(records),
                "sha256": ledger_hash,
            }
        },
        "next_gate": "frozen_terminal_biological_applicability_and_conservative_ranking",
        "nonclaims": [
            "The terminal admission rule does not repair or retry rejected molecules.",
            "Applicability enrichment does not guarantee biological applicability.",
            "Potency and route values did not alter molecular generation.",
            "Terminal chemical admission is not complete L2/L3 route closure.",
        ],
    }
    result["result_sha256"] = stable_sha256(result)
    return result, ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/model/phase1_ugi_constrained_stochastic_production_candidates_v2.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_constrained_stochastic_production_candidates_v2"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, ledger = collect(REPO, config)
    _atomic_bytes(output / "terminal_ledger.jsonl.gz", ledger)
    _atomic_json(output / "result.json", result)
    print(json.dumps(result["arms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
