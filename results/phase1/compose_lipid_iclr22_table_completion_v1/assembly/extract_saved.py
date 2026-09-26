"""Recover same-cohort assembly statistics without model or chemistry execution."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RESEARCH = Path("results/phase1/compose_lipid_iclr22_research_v1")
PAR = RESEARCH / "parallel_completion_v1"
CONFIRM = Path("results/phase1/compose_lipid_quality_confirmation_v1")
TRAIN = PAR / "training_evaluation"


def pin(path: Path) -> dict:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def identity(value: dict) -> str:
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected a nonempty recorded role/component tuple")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("Invalid recorded component identity")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def summarize_check(check: dict) -> dict:
    """Count every recorded executor/candidate, with an independently checked denominator."""
    if check.get("status") != "evaluated" or type(check.get("exact")) is not bool:
        raise ValueError("Missing evaluated exact verdict")
    if not isinstance(check.get("checks"), list) or not check["checks"]:
        raise ValueError("Missing executor-check ledger")
    counts = Counter(executor_checks=len(check["checks"]))
    accepted = set()
    inverse_bounds, replay_bounds = set(), set()
    for record in check["checks"]:
        inverse, candidates = record["inverse"], record["candidates"]
        if type(inverse.get("complete_search")) is not bool:
            raise ValueError("Missing inverse-search completion flag")
        if type(record.get("exact_registry_program_roundtrip")) is not bool:
            raise ValueError("Missing executor exact replay flag")
        returned = Counter(identity(x) for x in inverse["candidate_components"])
        checked = Counter(identity(x["components"]) for x in candidates)
        if returned != checked:
            raise ValueError("Returned inverse tuple does not have exactly one checked verdict")
        passed = []
        for candidate in candidates:
            if type(candidate.get("pass")) is not bool:
                raise ValueError("Nonboolean candidate verdict")
            counts["checked_candidates"] += 1
            counts["passing_candidates"] += candidate["pass"]
            counts["rejected_candidates"] += not candidate["pass"]
            if candidate["pass"]:
                passed.append(identity(candidate["components"]))
            replay_bounds.update(candidate.get("replay", {}).get("bound_reasons", []))
        if Counter(passed) != Counter(identity(x) for x in record["accepted_components"]):
            raise ValueError("Accepted components disagree with full checked-candidate ledger")
        if bool(passed) != record["exact_registry_program_roundtrip"]:
            raise ValueError("Executor exact flag disagrees with candidate verdicts")
        accepted.update(passed)
        counts["returned_inverse_candidates"] += sum(returned.values())
        counts["incomplete_inverse_searches"] += not inverse["complete_search"]
        inverse_bounds.update(inverse.get("bound_reasons", []))
    if bool(accepted) != check["exact"]:
        raise ValueError("Top-level exact verdict disagrees with accepted roots")
    return {
        **{
            k: counts[k]
            for k in (
                "executor_checks",
                "checked_candidates",
                "passing_candidates",
                "rejected_candidates",
                "returned_inverse_candidates",
                "incomplete_inverse_searches",
            )
        },
        "exact": check["exact"],
        "has_returned_candidate": counts["returned_inverse_candidates"] > 0,
        "distinct_accepted_component_tuples": len(accepted),
        "ambiguous_accepted_component_tuples": len(accepted) > 1,
        "all_inverse_searches_complete": counts["incomplete_inverse_searches"] == 0,
        "inverse_bound_reasons": sorted(inverse_bounds),
        "replay_bound_reasons": sorted(replay_bounds),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "executor_checks",
        "checked_candidates",
        "passing_candidates",
        "rejected_candidates",
        "returned_inverse_candidates",
        "incomplete_inverse_searches",
        "exact",
        "has_returned_candidate",
        "ambiguous_accepted_component_tuples",
        "all_inverse_searches_complete",
    ]
    result = {key: sum(row[key] for row in rows) for key in keys}
    result.update(
        requests=len(rows),
        no_returned_candidate=sum(not row["has_returned_candidate"] for row in rows),
        candidate_but_no_exact=sum(
            row["has_returned_candidate"] and not row["exact"] for row in rows
        ),
        requests_with_inverse_bounds=sum(bool(row["inverse_bound_reasons"]) for row in rows),
        requests_with_replay_bounds=sum(bool(row["replay_bound_reasons"]) for row in rows),
    )
    result["candidate_search_coverage"] = {
        "numerator": result["has_returned_candidate"],
        "denominator": len(rows),
        "fraction": result["has_returned_candidate"] / len(rows),
    }
    result["checked_candidate_acceptance"] = {
        "numerator": result["passing_candidates"],
        "denominator": result["checked_candidates"],
        "fraction": (
            result["passing_candidates"] / result["checked_candidates"]
            if result["checked_candidates"]
            else None
        ),
    }
    return result


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
    start = time.process_time()
    paths = {
        "selected": RESEARCH / "quality/all22_context_preserving/result.json",
        "selection_verification": RESEARCH / "quality/all22_context_preserving/verification.json",
        "candidate_ledger": CONFIRM / "compact_attempts.json",
        "candidate_diagnostics": CONFIRM / "assessment/d1_candidate_diagnostics.json",
        "generation_protocol": CONFIRM / "protocol.json",
        "generation_receipt": CONFIRM / "generation.json",
        "generation_producer": CONFIRM / "generate.py",
        "construction_producer": CONFIRM / "construct.py",
        "candidate_indexing_producer": Path(
            "results/phase1/compose_lipid_quality_selection_v2/run.py"
        ),
        "official_cohort": PAR / "routes/recount_closeout_v3/result.json",
        "cohort": Path("results/phase1/compose_lipid_training_cohort_v1/cohort.json"),
        "metadata_census": RESEARCH / "evidence_completion_v1/validation/metadata_census.json",
        "metadata_scope": RESEARCH / "evidence_completion_v1/validation/metadata_scope.json",
        "cyclic352": TRAIN / "cyclic_inference_qualification_v1/full_pair_v1/paired_result.json",
        "cyclic352_admission": TRAIN
        / "cyclic_inference_qualification_v1/full_pair_v1/root_admission.json",
        "null_readiness": TRAIN / "group_loss_evaluation_v1/control_readiness.json",
    }
    inputs = {name: pin(path) for name, path in paths.items()}
    protocol = {
        "schema": "forge.table_completion.saved_assembly.protocol.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_cap_seconds": 60,
        "inputs": inputs,
        "producer": pin(Path(__file__).relative_to(ROOT)),
        "population": "All1408 current selected requests; no filtering or resampling",
        "join": "request index, family, selected ordinal, exact stored SMILES and exact verdict",
        "candidate_indexing": "[branch.d1.raw, *branch.d1.proposals][selected_ordinal], as original selection producer",
        "coverage_definition": "Requests with at least one returned inverse component tuple /1408 requests",
        "precision_definition": "Passing recorded candidate checks / all returned-and-checked executor-specific candidates; computational replay/source acceptance, not experimental precision",
        "ambiguity_definition": "More than one distinct accepted constitutional role/component tuple across recorded executor checks; not all possible reaction traces",
        "unknowns": "True flow endpoint state was not saved; no same1408 cyclic or independently trained true-null result is substituted",
        "new_model_chemistry_TEST_network_calls": 0,
        "randomness": "None; saved original flow/layout seeds retained in source protocol",
    }
    if (HERE / "protocol.json").exists() or (HERE / "result.json").exists():
        raise FileExistsError("Preserve existing extraction; use a new version")
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    def read(key):
        return json.loads((ROOT / paths[key]).read_text())

    selected = read("selected")
    saved = read("candidate_ledger")
    official = read("official_cohort")
    cohort = read("cohort")
    metadata = read("metadata_census")
    generation = read("generation_receipt")
    cyclic = read("cyclic352")
    selections = {
        row["request"]: {"family": family, **row}
        for family, value in selected["by_family"].items()
        for row in value["selections"]
    }
    current = {row["index"]: row for row in official["rows"]}
    if not (len(selections) == len(current) == len(saved) == 1408):
        raise ValueError("Incomplete same1408 cohort")
    result_rows, first_draw = [], []
    for attempt in saved:
        index = attempt["index"]
        selection, target = selections[index], current[index]
        if selection["family"] != attempt["family"] or selection["family"] != target["family"]:
            raise ValueError("Family binding changed")
        if selection["ordinal"] != target["ordinal"] or selection["smiles"] != target["smiles"]:
            raise ValueError("Current selected molecule changed")
        branch = attempt["branches"]["d1"]
        candidate = [branch["raw"], *branch["proposals"]][selection["ordinal"]]
        if candidate.get("smiles") != selection["smiles"]:
            raise ValueError("Selected candidate molecule mismatch")
        details = summarize_check(candidate["check"])
        if details["exact"] != selection["exact"] or details["exact"] != target["exact_L1"]:
            raise ValueError("Selected exact verdict mismatch")
        result_rows.append(
            {
                "index": index,
                "family": attempt["family"],
                "selected_ordinal": selection["ordinal"],
                "selected_smiles_sha256": hashlib.sha256(selection["smiles"].encode()).hexdigest(),
                "source_pointer": f"{paths['candidate_ledger']}#/{index}/branches/d1",
                **details,
            }
        )
        first_draw.append(
            {
                "index": index,
                "family": attempt["family"],
                "d0_constrained_exact": attempt["branches"]["d0"]["raw"]["check"]["exact"],
                "d1_constrained_exact": branch["raw"]["check"]["exact"],
            }
        )
    families = cohort["families"]
    family_results = {
        family: aggregate([x for x in result_rows if x["family"] == family]) for family in families
    }
    all_results = aggregate(result_rows)
    assert all_results["exact"] == 1387 and all_results["requests"] == 1408
    metadata_rows = []
    for family in families:
        n = cohort["by_family"][family]
        present = metadata["train_with_source_study_metadata"].get(family, 0)
        metadata_rows.append(
            {
                "family": family,
                "TRAIN_records": n,
                "TRAIN_opaque_component_IDs": metadata["train_components_by_family"][family],
                "TRAIN_rows_with_recorded_study": present,
                "TRAIN_rows_without_recorded_study": n - present,
                "typed_core_states": metadata["conditioning_vocabulary"]["core_ids_by_family"][
                    family
                ],
                "independent_study_count": None,
            }
        )
    result = {
        "schema": "forge.table_completion.saved_assembly.result.v1",
        "complete": True,
        "protocol": pin((HERE / "protocol.json").relative_to(ROOT)),
        "totals": all_results,
        "by_family": family_results,
        "requests": result_rows,
        "first_draw_constrained_readout": {
            "not_true_flow_endpoint": True,
            "rows": first_draw,
            "totals": {
                k: sum(x[k] for x in first_draw)
                for k in ["d0_constrained_exact", "d1_constrained_exact"]
            },
            "by_family": {
                family: {
                    k: sum(x[k] for x in first_draw if x["family"] == family)
                    for k in ["d0_constrained_exact", "d1_constrained_exact"]
                }
                for family in families
            },
        },
        "same1408_missing_controls": {
            "true_flow_endpoint": "Unstored; deterministic sampler replay needed. Original generate.py discarded sample()'s t1-argmax state too and saved only terminal predictions.",
            "terminal_argmax": "Can reconstruct from saved final predictions using qualified masks and verify; this is distinct from true flow endpoint.",
            "cyclic": "No same1408 result admitted; paired replay with original5draw schedule needed.",
            "shared_null": "No independently trained true-null checkpoint/performance. Implementation readiness is not a control score.",
        },
        "separate352_cyclic": [
            {
                "family": x["family"],
                "readout": x["readout"],
                "requests": x["requests"],
                "conditioned_exact": x["metrics"]["exact_L1"]["conditioned"],
                "cyclic_exact": x["metrics"]["exact_L1"]["cyclic"],
                "gains": len(x["metrics"]["exact_L1"]["gains"]),
                "losses": len(x["metrics"]["exact_L1"]["losses"]),
            }
            for x in cyclic["paired"]
        ],
        "metadata_census": {
            "rows": metadata_rows,
            "known_recorded_study_IDs_all_TRAIN": metadata["train_study_ids"],
            "recorded_study_rows": sum(x["TRAIN_rows_with_recorded_study"] for x in metadata_rows),
            "non_null_program_IDs": metadata["conditioning_vocabulary"]["non_null_program_ids"],
            "independence_qualified": False,
            "interpretation": "Recorded metadata coverage, not an independent source-study census. Missing source IDs are unknown, not zero studies. Existing aggregate only; no TEST records read.",
        },
        "replay_cost_evidence": {
            k: generation[k]
            for k in [
                "draws",
                "requests",
                "forward_evaluations",
                "model_flow_calls",
                "seconds",
                "setup_seconds",
            ]
        },
        "same1408_final_selection_is_not_budget_matched_to_first_draw": True,
        "new_model_chemistry_TEST_network_calls": 0,
        "CPU_seconds": time.process_time() - start,
        "paper_edits": False,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"totals": all_results, "CPU_seconds": result["CPU_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
