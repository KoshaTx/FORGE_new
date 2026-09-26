"""Freeze distinct null estimands before observing trained-null molecular quality."""

import hashlib
import json
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
ROOT = (WORKTREE / "results").resolve().parent
BASE = ROOT / "results/phase1"
OUT = BASE / "compose_lipid_iclr22_parallel_improvement_v1/a_attribution/null_fullfit_readiness_v1"
NULL = BASE / "compose_lipid_iclr22_table_completion_v1/null_runtime"
ASSEMBLY = BASE / "compose_lipid_iclr22_table_completion_v1/assembly"


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    pipeline = read(ASSEMBLY / "full_pipeline_cyclic_readiness.json")
    replay = read(ASSEMBLY / "readout_recovery_v4/protocol.json")
    limits = read(BASE / "compose_lipid_quality_confirmation_v1/construction-protocol.json")[
        "limits"
    ]
    files = {
        "fit_protocol": OUT / "protocol.json",
        "fit_config": OUT / "configuration.json",
        "count_only_policy": NULL / "source/forge/model/compose_lipid_null.py",
        "count_only_sampler": NULL / "source/forge/model/compose_lipid_null_sampling.py",
        "checkpoint_sampler_cli": NULL / "count_only_evaluation.py",
        "null_forward": NULL / "source/forge/model/reaction_program_transformer.py",
        "shared_sampler": NULL / "source/forge/model/compose_lipid_sampling.py",
        "core_constructor": BASE / "compose_lipid_quality_confirmation_v1/construct.py",
        "constructor_protocol": BASE
        / "compose_lipid_quality_confirmation_v1/construction-protocol.json",
        "full_pipeline_readiness": ASSEMBLY / "full_pipeline_cyclic_readiness.json",
        "same1408_sampling_protocol": ASSEMBLY / "readout_recovery_v4/protocol.json",
        "original_replay_result": ASSEMBLY / "readout_recovery_v4/conditioned_v1/result.json",
        "cyclic_replay_result": ASSEMBLY / "readout_recovery_v4/cyclic_v1/result.json",
        "count_only_handoff": NULL / "count_only_handoff_v1/result.json",
    }
    original = read(files["original_replay_result"])
    cyclic = read(files["cyclic_replay_result"])
    record = {
        "schema": "forge.null_control_estimands_readiness.v1",
        "producer": pin(Path(__file__)),
        "inputs": {k: pin(p) for k, p in files.items()},
        "frozen_before_trained_null_quality": True,
        "new_fit_or_sampling_calls": 0,
        "fit_seed": 2026092401,
        "same_original_seed_not_independent_replication": True,
        "current_qualified_null": {
            "semantics": "Native forward zeros all nine semantic inputs; all physical target coordinates variable; family-balanced TRAIN loss/PCGrad unchanged; authenticated global prior.",
            "generation_information": "Global joint node/closure counts only, no request/core/role layouts; evaluation requests joined after generation.",
            "ready_API": "count_only_evaluation.generate + CountOnlySampler + assess_rows",
            "minimal_execution_work": [
                "Bind final fully admitted2794 checkpoint and config into exact count-only protocol.",
                "Freeze7040attempts grouped as1408requests×5draws, with identical64posthoc requests/family and no outcome-dependent reassignment.",
                "Store true flow endpoints separately via qualifiedt=1hook; existing sample() returns terminal argmax.",
                "Persist endpoint/logit/terminal ledgers incrementally; existing CountOnlySampler persists terminal states only.",
            ],
            "causal_scope": "Compares a genuinely unconditioned learned generator with the full conditioned system; not a pure learned-model effect because information, priors and fixed-core conditioning differ.",
            "no_valid_full_decoder_drop_in": "Current constructors require origin/core/component layouts absent from count-only API. Inserting originalfamilylayouts changes the estimand and violates this declared policy.",
        },
        "matched_structural_context_control": {
            "label": "Trained semantic-null model with common supplied structural context",
            "fit": "Reuse the same admitted null checkpoint; no extra fit.",
            "intervention": "Separate inference-only experiment: native forward zero9; same originalrequestlayouts/fixedcores, node/closure masks, role-conditioned source-noise tensors, drawseeds, flowsteps, constructorbank and selector as conditioned baseline.",
            "information_disclosure": "Null model receives fixed structure indirectly through state/noise/decoder even though explicit semantic features are zero. It must not be called count-only unconditional generation.",
            "causal_limit": "Differences reflect null training masks/prior/conditioning jointly, not isolated semantic embedding removal or modelvsrules.",
            "requires": [
                "Separate wrapper/protocol, without altering frozen null training policy/source.",
                "Admit checkpoint loading for this explicitly different inference diagnostic; do not route it through CountOnlySampler.",
                "Unchanged conditioned checkpoint/logit and constructor/selection fixture reproduction first.",
                "Forward hooks must prove zero9 and externalstructural tensors byte-identical across matched arms.",
                "Recompute each arm own candidate pool, baseline, gate and component/novelty floors; never inject conditioned outcomes.",
                "Reapply exact identity-specific tree assessment admission; unknown alternate trees remain unknown.",
            ],
            "available_apis": [
                "compose_lipid_sampling.sample(return_predictions=True)",
                "qualified same1408 raw capture hook",
                "construct_one / frozen donor and AEMA domain constructors",
                "candidate_pools / first_exact / support-aware and context-preserving selection",
            ],
        },
        "fixed_comparison_budgets": {
            "requests": 1408,
            "per_family": 64,
            "families": 22,
            "draws": 5,
            "trajectories": 7040,
            "flowsteps": replay["steps"],
            "batch_size": replay["batch_size"],
            "seed_schedule": "Use every original draw/offset seed in pinned same1408 protocol, not a new favorable schedule.",
            "constructor_limits": limits,
            "proposal_order": "Base draw0..4 then source-retained AEMA domain draw0..4; same finite caps and failures.",
            "primary_readouts": [
                "draw0 true endpointN1408",
                "draw0 terminalN1408",
                "final fully matchedpipelineN1408",
            ],
            "secondary_readout": "all5 trajectory yieldN7040; do not silently select bestexact.",
            "quality": "Exactness plus unchanged gates, full failure denominators, descriptors, role/product diversity and TRAINnovelty; independent heldout realism remains separate.",
        },
        "rules_or_count_prior_control": {
            "fit_needed": False,
            "ready": False,
            "missing": "Construct a registered nonlearned14-head score/structural sampler with legal-parent/closure support and authenticated TRAIN priors, then qualify unchanged-score replay before substitution.",
            "important_negative_finding": "Final selector has no model score; ordering permutation is not score removal. Donor ranking, mutation ranking and several structure readouts consume model logits upstream.",
            "minimal_discriminator": "Same layouts and candidate ceilings; compare learned versus TRAINcount/uniform-legal scores inside identical construction, explicitly labelled constructor-score control. A broader rules-onlygenerator needs independently defined count/layout policy.",
        },
        "cost": {
            "measured_conditioned_sampling_CPU_seconds": original["CPU_seconds"],
            "measured_conditioned_sampling_wall_seconds": original["wall_seconds"],
            "measured_cyclic_sampling_CPU_seconds": cyclic["CPU_seconds"],
            "measured_cyclic_sampling_wall_seconds": cyclic["wall_seconds"],
            "measured_original_post_sampling_wall_seconds": pipeline[
                "total_observed_original_post_generation_seconds"
            ],
            "evaluation_CPU_budget_proposed_per_arm": 5400,
            "limitations": "Existing hardware/workload measurements, not guaranteed null costs; genuine global-count null batches may be larger. Additional construction/assessment is separately measured534.117wallseconds; no new evaluation execution authorized by this readiness file.",
            "full_fit_projection": pin(OUT / "projection.json"),
        },
        "decision_rules": [
            "Fullfit completion never implies high molecular quality; audit all exposure/source/state receipts before sampling.",
            "No selected bestseed/checkpoint; final2794 is fixed, first fitseed pairedoriginal2026092401.",
            "A model contribution claim requires matched information/candidate/call budgets and quality/diversity outcomes, not merely exact yield.",
            "If rules/count-prior or null perform comparably, narrow claims to synthesis-constrained system and document weak learned contribution.",
            "Heldout and independent seed pairs remain required for papergeneralization regardless of development gains.",
        ],
    }
    with (OUT / "control_readiness.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"control_readiness": pin(OUT / "control_readiness.json")}))


if __name__ == "__main__":
    main()
