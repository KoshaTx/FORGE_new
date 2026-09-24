"""Authenticate integration identity and the predeclared fresh twelve-family population."""

import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1.multireaction import combinatorial_generation_pipeline as pipeline
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    inputs = []

    def load(path):
        inputs.append(pipeline._pin(path, ROOT))
        return pipeline._read(path)

    def result(name):
        r = load(ROOT / f"results/phase1/{name}/result.json")
        config_path = resolve_pin(r["config"], ROOT, label="pipeline evidence request")
        config, _ = pipeline.contract(ROOT, config_path)
        inputs.append(r["config"])
        stage_results = {}
        for stage in pipeline.STAGES:
            saved = r["stages"][stage]
            for p in saved.values():
                resolve_pin(p, ROOT, label="pipeline evidence stage")
                inputs.append(p)
            stage_results[stage] = pipeline._read(ROOT / saved["result"]["path"])
            for p in stage_results[stage]["artifacts"].values():
                resolve_pin(p, ROOT, label="pipeline evidence artifact")
                inputs.append(p)
        metrics, products = pipeline.summarize(ROOT, config, stage_results)
        assert all(r[k] == v for k, v in metrics.items())
        assert r["acceptance_passed"]
        stored = resolve_pin(r["qualified_products"], ROOT, label="full product export")
        inputs.append(r["qualified_products"])
        assert [json.loads(line) for line in stored.read_text().splitlines()] == products
        return r, config, stage_results

    replay, replay_config, replay_stages = result("combinatorial_generation_pipeline_replay_v2")
    fresh, fresh_config, fresh_stages = result("combinatorial_generation_pipeline_fresh_v1")
    checks = {}
    for stage, r in replay_stages.items():
        reference_pin = replay_config["reference_results"][stage]
        ref = load(resolve_pin(reference_pin, ROOT, label="historical stage"))
        assert {k: p["sha256"] for k, p in r["artifacts"].items()} == {
            k: p["sha256"] for k, p in ref["artifacts"].items()
        }
        for key in ("per_arm", "per_family_deltas", "total_exact_gain"):
            assert r[key] == ref[key]
        checks[stage] = "scientific_comparisons_and_all_artifact_hashes_identical"
    assert fresh_config["population_mode"] == "fresh_train"
    assert replay_config["population_mode"] == "reference_replay"
    assert all(
        fresh_config["sampling"][key] != replay_config["sampling"][key]
        for key in ("layout_seed", "flow_seed")
    )
    # A fresh request need not pick unseen TRAIN layouts. Report overlap rather than claiming it.
    reference_layouts = {
        r["record_id"]
        for r in load(ROOT / replay_stages["graph"]["artifacts"]["layouts.json"]["path"])
    }
    fresh_layouts = load(ROOT / fresh_stages["graph"]["artifacts"]["layouts.json"]["path"])
    assert len(fresh_layouts) == fresh["generated_attempts"] == 768
    assert all(r["fold"] == "train" for r in fresh_layouts)

    def totals(r):
        return {
            arm: {
                key: sum(f[key] for f in families.values())
                for key in (
                    "attempts",
                    "valid_connected",
                    "exact_program",
                    "unique_valid_products",
                    "novel_valid_attempts_vs_all_cache_train",
                    "unique_novel_exact_products",
                )
            }
            for arm, families in r["comparison_vs_raw"]["per_arm"].items()
        }

    initial = ROOT / "results/phase1/combinatorial_generation_pipeline_replay_v1"
    assert not (initial / "result.json").exists()
    recovery = load(
        ROOT / "results/phase1/combinatorial_generation_pipeline_replay_v2/recovery.json"
    )
    for p in [recovery["source"], *recovery["inputs"]]:
        resolve_pin(p, ROOT, label="recovery proof")
        inputs.append(p)
    payload = {
        "schema_version": "forge.combinatorial_generation_pipeline_evidence.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs + [pipeline._pin(Path(__file__), ROOT)],
        "integration_replay": {"checks": checks, "totals": totals(replay)},
        "fresh_population": {
            "sampling": fresh_config["sampling"],
            "totals": totals(fresh),
            "per_family": fresh["comparison_vs_raw"]["per_arm"],
            "per_family_deltas": fresh["comparison_vs_raw"]["per_family_deltas"],
            "stage_exact_gains": fresh["stage_exact_gains"],
            "checks": fresh["checks"],
            "qualified_products": fresh["qualified_products"],
            "layout_attempts_overlapping_reference_layout_ids": sum(
                r["record_id"] in reference_layouts for r in fresh_layouts
            ),
            "all_neural_terminal_states_reproduced_in_source_core_stage": fresh[
                "original_neural_terminals_reproduced"
            ],
        },
        "interrupted_integration_attempt_retained": True,
        "completed_stages_reused_after_exact_config_equivalence_check": list(pipeline.STAGES),
        "goal_complete": False,
        "promoted": False,
        "limits": [
            "Fresh layout and noise seeds were fixed before the new population was generated. Context is still sampled from TRAIN; it is not a heldout-component or source-independent evaluation.",
            "The method and eight per-family preservation metrics are unchanged. Admission uses explicit TRAIN membership and batch multiplicities; this is not preservation of every possible notion of chemical diversity.",
            "The final export contains every exactly reconstructed product, retains duplicate attempts and source scopes, and is not a prospective candidate panel.",
            "Stage execution and semantic verification add compute. This is a capability/integrity check and a new-population generation comparison, not a matched-total-cost claim.",
        ],
    }
    target = OUT / "evidence.json"
    if target.exists():
        raise ValueError("evidence already exists")
    pipeline._write(target, payload)
    print(
        json.dumps(
            {"totals": totals(fresh), "stage_exact_gains": fresh["stage_exact_gains"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
