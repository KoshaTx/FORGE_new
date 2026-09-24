"""Record the generated-context audit, expansion limits and rejected count-prior pilot."""

import json
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    inputs = {}

    def pin(path):
        value = {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}
        inputs[value["path"]] = value
        return value

    def read(path):
        pin(path)
        return json.loads(path.read_text())

    def result(name):
        r = read(ROOT / f"results/phase1/{name}/result.json")
        config = read(resolve_pin(r["config"], ROOT, label="findings config"))
        assert config["source_pins"] == r["sources"]
        for p in [*r["sources"], *r["artifacts"].values()]:
            pin(resolve_pin(p, ROOT, label="findings source/artifact"))
        return r

    audit = result("combinatorial_generated_layouts_v1")
    expansion = result("combinatorial_context_expansion_v1")
    count = result("combinatorial_count_layout_generation_v1")
    replay = result("combinatorial_count_layout_generation_replay_v1")
    keys = set(count) - {"created_at_utc", "duration_seconds", "artifacts"}
    assert set(replay) - {"created_at_utc", "duration_seconds", "artifacts"} == keys
    assert all(count[k] == replay[k] for k in keys)
    assert {k: p["sha256"] for k, p in count["artifacts"].items()} == {
        k: p["sha256"] for k, p in replay["artifacts"].items()
    }
    rows = [
        json.loads(line)
        for line in (ROOT / count["artifacts"]["attempts.jsonl"]["path"]).read_text().splitlines()
    ]
    assert len(rows) == count["neural_attempts"] == 768
    assert all(row["layout_record_id"].startswith("factorized-layout-") for row in rows)
    assert not any(
        "exact_target_graph" in row or "tensor_reconstruction_exact" in row for row in rows
    )
    by_family = {}
    for family in sorted(count["by_family"]):
        c = count["by_family"][family]
        a = audit["by_family"][family]
        e = expansion["by_family"][family]
        by_family[family] = {
            "generated_unique_new_model_contexts": a["unique_novel_sparse_flow_contexts"],
            "generated_unique_new_sampler_contexts": a["unique_novel_sampler_contexts"],
            "expanded_unique_new_model_contexts": e["unique_novel_sparse_flow_contexts"],
            "count_prior_attempts": c["attempts"],
            "count_prior_valid": c["valid_connected"],
            "count_prior_exact": c["exact_program"],
        }
    metrics = (
        "attempts",
        "valid_connected",
        "exact_program",
        "unique_valid_products",
        "novel_valid_attempts_vs_all_cache_train",
        "unique_novel_exact_products",
    )
    totals = {
        arm: {m: sum(r[m] for r in values.values()) for m in metrics}
        for arm, values in {
            "source_layout_baseline": count["comparison_vs_source_layouts"]["per_arm"]["baseline"],
            "existing_completed_pipeline": count["comparison_vs_previous_completion"]["per_arm"][
                "baseline"
            ],
            "count_only_prior": count["by_family"],
        }.items()
    }
    preflight = read(OUT / "existing_prior_preflight.json")
    for p in preflight["inputs"]:
        pin(resolve_pin(p, ROOT, label="prior preflight"))
    pin(Path(__file__))
    payload = {
        "schema_version": "forge.combinatorial_generated_layout_findings.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "negative_count_prior_preserved",
        "inputs": inputs,
        "by_family": by_family,
        "totals": totals,
        "generated_context_coverage_families": sum(
            r["generated_unique_new_model_contexts"] > 0 for r in by_family.values()
        ),
        "expanded_context_coverage_families": sum(
            r["expanded_unique_new_model_contexts"] > 0 for r in by_family.values()
        ),
        "expansion_forward_calls": expansion["forward_expansions"],
        "count_prior_support": count["prior_support"],
        "urea_topology_counts_by_depth": count["source_topology_counts_by_depth"][
            "urea_amine_isocyanate"
        ],
        "count_replay_all_scientific_fields_and_artifact_hashes_identical": True,
        "count_prior_preserves_source_layout_baseline": count["comparison_vs_source_layouts"][
            "all_family_preservation_screen_passed"
        ],
        "count_prior_preserves_completed_baseline": count["comparison_vs_previous_completion"][
            "all_family_preservation_screen_passed"
        ],
        "promoted": False,
        "next_test": "Paired layouts with and without an explicitly qualified reaction-core scaffold; identical external atom slots, vocabulary, size/closure support, exact program checker, novelty/diversity metrics and generation budget. Preserve every failed attempt.",
        "limits": [
            "New neural context is an exploratory representation diagnostic, not a replacement for product/component novelty.",
            "A count prior may legitimately sample a previously observed count combination while producing a novel graph. Unseen conditioning combinations in every family are not required by the user goal.",
            "Registry recombination products are context proposals and are never counted as neural outputs.",
            "The count-prior/source-layout comparison changes conditioning and tensor shapes; the Ugi contrast does not isolate the causal benefit of fixed reaction cores.",
            "All unique totals sum per-family counts. No model promotion, all-family chemical realism, experimental synthesis or complete L2/L3 closure is established.",
        ],
    }
    for p in inputs.values():
        resolve_pin(p, ROOT, label="final findings input")
    path = OUT / "findings.json"
    if path.exists():
        raise ValueError("findings output exists")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "totals": totals,
                "generated_context_coverage_families": payload[
                    "generated_context_coverage_families"
                ],
                "expanded_context_coverage_families": payload["expanded_context_coverage_families"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
