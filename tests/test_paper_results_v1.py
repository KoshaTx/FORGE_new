from __future__ import annotations

from forge_paper.results_v1 import (
    COMMON_METRICS,
    COMMON_ROW_SCHEMA,
    MECHANISM_METRICS,
    MECHANISM_ROW_SCHEMA,
    render_v1_results,
    write_v1_result_rows,
)

from forge.core.hashing import pin_record
from forge.core.io import write_json


def test_v1_renderer_writes_only_hash_verified_seed_rows(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.json"
    write_json(source, {"status": "pass"})
    config = repo / "renderer.json"
    write_json(
        config,
        {
            "schema_version": "forge.natbiotech_v1_renderer_config.v1",
            "bootstrap_seed": 3,
            "bootstrap_resamples": 100,
            "expected_seeds": [1],
            "common_method_order": ["method"],
            "mechanism_arm_order": ["arm"],
            "display_names": {"method": "Method", "arm": "Arm"},
            "held_reaction_family_is_secondary_not_hard_gate": True,
            "candidate_selection": False,
        },
    )
    rows = repo / "rows.jsonl.gz"
    source_pin = pin_record(source, repo)
    write_v1_result_rows(
        rows,
        [
            {
                "schema_version": COMMON_ROW_SCHEMA,
                "method_id": "method",
                "seed": 1,
                "metrics": {metric: 1.0 for metric in COMMON_METRICS},
                "source": source_pin,
                "route_source": source_pin,
                "route_scope": "bounded_forge_candidate_index",
            },
            {
                "schema_version": MECHANISM_ROW_SCHEMA,
                "arm_id": "arm",
                "seed": 1,
                "metrics": {metric: 2.0 for metric in MECHANISM_METRICS},
                "source": source_pin,
            },
        ],
    )
    output = repo / "generated"
    result = render_v1_results(config, repo, rows, output, strict=False)
    assert result["status"] == "partial_nonpublication_preview"
    assert (output / "common_ugi_benchmark_rows.tex").read_text().startswith("Method & 1.00")
    assert result["candidate_selection"] is False
