from __future__ import annotations

import json
from pathlib import Path

from forge.product.defog_feasibility import sha256_file

REPO = Path(__file__).resolve().parents[1]


def test_v5_representation_config_is_hash_pinned_and_nonredundant() -> None:
    config = json.loads((REPO / "configs/model/phase1_product_morphology_v5.json").read_text())

    assert config["status"] == "representation_selected_for_bounded_implementation_gate"
    for record in config["inputs"].values():
        path = REPO / record["path"]
        assert path.is_file()
        assert sha256_file(path) == record["sha256"]

    canonical = json.loads(
        (REPO / config["inputs"]["canonical_representation_audit"]["path"]).read_text()
    )
    traversal = json.loads(
        (REPO / config["inputs"]["tree_traversal_comparison"]["path"]).read_text()
    )
    assert canonical["schema_version"] == "phase1_v5_canonical_representation_audit.v2"
    assert canonical["status"] == "pass"
    assert canonical["policy"]["tree_traversal"] == "breadth_first_tree_preorder"
    assert "offspring" in canonical["policy"]["signature_arrays"]
    assert "parents" not in canonical["policy"]["signature_arrays"]
    assert "edges" not in canonical["policy"]["signature_arrays"]
    assert traversal["schema_version"] == "phase1_v5_traversal_representation_audit.v2"
    assert traversal["status"] == "pass"
    assert traversal["policy"]["selection_metrics_fold"] == "R0_cal"
    assert traversal["policy"]["r0_heldout_use"] == (
        "exact_invariance_checks_only_not_architecture_selection"
    )
    hybrid = traversal["traversals"]["breadth_first_tree_preorder"]
    pure_dfs = traversal["traversals"]["depth_first_preorder"]
    assert hybrid["r0_metrics"]["mean_root_distance_stretch"]["mean"] == 0.0
    assert pure_dfs["r0_metrics"]["mean_root_distance_stretch"]["mean"] > 0.0
    for artifact in (canonical, traversal):
        for implementation in artifact["implementation"].values():
            assert sha256_file(Path(implementation["path"])) == implementation["sha256"]

    representation = config["representation"]
    assert representation["spanning_tree"] == "deterministic_breadth_first"
    assert representation["serialization"] == "depth_first_preorder"
    assert representation["learned_parent_pointers"] is False
    assert representation["maximum_heavy_atoms"] == 282

    complexity = config["complexity_contract"]
    assert complexity["stored_graph_state"] == "O(N_plus_K)"
    assert complexity["end_to_end_linear_runtime_claimed"] is False
    assert complexity["dense_pairwise_ring_size_matrix_allowed_in_v5_sampling"] is False

    program = config["global_program"]
    assert "heavy_atom_count" in program["derived"]
    assert program["independent_heavy_atom_count_head"] is False
    assert "cycle_rank" in program["generated"]

    closures = config["closures"]
    assert closures["primary_representation"] == "sequential_unordered_endpoint_pairs"
    assert closures["closure_stub_sequence"] == "ablation_only"
    assert closures["completion_order"] == "strict_lexicographic_endpoint_pairs"
    assert closures["prefix_policy"] == (
        "renormalize_only_over_edges_with_an_exact_residual_completion"
    )
    assert closures["terminal_repair_allowed"] is False

    deferred = config["deferred"]
    assert deferred["biological_guidance"] is False
    assert deferred["l2_route_generation"] is False
    assert deferred["synthesis_value_guidance"] is False
