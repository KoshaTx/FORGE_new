"""Allow-listed product and L1 experiment specifications."""

from __future__ import annotations

PRODUCT_L1_SPECIFICATIONS: dict[str, str] = {
    "phase1-sampling": "experiments/phase1/product_l1/sampling.json",
    "phase1-training-production": "experiments/phase1/product_l1/training_production.json",
    "phase1-training-smoke": "experiments/phase1/product_l1/training_smoke.json",
    "phase1-ugi-v0-port-transformer-smoke-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_smoke_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-h100-preflight-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_h100_preflight_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-seed0-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_seed0_h100_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-seed0-sampling-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_seed0_sampling_v1.json"
    ),
    "phase1-ugi-tree-transformer-challenger-smoke-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_challenger_smoke_v1.json"
    ),
    "phase1-ugi-tree-transformer-calibration-program-draw-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_calibration_program_draw_v1.json"
    ),
    "phase1-ugi-tree-transformer-development-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_development_h100_v1.json"
    ),
    "phase1-ugi-tree-transformer-calibration-h100-preflight-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_calibration_h100_preflight_v1.json"
    ),
    "phase1-ugi-tree-calibration-dense-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_dense_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-relations-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_relations_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-consistency-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_consistency_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-masking-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_masking_h100_v1.json"
    ),
    "phase1-ugi-v0-calibration-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_calibration_h100_v1.json"
    ),
    "phase1-ugi-v0-current-program-comparison-h100-preflight-v1": (
        "experiments/phase1/product_l1/ugi_v0_current_program_comparison_h100_preflight_v1.json"
    ),
    "phase1-ugi-v0-current-program-comparison-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_current_program_comparison_h100_v1.json"
    ),
    "phase1-ugi-transformer-morphology-projection-comparison-h100-v1": (
        "experiments/phase1/product_l1/"
        "ugi_transformer_morphology_projection_comparison_h100_v1.json"
    ),
    "phase1-ugi-tree-relational-production-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_relational_production_h100_v1.json"
    ),
    "phase1-ugi-tree-relational-edge-constrained-resampling-seed0-h100-v1": (
        "experiments/phase1/product_l1/"
        "ugi_tree_relational_edge_constrained_resampling_seed0_h100_v1.json"
    ),
}


__all__ = ["PRODUCT_L1_SPECIFICATIONS"]
