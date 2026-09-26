"""Allow-listed multireaction experiment specifications."""

from __future__ import annotations

FOUNDATION_SPECIFICATIONS: dict[str, str] = {
    "phase1-multireaction-corpus": "experiments/phase1/multireaction/experiment.json",
    "phase1-bl-lx-reaction-enumerated-expansion": (
        "experiments/phase1/multireaction/bl_lx_reaction_enumerated_expansion.json"
    ),
    "phase1-bl-lx-mixed-repeat-expansion": (
        "experiments/phase1/multireaction/bl_lx_mixed_repeat_expansion.json"
    ),
    "phase1-bl-lx-model-support-mixed-repeat-expansion": (
        "experiments/phase1/multireaction/bl_lx_mixed_repeat_expansion_v2.json"
    ),
    "phase1-multireaction-training-smoke": ("experiments/phase1/multireaction/training_smoke.json"),
    "phase1-multireaction-overfit": "experiments/phase1/multireaction/overfit.json",
    "phase1-reaction-program-transformer-overfit": (
        "experiments/phase1/multireaction/transformer_overfit.json"
    ),
    "phase1-reaction-program-transformer-overfit-v2": (
        "experiments/phase1/multireaction/transformer_overfit_v2.json"
    ),
    "phase1-reaction-program-transformer-overfit-v3": (
        "experiments/phase1/multireaction/transformer_overfit_v3.json"
    ),
    "phase1-transformer-synthesis-program-production-design": (
        "experiments/phase1/multireaction/transformer_production_design.json"
    ),
    "phase1-transformer-synthesis-program-production": (
        "experiments/phase1/multireaction/transformer_production.json"
    ),
    "phase1-transformer-synthesis-program-production-h100": (
        "experiments/phase1/multireaction/transformer_production_h100.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-l4": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_l4.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-a100-40gb": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_a100_40gb.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-h100": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_h100.json"
    ),
    "phase1-finite-component-catalogue-baseline": (
        "experiments/phase1/multireaction/finite_component_catalogue_baseline.json"
    ),
    "phase1-conditional-role-dependence": (
        "experiments/phase1/multireaction/conditional_role_dependence.json"
    ),
    "phase1-learned-inventory-selector": (
        "experiments/phase1/multireaction/learned_inventory_selector.json"
    ),
    "phase1-learned-inventory-selector-h100-preflight": (
        "experiments/phase1/multireaction/learned_inventory_selector_h100_preflight.json"
    ),
    "phase1-local-chemistry-support": (
        "experiments/phase1/multireaction/local_chemistry_support.json"
    ),
    "phase1-local-morphology-support-v2": (
        "experiments/phase1/multireaction/local_morphology_support_v2.json"
    ),
    "phase1-local-morphology-resampling-smoke-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_smoke_v2.json"
    ),
    "phase1-local-morphology-resampling-h100-preflight-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_h100_preflight_v2.json"
    ),
    "phase1-local-morphology-resampling-seed0-h100-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_seed0_h100_v2.json"
    ),
    "phase1-local-chemistry-resampling": (
        "experiments/phase1/multireaction/local_chemistry_resampling.json"
    ),
    "phase1-local-chemistry-resampling-h100-preflight": (
        "experiments/phase1/multireaction/local_chemistry_resampling_h100_preflight.json"
    ),
    "phase1-transformer-mechanism-study": (
        "experiments/phase1/multireaction/transformer_mechanism_study.json"
    ),
    "phase1-transformer-mechanism-study-h100-preflight": (
        "experiments/phase1/multireaction/transformer_mechanism_study_h100_preflight.json"
    ),
    "phase1-bl-lx-repair-calibration-smoke": (
        "experiments/phase1/multireaction/bl_lx_repair_calibration_smoke.json"
    ),
    "phase1-bl-lx-repair-calibration-h100": (
        "experiments/phase1/multireaction/bl_lx_repair_calibration_h100.json"
    ),
    "phase1-bl-core-constraint-calibration-smoke": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_smoke.json"
    ),
    "phase1-bl-core-constraint-calibration-h100": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_h100.json"
    ),
    "phase1-bl-core-constraint-calibration-h100-preflight": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_h100_preflight.json"
    ),
    "phase1-ugi-train-exposure-calibration-smoke": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_smoke.json"
    ),
    "phase1-ugi-train-exposure-calibration-h100-preflight": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_h100_preflight.json"
    ),
    "phase1-ugi-train-exposure-calibration-h100": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_h100.json"
    ),
    "phase1-reaction-specialist-preflight-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_preflight_seed0_h100.json"
    ),
    "phase1-reaction-specialist-ugi-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_ugi_seed0_h100.json"
    ),
    "phase1-reaction-specialist-ugi-v0-evaluation-recovery-seed0-h100": (
        "experiments/phase1/multireaction/"
        "reaction_specialist_ugi_v0_evaluation_recovery_seed0_h100.json"
    ),
    "phase1-reaction-specialist-bl-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_bl_seed0_h100.json"
    ),
    "phase1-reaction-specialist-lx-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_lx_seed0_h100.json"
    ),
    "phase1-reaction-topology-specialist-ugi-h100-preflight-v2": (
        "experiments/phase1/multireaction/reaction_topology_specialist_ugi_h100_preflight_v2.json"
    ),
    "phase1-reaction-topology-specialist-ugi-smoke-v2": (
        "experiments/phase1/multireaction/reaction_topology_specialist_ugi_smoke_v2.json"
    ),
    "phase1-reaction-topology-specialist-ugi-seed0-h100-v2": (
        "experiments/phase1/multireaction/reaction_topology_specialist_ugi_seed0_h100_v2.json"
    ),
    "phase1-ugi-chemistry-specialist-seed0-h100-v3": (
        "experiments/phase1/multireaction/ugi_chemistry_specialist_seed0_h100_v3.json"
    ),
    "phase1-ugi-chemistry-specialist-evaluation-recovery-seed0-h100-v4": (
        "experiments/phase1/multireaction/"
        "ugi_chemistry_specialist_evaluation_recovery_seed0_h100_v4.json"
    ),
    "phase1-ugi-role-chemistry-prior-h100-preflight-v1": (
        "experiments/phase1/multireaction/ugi_role_chemistry_prior_h100_preflight_v1.json"
    ),
    "phase1-ugi-terminal-chemistry-temperature-h100-preflight-v1": (
        "experiments/phase1/multireaction/ugi_terminal_chemistry_temperature_h100_preflight_v1.json"
    ),
    "phase1-ugi-topology-conditioned-chemistry-flow-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_topology_conditioned_chemistry_flow_h100_preflight_v1.json"
    ),
    "phase1-ugi-learned-topology-then-chemistry-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_learned_topology_then_chemistry_h100_preflight_v1.json"
    ),
    "phase1-ugi-contextual-chemistry-specialist-seed0-h100-v4": (
        "experiments/phase1/multireaction/ugi_contextual_chemistry_specialist_seed0_h100_v4.json"
    ),
    "phase1-ugi-joint-lipid-specialist-seed0-h100-v5": (
        "experiments/phase1/multireaction/ugi_joint_lipid_specialist_seed0_h100_v5.json"
    ),
    "phase1-ugi-role-local-decoder-seed0-h100-v6": (
        "experiments/phase1/multireaction/ugi_role_local_decoder_seed0_h100_v6.json"
    ),
    "phase1-ugi-measured-only-full-model-seed0-h100-v7": (
        "experiments/phase1/multireaction/ugi_measured_only_full_model_seed0_h100_v7.json"
    ),
    "phase1-ugi-structured-topology-specialist-seed0-h100-v8": (
        "experiments/phase1/multireaction/ugi_structured_topology_specialist_seed0_h100_v8.json"
    ),
    "phase1-ugi-group-balanced-program-prior-seed0-h100-v1": (
        "experiments/phase1/multireaction/ugi_group_balanced_program_prior_seed0_h100_v1.json"
    ),
    "phase1-ugi-amine-semantic-program-seed0-h100-v1": (
        "experiments/phase1/multireaction/ugi_amine_semantic_program_seed0_h100_v1.json"
    ),
    "phase1-ugi-amine-semantic-joint-support-seed0-h100-v2": (
        "experiments/phase1/multireaction/ugi_amine_semantic_joint_support_seed0_h100_v2.json"
    ),
    "phase1-ugi-all-role-semantic-seed0-h100-v1": (
        "experiments/phase1/multireaction/ugi_all_role_semantic_seed0_h100_v1.json"
    ),
    "phase1-ugi-joint-all-role-semantic-seed0-h100-v2": (
        "experiments/phase1/multireaction/ugi_joint_all_role_semantic_seed0_h100_v2.json"
    ),
    "phase1-ugi-mog-semantic-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/ugi_mog_semantic_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-mog-joint-realism-seed0-h100-preflight-v2": (
        "experiments/phase1/multireaction/ugi_mog_joint_realism_seed0_h100_preflight_v2.json"
    ),
    "phase1-ugi-local-chemistry-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/" "ugi_local_chemistry_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-atom-local-chemistry-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_atom_local_chemistry_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-atom-trust-region-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/" "ugi_atom_trust_region_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-context-support-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/" "ugi_context_support_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-whole-head-support-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_whole_head_support_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-binary-whole-head-support-mog-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_binary_whole_head_support_mog_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-whole-head-trajectory-trust-region-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_whole_head_trajectory_trust_region_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-morphology-diversity-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/" "ugi_morphology_diversity_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-amine-substitution-semantic-seed0-h100-preflight-v3": (
        "experiments/phase1/multireaction/"
        "ugi_amine_substitution_semantic_seed0_h100_preflight_v3.json"
    ),
    "phase1-ugi-amine-donor-branch-factorial-seed0-h100-preflight-v1": (
        "experiments/phase1/multireaction/"
        "ugi_amine_donor_branch_factorial_seed0_h100_preflight_v1.json"
    ),
    "phase1-ugi-complete-semantic-seed0-h100-v1": (
        "experiments/phase1/multireaction/ugi_complete_semantic_seed0_h100_v1.json"
    ),
    "phase1-bl-core-constrained-production-smoke": (
        "experiments/phase1/multireaction/bl_core_constrained_production_smoke.json"
    ),
    "phase1-bl-core-constrained-production-h100-preflight": (
        "experiments/phase1/multireaction/bl_core_constrained_production_h100_preflight.json"
    ),
    "phase1-bl-core-constrained-production-h100": (
        "experiments/phase1/multireaction/bl_core_constrained_production_h100.json"
    ),
    "phase1-program-semantic-intervention": (
        "experiments/phase1/multireaction/program_semantic_intervention.json"
    ),
    "phase1-final-program-semantic-intervention-h100-preflight": (
        "experiments/phase1/multireaction/final_program_semantic_intervention_h100_preflight.json"
    ),
    "phase1-final-program-semantic-intervention": (
        "experiments/phase1/multireaction/final_program_semantic_intervention.json"
    ),
    "phase1-held-reaction-family-study": (
        "experiments/phase1/multireaction/held_reaction_family_study.json"
    ),
    "phase1-external-ugi-common-export": (
        "experiments/phase1/multireaction/external_ugi_common_export.json"
    ),
}


SHARED_PROGRAM_SPECIFICATIONS: dict[str, str] = {
    "phase1-shared-synthesis-program-representation": (
        "experiments/phase1/multireaction/shared_representation.json"
    ),
    "phase1-shared-synthesis-program-mixed-representation": (
        "experiments/phase1/multireaction/shared_mixed_representation.json"
    ),
    "phase1-shared-synthesis-program-mixed-training-design": (
        "experiments/phase1/multireaction/shared_mixed_training_design.json"
    ),
    "phase1-shared-synthesis-program-mixed-production-cache": (
        "experiments/phase1/multireaction/shared_mixed_production_cache.json"
    ),
    "phase1-shared-synthesis-program-mixed-training-smoke": (
        "experiments/phase1/multireaction/shared_mixed_training_smoke.json"
    ),
    "phase1-shared-synthesis-program-mixed-h100-training-profile": (
        "experiments/phase1/multireaction/shared_mixed_h100_training_profile.json"
    ),
    "phase1-shared-synthesis-program-mixed-h100-tf32-profile": (
        "experiments/phase1/multireaction/shared_mixed_h100_tf32_profile.json"
    ),
    "phase1-shared-synthesis-program-mixed-primary-seed0-h100": (
        "experiments/phase1/multireaction/shared_mixed_primary_seed0_h100.json"
    ),
    "phase1-shared-synthesis-program-mixed-primary-seed0-evaluation-h100": (
        "experiments/phase1/multireaction/shared_mixed_primary_seed0_evaluation_h100.json"
    ),
    "phase1-shared-mixed-role-morphology-smoke": (
        "experiments/phase1/multireaction/shared_mixed_role_morphology_smoke.json"
    ),
    "phase1-shared-mixed-role-morphology-seed0-h100": (
        "experiments/phase1/multireaction/shared_mixed_role_morphology_seed0_h100.json"
    ),
    "phase1-shared-bias-end-to-end-smoke": (
        "experiments/phase1/multireaction/shared_bias_end_to_end_smoke.json"
    ),
    "phase1-shared-bias-end-to-end-seed0-h100": (
        "experiments/phase1/multireaction/shared_bias_end_to_end_seed0_h100.json"
    ),
    "phase1-shared-bias-parallel-global-seed0-h100-v2": (
        "experiments/phase1/multireaction/shared_bias_parallel_global_seed0_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-global-seed0-evaluation-h100-v2": (
        "experiments/phase1/multireaction/shared_bias_parallel_global_seed0_evaluation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed1-core-saturation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed1_core_saturation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed2-core-saturation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed2_core_saturation_h100_v2.json"
    ),
    "phase1-shared-bias-cyclic-core-saturation-h100": (
        "experiments/phase1/multireaction/shared_bias_cyclic_core_saturation_h100.json"
    ),
    "phase1-shared-bias-shared-null-core-saturation-h100": (
        "experiments/phase1/multireaction/shared_bias_shared_null_core_saturation_h100.json"
    ),
    "phase1-shared-bias-ugi-only-core-saturation-h100": (
        "experiments/phase1/multireaction/shared_bias_ugi_only_core_saturation_h100.json"
    ),
    "phase1-shared-bias-program-role-core-saturation-seed1-h100": (
        "experiments/phase1/multireaction/shared_bias_program_role_core_saturation_seed1_h100.json"
    ),
    "phase1-shared-bias-program-role-core-saturation-seed2-h100": (
        "experiments/phase1/multireaction/shared_bias_program_role_core_saturation_seed2_h100.json"
    ),
    "phase1-transformer-mechanism-ablation-core-saturation-seed0-h100": (
        "experiments/phase1/multireaction/"
        "transformer_mechanism_ablation_core_saturation_seed0_h100.json"
    ),
    "phase1-transformer-mechanism-ablation-core-saturation-seed1-h100": (
        "experiments/phase1/multireaction/"
        "transformer_mechanism_ablation_core_saturation_seed1_h100.json"
    ),
    "phase1-transformer-mechanism-ablation-core-saturation-seed2-h100": (
        "experiments/phase1/multireaction/"
        "transformer_mechanism_ablation_core_saturation_seed2_h100.json"
    ),
    "phase1-transformer-production-ablation-core-saturation-h100": (
        "experiments/phase1/multireaction/transformer_production_ablation_core_saturation_h100.json"
    ),
    "phase1-transformer-production-ablation-core-saturation-seed0-h100": (
        "experiments/phase1/multireaction/"
        "transformer_production_ablation_core_saturation_seed0_h100.json"
    ),
    "phase1-transformer-production-ablation-core-saturation-seed1-h100": (
        "experiments/phase1/multireaction/"
        "transformer_production_ablation_core_saturation_seed1_h100.json"
    ),
    "phase1-transformer-production-ablation-core-saturation-seed2-h100": (
        "experiments/phase1/multireaction/"
        "transformer_production_ablation_core_saturation_seed2_h100.json"
    ),
    "phase1-shared-bias-parallel-global-seed0-core-saturation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_global_seed0_core_saturation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-evaluation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed0_evaluation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-core-saturation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed0_core_saturation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-h100-v2": (
        "experiments/phase1/multireaction/shared_bias_parallel_program_role_seed0_h100_v2.json"
    ),
    "phase1-shared-synthesis-program-integration": (
        "experiments/phase1/multireaction/shared_integration.json"
    ),
    "phase1-shared-synthesis-program-production-design": (
        "experiments/phase1/multireaction/shared_production_design.json"
    ),
    "phase1-shared-synthesis-program-production-cache": (
        "experiments/phase1/multireaction/shared_production_cache.json"
    ),
    "phase1-shared-synthesis-program-production-training": (
        "experiments/phase1/multireaction/shared_production_training.json"
    ),
    "phase1-shared-synthesis-program-production-accelerator-benchmark-a100-40gb": (
        "experiments/phase1/multireaction/shared_production_accelerator_benchmark_a100_40gb.json"
    ),
    "phase1-shared-synthesis-program-production-accelerator-benchmark-l4": (
        "experiments/phase1/multireaction/shared_production_accelerator_benchmark_l4.json"
    ),
}


__all__ = ["FOUNDATION_SPECIFICATIONS", "SHARED_PROGRAM_SPECIFICATIONS"]
