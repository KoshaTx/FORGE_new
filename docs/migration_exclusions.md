# Modules that must not be refactored

Two separate mechanisms freeze code in this repository. Both were discovered by a gate catching a
migration batch, not by reading, so treat this list as authoritative over intuition.

## 1. Hash-pinned source files

An artifact recorded the SHA-256 of the code that produced it. Editing one of these -- even a pure
refactor changing no behaviour -- breaks that artifact's provenance and fails `make verify-pins`.

Note `bio/liver.py`, `bio/muscle.py` and `bio/vaccine.py` appear here despite being imported by no
module, script or test. They look like dead code and are not removable.
- `configs/assembly/m0_09_ugi3_assembly_qualification.json`
- `configs/assembly/ugi_variant.yaml`
- `configs/bio/m0_07_auxiliary_supervision.json`
- `configs/bio/m0_07_lantern_reproduction.json`
- `configs/bio/m0_07_oracle_classical.json`
- `configs/bio/m0_07_oracle_freeze.json`
- `configs/bio/m0_07_oracle_graph_cache.json`
- `configs/bio/m0_07_oracle_graph_corpus.json`
- `configs/bio/m0_07_oracle_graph_matrix.json`
- `configs/bio/m0_07_oracle_graph_pretraining.json`
- `configs/bio/m0_07_oracle_graph_profile.json`
- `configs/bio/m0_07_oracle_graph_transfer.json`
- `configs/bio/m0_07_oracle_graph_transfer_decision.json`
- `configs/bio/m0_07_oracle_production.json`
- `configs/bio/m0_07_oracle_splits.json`
- `configs/bio/m0_07_ugi_semantic_annotations.json`
- `configs/bio/m0_08_endpoint_decision.json`
- `configs/bio/phase1_oracle_campaign_selection.json`
- `configs/bio/phase1_ugi_branch_exploration_applicability_v1.json`
- `configs/bio/phase1_ugi_distributional_applicability_v3.json`
- `configs/bio/phase1_ugi_hela_potency_diagnostic_authorization_v1.json`
- `configs/bio/phase1_ugi_interpolative_conformal_v1.json`
- `configs/bio/phase1_ugi_morphology_proposal_challenger_adjudication_v1.json`
- `configs/bio/phase1_ugi_morphology_proposal_challenger_confirmation_analysis_v1.json`
- `configs/bio/phase1_ugi_morphology_proposal_confirmation_analysis_v1.json`
- `configs/bio/phase1_ugi_morphology_proposal_strength_sweep_v1.json`
- `configs/bio/phase1_ugi_production_full_support_rescoring_v3.json`
- `configs/bio/phase1_ugi_prospective_panel_v6.json`
- `configs/corpus/m0_03_r0_reconciliation.json`
- `configs/corpus/m0_04_r1_prime_audit.json`
- `configs/corpus/m0_05_decomposition_precision_audit.json`
- `configs/corpus/m0_05_source_evidence_adjudication.json`
- `configs/model/m0_06_defog_feasibility.json`
- `configs/model/m0_06_lipid_ring_support.json`
- `configs/model/m0_06_sparse_topology_feasibility.json`
- `configs/model/m0_06_sparse_topology_full_support.json`
- `configs/model/phase1_product_cuda_preflight.json`
- `configs/model/phase1_product_cuda_preflight_v2.json`
- `configs/model/phase1_product_cuda_preflight_v3.json`
- `configs/model/phase1_product_l1_data.json`
- `configs/model/phase1_product_prelaunch_audit.json`
- `configs/model/phase1_product_pretrain.json`
- `configs/model/phase1_product_pretrain_v2.json`
- `configs/model/phase1_product_pretrain_v3.json`
- `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v2.json`
- `configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json`
- `configs/model/phase1_ugi_candidate_eligibility_audit_v1.json`
- `configs/model/phase1_ugi_candidate_eligibility_audit_v2.json`
- `configs/model/phase1_ugi_component_expansion.json`
- `configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_v1.json`
- `configs/model/phase1_ugi_expanded_enumeration.json`
- `configs/model/phase1_ugi_expanded_exemplars.json`
- `configs/model/phase1_ugi_joint_sparse_balanced.json`
- `configs/model/phase1_ugi_joint_sparse_balanced_v2.json`
- `configs/model/phase1_ugi_joint_sparse_size_only_v1.json`
- `configs/model/phase1_ugi_morphology_proposal_challenger_confirmation_v1.json`
- `configs/model/phase1_ugi_morphology_proposal_challenger_schedule_v1.json`
- `configs/model/phase1_ugi_morphology_proposal_confirmation_v1.json`
- `configs/model/phase1_ugi_morphology_proposal_schedule_v1.json`
- `configs/model/phase1_ugi_postselection_provenance_audit_v1.json`
- `configs/model/phase1_ugi_product_l1_postselection_held_component_stress_v1.json`
- `configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json`
- `configs/route/graph2edits_l2_forward_resolver_v1.json`
- `configs/route/graph2edits_proposal_backend_candidate_v1.json`
- `configs/route/graph2edits_runtime_qualification_macos_arm64_py311_v1.json`
- `configs/route/graph2edits_usp50k_training_corpus_manifest_v1.json`
- `configs/route/m0_09_agile_virtual_ugi3_capability.json`
- `configs/route/m0_09_hydrophobic_motif_transfer.json`
- `configs/route/m0_09_l2_supervision_decision.json`
- `configs/route/m0_09_lnpdb_head_transfer.json`
- `configs/route/m0_09_source_priority.json`
- `configs/route/m0_09_ugi3_virtual_terminal_procurement.json`
- `configs/route/phase1_ugi3_complete_computational_dossiers.json`
- `configs/route/phase1_ugi3_exact_c16_qualified_reactions_v1.json`
- `configs/route/phase1_ugi3_exact_c18_qualified_reactions_v1.json`
- `configs/route/phase1_ugi3_exact_c18_route_v1.json`
- `configs/route/phase1_ugi3_exact_evidence_source.json`
- `configs/route/phase1_ugi3_exact_source_forward_verification.json`
- `configs/route/phase1_ugi3_hybrid_search.json`
- `configs/route/phase1_ugi3_production_registry_route_readiness.json`
- `configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json`
- `configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json`
- `configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json`
- `configs/route/phase1_ugi_bounded_hybrid_route_cascade_v1.json`
- `configs/route/phase1_ugi_synthesis_guidance_failure_audit_v1.json`
- `configs/route/single_step_proposal_lane_qualification_benchmark_v1.json`
- `configs/route/variants/ugi3_upstream_amine_formylation_exact_source_v1.json`
- `configs/route/variants/ugi3_upstream_esterification_exact_source_v1.json`
- `configs/route/variants/ugi3_upstream_formamide_dehydration_exact_source_v1.json`
- `configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json`
- `configs/verify/m0_10_flower_transfer_pilot.json`
- `scripts/build_single_step_proposal_benchmark_manifest.py`
- `scripts/modal_phase1_product_cuda_preflight.py`
- `scripts/modal_phase1_ugi_joint_sparse_size_only_v1.py`
- `scripts/modal_phase1_ugi_joint_sparse_v2.py`
- `scripts/phase1_adjudicate_ugi_morphology_proposal_challenger.py`
- `scripts/phase1_analyze_ugi_morphology_proposal_confirmation.py`
- `scripts/phase1_audit_ugi_candidate_eligibility.py`
- `scripts/phase1_audit_ugi_distributional_applicability_v3.py`
- `scripts/phase1_audit_ugi_interpolative_conformal.py`
- `scripts/phase1_freeze_ugi_morphology_proposal_challenger_schedule.py`
- `scripts/phase1_freeze_ugi_morphology_proposal_schedule.py`
- `scripts/phase1_rescore_ugi_branch_exploration_applicability_v1.py`
- `scripts/phase1_sweep_ugi_morphology_proposal_strength.py`
- `src/forge/bio/endpoint.py`
- `src/forge/bio/liver.py`
- `src/forge/bio/muscle.py`
- `src/forge/bio/ugi_branch_exploration_applicability_v1.py`
- `src/forge/bio/ugi_distributional_applicability_v3.py`
- `src/forge/bio/ugi_interpolative_conformal.py`
- `src/forge/bio/ugi_morphology_proposal_challenger_adjudication.py`
- `src/forge/bio/ugi_morphology_proposal_confirmation_analysis.py`
- `src/forge/bio/ugi_morphology_proposal_strength_sweep.py`
- `src/forge/bio/vaccine.py`
- `src/forge/product/ugi_candidate_eligibility.py`
- `src/forge/product/ugi_morphology_proposal_challenger_schedule.py`
- `src/forge/product/ugi_morphology_proposal_schedule.py`
- `src/forge/route/graph2edits_backend.py`
- `src/forge/route/planner.py`
- `src/forge/route/planner_cache.py`
- `src/forge/route/proposal_engine.py`
- `src/forge/route/qualified_forward.py`
- `src/forge/route/single_step_benchmark_manifest.py`
- `src/forge/route/ugi3_exact_evidence_source.py`
- `src/forge/route/ugi3_hybrid_search.py`
- `tests/test_m0_08_endpoints.py`
- `tests/test_ugi_branch_exploration_applicability_v1.py`
- `tests/test_ugi_candidate_eligibility.py`
- `tests/test_ugi_distributional_applicability_v3.py`
- `tests/test_ugi_interpolative_conformal.py`
- `tests/test_ugi_morphology_proposal_challenger_adjudication.py`
- `tests/test_ugi_morphology_proposal_challenger_schedule.py`
- `tests/test_ugi_morphology_proposal_confirmation_analysis.py`
- `tests/test_ugi_morphology_proposal_schedule.py`
- `tests/test_ugi_morphology_proposal_strength_sweep.py`

## 2. Modules inside the blinded-execution dependency manifest

`route/ugi3_route_saturation_blinded_execution.py` declares `RUNTIME_DEPENDENCY_MODULES`, the exact
set of `forge.*` modules its runner is permitted to load, and validates a manifest artifact recording
each one's digest by **exact set equality**. The point is to prove which code ran during a sealed
holdout -- that no renderer was loaded and nothing undeclared executed.

Adding an import to any module in this graph pulls `forge.core` into the loaded set and breaks that
equality. That is not a lint failure to fix by extending the list: changing the declared set is a
contract change that would require re-running the blinded execution, which a refactor may not do.

So these modules keep their local helpers until the sealed protocol is deliberately revisited.

- `forge.bio`
- `forge.bio.ugi_semantic_annotations`
- `forge.chemistry`
- `forge.data`
- `forge.data.r0_splits`
- `forge.data.r1_prime_audit`
- `forge.product`
- `forge.product.adapter_node_conditioning`
- `forge.product.canonical_representation_audit`
- `forge.product.defog_feasibility`
- `forge.product.lipid_context`
- `forge.product.lipid_support_skeleton`
- `forge.product.phase1_flow`
- `forge.product.phase1_tree_topology_flow`
- `forge.product.sparse_topology_feasibility`
- `forge.product.ugi_adapter_features`
- `forge.product.ugi_blinded_headless_sampling`
- `forge.product.ugi_chemistry_corpus`
- `forge.product.ugi_chemistry_flow`
- `forge.product.ugi_chemistry_interface`
- `forge.product.ugi_closure_placement`
- `forge.product.ugi_component_expansion`
- `forge.product.ugi_generated_components`
- `forge.product.ugi_held_component_gate`
- `forge.product.ugi_joint_sparse_flow`
- `forge.product.ugi_morphology_flow`
- `forge.product.ugi_morphology_program`
- `forge.product.ugi_training_cache`
- `forge.product.v5_sparse_representation`
- `forge.route`
- `forge.route.planner`
- `forge.route.qualified_forward`
- `forge.route.ugi3_exact_c18_route`
- `forge.route.ugi3_route_registry_pair_contract`
- `forge.route.ugi3_route_saturation_blinded_execution`
- `forge.value`
- `forge.value.synthesis`

### Consequence for the plan

`data/r1_prime_audit.py` is in this set. It is also the keystone the restructuring plan schedules for
decomposition -- 1,836 lines, 121 import sites. **That decomposition cannot proceed as written.**
Splitting it changes what the blinded runner loads, so it needs the sealed protocol addressed first,
or the split has to preserve the module as a facade that imports nothing new.
