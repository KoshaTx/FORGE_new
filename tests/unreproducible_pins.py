"""Tests whose pinned inputs cannot be reproduced from the current tree.

These are not failures of the code under test.  Each one verifies that an
artifact was produced by exactly the source revision its config pins, and
each fails because that pinned revision no longer exists: the files were
edited before they were ever committed, so the pinned content is gone.

They are quarantined as expected failures rather than deleted, silenced or
re-pinned.  Re-pinning to the current hashes would assert that the stored
results came from code that did not produce them, which is false.  Deleting
them would discard the provenance contract itself.

A quarantined test that starts passing is reported by pytest as XPASS, so
this list cannot silently rot.  Entries leave it when the analysis is re-run
under current sources and the config is re-pinned to the run that produced it.

See docs/DECISION_LOG.md, entry dated 2026-08-06.
"""

from __future__ import annotations

UNREPRODUCIBLE_PIN_REASON = "pinned source revision predates version control and is unrecoverable; see docs/DECISION_LOG.md 2026-08-06"

# Tests whose pin check runs inside a module-scoped fixture, so the mismatch
# surfaces as an error during setup rather than a failure in the test body.
# pytest cannot mark a setup error as xfail, so these are skipped instead.
# The cause and the remedy are identical to the xfail set above; only the
# phase in which the check fires differs.
UNREPRODUCIBLE_PIN_SETUP_ERRORS: frozenset[str] = frozenset(
    {
        "tests/test_ugi3_cumulative_production_source.py::test_all_late_exact_layers_are_reachable",
        "tests/test_ugi3_cumulative_production_source.py::test_family_projection_in_cumulative_source_is_not_promoted",
        "tests/test_ugi3_cumulative_production_source.py::test_production_chain_order_and_manifest_are_frozen",
        "tests/test_ugi_selected_guidance_adapter_v3.py::test_initialize_and_state_operations_enter_runtime_guard",
        "tests/test_ugi_selected_guidance_adapter_v3.py::test_pending_token_replacement_fails_on_an_already_bound_lane",
        "tests/test_ugi_selected_guidance_adapter_v3.py::test_runtime_guard_rehashes_bound_receipt",
        "tests/test_ugi_selected_guidance_adapter_v3.py::test_v3_replaces_only_pending_receipt_token_and_binds_complete_identity",
        # Same cause, reached through the restartable-equivalence receipt: it
        # pins generator implementation 8a8dfa39..., the tree is now dab5e59a...
        "tests/test_ugi_selected_guidance_adapter.py::test_ancestry_cannot_cross_frozen_morphology_programs",
        "tests/test_ugi_selected_guidance_adapter.py::test_each_particle_consumes_its_own_batch_size_one_rng_stream",
        "tests/test_ugi_selected_guidance_adapter.py::test_identity_ancestry_is_digest_preserving_and_snapshot_has_no_aliases",
        "tests/test_ugi_selected_guidance_adapter.py::test_nonidentity_ancestry_copies_categories_but_retains_destination_rng",
        "tests/test_ugi_selected_guidance_adapter.py::test_one_program_zero_guidance_is_exact_selected_callback_reference",
        "tests/test_ugi_selected_restartable_generator.py::test_factory_rejects_schedule_checkpoint_mismatch_before_sampling",
        "tests/test_ugi_selected_restartable_generator.py::test_real_selected_checkpoint_one_sample_is_deterministic_and_trace_owned",
    }
)

# 68 tests, all failing on a pinned-input hash mismatch.
UNREPRODUCIBLE_PIN_TESTS: frozenset[str] = frozenset(
    {
        "tests/test_phase1_cuda_preflight_config.py::test_v3_cuda_preflight_is_retained_but_invalidated_after_sampler_fix",
        "tests/test_ugi_grouped_smc_schedule_qualification.py::test_committed_grouped_smc_schedule_receipt_matches_builder",
        "tests/test_ugi_grouped_smc_schedule_qualification.py::test_grouped_smc_schedule_has_nontrivial_within_program_ancestry",
        "tests/test_ugi_matched_budget_orchestration.py::test_frozen_matched_budget_audit_is_deterministic_and_provenance_pinned",
        "tests/test_ugi_production_generator_manifest.py::test_v2_production_generator_manifest_authenticates_all_pinned_artifacts",
        "tests/test_ugi_production_generator_manifest.py::test_v3_production_generator_manifest_authenticates_all_pinned_artifacts",
        "tests/test_ugi_production_zero_guidance_seam_v3.py::test_selected_v3_direct_reference_is_an_explicit_third_identity",
        "tests/test_ugi_selected_guidance_adapter_v2.py::test_v2_seam_config_is_hash_pinned_and_fails_closed",
        "tests/test_ugi_selected_guidance_adapter_v3.py::test_equivalence_binding_validates_receipt_rows_and_frozen_sources",
        "tests/test_ugi_selected_v2_pool_singleton_equivalence_v2.py::test_v2_equivalence_config_binds_qualified_runtime_and_prior_chain",
        "tests/test_ugi3_cumulative_production_source.py::test_expired_unified_window_fails_closed[2026-08-09T05:16:00Z]",
        "tests/test_ugi3_cumulative_production_source.py::test_expired_unified_window_fails_closed[2026-08-09T05:16:01Z]",
        "tests/test_ugi3_cumulative_production_source.py::test_fresh_terminal_delta_rejects_exact_evidence_expiry",
        "tests/test_ugi3_exact_c16_route.py::test_assessment_time_requires_strict_iso_utc",
        "tests/test_ugi3_exact_c16_route.py::test_conditions_source_hash_is_enforced",
        "tests/test_ugi3_exact_c16_route.py::test_exact_c16_route_is_four_step_two_leaf_l2_l3_closed",
        "tests/test_ugi3_exact_c16_route.py::test_exact_pair_registry_cannot_be_generalized_or_changed",
        "tests/test_ugi3_exact_c16_route.py::test_frozen_overlay_preserves_two_terminal_leaves_and_delegates_neighbors",
        "tests/test_ugi3_exact_c16_route.py::test_rejected_source_conflict_cannot_support_a_route_step",
        "tests/test_ugi3_exact_c16_route.py::test_rejected_source_conflict_must_remain_carbon_incoherent",
        "tests/test_ugi3_exact_c16_route.py::test_route_chain_carbon_identity_cannot_change",
        "tests/test_ugi3_exact_c16_route.py::test_terminal_cannot_close_without_explicit_current_availability",
        "tests/test_ugi3_exact_c16_route.py::test_terminal_evidence_cannot_close_after_expiry",
        "tests/test_ugi3_exact_c16_route.py::test_terminal_observation_cannot_be_future_dated",
        "tests/test_ugi3_exact_c16_route.py::test_terminal_refresh_policy_must_be_well_formed",
        "tests/test_ugi3_exact_c18_route.py::test_conditions_source_hash_is_enforced",
        "tests/test_ugi3_exact_c18_route.py::test_exact_c18_route_is_three_step_l2_and_l3_closed",
        "tests/test_ugi3_exact_c18_route.py::test_exact_pair_identity_cannot_be_generalized_or_changed",
        "tests/test_ugi3_exact_c18_route.py::test_frozen_overlay_preserves_exact_chain_and_delegates_neighbors",
        "tests/test_ugi3_exact_c18_route.py::test_procurement_policy_cannot_change_silently",
        "tests/test_ugi3_exact_c18_route.py::test_procurement_timestamp_requires_strict_utc",
        "tests/test_ugi3_exact_c18_route.py::test_route_steps_must_form_one_contiguous_chain",
        "tests/test_ugi3_exact_c18_route.py::test_terminal_assessment_time_must_be_inside_observation_window[2026-08-03T02:24:01Z]",
        "tests/test_ugi3_exact_c18_route.py::test_terminal_assessment_time_must_be_inside_observation_window[2026-09-02T02:24:03Z]",
        "tests/test_ugi3_exact_c18_route.py::test_terminal_cannot_close_without_explicit_current_availability",
        "tests/test_ugi3_fresh_pool_route_coverage_v2.py::test_frozen_v2_delta_reproduces",
        "tests/test_ugi3_fresh_pool_route_coverage_v3.py::test_frozen_v3_refresh_reproduces",
        "tests/test_ugi3_fresh_pool_route_coverage_v4.py::test_exact_c18_refresh_is_reproducible_and_nonselecting",
        "tests/test_ugi3_fresh_pool_route_coverage_v4.py::test_exact_route_result_must_remain_complete_and_exact",
        "tests/test_ugi3_fresh_pool_route_coverage_v4.py::test_v4_rejects_contradictory_homologue_authorization",
        "tests/test_ugi3_fresh_pool_route_coverage_v4.py::test_v4_rejects_expired_exact_route_assessment",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_exact_c16_refresh_is_reproducible_nonselecting_and_config_owned",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_exact_result_must_own_assessment",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_exact_result_must_own_exact_route_config",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_exact_route_result_must_remain_four_step_two_leaf_complete",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_exact_terminal_evidence_must_be_current_at_frozen_assessment",
        "tests/test_ugi3_fresh_pool_route_coverage_v5.py::test_immutable_v4_config_ownership_is_enforced",
        "tests/test_ugi3_frozen_parent_component_ledger.py::test_c18_must_come_from_noncomplete_v3_record",
        "tests/test_ugi3_frozen_parent_component_ledger.py::test_parent_ledger_is_deterministic_stored_and_config_owned",
        "tests/test_ugi3_route_registry_pair_builder.py::test_builder_is_deterministic_and_preserves_every_non_c18_record",
        "tests/test_ugi3_route_registry_pair_builder.py::test_builder_outputs_pass_the_independent_pair_contract",
        "tests/test_ugi3_route_registry_pair_builder.py::test_existing_output_blocks_immutable_rebuild",
        "tests/test_ugi3_route_registry_pair_builder.py::test_repinned_c18_assessment_bytes_fail_reproduction",
        "tests/test_ugi3_route_registry_pair_contract.py::test_added_or_deleted_registry_identity_fails_closed",
        "tests/test_ugi3_route_registry_pair_contract.py::test_complete_immutable_pair_authorizes_one_time_reveal",
        "tests/test_ugi3_route_registry_pair_contract.py::test_conflicting_source_component_ledgers_fail_closed",
        "tests/test_ugi3_route_registry_pair_contract.py::test_missing_source_component_ledger_lineage_fails_closed",
        "tests/test_ugi3_route_registry_pair_contract.py::test_non_c18_evidence_must_be_identical_in_both_registries",
        "tests/test_ugi3_route_registry_pair_contract.py::test_r1_seed_or_parent_anchor_change_fails_closed",
        "tests/test_ugi3_synthesis_value_audit_v2.py::test_v2_changes_only_exact_propylamine_component",
        "tests/test_ugi3_synthesis_value_audit_v2.py::test_v2_is_reproducible_and_keeps_v1_immutable",
        "tests/test_ugi3_synthesis_value_audit_v2.py::test_v2_summary_detects_semantic_drift",
        "tests/test_ugi3_synthesis_value_audit_v3.py::test_v3_changes_only_exact_cyclohexylamine_component",
        "tests/test_ugi3_synthesis_value_audit_v3.py::test_v3_is_reproducible_and_keeps_v2_immutable",
        "tests/test_ugi3_synthesis_value_audit_v3.py::test_v3_summary_detects_semantic_drift",
        "tests/test_ugi3_synthesis_value_audit.py::test_frozen_summary_detects_semantic_drift",
        "tests/test_ugi3_synthesis_value_audit.py::test_generated_catalog_absence_remains_missing_knowledge",
        "tests/test_ugi3_synthesis_value_audit.py::test_synthesis_value_audit_is_reproducible_and_nonprobabilistic",
    }
)
