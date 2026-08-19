# Modules that must not be refactored

Two separate mechanisms freeze code in this repository. Both were discovered by a gate catching a
migration batch, not by reading, so treat this list as authoritative over intuition.

## 1. Hash-pinned source files

An artifact recorded the SHA-256 of the code that produced it. Editing one of these -- even a pure
refactor changing no behaviour -- breaks that artifact's provenance and fails `make verify-pins`.

Note `bio/liver.py`, `bio/muscle.py` and `bio/vaccine.py` appear here despite being imported by no
module, script or test. They look like dead code and are not removable.

- `scripts/modal_phase1_product_cuda_preflight.py`
- `scripts/modal_phase1_ugi_joint_sparse_v2.py`
- `scripts/phase1_adjudicate_ugi_morphology_proposal_challenger.py`
- `scripts/phase1_audit_ugi_distributional_applicability_v3.py`
- `scripts/phase1_audit_ugi_interpolative_conformal.py`
- `src/forge/bio/endpoint.py`
- `src/forge/bio/liver.py`
- `src/forge/bio/muscle.py`
- `src/forge/bio/ugi_distributional_applicability_v3.py`
- `src/forge/bio/ugi_interpolative_conformal.py`
- `src/forge/bio/ugi_morphology_proposal_challenger_adjudication.py`
- `src/forge/bio/vaccine.py`
- `src/forge/route/planner.py`
- `src/forge/route/planner_cache.py`
- `src/forge/route/qualified_forward.py`
- `src/forge/route/ugi3_exact_evidence_source.py`
- `src/forge/route/ugi3_hybrid_search.py`

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
