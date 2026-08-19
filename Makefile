.PHONY: help vendor vendor-partial verify verify-partial verify-pins verify-pins-code archive-pins m0-03-r0-reconcile m0-03 m0-03-reproduce \
	m0-04 m0-05 m0-06 \
	m0-06-sparse m0-06-sparse-full-support m0-06-ring-support \
	m0-05-score \
	m0-04-source-platform-audit m0-09 \
	m0-07-agile-reconciliation m0-07-oracle-splits m0-07-oracle-classical \
	m0-07-lantern-reproduction m0-07-auxiliary-supervision \
	m0-07-oracle-graph-corpus m0-07-oracle-graph-profile \
	m0-07-oracle-graph-cache m0-07-oracle-graph-pretraining-profile \
	m0-07-oracle-graph-pretraining m0-07-oracle-graph-matrix \
	m0-07-oracle-graph-transfer m0-07-oracle-graph-transfer-decision \
	m0-07-ugi-semantic-annotations \
	m0-07-oracle-freeze \
	m0-07-oracle-production \
	m0-08-endpoint-decision \
	m0-09-route-sources m0-09-source-priority m0-09-route-map m0-09-pmc-sources \
	m0-09-publisher-sources m0-09-source-review-index \
	m0-09-paper-route-reviews m0-09-ugi3-capability \
	m0-09-ugi3-assembly-qualification \
	m0-09-agile-virtual-smiles m0-09-agile-virtual-ugi3-capability \
	m0-09-agile-virtual-ugi3-component-programs \
	m0-09-agile-virtual-ugi3-terminal-queue \
	m0-09-ugi3-aldehyde-head-capability \
	m0-09-hydrophobic-motif-transfer m0-09-lnpdb-head-transfer \
	m0-09-l2-supervision-decision \
	m0-10-flower-transfer-pilot \
	phase1-data phase1-product-prelaunch-audit phase1-oracle-campaign-selection \
	phase1-v5-canonical-representation-audit phase1-v5-tree-traversal-comparison \
	phase1-product-overfit phase1-product-smoke phase1-product-cuda-preflight \
	phase1-product-pretrain \
	manuscript-pdf manuscript-iclr paper-verify paper-bundle code-survey test-baseline-report \
	doctor experiment-smoke phase1-corpus-run phase1-training-smoke \
	phase1-training-modal-plan phase1-sampling-smoke phase1-sampling-reproduce \
	typecheck check-core \
	test lint fmt clean

help:
	@echo "FORGE — authorized M0 and bounded Phase 1 work. Read AGENTS.md first."
	@echo ""
	@echo "  make vendor          copy hash-pinned source assets into data/vendor/"
	@echo "  make vendor-partial  same, tolerating assets only the origin workstation holds"
	@echo "  make verify          re-hash vendored assets against MANIFEST.json"
	@echo "  make verify-partial  same, treating an absent optional asset as not a failure"
	@echo "  make verify-pins     re-hash every input a result artifact pins; drift is fatal"
	@echo "  make verify-pins-code same, extended over configs/ (which pin their own source)"
	@echo "  make archive-pins    recover pinned code bytes from git into the frozen-code archive"
	@echo "  make m0-03-r0-reconcile build the corrected constitutional R0 corpus"
	@echo "  make m0-03           validate the frozen R0 splits (create only if absent)"
	@echo "  make m0-03-reproduce rebuild M0-03 in memory and require byte identity"
	@echo "  make m0-04           run the non-circular R1-prime anchoring audit"
	@echo "  make m0-04-source-platform-audit explain source-platform registry coverage"
	@echo "  make m0-05           build the blinded decomposition-precision review packet"
	@echo "  make m0-05-adjudicate build the source-grounded chemistry admission gate"
	@echo "  make m0-05-score REVIEWS='reviewer1.csv reviewer2.csv' score human reviews"
	@echo "  make m0-06           run the bounded N=64/N=96 DeFoG feasibility gate"
	@echo "  make m0-06-sparse    test the sparse hierarchical topology replacement"
	@echo "  make m0-06-sparse-full-support test F, Si, and aromatic restoration support"
	@echo "  make m0-06-ring-support freeze the lipid-native Ugi ring-topology gate"
	@echo "  make m0-07-agile-reconciliation reconcile AGILE assays before oracle fitting"
	@echo "  make m0-07-oracle-splits freeze leak-aware oracle evaluation splits"
	@echo "  make m0-07-oracle-classical run the classical oracle matrix lane"
	@echo "  make m0-07-lantern-reproduction reproduce the audit-only released checkpoint"
	@echo "  make m0-07-auxiliary-supervision audit compatible external oracle supervision"
	@echo "  make m0-07-oracle-graph-corpus freeze leakage-safe graph inputs"
	@echo "  make m0-07-oracle-graph-profile run the train-only graph runtime gate"
	@echo "  make m0-07-oracle-graph-cache build the deterministic numeric tensor cache"
	@echo "  make m0-07-oracle-graph-pretraining-profile profile label-free R0 pretraining"
	@echo "  make m0-07-oracle-graph-pretraining run full label-free R0 pretraining"
	@echo "  make m0-07-oracle-graph-matrix fit the leakage-safe supervised graph matrix"
	@echo "  make m0-07-oracle-graph-transfer compare frozen and fine-tuned R0 encoders"
	@echo "  make m0-07-oracle-graph-transfer-decision freeze transfer-lane disposition"
	@echo "  make m0-07-ugi-semantic-annotations freeze exact-source Ugi graph semantics"
	@echo "  make m0-07-oracle-freeze freeze one oracle and its applicability policy"
	@echo "  make m0-07-oracle-production refit and hash the selected production oracle"
	@echo "  make m0-08-endpoint-decision freeze the endpoint package without selecting one"
	@echo "  make m0-09           build the L2 supervision inventory"
	@echo "  make m0-09-route-sources build the LNPDB source and component ledgers"
	@echo "  make m0-09-source-priority rank source review using explicit evidence only"
	@echo "  make m0-09-route-map build AGILE routes and the all-LNPDB route-awareness map"
	@echo "  make m0-09-pmc-sources acquire and hash the 26 PMC-linked source records"
	@echo "  make m0-09-publisher-sources acquire publisher-hosted supplementary files"
	@echo "  make m0-09-source-review-index build the verified chemistry review queue"
	@echo "  make m0-09-paper-route-reviews validate reviewed subcomponent routes"
	@echo "  make m0-09-ugi3-capability build the bounded Ugi-3 precursor capability audit"
	@echo "  make m0-09-ugi3-assembly-qualification qualify all 1,200 AGILE assemblies"
	@echo "  make m0-09-agile-virtual-smiles extract the 12,276 virtual candidate SMILES"
	@echo "  make m0-09-agile-virtual-ugi3-capability audit their Ugi components and closure"
	@echo "  make m0-09-agile-virtual-ugi3-component-programs project recursive component routes"
	@echo "  make m0-09-agile-virtual-ugi3-terminal-queue build the deduplicated L3 queue"
	@echo "  make m0-09-ugi3-aldehyde-head-capability audit every aldehyde and amine head"
	@echo "  make m0-09-hydrophobic-motif-transfer run the fixed cross-platform tail pilot"
	@echo "  make m0-09-lnpdb-head-transfer freeze the bounded LNPDB head-transfer queue"
	@echo "  make m0-09-l2-supervision-decision freeze architecture and stopping gates"
	@echo "  make m0-10-flower-transfer-pilot audit FlowER transfer readiness and demotion"
	@echo "  make phase1-data      freeze broad product and exact Ugi-L1 training inputs"
	@echo "  make phase1-product-prelaunch-audit scan every R1 product before GPU training"
	@echo "  make phase1-v5-canonical-representation-audit test sparse encoding invariance"
	@echo "  make phase1-v5-tree-traversal-comparison compare BFS and preorder encodings"
	@echo "  make phase1-oracle-campaign-selection select the HeLa campaign oracle"
	@echo "  make phase1-product-overfit run the mandatory tiny-overfit gate"
	@echo "  make phase1-product-smoke run bounded local sparse-flow training"
	@echo "  make phase1-product-cuda-preflight run the deterministic L4 device gate"
	@echo "  make phase1-product-pretrain run recoverable full-corpus L4 training"
	@echo "  make doctor          verify all registered experiment inputs"
	@echo "  make experiment-smoke run and verify the local runner smoke test"
	@echo "  make phase1-corpus-run rebuild Phase 1 corpus artifacts in runs/"
	@echo "  make phase1-training-smoke verify the resumable CPU training DAG"
	@echo "  make phase1-training-modal-plan dry-run the pinned production L4 DAG"
	@echo "  make phase1-sampling-smoke run and verify four diagnostic samples"
	@echo "  make phase1-sampling-reproduce require byte-identical sampling outputs"
	@echo "  make check-core      fast architecture, runner, and provenance gate"
	@echo "  make paper-verify    verify the exact paper and recursive evidence graph"
	@echo "  make manuscript-pdf build the authoritative ICLR paper in isolation"
	@echo "  make paper-bundle    build and compile-check a deterministic Overleaf bundle"
	@echo "  make code-survey     classify code against the paper and supported CLI roots"
	@echo "  make test-baseline-report compare the last clean-cache pytest run with known blockers"
	@echo "  make test            pytest"
	@echo "  make lint / fmt      ruff / black"

vendor:
	python3 -m forge.cli data vendor

vendor-partial:
	python3 -m forge.cli data vendor --allow-partial

verify:
	python3 -m forge.cli data verify

verify-partial:
	python3 -m forge.cli data verify --allow-partial

# Re-hash every input a result artifact declares. Drift means a supposedly-frozen byte moved and
# is always fatal; absence is reported but tolerated, because many pinned inputs live only on the
# workstation that produced them. Run this after any refactor.
verify-pins:
	python3 -m forge.cli provenance verify --expect-verified $(EXPECT_PINS)

# The same check extended over `configs/`. A frozen config pins the source that produced it, and
# those pins outnumber the result-declared ones ~4:1 but went unscanned, so drift in them was
# invisible: 216 pins over 117 files had gone stale without any gate noticing. Archiving the
# recoverable bytes retired 124 of them; the rest exist in no commit and need a reviewed
# `docs/known_artifact_drift.json` entry each, which is a provenance judgment, not a chore.
# The ratchet holds that backlog flat -- it fails the moment drift grows.
verify-pins-code:
	python3 -m forge.cli provenance verify --code \
		--expect-verified $(EXPECT_PINS_ALL) --allow-drift $(CODE_DRIFT_BACKLOG)

# Recover pinned code/config bytes from git objects into the content-addressed archive. Idempotent
# and additive: a blob is admitted on SHA-256 equality alone. Run after any refactor that edits a
# file some frozen config or result pins.
archive-pins:
	python3 -m forge.cli provenance archive

doctor:
	python3 -m forge.cli doctor

experiment-smoke:
	python3 -m forge.cli experiment run installation-smoke --profile smoke --resume
	python3 -m forge.cli experiment verify installation-smoke --profile smoke

phase1-corpus-run:
	python3 -m forge.cli experiment run phase1-corpus --profile full --resume
	python3 -m forge.cli experiment verify phase1-corpus --profile full

phase1-training-smoke:
	python3 -m forge.cli experiment run phase1-training-smoke --profile smoke --resume
	python3 -m forge.cli experiment verify phase1-training-smoke --profile smoke

phase1-training-modal-plan:
	python3 -m forge.cli experiment plan phase1-training-production --profile full --backend modal

phase1-sampling-smoke:
	python3 -m forge.cli experiment run phase1-sampling --profile smoke --resume
	python3 -m forge.cli experiment verify phase1-sampling --profile smoke

phase1-sampling-reproduce:
	python3 -m forge.cli experiment reproduce phase1-sampling --profile smoke

typecheck:
	python3 -m mypy src/forge/core src/forge/chem src/forge/assembly \
		src/forge/bio src/forge/corpus src/forge/generate src/forge/experiment \
		src/forge/maintenance src/forge/paper src/forge/provenance src/forge/cli.py

check-core: verify-pins typecheck
	python3 -m ruff check src/forge/core src/forge/chem src/forge/assembly \
		src/forge/bio src/forge/potency src/forge/corpus src/forge/generate \
		src/forge/experiment src/forge/maintenance \
		src/forge/paper src/forge/provenance src/forge/cli.py \
		src/forge/data/vendor.py src/forge/experiment/modal_app.py \
		src/forge/provenance tests/test_architecture_boundaries.py \
		tests/test_assembly_ugi3.py tests/test_core_hashing.py \
		tests/test_core_provenance_archive.py tests/test_experiment_runner.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_seed.py tests/test_maintenance_test_baseline.py \
		tests/test_experiment_spec.py tests/test_paper_reproduction.py
	python3 -m pytest -q tests/test_architecture_boundaries.py tests/test_assembly_ugi3.py \
		tests/test_core_hashing.py tests/test_core_provenance_archive.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_runner.py \
		tests/test_experiment_seed.py tests/test_maintenance_test_baseline.py \
		tests/test_experiment_spec.py tests/test_paper_reproduction.py \
		tests/test_phase1_product_l1_data.py tests/test_pinned_sources_are_tracked.py

EXPECT_PINS ?= 742

# `verify-pins-code` covers results/ + docs/provenance/ + configs/. Both numbers are ratchets:
# raise EXPECT_PINS_ALL as pins are recovered, lower CODE_DRIFT_BACKLOG as unrecoverable ones get
# reviewed entries. Never raise CODE_DRIFT_BACKLOG -- a growing count means a frozen byte moved.
EXPECT_PINS_ALL ?= 2092
# All 14 remaining sit under eight configs whose schema carries no `status` field, so "the artifact
# is frozen" is not asserted and the acceptance bar cannot be applied. Repairing that schema
# regression -- `fresh_pool_route_coverage` carries `status` and `task` at v1-v3 and drops both from
# v4 -- is the prerequisite for retiring them.
CODE_DRIFT_BACKLOG ?= 14

m0-03-r0-reconcile:
	PYTHONPATH=src python3 scripts/m0_03_reconcile_r0.py

m0-03:
	PYTHONPATH=src python3 scripts/m0_03_freeze_r0_splits.py

m0-03-reproduce:
	PYTHONPATH=src python3 scripts/m0_03_freeze_r0_splits.py --reproduce

m0-04:
	PYTHONPATH=src python3 scripts/m0_04_r1_prime_audit.py

m0-04-source-platform-audit:
	PYTHONPATH=src python3 scripts/m0_04_source_platform_registry_audit.py

m0-05:
	PYTHONPATH=src python3 scripts/m0_05_decomposition_precision_audit.py

m0-05-adjudicate: m0-07-agile-reconciliation
	PYTHONPATH=src python3 scripts/m0_05_source_evidence_adjudication.py

m0-05-score:
	@test -n "$(REVIEWS)" || (echo "Set REVIEWS='reviewer1.csv reviewer2.csv'"; exit 2)
	PYTHONPATH=src python3 scripts/m0_05_score_decomposition_reviews.py $(REVIEWS)

m0-06:
	PYTHONPATH=src python3 scripts/m0_06_defog_feasibility.py

m0-06-sparse:
	PYTHONPATH=src python3 scripts/m0_06_sparse_topology_feasibility.py

m0-06-sparse-full-support:
	PYTHONPATH=src python3 scripts/m0_06_sparse_topology_feasibility.py \
		--config configs/model/m0_06_sparse_topology_full_support.json \
		--output-dir results/m0_06_sparse_full_support

m0-06-ring-support:
	PYTHONPATH=src python3 scripts/m0_06_lipid_ring_support.py

m0-07-agile-reconciliation:
	PYTHONPATH=src python3 scripts/m0_07_agile_label_reconciliation.py

m0-07-oracle-splits: m0-07-agile-reconciliation
	PYTHONPATH=src python3 scripts/m0_07_freeze_oracle_splits.py

m0-07-oracle-classical: m0-07-oracle-splits
	PYTHONPATH=src python3 scripts/m0_07_run_classical_oracle_matrix.py

m0-07-lantern-reproduction: m0-07-oracle-splits
	PYTHONPATH=src python3 scripts/m0_07_reproduce_lantern.py

m0-07-auxiliary-supervision:
	PYTHONPATH=src python3 scripts/m0_07_auxiliary_supervision_audit.py

m0-07-oracle-graph-corpus:
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_corpus.py

m0-07-oracle-graph-profile: m0-07-oracle-graph-corpus
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_profile.py

m0-07-oracle-graph-cache: m0-07-oracle-graph-profile
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_cache.py

m0-07-oracle-graph-pretraining-profile: m0-07-oracle-graph-profile
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_pretraining.py

m0-07-oracle-graph-pretraining: m0-07-oracle-graph-profile
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_pretraining.py --full-training

m0-07-oracle-graph-matrix: m0-07-oracle-graph-profile
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_matrix.py

m0-07-oracle-graph-transfer: m0-07-oracle-graph-pretraining
	PYTHONPATH=src python3 scripts/m0_07_oracle_graph_transfer.py \
		--config configs/bio/m0_07_oracle_graph_transfer.json

m0-07-oracle-graph-transfer-decision:
	PYTHONPATH=src python3 scripts/m0_07_freeze_oracle_graph_transfer.py

m0-07-ugi-semantic-annotations:
	PYTHONPATH=src python3 scripts/m0_07_ugi_semantic_annotations.py

m0-07-oracle-freeze:
	PYTHONPATH=src python3 scripts/m0_07_freeze_oracle.py

m0-07-oracle-production:
	PYTHONPATH=src python3 scripts/m0_07_refit_oracle.py

m0-08-endpoint-decision:
	PYTHONPATH=src python3 scripts/m0_08_freeze_endpoint_decision.py

m0-09:
	PYTHONPATH=src python3 scripts/m0_09_l2_supervision_inventory.py

m0-09-route-sources:
	PYTHONPATH=src python3 scripts/m0_09_lnpdb_route_source_ledger.py

m0-09-source-priority:
	PYTHONPATH=src python3 scripts/m0_09_source_priority_queue.py

m0-09-route-map:
	PYTHONPATH=src python3 scripts/m0_09_route_awareness_map.py

m0-09-pmc-sources:
	PYTHONPATH=src python3 scripts/fetch_m0_09_pmc_source_packages.py

m0-09-publisher-sources:
	PYTHONPATH=src python3 scripts/fetch_m0_09_publisher_sources.py

m0-09-source-review-index:
	PYTHONPATH=src python3 scripts/m0_09_source_review_index.py

m0-09-paper-route-reviews:
	PYTHONPATH=src python3 scripts/m0_09_lnpdb_paper_route_reviews.py

m0-09-ugi3-capability:
	PYTHONPATH=src python3 scripts/m0_09_ugi3_precursor_capability.py

m0-09-ugi3-assembly-qualification:
	PYTHONPATH=src python3 scripts/m0_09_ugi3_assembly_qualification.py

m0-09-agile-virtual-smiles:
	PYTHONPATH=src python3 scripts/m0_09_extract_agile_virtual_smiles.py

m0-09-agile-virtual-ugi3-capability:
	PYTHONPATH=src python3 scripts/m0_09_agile_virtual_ugi3_capability.py

m0-09-agile-virtual-ugi3-component-programs: m0-09-agile-virtual-ugi3-capability
	PYTHONPATH=src python3 scripts/m0_09_agile_virtual_ugi3_component_programs.py

m0-09-agile-virtual-ugi3-terminal-queue: m0-09-agile-virtual-ugi3-component-programs
	PYTHONPATH=src python3 scripts/m0_09_agile_virtual_ugi3_terminal_queue.py

m0-09-ugi3-aldehyde-head-capability: m0-09-ugi3-assembly-qualification
	PYTHONPATH=src python3 scripts/m0_09_ugi3_aldehyde_head_capability.py

m0-09-hydrophobic-motif-transfer: m0-09-agile-virtual-ugi3-capability
	PYTHONPATH=src python3 scripts/m0_09_hydrophobic_motif_transfer.py

m0-09-lnpdb-head-transfer:
	PYTHONPATH=src python3 scripts/m0_09_lnpdb_head_transfer.py

m0-09-l2-supervision-decision:
	PYTHONPATH=src python3 scripts/m0_09_l2_supervision_decision.py

m0-10-flower-transfer-pilot:
	PYTHONPATH=src python3 scripts/m0_10_flower_transfer_pilot.py

phase1-data:
	PYTHONPATH=src python3 scripts/phase1_freeze_product_l1_data.py

phase1-product-prelaunch-audit:
	PYTHONPATH=src python3 scripts/phase1_audit_product_prelaunch.py

phase1-oracle-campaign-selection:
	PYTHONPATH=src python3 scripts/phase1_select_oracle_campaign.py

phase1-v5-canonical-representation-audit:
	PYTHONPATH=src python3 scripts/phase1_audit_canonical_representation.py

phase1-v5-tree-traversal-comparison:
	PYTHONPATH=src python3 scripts/phase1_compare_tree_traversals.py

phase1-product-overfit:
	PYTHONPATH=src python3 scripts/phase1_train_product_pretrain.py \
		--smoke --overfit-only --overwrite \
		--output-dir results/phase1/product_pretrain_overfit

phase1-product-smoke:
	PYTHONPATH=src python3 scripts/phase1_train_product_pretrain.py \
		--smoke --overwrite \
		--output-dir results/phase1/product_pretrain_smoke

phase1-product-cuda-preflight:
	python3 -m forge.cli experiment plan phase1-training-production --profile full --backend modal

phase1-product-pretrain:
	python3 -m forge.cli experiment run phase1-training-production --profile full --backend modal

paper-verify:
	python3 -m forge.cli paper verify

manuscript-pdf manuscript-iclr:
	python3 -m forge.cli paper build

paper-bundle:
	python3 -m forge.cli paper bundle

code-survey:
	python3 -m forge.cli maintenance survey \
		--output provenance/code-retirement/iclr2027.json

test-baseline-report:
	python3 -m forge.cli maintenance test-report \
		--output results/maintenance/bio_to_potency_migration_v1/test_baseline.json

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	python3 -m ruff check src tests scripts

fmt:
	python3 -m black src tests scripts && python3 -m ruff check --fix src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
