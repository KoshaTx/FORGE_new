.PHONY: help vendor vendor-partial verify m0-03-r0-reconcile m0-03 m0-03-reproduce \
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
	manuscript-pdf \
	manuscript-iclr \
	test lint fmt clean

help:
	@echo "FORGE — Milestone M0 only. Read AGENTS.md before working here."
	@echo ""
	@echo "  make vendor          copy hash-pinned source assets into data/vendor/"
	@echo "  make vendor-partial  same, tolerating the absent 96 MB R1 file"
	@echo "  make verify          re-hash vendored assets against MANIFEST.json"
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
	@echo "  make manuscript-pdf  build the LaTeX FORGE working manuscript PDF"
	@echo "  make manuscript-iclr build the ICLR-format version of the same manuscript"
	@echo "  make test            pytest"
	@echo "  make lint / fmt      ruff / black"

vendor:
	python3 scripts/vendor.py

vendor-partial:
	python3 scripts/vendor.py --allow-partial

verify:
	python3 scripts/vendor.py --verify

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
	modal run scripts/modal_phase1_product_cuda_preflight.py

phase1-product-pretrain:
	modal run scripts/modal_phase1_product_pretrain.py

manuscript-pdf:
	pandoc manuscript/FORGE_Nature_Biotechnology_working_draft.md \
		--from=markdown --to=latex --standalone \
		--template=manuscript/latex/forge_natbiotech.template.tex \
		--lua-filter=manuscript/latex/forge_natbiotech_filter.lua \
		--output=manuscript/FORGE_Nature_Biotechnology_working_draft.tex
	xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=manuscript \
		manuscript/FORGE_Nature_Biotechnology_working_draft.tex
	xelatex -interaction=nonstopmode -halt-on-error \
		-output-directory=manuscript \
		manuscript/FORGE_Nature_Biotechnology_working_draft.tex

manuscript-iclr:
	pandoc manuscript/FORGE_ICLR2027_submission.md \
		--from=markdown --to=latex --standalone \
		--template=manuscript/latex/forge_iclr.template.tex \
		--lua-filter=manuscript/latex/forge_iclr_filter.lua \
		--output=manuscript/FORGE_ICLR2027_submission.tex
	TEXINPUTS=manuscript/latex/iclr2027:$$TEXINPUTS pdflatex -interaction=nonstopmode \
		-halt-on-error -output-directory=manuscript \
		manuscript/FORGE_ICLR2027_submission.tex
	TEXINPUTS=manuscript/latex/iclr2027:$$TEXINPUTS pdflatex -interaction=nonstopmode \
		-halt-on-error -output-directory=manuscript \
		manuscript/FORGE_ICLR2027_submission.tex

test:
	PYTHONPATH=src python3 -m pytest -q

lint:
	python3 -m ruff check src tests scripts

fmt:
	python3 -m black src tests scripts && python3 -m ruff check --fix src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
