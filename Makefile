.PHONY: help vendor vendor-partial verify verify-partial verify-pins verify-pins-code \
	archive-pins doctor experiment-list experiment-smoke phase1-corpus-run \
	phase1-multireaction-corpus \
	phase1-bl-lx-reaction-enumerated-expansion phase1-bl-lx-mixed-repeat-expansion \
	phase1-bl-lx-model-support-mixed-repeat-expansion \
	phase1-potency-study-corpus \
	phase1-multireaction-training-smoke \
	phase1-multireaction-overfit \
	phase1-transformer-production-modal-plans \
	phase1-transformer-production-accelerator-benchmark-plans \
	phase1-finite-component-catalogue-smoke \
	phase1-finite-component-catalogue-full \
	phase1-shared-synthesis-program-representation \
	phase1-shared-synthesis-program-mixed-representation \
	phase1-shared-synthesis-program-mixed-training-design \
	phase1-shared-synthesis-program-mixed-production-cache \
	phase1-shared-synthesis-program-mixed-training-smoke \
	phase1-shared-synthesis-program-integration \
	phase1-shared-synthesis-program-production-design \
	phase1-shared-synthesis-program-production-cache \
	phase1-shared-synthesis-program-production-accelerator-benchmark-plans \
	phase1-shared-synthesis-program-production-modal-plan \
	phase1-training-smoke phase1-training-modal-plan phase1-sampling-smoke \
	phase1-sampling-reproduce paper-experiment-readiness paper-verify manuscript-pdf manuscript-iclr \
	paper-bundle code-survey test-baseline-report assessment-benchmark typecheck check-core \
	test lint fmt clean

EXPECT_PINS ?= 822
EXPECT_PINS_ALL ?= 2416
CODE_DRIFT_BACKLOG ?= 4
UV_RUN ?= uv run

help:
	@echo "FORGE — authorized M0 and bounded Phase 1 work. Read AGENTS.md first."
	@echo ""
	@echo "Data and provenance"
	@echo "  make vendor / verify             materialize and verify hash-pinned inputs"
	@echo "  make verify-pins                 verify result-declared inputs"
	@echo "  make verify-pins-code            include config-declared source pins"
	@echo "  make archive-pins                archive recoverable historical source bytes"
	@echo ""
	@echo "Experiments"
	@echo "  make doctor                      verify all registered experiment inputs"
	@echo "  make experiment-list             list the supported experiment catalog"
	@echo "  make experiment-smoke            exercise the installation/runtime pipeline"
	@echo "  make phase1-corpus-run            rebuild the Phase 1 corpus DAG"
	@echo "  make phase1-multireaction-corpus  rebuild source-grounded multi-reaction programs"
	@echo "  make phase1-bl-lx-reaction-enumerated-expansion  build BL/LX virtual program support"
	@echo "  make phase1-bl-lx-mixed-repeat-expansion  build heterogeneous BL/LX program support"
	@echo "  make phase1-bl-lx-model-support-mixed-repeat-expansion  rebuild support-qualified BL/LX data"
	@echo "  make phase1-potency-study-corpus  rebuild the single-source LNPDB study view"
	@echo "  make phase1-multireaction-training-smoke  run conditioned train/sample smoke"
	@echo "  make phase1-multireaction-overfit  run the fail-closed reaction-core overfit gate"
	@echo "  make phase1-transformer-production-accelerator-benchmark-plans  inspect matched A100/H100 benchmarks"
	@echo "  make phase1-transformer-production-modal-plans  inspect A100/H100 full-run alternatives"
	@echo "  make phase1-finite-component-catalogue-smoke  verify the catalogue baseline"
	@echo "  make phase1-finite-component-catalogue-full  run all three matched CPU replicates"
	@echo "  make phase1-shared-synthesis-program-representation  qualify the full shared graph"
	@echo "  make phase1-shared-synthesis-program-mixed-representation  qualify expanded BL/LX graph support"
	@echo "  make phase1-shared-synthesis-program-mixed-training-design  freeze expanded-data training design"
	@echo "  make phase1-shared-synthesis-program-mixed-production-cache  rebuild expanded-data packed cache"
	@echo "  make phase1-shared-synthesis-program-mixed-training-smoke  run local expanded-data optimizer smoke"
	@echo "  make phase1-shared-synthesis-program-integration  qualify shared cache/model/sampling"
	@echo "  make phase1-shared-synthesis-program-production-design  freeze matched design only"
	@echo "  make phase1-shared-synthesis-program-production-cache  rebuild the packed full cache"
	@echo "  make phase1-shared-synthesis-program-production-accelerator-benchmark-plans  inspect L4/A100 benchmarks"
	@echo "  make phase1-shared-synthesis-program-production-modal-plan  inspect three full replicates"
	@echo "  make phase1-training-smoke        run and verify resumable CPU training"
	@echo "  make phase1-training-modal-plan   dry-run the pinned production L4 DAG"
	@echo "  make phase1-sampling-smoke        run and verify diagnostic sampling"
	@echo "  make phase1-sampling-reproduce    require byte-identical sampling outputs"
	@echo ""
	@echo "Quality and paper"
	@echo "  make check-core                   architecture, typing, provenance, and tests"
	@echo "  make paper-experiment-readiness   audit every v1 result and baseline obligation"
	@echo "  make test / lint / fmt            repository quality gates"
	@echo "  make code-survey                  classify historical code without deleting it"
	@echo "  make assessment-benchmark         time and digest the CPU Ugi assessor suite"
	@echo "  make paper-verify / paper-bundle  verify or package paper evidence"

vendor:
	python3 -m cli data vendor

vendor-partial:
	python3 -m cli data vendor --allow-partial

verify:
	PYTHONPATH=.:tools python3 -m cli data verify

verify-partial:
	PYTHONPATH=.:tools python3 -m cli data verify --allow-partial

# Missing workstation-only artifacts are reported but tolerated; byte drift is always fatal.
verify-pins:
	PYTHONPATH=.:tools python3 -m cli provenance verify --expect-verified $(EXPECT_PINS)

verify-pins-code:
	PYTHONPATH=.:tools python3 -m cli provenance verify --code \
		--expect-verified $(EXPECT_PINS_ALL) --allow-drift $(CODE_DRIFT_BACKLOG)

archive-pins:
	PYTHONPATH=.:tools python3 -m cli provenance archive

doctor:
	python3 -m cli doctor

experiment-list:
	python3 -m cli experiment list

experiment-smoke:
	python3 -m cli experiment run installation-smoke --profile smoke --resume
	python3 -m cli experiment verify installation-smoke --profile smoke

phase1-corpus-run:
	python3 -m cli experiment run phase1-corpus --profile full --resume
	python3 -m cli experiment verify phase1-corpus --profile full

phase1-multireaction-corpus:
	python3 -m cli experiment run phase1-multireaction-corpus --profile full --resume
	python3 -m cli experiment verify phase1-multireaction-corpus --profile full
	python3 -m cli experiment reproduce phase1-multireaction-corpus --profile full

phase1-bl-lx-reaction-enumerated-expansion:
	python3 -m cli experiment run phase1-bl-lx-reaction-enumerated-expansion --profile full --resume
	python3 -m cli experiment verify phase1-bl-lx-reaction-enumerated-expansion --profile full
	python3 -m cli experiment reproduce phase1-bl-lx-reaction-enumerated-expansion --profile full

phase1-bl-lx-mixed-repeat-expansion:
	python3 -m cli experiment run phase1-bl-lx-mixed-repeat-expansion --profile full --resume
	python3 -m cli experiment verify phase1-bl-lx-mixed-repeat-expansion --profile full
	python3 -m cli experiment reproduce phase1-bl-lx-mixed-repeat-expansion --profile full

phase1-bl-lx-model-support-mixed-repeat-expansion:
	python3 -m cli experiment run phase1-bl-lx-model-support-mixed-repeat-expansion --profile full --resume
	python3 -m cli experiment verify phase1-bl-lx-model-support-mixed-repeat-expansion --profile full
	python3 -m cli experiment reproduce phase1-bl-lx-model-support-mixed-repeat-expansion --profile full

phase1-potency-study-corpus:
	python3 -m cli experiment run phase1-potency-study-corpus --profile full --resume
	python3 -m cli experiment verify phase1-potency-study-corpus --profile full
	python3 -m cli experiment reproduce phase1-potency-study-corpus --profile full

phase1-multireaction-training-smoke:
	python3 -m cli experiment run phase1-multireaction-training-smoke --profile smoke --resume
	python3 -m cli experiment verify phase1-multireaction-training-smoke --profile smoke
	python3 -m cli experiment reproduce phase1-multireaction-training-smoke --profile smoke

phase1-multireaction-overfit:
	python3 -m cli experiment run phase1-multireaction-overfit --profile smoke --resume
	python3 -m cli experiment verify phase1-multireaction-overfit --profile smoke
	python3 -m cli experiment reproduce phase1-multireaction-overfit --profile smoke

phase1-transformer-production-modal-plans:
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production --profile full --backend modal --replicate 0
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production --profile full --backend modal --replicate 1
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production --profile full --backend modal --replicate 2
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production-h100 --profile full --backend modal --replicate 0
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production-h100 --profile full --backend modal --replicate 1
	python3 -m cli experiment plan phase1-transformer-synthesis-program-production-h100 --profile full --backend modal --replicate 2

phase1-transformer-production-accelerator-benchmark-plans:
	python3 -m cli experiment plan phase1-transformer-production-accelerator-benchmark-a100-40gb --profile smoke --backend modal
	python3 -m cli experiment plan phase1-transformer-production-accelerator-benchmark-h100 --profile smoke --backend modal

phase1-finite-component-catalogue-smoke:
	python3 -m cli experiment run phase1-finite-component-catalogue-baseline --profile smoke --resume
	python3 -m cli experiment verify phase1-finite-component-catalogue-baseline --profile smoke
	python3 -m cli experiment reproduce phase1-finite-component-catalogue-baseline --profile smoke

phase1-finite-component-catalogue-full:
	python3 -m cli experiment run phase1-finite-component-catalogue-baseline --profile full --replicate 0 --resume
	python3 -m cli experiment verify phase1-finite-component-catalogue-baseline --profile full --replicate 0
	python3 -m cli experiment run phase1-finite-component-catalogue-baseline --profile full --replicate 1 --resume
	python3 -m cli experiment verify phase1-finite-component-catalogue-baseline --profile full --replicate 1
	python3 -m cli experiment run phase1-finite-component-catalogue-baseline --profile full --replicate 2 --resume
	python3 -m cli experiment verify phase1-finite-component-catalogue-baseline --profile full --replicate 2

phase1-shared-synthesis-program-representation:
	python3 -m cli experiment run phase1-shared-synthesis-program-representation --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-representation --profile full
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-representation --profile full

phase1-shared-synthesis-program-mixed-representation:
	python3 -m cli experiment run phase1-shared-synthesis-program-mixed-representation --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-mixed-representation --profile full
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-mixed-representation --profile full

phase1-shared-synthesis-program-mixed-training-design:
	python3 -m cli experiment run phase1-shared-synthesis-program-mixed-training-design --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-mixed-training-design --profile full
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-mixed-training-design --profile full

phase1-shared-synthesis-program-mixed-production-cache:
	python3 -m cli experiment run phase1-shared-synthesis-program-mixed-production-cache --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-mixed-production-cache --profile full
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-mixed-production-cache --profile full

phase1-shared-synthesis-program-mixed-training-smoke:
	python3 -m cli experiment run phase1-shared-synthesis-program-mixed-training-smoke --profile smoke --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-mixed-training-smoke --profile smoke
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-mixed-training-smoke --profile smoke

phase1-shared-synthesis-program-integration:
	python3 -m cli experiment run phase1-shared-synthesis-program-integration --profile smoke --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-integration --profile smoke
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-integration --profile smoke

phase1-shared-synthesis-program-production-design:
	python3 -m cli experiment run phase1-shared-synthesis-program-production-design --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-production-design --profile full
	python3 -m cli experiment reproduce phase1-shared-synthesis-program-production-design --profile full

phase1-shared-synthesis-program-production-cache:
	python3 -m cli experiment run phase1-shared-synthesis-program-production-cache --profile full --resume
	python3 -m cli experiment verify phase1-shared-synthesis-program-production-cache --profile full

phase1-shared-synthesis-program-production-accelerator-benchmark-plans:
	python3 -m cli experiment plan \
		phase1-shared-synthesis-program-production-accelerator-benchmark-l4 \
		--profile smoke --backend modal
	python3 -m cli experiment plan \
		phase1-shared-synthesis-program-production-accelerator-benchmark-a100-40gb \
		--profile smoke --backend modal

phase1-shared-synthesis-program-production-modal-plan:
	python3 -m cli experiment plan phase1-shared-synthesis-program-production-training \
		--profile full --backend modal --replicate 0
	python3 -m cli experiment plan phase1-shared-synthesis-program-production-training \
		--profile full --backend modal --replicate 1
	python3 -m cli experiment plan phase1-shared-synthesis-program-production-training \
		--profile full --backend modal --replicate 2

phase1-training-smoke:
	python3 -m cli experiment run phase1-training-smoke --profile smoke --resume
	python3 -m cli experiment verify phase1-training-smoke --profile smoke
	python3 -m cli experiment reproduce phase1-training-smoke --profile smoke

phase1-training-modal-plan:
	python3 -m cli experiment plan phase1-training-production --profile full --backend modal

phase1-sampling-smoke:
	python3 -m cli experiment run phase1-sampling --profile smoke --resume
	python3 -m cli experiment verify phase1-sampling --profile smoke

phase1-sampling-reproduce:
	python3 -m cli experiment reproduce phase1-sampling --profile smoke

paper-verify:
	PYTHONPATH=.:tools:paper python3 -m cli paper verify

paper-experiment-readiness:
	PYTHONPATH=.:tools:paper python3 -m cli paper experiments

manuscript-pdf manuscript-iclr:
	PYTHONPATH=.:tools:paper python3 -m cli paper build

paper-bundle:
	PYTHONPATH=.:tools:paper python3 -m cli paper bundle

code-survey:
	PYTHONPATH=.:tools:paper python3 -m forge_maintenance survey \
		--output provenance/code-retirement/iclr2027.json

test-baseline-report:
	PYTHONPATH=.:tools:paper python3 -m forge_maintenance test-report \
		--output results/maintenance/root_package_architecture_v1/test_baseline.json

# CPU only. Times the frozen Ugi assessor suite over three 3,072-attempt ledgers and digests every
# emitted row and metric, so a performance change can be shown to be output-exact.
assessment-benchmark:
	PYTHONPATH=.:tools python3 -m forge_maintenance assessment-benchmark \
		--output build/assessment_benchmark/result.json

typecheck:
	MYPYPATH=.:tools:paper python3 -m mypy forge/core forge/chemistry forge/assembly cli
	MYPYPATH=. python3 -m mypy experiments/_runtime experiments/catalog.py \
		experiments/phase1/product_l1/stages.py experiments/phase1/multireaction/stages.py \
		experiments/phase1/hela_potency/stages.py \
		experiments/phase1/multireaction/training.py \
		experiments/phase1/multireaction/sampling.py \
		experiments/phase1/multireaction/qualification.py \
		experiments/phase1/multireaction/shared_training.py \
		experiments/phase1/multireaction/shared_sampling.py \
		experiments/phase1/multireaction/shared_qualification.py \
		experiments/phase1/multireaction/production_design.py \
		experiments/phase1/multireaction/production_training.py \
		experiments/phase1/multireaction/production_evaluation.py \
		experiments/phase1/multireaction/catalogue_baseline.py \
		experiments/phase1/multireaction/benchmark_studies.py \
		experiments/phase1/multireaction/common_assessment.py \
		experiments/phase1/multireaction/lipid_realism_assessment.py \
		experiments/phase1/multireaction/lipid_realism_aggregation.py \
		experiments/phase1/multireaction/external_baselines.py \
		experiments/phase1/multireaction/external_ugi_contract.py \
		experiments/phase1/multireaction/native_baseline_ports.py \
		experiments/phase1/multireaction/native_baseline_runtime.py \
		experiments/phase1/multireaction/method_blind_route_union.py \
		experiments/phase1/multireaction/mechanism_study.py \
		experiments/phase1/multireaction/production_preflight.py \
		experiments/phase1/multireaction/accelerator_selection.py \
		experiments/phase1/multireaction/production_adjudication.py \
		experiments/phase1/multireaction/final_production_adjudication.py \
		forge/model/vocabulary.py \
		forge/model/eligibility.py forge/model/adapter_node_conditioning.py \
		forge/model/reaction_program_conditioning.py \
		forge/model/reaction_program_graph.py forge/model/reaction_program_flow.py \
		forge/model/synthesis_program_graph.py \
		forge/model/synthesis_program_sampling.py forge/model/tensor_checkpoint.py \
		forge/model/reaction_program_sampling.py \
		forge/model/reaction_program_evaluation.py \
		forge/model/common_ugi_benchmark.py forge/model/common_lipid_realism.py \
		forge/model/conditional_role_dependence.py \
		forge/model/finite_component_catalogue.py forge/model/learned_inventory_selector.py \
		forge/model/reaction_program_transformer.py forge/model/training_restart.py \
		forge/synthesis/assessment/common_route_evidence.py \
		forge/corpus/training_cache.py \
		forge/corpus/lnpdb.py forge/corpus/multireaction.py \
		forge/corpus/component_splits.py forge/corpus/multireaction_expansion.py \
		forge/corpus/multireaction_mixed_expansion.py \
		forge/corpus/reaction_program_records.py \
		forge/corpus/reaction_program_training.py \
		forge/corpus/synthesis_program_representation.py \
		forge/corpus/synthesis_program_training.py forge/potency/study_data.py
	MYPYPATH=.:paper python3 -m mypy paper/forge_paper
	MYPYPATH=.:tools python3 -m mypy tools/forge_provenance tools/forge_data tools/forge_maintenance

check-core: verify-pins typecheck
	python3 -m ruff check forge cli experiments/_runtime experiments/phase1 tools \
		tests/test_architecture_boundaries.py tests/test_assembly_ugi3.py \
		tests/test_reaction_program.py tests/test_multireaction_corpus.py \
		tests/test_multireaction_expansion.py \
		tests/test_multireaction_mixed_expansion.py \
		tests/test_reaction_program_records.py \
		tests/test_multireaction_qualification.py \
		tests/test_multireaction_training.py \
		tests/test_finite_component_catalogue_baseline.py \
		tests/test_reaction_program_evaluation.py \
		tests/test_reaction_program_conditioning.py tests/test_reaction_program_flow.py \
		tests/test_synthesis_program_graph.py \
		tests/test_shared_synthesis_program_representation.py \
		tests/test_shared_synthesis_program_integration.py \
		tests/test_shared_synthesis_program_production_design.py \
		tests/test_shared_synthesis_program_production_preflight.py \
		tests/test_shared_synthesis_program_production_adjudication.py \
		tests/test_final_production_adjudication.py \
		tests/test_common_lipid_realism.py \
		tests/test_lnpdb_catalogue.py \
		tests/test_potency_study_data.py \
		tests/test_core_hashing.py tests/test_core_provenance_archive.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_runner.py tests/test_experiment_seed.py \
		tests/test_experiment_spec.py tests/test_maintenance_test_baseline.py \
		tests/test_paper_experiment_matrix.py tests/test_paper_reproduction.py \
		tests/test_phase1_product_l1_data.py \
		tests/test_pinned_sources_are_tracked.py
	python3 -m pytest -q tests/test_architecture_boundaries.py tests/test_assembly_ugi3.py \
		tests/test_reaction_program.py tests/test_multireaction_corpus.py \
		tests/test_multireaction_expansion.py \
		tests/test_multireaction_mixed_expansion.py \
		tests/test_reaction_program_records.py \
		tests/test_multireaction_qualification.py \
		tests/test_multireaction_training.py \
		tests/test_finite_component_catalogue_baseline.py \
		tests/test_reaction_program_evaluation.py \
		tests/test_reaction_program_conditioning.py tests/test_reaction_program_flow.py \
		tests/test_synthesis_program_graph.py \
		tests/test_shared_synthesis_program_representation.py \
		tests/test_shared_synthesis_program_integration.py \
		tests/test_shared_synthesis_program_production_design.py \
		tests/test_shared_synthesis_program_production_preflight.py \
		tests/test_shared_synthesis_program_production_adjudication.py \
		tests/test_final_production_adjudication.py \
		tests/test_common_lipid_realism.py \
		tests/test_lnpdb_catalogue.py \
		tests/test_potency_study_data.py \
		tests/test_core_hashing.py tests/test_core_provenance_archive.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_runner.py tests/test_experiment_seed.py \
		tests/test_experiment_spec.py tests/test_maintenance_test_baseline.py \
		tests/test_paper_experiment_matrix.py tests/test_paper_reproduction.py \
		tests/test_phase1_product_l1_data.py \
		tests/test_pinned_sources_are_tracked.py

test:
	PYTHONPATH=. $(UV_RUN) python -m pytest -q

lint:
	python3 -m ruff check forge cli experiments/_runtime experiments/phase1 tools tests

fmt:
	python3 -m black forge cli experiments/_runtime experiments/phase1 tools tests
	python3 -m ruff check --fix forge cli experiments/_runtime experiments/phase1 tools tests

clean:
	find forge cli experiments tools tests -type f -path '*/__pycache__/*' -delete
	find forge cli experiments tools tests -type d -name __pycache__ -empty -delete
	rm -f paper/*.aux paper/*.blg paper/*.fdb_latexmk paper/*.fls paper/*.log \
		paper/*.out paper/*.toc paper/*.synctex.gz
