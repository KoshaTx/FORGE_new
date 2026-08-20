.PHONY: help vendor vendor-partial verify verify-partial verify-pins verify-pins-code \
	archive-pins doctor experiment-list experiment-smoke phase1-corpus-run \
	phase1-training-smoke phase1-training-modal-plan phase1-sampling-smoke \
	phase1-sampling-reproduce paper-verify manuscript-pdf manuscript-iclr \
	paper-bundle code-survey test-baseline-report typecheck check-core test lint fmt clean

EXPECT_PINS ?= 758
EXPECT_PINS_ALL ?= 2117
CODE_DRIFT_BACKLOG ?= 4

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
	@echo "  make phase1-training-smoke        run and verify resumable CPU training"
	@echo "  make phase1-training-modal-plan   dry-run the pinned production L4 DAG"
	@echo "  make phase1-sampling-smoke        run and verify diagnostic sampling"
	@echo "  make phase1-sampling-reproduce    require byte-identical sampling outputs"
	@echo ""
	@echo "Quality and paper"
	@echo "  make check-core                   architecture, typing, provenance, and tests"
	@echo "  make test / lint / fmt            repository quality gates"
	@echo "  make code-survey                  classify historical code without deleting it"
	@echo "  make paper-verify / paper-bundle  verify or package paper evidence"

vendor:
	python3 -m cli data vendor

vendor-partial:
	python3 -m cli data vendor --allow-partial

verify:
	python3 -m cli data verify

verify-partial:
	python3 -m cli data verify --allow-partial

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

typecheck:
	MYPYPATH=.:tools:paper python3 -m mypy forge/core forge/chemistry forge/assembly cli
	MYPYPATH=. python3 -m mypy experiments/_runtime experiments/catalog.py \
		experiments/phase1/product_l1/stages.py forge/model/vocabulary.py \
		forge/model/eligibility.py forge/corpus/training_cache.py
	MYPYPATH=.:paper python3 -m mypy paper/forge_paper
	MYPYPATH=.:tools python3 -m mypy tools/forge_provenance tools/forge_data tools/forge_maintenance

check-core: verify-pins typecheck
	python3 -m ruff check forge cli experiments/_runtime experiments/phase1 tools \
		tests/test_architecture_boundaries.py tests/test_assembly_ugi3.py \
		tests/test_core_hashing.py tests/test_core_provenance_archive.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_runner.py tests/test_experiment_seed.py \
		tests/test_experiment_spec.py tests/test_maintenance_test_baseline.py \
		tests/test_paper_reproduction.py tests/test_phase1_product_l1_data.py \
		tests/test_pinned_sources_are_tracked.py
	python3 -m pytest -q tests/test_architecture_boundaries.py tests/test_assembly_ugi3.py \
		tests/test_core_hashing.py tests/test_core_provenance_archive.py \
		tests/test_experiment_modal.py tests/test_experiment_model_pipelines.py \
		tests/test_experiment_runner.py tests/test_experiment_seed.py \
		tests/test_experiment_spec.py tests/test_maintenance_test_baseline.py \
		tests/test_paper_reproduction.py tests/test_phase1_product_l1_data.py \
		tests/test_pinned_sources_are_tracked.py

test:
	PYTHONPATH=. python3 -m pytest -q

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
