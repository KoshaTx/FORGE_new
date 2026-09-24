# COMPOSE-Lipid: current handoff

Updated 2026-09-18. This is the authoritative entry point. Older dated reports
remain provenance and decision history; their partial counts do not override
this handoff.

The 2026-09-17/18 session record — what was verified, what the v8 pipeline's
untracked state meant, the corpus provenance findings, and an explicit list of
what was *not* checked — is in
[`HANDOFF_CLAUDE_2026-09-18.md`](HANDOFF_CLAUDE_2026-09-18.md).

## Current outcome

The user-directed, from-scratch corpus rebuild is complete across **23 exact
reaction-program families**. Each family uses complete family-specific
subcomponents and its own permitted knobs: head topology/spacer/substitution,
tail length, branch position and size, positional unsaturation, internal-ester
position, occupancy, site, stage order, symmetry/asymmetry, and other
family-specific chemistry where applicable. Coupled or repeated arms remain
coupled. No global reactive-handle pool was used as the design rule.

The rebuild yielded:

- **3,182,837** globally unique constitutional candidates;
- **2,936,799** within current <=80-heavy-atom model support;
- **246,038** explicit size holds;
- **11,267** retained reported-source anchors;
- **zero** retained virtual rows from the superseded corpus.

The new balanced release contains exactly **200,000 unique constitutions**:

- **11,267 source anchors**;
- **188,733 new exact virtual constructions**;
- all source anchors pinned;
- family capacity tempered rather than copied as raw abundance;
- component coverage selected before family-axis filling.

The complete family table and selection rationale are in
[`POST_INSTRUCTION_CAPACITY_STATUS_2026-09-16.md`](POST_INSTRUCTION_CAPACITY_STATUS_2026-09-16.md).

## Authoritative artifacts

- Exact family capacity:
  `artifacts/corpus_build_v2/post_instruction_family_enumeration_v8_final/`
- Globally deduplicated universe:
  `artifacts/corpus_build_v2/post_instruction_global_union_v8/`
- Balanced 200k release:
  `artifacts/corpus_build_v2/post_instruction_balanced_release_v8/`
- Exact supervision manifest:
  `artifacts/corpus_build_v2/post_instruction_supervision_manifest_v8_2/`
- Canonical complete-component manifest:
  `artifacts/corpus_build_v2/post_instruction_component_manifest_v8_1/`
- Corrected grouped train/calibration/test split:
  `artifacts/corpus_build_v2/post_instruction_generator_splits_v8_1/`
- Frozen train-only target measure:
  `artifacts/corpus_build_v2/post_instruction_target_measure_v8_1/`
- Release policy:
  `configs/corpus/post_instruction_balanced_release_v8.json`
- Split policy:
  `configs/corpus/post_instruction_generator_splits_v8_1.json`
- Target-measure policy:
  `configs/corpus/post_instruction_target_measure_v8_1.json`

The release is chemistry-only. Virtual products are exact graph constructions,
not claims of synthesis, yield, availability, activity, or historical routes.

## Supervision integrity

All **200,000** targets have one supervision disposition:

- 7,434 prior exact source programs;
- 3,833 compatible source-decomposition replays;
- 172,877 exact virtual family-task replays;
- 15,856 exact bound Ugi-grid replays.

Every named virtual executor is importable. A representative release target
from every one of the 23 families was reconstructed from its referenced task or
complete precursor records. The focused corpus/replay suite passes **21/21** (see **Validation command**).

The current supervision artifact is `v8_2`. Earlier locator manifests are under
`artifacts/corpus_build_v2/superseded_post_instruction_supervision_manifests/`
and must not be used.

## Component, split, and target-measure integrity

The original v8 split is preserved as provenance but superseded for model-facing
evaluation. Its family metadata IDs did not uniformly represent complete
component structures; **3,734 of 7,002** old unseen-precursor rows had no
structurally unseen component. The corpus chemistry itself was not changed.

The v8.1 component manifest resolves all targets into canonical complete
component constitutions: **45,515** globally unique structures with roles and
multiplicity preserved. The corrected grouped split is frozen over all
**200,000** targets:

- train **159,782**; calibration **19,975**; test **19,975**; reference **268**;
- 23 formal evaluation families; the 3 `reported_*` families are reference-only
  and never enter train, calibration, or test.

Corrected held-out test panels:

- unseen component structure **9,259** targets;
- unseen exact component combination **4,676** targets;
- unseen coarse regional morphology **5,512** targets;
- globally held source-study transfer **532** targets.

The independent v8.1 audit verifies output and producer hashes, zero component
structure leakage, zero selected-PMID leakage across all formal families, exact
combination components seen in the same family training set, and complete named
panel coverage. The morphology panel is explicitly coarse morphology, not exact
graph topology.

The train-only measure is also frozen and independently audited: all **159,782**
train targets, **23** equal-mass supported chemistry families, bounded source
confidence, **15,162** joint regional-conditioning groups, no calibration/test
fit, and no Michael/BEAE family bias. Its realized global source mass is
**0.1352664556**.

Full results and the per-family evaluation-strength table are in
[`MODELING_READINESS_V8_1_2026-09-18.md`](MODELING_READINESS_V8_1_2026-09-18.md).

The split is chemistry-only: no biology, no BEAE outcomes, no GPU, and
`training_admissible: false`.

## What remains before training

The corpus molecules and their replay supervision are complete. The release is
still marked `training_admissible: false` because model-facing preparation has
not been frozen.

The corrected grouped split, held-out panels, and train-only measure are frozen
and audited. Proceed from there in this order:

1. project the weighted supervision records into exact model programs and materialize
   training paths;
2. prepare candidate batches and pass CPU reconstruction/runtime checks;
3. run the bounded one-seed regional-context health comparison;
4. freeze the production chemistry process only after chemistry-only evaluation.

The 2,297 weighted compatible-decomposition rows must be projected rather than
silently dropped or replaced. Calibration and test assignments must not be
recomputed.

No GPU training has been launched for this release. Local preparation must stay
memory-safe: one CPU per worker, at most four local workers, restartable 30-shard
artifacts, and no large in-memory union.

## Model and controller boundary

Train the base chemistry process first:

`regional lipid state -> frozen reaction-grounded R_theta^lipid -> exact endpoint`.

The base process uses anonymous region/port embeddings, canonical reaction
roles, joint regional context, exact scaffold/chemistry constraints, and soft
morphology/composition signals. Biological labels and BEAE outcomes do not
enter this model.

Only after the base process freezes should the newer controller be ported. Its
high-level action is a short, dynamically composed regional program, not one
primitive edit. Programs must bind by structural region/port, compile into exact
supported transitions, maintain validity at every intermediate, and receive the
expensive biological score only after the full program completes. The archive
stores productive context-program-binding tuples. The matched comparison keeps
the base model, oracle, scaffold, support, seeds, and budgets fixed.

BEAE remains a sealed prospective use case. Broad family chemistry may train
the base process; historical BEAE outcomes remain excluded until the declared
one-time evaluation.

## Validation command

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q -o addopts='' \
  tests/unit/test_post_instruction_supervision_manifest_v8.py \
  tests/unit/test_post_instruction_balanced_release_v8.py \
  tests/unit/test_post_instruction_generator_splits_v8.py \
  tests/unit/test_post_instruction_component_manifest_v8_1.py \
  tests/unit/test_generator_splits_v2.py \
  tests/unit/test_post_instruction_generator_splits_v8_1.py \
  tests/unit/test_post_instruction_target_measure_v8_1.py \
  tests/unit/test_family_20_maleate_enumeration_v8.py \
  tests/unit/test_family_21_acid_epoxide_diester_enumeration_v8.py \
  tests/unit/test_family_22_passerini_enumeration_v8.py \
  tests/unit/test_family_23_vitamin_b5_enumeration_v8.py
```

The v8.1 focused additions currently pass `10/10`; run the full command before
committing or launching preparation.
