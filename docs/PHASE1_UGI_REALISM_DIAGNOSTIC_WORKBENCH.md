# Ugi realism diagnostic workbench

Date: 2026-09-08. Scope: the authorized computational Phase 1 product/L1 model.

## Question and decision

The present question is where the gap between measured Ugi structures and generated structures
first appears: empirical support, learned conditional predictions, terminal decoding, or evaluation.
The workbench measures these separately before another decoder sweep or model specialization.
It produces development evidence, not a model promotion or a synthesis-success claim.

The user authorized this attribution study and the reusable tools needed to improve model research.
All work here is local CPU analysis of the frozen checkpoint, measured training examples, and
existing generated ledgers. The existing scientific gates, datasets, holdouts, model checkpoints,
registry, and historical experiment configurations remain frozen.

## Research contract

| Hypothesis | Alternative explanation | Discriminating evidence | Consequence |
| --- | --- | --- | --- |
| Decoder support excludes important measured structures. | Exclusions have little effect on the reference distribution. | Attribute every exclusion; compare admitted measured structures with the unchanged full train-development reference. | Separate empirical coverage from conditional realization quality. An admitted-reference view never replaces full-reference reporting. |
| Neural predictions lose useful structure through the sampling contract. | The checkpoint already assigns poor probabilities before decoding. | Compare identical corruptions with noisy versus target topology; probe neural and rank/uniform probability laws on the same targets. | Locate the first measurable loss before choosing a model intervention. Teacher-forced probes do not establish rollout quality. |
| Coarse descriptors miss differences between constitutional arrangements. | Existing descriptor improvements reflect broadly improved structure. | Inspect descriptor collisions and complementary graph fingerprints, including novel-component strata. | Treat descriptor and graph evidence as separate development signals. Nearest-reference similarity alone cannot certify realism. |
| Visual preferences depend on depiction rather than graph structure. | Review is consistent under depictions and recognizes measured structures. | Prepare blinded identical-graph redraw pairs and measured positive controls. | Report pending review explicitly; require calibration before treating appearance preferences as independent structural evidence. |

Each numerical stage uses versioned configurations, explicit seeds, input hashes, and implementation
hashes. Scientific results use deterministic serialization. Unfinished or failed stages are never
represented as completed evidence. The suite verifies persisted stage artifacts before reusing them.

## Independent responsibilities

- Support audit: quantify measured product, component, and family admission under the existing policy.
- Model attribution: trace the frozen checkpoint on bounded measured training examples and record
  per-role, per-channel denominators and confidence.
- Evaluator audit: compare structural representations and prepare reviewer-calibration controls.
- Integration: verify artifacts, preserve contradictions, and connect each finding to a falsifiable
  next experiment.

The engineering skill governs provenance and tests. The ML research skill governs comparisons,
competing explanations, and claim limits. Independent agents own the three audit modules; the main
agent reviews their integration and conclusions. No additional agent framework is introduced.

## How results should change model development

1. Establish an empirical control using admitted measured examples and the full reference. This is
   not an oracle upper bound on generated realism: admitted measured examples are only a subset of
   the permitted graph space, and the evaluated statistics are not a complete description of it.
2. Establish what the checkpoint predicts before terminal constraints and stochastic reweighting.
   Count actually corrupted coordinates separately so easy unchanged atoms do not hide failures.
3. Separate novelty from similarity. A method that produces more known components can become closer
   to training structures without learning better new components. Report both populations.
4. Calibrate the evaluator using graph identities and measured controls. Drawing style, saturation,
   and chemical realism are distinct concepts. Uncertainty from training seeds is not estimated by
   resampling molecules from a single checkpoint.
5. Choose a single subsequent intervention from the first unresolved failure. Potential changes
   include richer anonymous semantic conditioning, training on the same admissible structural
   choices used by decoding, or qualified component supervision. These remain hypotheses until a
   bounded comparison separates their claimed mechanism from extra capacity, exposure, or tuning.

The prior failed measured-only, contextual chemistry, structured-topology, and additional chemistry
flow experiments remain negative evidence. This work does not justify repeating them unchanged.
Broad pretraining remains subject to the existing Ugi-first decision gate. No paid execution,
biological optimization, prospective candidate selection, or wet-lab work is initiated here.

## Acceptance criteria

- All three independent audits have runnable local entry points and precise input pins.
- Every admitted or excluded measured example is accounted for; train filtering precedes molecular
  parsing and calibration/heldout structures are not consumed.
- Teacher-forced results distinguish prediction, probability-law probes, and unmeasured rollout
  behavior; graph comparison distinguishes familiar and novel components.
- The calibration packet records identical graph identities under redraws and leaves reviewer
  judgments pending. No review result is fabricated.
- The suite supports validated reuse, failure receipts, and a read-only verification command.
- Focused tests cover provenance tampering, scientific denominators, graph invariance, and restart
  behavior. Required repository checks are run and their actual outcome reported.
- Numeric findings and hashes are persisted under `results/phase1/`; the decision log records both
  useful and negative results. No existing promotion gate is relaxed.

## Measured findings

### Empirical support is already shifted before learning

The audit reproduces the pinned program-draw census exactly: **231 of 480** measured training
products are admitted and **249** excluded. All exclusions are attributable to existing exterior
atom-count bounds; there are no additional topology exclusions in this measured population.
Overlapping exclusion counts are 96 amine-above-maximum, 32 amine-below-minimum, 120
isocyanide-above-maximum, and 60 aldehyde-below-minimum. These overlap and must not be summed
as distinct excluded products. The full ledger retains the mutually exclusive reason patterns.

Measured component counts with any admitted product fall from 15/8/4 to 11/7/3 for
amine/aldehyde/isocyanide; represented family triples fall from 28 to 20. Holding equal
family-group weighting and the full frozen evaluation reference fixed gives:

| Empirical population | Mean normalized Wasserstein | Energy distance | MMD squared |
| --- | ---: | ---: | ---: |
| All measured train products | 0.042030 | 0.060523 | 0.003695 |
| Admitted measured train products | 0.106725 | 0.178539 | 0.014479 |

This establishes an empirical distribution shift caused by admission, not an attainable lower
bound for a generator. The populations overlap the train-development reference; these are not
held-out generalization estimates. The full evaluation reference has 56 rows, of which 32 are
admitted. The supplementary admitted-reference diagnostic neither replaces nor rebalances that
reference. No support restriction is relaxed by this audit.

### Frozen-checkpoint errors depend on topology, with residual chemistry errors

Six CPU float32 forward passes cover 16 deliberately component-diverse measured TRAIN products,
including all 15/8/4 measured components. At flow time 0.5, replacing noisy topology with target
topology while keeping the same noisy chemistry changes the following genuinely corrupted
variable-coordinate predictions:

| Coordinates | Noisy topology | Target topology |
| --- | ---: | ---: |
| Amine atoms | 12/22 (54.5%) | 20/22 (90.9%) |
| Aldehyde atoms | 15/28 (53.6%) | 19/28 (67.9%) |
| Aldehyde bonds | 20/34 (58.8%) | 31/34 (91.2%) |

Under noisy topology, changed amine offspring-count accuracy is 29/42 and properly masked
parent-pointer accuracy is 40/45. These denominators are coordinates, not independent molecules.
The component-covering panel is not a random population estimate, and supplied layout, role,
morphology, and serialization context prevent interpreting these scores as rollout success.

An important correction to the initial architectural hypothesis: the authenticated checkpoint
already contains `model_config.semantic_objective.topology_conditioned_chemistry_weight = 1.0`
and whole-graph attention. Adding target-topology supervision or claiming independent learned
coordinates would misdescribe the existing model. The probe demonstrates topology sensitivity
and residual errors under an already-trained condition. It does not identify the best successor
architecture. Detailed decoder composition, ester-arm, and unsaturation targets remain absent
from explicit neural conditioning; their causal contribution is unmeasured.

The rank/uniform probe isolates the existing probability law on identical **categorical training
supports**, with other scores held equal. Ranking can discard logit-gap information, but it can
either increase or decrease confidence. These are not complete feasible terminal arrangements;
the probe does not quantify actual decoder damage or improvement. Actual sampling-trajectory
attribution remains unmeasured.

### Evaluator resolution and reviewer calibration

The evaluator stage reports descriptor collisions, supplementary per-role graph proximity,
novelty-stratified comparisons, and measured leave-exact-self-out controls. Its drawing packet
checks graph/descriptor/fingerprint invariance mechanically; reviewer outcomes remain pending.
Measured source evidence is not a ground-truth preference for every generated-versus-measured
pair, and a preference for a saturated or compact drawing is not itself a chemical-validity test.

Among 803 distinct products across the measured reference and saved arms, 183 fall into 81
collision classes under the frozen 36-dimensional descriptor vector. Measured products alone have
no collisions in this population; baseline has 15 classes spanning 30/170 distinct products and
treatment has 30 classes spanning 68/215. This demonstrates limited descriptor resolution, not
that every collision is chemically implausible or that greater collision count means poorer quality.

Complementary graph distances expose a novelty tradeoff. Negative differences mean treatment
is closer to the nearest unique measured training component:

| Role and paired population | Paired attempts | Morgan-count distance difference | Atom-pair-count distance difference |
| --- | ---: | ---: | ---: |
| Amine, all eligible | 256 | -0.14379 | -0.11642 |
| Amine, both novel versus all train | 124 | +0.00018 | +0.00289 |
| Aldehyde, all eligible | 256 | +0.09459 | -0.00655 |
| Aldehyde, both novel versus all train | 64 | +0.30135 | +0.20240 |

Novel amine attempts fall from 189 to 137 and novel aldehyde attempts from 186 to 86. Unique
component counts nevertheless increase (amine 25 to 51; aldehyde 46 to 76), illustrating why
unique count and novelty incidence must not be conflated. The novel-amine intervals include zero
for both fingerprints; the novel-aldehyde intervals are positive for both. Isocyanides remain
three familiar components in both arms, with no novel-pair contrast available.

These novelty strata are defined **after treatment** and the intervals resample saved attempts
from one seed. They do not isolate a causal effect, estimate across-training uncertainty, or
calibrate chemical realism. Fingerprints can also collide; in particular, local Morgan environments
can be insensitive to some positional changes. The atom-pair representation omits paths longer
than 30 bonds while retaining each complete graph. Neither representation establishes graph identity.

The new calibration packet contains eight identical-graph redraw pairs and four measured controls.
Pair kinds, A/B positions, and depiction styles are randomized without encoding measured/generated
origin. All mechanical identity checks pass. A representative SVG was rendered and visually checked
for legibility; no scientific reviewer judgment was entered.

## Decision boundary for the next intervention

Do not promote the current arm, run another scalar guidance sweep against the same review rows,
or treat this workbench as evidence that model realism improved. Three questions remain separate:

1. Why the count-support restriction exists and whether it is essential to decoder correctness.
   Any proposed revision must preserve qualified structural support and exact assembly; full
   reference reporting remains mandatory. This study does not authorize weakening a gate.
2. Where errors first appear in an actual terminal trajectory. A subsequent passive trace should
   distinguish the network's prediction, the legal candidate set, probability reweighting, and
   the accepted state using the same requests and randomness. Only then can a bounded intervention
   test a specified mechanism, with historical failed interventions retained as controls.
3. Whether structural improvements are detected consistently without memorization or drawing-style
   dependence. Supplementary graph distances and the pending calibration packet cannot replace
   the frozen promotion criteria.

This is diagnostic infrastructure plus measured attribution, not a trained successor model or a
completed reviewer calibration. No new paid computation or biological experiment is initiated.

## Reuse and verification

Each audit runs independently through its matching module under
`experiments.phase1.multireaction.ugi_realism_*`, with `--repo-root`, `--config`, and
`--output-dir`. Individual audit outputs must be fresh; do not overwrite a completed result.

The integrated entry point runs only missing local stages and verifies every persisted receipt
before reuse:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python -m \
  experiments.phase1.multireaction.ugi_realism_diagnostic_suite \
  --repo-root . \
  --config configs/multireaction/ugi_realism_diagnostic_suite_v1.json \
  --output-dir results/phase1/ugi_realism_diagnostic_replay_v1
```

Repeat the same command after an interruption. Completed stages are reused only if their config,
inputs, original source pins, implementation inventory, and artifact inventories verify. Failed
attempt output is preserved under a numbered attempt directory. This is a local CPU workflow;
it is not a retry mechanism for paid remote computation. Use one writer per output directory.

Add `--collect` to authenticate already completed results pinned by the suite config without
repeating their computations. Collected-result receipts must match the exact declared result,
including its original execution provenance. Collection does not retroactively assert that old
results were produced by the current implementation snapshot.

Read-only verification is separate:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python -m \
  experiments.phase1.multireaction.ugi_realism_diagnostic_suite \
  --repo-root . \
  --verify results/phase1/ugi_realism_diagnostic_suite_v1/result.json
```

The implementation inventory conservatively covers Python under `forge/` and `experiments/`,
plus the project and lock files. Code drift invalidates reuse; use a new versioned output and
configuration rather than editing a historical receipt. Hashes authenticate artifacts, not
scientific claims: suite completion explicitly leaves model improvement unestablished and visual
review pending.

## Validation outcome

- `make verify`: all 30 vendored assets verified.
- Focused diagnostics and source-provenance checks: **53 passed**, no failures or skips.
- Black and Ruff: all 11 touched Python files pass; whitespace checks pass.
- Collection, read-only verification, fresh integrated execution, and verified reuse all succeed.
  Compared scientific fields and all evaluator artifact hashes are identical between standalone
  execution and fresh integrated replay. See
  `results/phase1/ugi_realism_diagnostic_suite_v1/replay_equivalence.json` for exact comparison
  commands and input hashes.
- Final `make test`: **2,052 passed, 170 failed, 21 setup errors, 92 skipped/xfail** across 2,335
  test cases. All new audit tests pass in this run. Remaining failures include missing historical
  result artifacts, an unchanged stale project-file pin, the tracked-but-uncatalogued specialist
  specification, and the existing M0 transfer-audit contract error. Their checks were not weakened
  or skipped. The initial full run is retained too; its two new source-tracking failures were
  resolved by staging the pinned new source files.

Full logs, JUnit reports, test/source hashes, commands, and failure examples are in
`results/phase1/ugi_realism_diagnostic_validation_v1/`. The validation result SHA-256 is
`cdf2c7c5abc7c20966650e66d3d1568ddcb11458f6590e2a88081c480403b992`.
Repository-wide Phase 1 definition of done is **not met** because `make test` remains failing;
the completed deliverable here is the implemented and exercised diagnostic workbench, with that
validation limitation explicitly preserved. This is not release readiness or model promotion.

## Validation follow-up, 2026-09-08

The continuation corrected two test assumptions while preserving the workbench implementation,
frozen configurations, model weights, and historical results:

- Architecture discovery had treated a `forge.modal_experiment_group.v1` batch manifest as an
  individual experiment. Explicit group ownership now covers that manifest without filtering
  unknown JSON files out of discovery. Both schemas are parsed, and every group member must
  resolve to a catalogued experiment with the declared identity, hash, profile, and replicate.
- The FlowER end-to-end test unconditionally read `hash_matches`, although missing optional
  evidence deliberately has no observed hash. An isolated temporary workspace now retains exact
  copies of the three required pinned inputs and tests absent, matching, and mismatching optional
  evidence. Missing evidence remains missing, mismatched evidence retains its blocker, and all
  cases preserve the frozen demotion. Required-input mismatch still fails before writing a result.

The original installation-smoke project-file pin remains unchanged and fails authentication. Its
failure is retained; adjusting that historical pin would not establish reproducibility.

The saved baseline's 191 failing/error nodes now have a first-visible-cause inventory: 40 legacy
source identity/resolver failures, 138 historical result/cache failures or errors, ten sealed-artifact
failures or errors, and the three separately diagnosed catalog/FlowER/Modal cases. These categories
describe the first observed failure, not all transitive prerequisites. Exact historical source
copies were found locally, including five identities absent from the source archive. Restoring
those identities requires archival resolution; recreating the retired `src/` layout is invalid.
No matching missing result artifacts were found in the scoped search. Sealed structure contents
were not opened, and no historical source or data artifact was restored or regenerated.

The inventory, full expected hashes, declaration locations, scoped search method, and unresolved
paths are preserved in
`results/phase1/ugi_realism_diagnostic_validation_followup_v1/artifact_triage.json`.

Follow-up verification reports **75 focused checks passed**, all **30** vendored assets verified,
and passing Black/Ruff checks for both repaired test files. The full run reports **2,057 passed,
168 failed, 21 setup errors, and 92 skipped/xfail** across **2,338** cases. Compared with the saved
baseline, the two targeted failing nodes are resolved and there are **no new failed nodes**. The
FlowER end-to-end case is now represented by its three explicit optional-evidence cases; all pass.
The original 53 diagnostic/provenance tests are included in the passing focused checks. Both the
saved suite and replay still verify against their original source inventory.

Full logs, JUnit reports, the failure-node comparison, input hashes, and a runnable summary script
are in `results/phase1/ugi_realism_diagnostic_validation_followup_v1/`. The new `result.json` SHA-256
is `71ee98e1b818597f9cac1fa4e1f40f50bbe476ddde227f4bc6d68bfc3025cc50`.
The original validation record is preserved. Repository-wide Phase 1 definition of done remains
**unmet**; this continuation establishes two test-contract repairs and explicit failure accounting,
not restored historical evidence or improved model realism.
## Passive terminal-choice attribution follow-up

On 2026-09-08 the active objective became improved realism with no observed loss of validity,
exact Ugi reconstruction, diversity, or novelty. The additive requirements and unchanged frozen
gates are in `docs/PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md`.

The bounded replay uses the complete first original batch of 128 requests from the audited
amine baseline and preceding tiered-head treatment. It records decisions for the first 16
prespecified attempts. Both endpoint-only controls and full traces reproduce the historical
products, topology, and abstentions exactly. Within each arm, every output row, sampler summary,
terminal tensor, and captured random-generator state is identical with and without decision
tracing. The observer adds no model evaluation or random draw.

| First 128 requests | Baseline | Prior treatment |
| --- | ---: | ---: |
| Valid and exact-L1 products | 128 | 128 |
| Distinct exact-L1 products | 100 | 114 |
| Effective component count | 29.5685 | 34.4265 |
| Mean pairwise ECFP4 distance | 0.412159 | 0.461797 |
| Products containing any novel component | 119 | 86 |
| Whole products novel to TRAIN | 119 | 86 |
| Distinct exact-L1 products with a novel component | 96 | 81 |
| Novel amine-component attempts | 99 | 69 |
| Novel aldehyde-component attempts | 92 | 46 |

All incidence denominators are 128; failed or ambiguous attempts would remain in those
denominators. Distinct amines increase 19 to 40 and distinct aldehydes 35 to 49, despite lower
novelty incidence. Aldehyde effective count decreases 21.6278 to 17.9748. Isocyanides remain
three familiar components in both arms. The previous treatment fails the active objective.

Among the 16 traced attempts, the baseline selects a neural-score maximum in 13/14 amine
topology decisions with multiple candidates, compared with 7/14 for treatment. Mean actual
probability on neural maxima is 0.8063 versus 0.2888. In the ten amine-chemistry decisions with
multiple candidates, the baseline uses deterministic neural maxima; treatment selects a neural
maximum in 2/10 decisions, with mean probability 0.3078 on those maxima. These observations
describe how the selection law changes preferences; a neural maximum is not a realism label.

The treatment's 16 aldehyde unsaturation-count draws use measured frequencies independently of
their captured neural group scores. Eleven choose the group with maximal neural group score.
Only three subsequent position draws have multiple candidates; none selects a neural maximum
within its sampled count group. Those conditional draws must not be treated as an unconditional
tail distribution. The sixteen isocyanide count populations each contain just one candidate.

The analysis summarizes 64 baseline and 128 treatment topology/joint-terminal decisions. It
excludes candidate-internal scoring, duplicate commit records, coordinate choices, and pattern
group choices; it is not a count of every terminal operation. It distinguishes candidate
alternatives from stochastic choices with multiple positive-probability alternatives. Raw
candidate sums and group scores are not calibrated molecular probabilities.

Reproduction commands (use a fresh output directory for a new execution):

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python -m experiments.phase1.multireaction.ugi_sampling_trace \
  --repo-root . --config configs/multireaction/ugi_sampling_trace_v1.json \
  --output-dir results/phase1/ugi_sampling_trace_new
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python -m experiments.phase1.multireaction.ugi_sampling_trace_analysis \
  --repo-root . --trace-result results/phase1/ugi_sampling_trace_v2/result.json \
  --output-dir results/phase1/ugi_sampling_trace_analysis_new
```

The completed replay result SHA-256 is
`84ee1e1a394f4adda02b7d71485c84fc64eb83f17d2823d338875a449b07ce60`;
the analysis result is
`9f1cbd61829d757cadd69fe169077d660ce9ac304203fdb75b81d52224f02274`.
The first run's final receipt failed on a pin-record metadata mismatch after both arms completed;
its failure and completed stage artifacts are preserved in `results/phase1/ugi_sampling_trace_v1/`.
The corrected second run completed in 159.5 seconds including preparation and four sampling
passes. The offline analysis was corrected after sampling to use all-attempt novelty denominators,
reject impossible zero-probability selections, and distinguish deterministic candidate alternatives.
Its earlier, unused-by-sampling source is preserved under the replay's `source_archive/`, matching
the conservative whole-repository snapshot. The final analysis records its own source hash.

The next hypothesis is one bond-logit refresh after all atom states are committed and before
variable bond selection. The first test is a frozen-model TRAIN response probe with exact
topology, paired corrupted bonds, and correct atom/forced-ester commitments. It cannot establish
robustness to incorrect generated commitments or improve already selected head placement.
Neither new model training nor a new sampling intervention has occurred in this follow-up.

Trace validation records are in `results/phase1/ugi_sampling_trace_validation_v1/`: **96** final
focused checks and **30** vendor checks pass. Black/Ruff pass for the eight new Python files.
Full tests report **2,097 passed, 168 failed, 21 setup errors, and 92 skipped/xfail**, with no new
failed nodes. The full run predates the final offline-analysis corrections and three additional
regression tests; the final focused run covers those corrections. This receipt excludes the
subsequent committed-bond probe. Global readiness and realism improvement remain unestablished.


## Committed-chemistry prerequisite result

The subsequent fixed TRAIN probe is complete; it **does not advance** to a sampling intervention.
It covers all 11/7/3 admitted measured component identities in sixteen products, preserving the
231/480 admission census and all evaluated uncommitted bond inputs.

| Flow time | Corrupted amine bonds, correct | Corrupted aldehyde bonds, correct | Corrupted isocyanide bonds, correct | Equal-role target NLL | Gate |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 7/9 → 9/9 | 11/14 → 11/14 | 8/8 → 8/8 | 0.646453 → 0.350024 | Pass |
| 0.9 | 2/2 → 2/2 | 4/4 → 4/4 | 1/1 → 1/1 | 0.050344 → 0.048037 | Fail: no additional correction |

Both times were required to pass, and the latter condition was already at an accuracy ceiling
on its seven corrupted coordinates. The criterion is retained. Repeated inputs produce identical
logits; no per-role accuracy regression occurred. Exact all-uncommitted bond-vector recovery is
9/16 → 10/16 at time 0.5 and 13/16 → 13/16 at 0.9.

The result combines correct atom states with correct forced ester commitments. It does not
isolate the contribution of atoms, test erroneous generated commitments, or establish terminal
sampling benefit. No new sampler intervention or training was run. Full config, selection,
source/input hashes, raw TRAIN tensor evidence, results and the non-advancement decision are in
`results/phase1/ugi_committed_bond_probe_v1/`; the result SHA-256 is
`7486e37ba6bea4bf7f3fe41b124bb54f42c8a383ee2d8dd9693d1433d82b7164`.

Final validation, including the probe and corrected trace analysis: **106 focused tests pass**;
**30 vendor assets verify**. The full run on current Python/test files reports **2,110 passed,
168 failed, 21 setup errors, and 92 skipped/xfail**, with no newly failing nodes. The separate
validation report and executable summary are in
`results/phase1/ugi_committed_bond_probe_validation_v1/`; result SHA-256
`9e57a27b80bc97bf55ed506dceb9999673fac50fbe55167ee0be0b6d1a6aab45`.
The failed scientific advancement criterion and failing global test status both remain explicit.
