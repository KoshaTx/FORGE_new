# AGENTS.md — FORGE

Read this file completely before doing anything. It is the contract for work in this repository.

## What this project is

The reviewer-facing reference is now the supplied **FORGE: Reaction-Guided Generative Design of
Ionizable Lipids** manuscript, preserved at [`paper/submission/FORGE.pdf`](paper/submission/FORGE.pdf).
Read [`paper/submission/README.md`](paper/submission/README.md) and its evidence/artifact guides
for the shared three-family study, author-reported in vivo measurements and reproducibility limits.
The older `paper/v1_iclr` source is a related revision, not the exact source of the supplied PDF.
The separate 22-family study remains documented in `docs/STUDIES.md`.

The historical scientific plan is [`docs/PLAN.md`](docs/PLAN.md). Read §1–§5 and §13 before
changing scientific code, preserving the frozen chemistry, splits and evidence contracts.
Its Nature Biotechnology/computational-only framing and the dated authorizations below are
historical context, not descriptions of the newly supplied manuscript. Inclusion of author-reported
experimental data does not authorize new wet-lab work, training or cloud jobs for a repository cleanup.

## Authorized scope — READ THIS

The plan has been **scientifically approved in principle but is NOT implementation-frozen**.

Milestone M0 is complete. On 2026-07-30, the user explicitly authorized
**Phase 1 product plus L1 training** after reviewing the M0 result and training
mixture. The authoritative closeout is [`docs/M0_REPORT.md`](docs/M0_REPORT.md).

On 2026-08-01, after the product-plus-L1 generator and post-selection
provenance audits were frozen, the user explicitly authorized the next
**synthesis-routing and synthesis-guidance milestone**. This authorization is
incremental and fail-closed: implement the structured synthesis-value contract,
planner/cache and restartable-sampler plumbing, qualify a bounded hybrid L2
planner, and only then run matched synthesis-guided versus post-hoc experiments.

On 2026-08-03, the user explicitly prioritized and authorized one bounded,
diagnostic **HeLa mTP potency-guidance pilot** before the nonzero synthesis-guidance
run. This does not authorize unrestricted biological optimization. The pilot must
use the already selected frozen HeLa ensemble, a new hash-pinned applicability and
uncertainty policy, matched in-trajectory and post-hoc arms, and zero synthesis or
proposal-engine calls. Unsupported chemical shifts must abstain, and the run cannot
lock candidates, access sealed holdouts or support an in-vivo efficacy claim.

On 2026-08-21, the user explicitly reconfirmed that the current paper is computational only.
Candidate-panel recovery, procurement, synthesis, formulation, in-vitro testing and in-vivo testing
are not paper-readiness tasks and must not be presented as blockers or next steps.

On 2026-09-25, the user explicitly changed the primary L2/L3 success criterion for the
22-family study to the older repository's **computational makeability** criterion:
every required component is either vendor-listed directly or has a complete computational
route to vendor-listed terminal materials. Planner-generated paths and forward-applied
published-transform constructions can contribute without exact-substrate experimental
evidence. The versioned contract is
`configs/route/compose_lipid_computational_makeability_v1.json`, explained in
`docs/COMPOSE_LIPID_COMPUTATIONAL_MAKEABILITY.md`. Exact-source/current-item dossier
closure remains a separately reported secondary metric with its original checks intact.
This changes the primary evaluation outcome; it does not relabel hypotheses as executed
chemistry or listings as verified stock, and does not authorize production synthesis
guidance. Keep old results under their original metric, preserve full denominators,
and leave the manuscript unchanged until separately requested.

**You are authorized to:**

- Implement and train the sparse whole-lipid discrete-flow backbone selected by M0-06.
- Prepare corrected constitutional R0 for an optional broad-pretraining ablation; do not launch that
  arm until the frozen Ugi-first decision gate identifies a gap that broad structure-only data could
  plausibly close.
- Implement the source-balanced masked Ugi L1 adapter and exact forward round-trip objective.
- Run a bounded Ugi-from-scratch production arm as soon as the support-skeleton and atom-origin
  representation gates pass. Launch the identical broad-pretrained backbone specialized to Ugi only
  if the frozen Ugi-first decision gate identifies a representation, sample-efficiency or
  held-component gap that broad structure-only data could plausibly close. Broad pretraining is a
  decision-gated ablation, not a prerequisite that blocks the first Ugi arm.
- If that decision gate passes, run bounded CPU smoke tests, overfit gates and GPU training for the
  prespecified broad-to-Ugi replay comparison; otherwise preserve the plan as an unexecuted ablation.
- Evaluate validity, connectedness, broad-corpus coverage, diversity, effective component count,
  component novelty, and exact Ugi L1 round-trip consistency.
- Freeze and implement the structured pre-prospective synthesis-value contract,
  planner assessment/cache, restartable product-sampler interface and matched
  budget ledger described in `docs/PHASE1_SYNTHESIS_GUIDANCE_READINESS.md`.
- Implement and independently qualify a bounded hybrid L2 planner without
  promoting family projections, motif similarity, handle qualification or
  provenance beyond their admitted evidence tiers.
- Run zero-guidance equivalence and small diagnostic coupling tests before any
  production guidance sweep. Run the matched synthesis-guided versus post-hoc
  experiment only after the planner and dossier gates pass.
- Implement and run the single bounded HeLa potency-only diagnostic authorized
  above after its versioned applicability policy, conservative utility and matched
  lambda-zero identity control pass. Preserve every abstention and call the result
  signal-limited if no eligible nonuniform guidance contrast occurs.

**You are NOT authorized to:**

- Add biological tilting outside the single versioned HeLa diagnostic above, or
  allow a boundary, extrapolative, unsupported-role or invalid completion to affect
  biological guidance weights.
- Treat raw route likelihood, static registry membership or structural novelty
  as a synthesis-success probability.
- Run production synthesis-value tilting before a nontrivial generated-candidate
  L1/L2/L3 dossier set closes and the zero-guidance equivalence gate passes.
- Build Phases 2–8 from the plan.
- Create the `PREREGISTRATION.md` freeze.
- Recover or lock a prospective candidate panel for this paper, initiate procurement, or plan or
  execute synthesis, formulation, in-vitro or in-vivo experiments as part of paper readiness.

Treat all M0 artifacts, splits, evidence gates, and negative results as frozen inputs. If Phase 1
appears to require out-of-scope work, **stop and report** rather than expanding scope.

## Hard constraints — an agent optimizing for green tests will violate these by default

These are scientific-validity rules, not style preferences. Violating any of them invalidates the
paper's central claims. **Never relax a gate to make it pass. Fail loudly and report.**

The explicit September 25 change above introduces a separately named primary outcome;
it does not change the meaning or acceptance checks of existing strict dossier fields.

1. **Never sample the R1 corpus by raw `reaction_family` counts.** Ugi is 222,768 / 464,265 (48%) and
   would collapse the prior onto one chemistry. Use the `realism_weight` column.
2. **Never extract blocks from all of R0.** Blocks come from `R0_train` only, split *before*
   decomposition. Harvesting from all of R0 and then measuring R0 recovery is circular and is the
   specific defect M0-03/M0-04 exist to prevent.
3. **Never report the `reductive_amination` substructure hit rate.** Its product motif is essentially
   any C–N bond; it matched 99.8% of R0 and is a degenerate artifact, not a finding.
4. **Never call the current R1 "route-certified."** It is **"reaction-enumerated support."** Nothing is
   route-certified until L2 (subcomponent synthesis) and L3 (procurement) close.
5. **Never claim "any graph is expressible."** The correct statement is: *any graph within the declared
   bounded atom vocabulary, bond vocabulary, and size support is representable.*
6. **Never silently reduce the corpus to small molecules.** If dense edge flow fails at 96 atoms
   (M0-06), report it and propose sparse parameterization. Do not quietly cap `N_max` and proceed.
7. **Never make `held_reaction_family` a hard pass/fail gate.** It is a secondary robustness stress
   test. Zero-shot unseen-family retrosynthesis is not required for the paper's claim.
8. **Report both coverage and precision** for retro-decomposition. Coverage alone is not evidence that
   a disconnection is chemically sensible.
9. **Do not fork or modify `compose_v4` / `compose_rgm`.** FORGE *consumes* their corpus and registry as
   hash-pinned vendored data. This boundary keeps Paper 3 from competing with Paper 2.

## Settled decisions — do not relitigate

- **Ugi variant: AGILE-type amine–aldehyde–isocyanide 3CR.** The transform has three reactants and no
  carboxylic-acid reactant component. Do not call it simply `acid-free`: the AGILE procedure uses an
  acidic phosphorus catalyst. See `docs/DECISION_LOG.md` and PLAN §2. The ester in AGILE lipids comes
  from the **aldehyde** component, which is why ester construction is an L2 problem. Config:
  `configs/assembly/ugi_variant.yaml`.
- **Support tiers are E0/E1/E2/E3**, not the older I/F/B/N. E2 (known assembly, generated precursor) is
  the primary open-endedness claim; E3 is exploratory. PLAN §3.
- **Phase 1 model identity is constitutional and stereo-free.** Train on the 12,386 unique
  constitutional Ugi graphs, not the earlier 12,800 source-string union. Preserve all 13,376 source
  rows in the constitutional provenance ledger; never duplicate model weight because two source
  strings collapse to one graph.
- **Ugi origin and reaction-core membership are orthogonal.** Use exact atom-mapped precursor origins,
  a separate core-position state, and adapter-defined role anchors. The assembly-introduced amide
  oxygen is not falsely assigned to a precursor. A core root exists for serialization only; root
  depth is not molecular semantics. Broad non-Ugi records receive no fabricated Ugi annotations.
- **Ugi branch budgets are precursor-origin and component weighted.** Current aldehyde-derived
  components contain at most one exterior branch node and current isocyanide tails contain none.
  Do not let Cartesian product frequency or an unconstrained generic decoder create repeated tail
  junction chains. Broader tail branching requires qualified component supervision.
- **Biological endpoint** is outside the current computational paper. Keep the `Endpoint` interface
  generic for possible future work; do not hard-code an endpoint or treat endpoint selection as a
  current blocker.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
make vendor      # pulls hash-pinned source data (see docs/DATA_PROVENANCE.md)
make verify      # confirms every vendored file matches its recorded sha256
make test
```

**Data access is the most likely blocker.** The source corpus lives at absolute paths on the original
workstation (a git worktree of `compose_rgm`). `make vendor` copies from those paths. If they are not
present — e.g. you are running in a cloud container — `make vendor` will fail with a clear message
listing exactly what is missing and its expected sha256. Do not fabricate, synthesize, or download
substitute data. Report the missing paths and stop.

## Definition of done for any M0 or Phase 1 task

A task is complete when **all** of these hold:

1. Its acceptance criteria in `docs/M0_TASKS.md`, `docs/M0_REPORT.md`, or the frozen Phase 1 config are
   met and demonstrated by a runnable command.
2. Relevant current implementation checks and input verification pass. For the new COMPOSE
   training pipeline, use `make test-training OUTPUT=results/phase1/<new-validation-directory>`.
   On 2026-09-22 the user explicitly removed the requirement to recover old experiment outputs or
   pass the historical full-repository suite before training. Historical reproduction remains a
   separate task; its missing outputs are not training blockers. Current chemistry qualification,
   frozen holdouts, data integrity, molecular-size support and training correctness still apply.
3. Its numeric findings are written to `results/<task_id>/` as JSON, with the input file sha256s
   recorded so the result is reproducible and attributable.
4. `docs/DECISION_LOG.md` has a dated entry stating what was decided or measured — including negative
   results. **Negative results are first-class deliverables here.** A task that discovers the approach
   does not work has succeeded.

## Test iteration

Default to the affected test case or owning test file, using `make test-one TEST=path[::case]`.
After that check passes, stop testing unless a failure, shared-interface change or unresolved
concern justifies broader coverage. Documentation/status-only changes need no pytest run.
Validate changed config/input pins and their affected contracts without rebuilding unrelated data.

`make test-preparation` is an integration checkpoint for changes spanning the preparation pipeline,
not an after-every-edit check. It covers ComposeLipid plus source-coordinate, atom-origin, graph
and cache checks. Use the relevant subsystem checks for other integration changes. See
[`docs/TESTING.md`](docs/TESTING.md) for commands and scope.

Do not launch the full suite after every intermediate preparation change or status request.
Use the relevant current pipeline validation at its readiness checkpoint. `make test-full` remains
available for repository-wide and historical reproduction, but is not a prerequisite for new
training. A focused pass alone is not data admission. Reuse existing results only for their exact
source/input hashes; monitor an already running validation instead of starting another copy.

## Style

- Python ≥3.10, RDKit, PyTorch. `ruff` + `black`, line length 100.
- Prefer plain functions and dataclasses over frameworks. No hydra/lightning in M0.
- Every script that produces a number must record the sha256 of its inputs.
- Deterministic seeds everywhere; record them.
- Chemistry constants (SMARTS, roles, policies) are **read from the vendored registry**, never
  hardcoded or retyped from memory. The registry is the source of truth.

## Compute efficiency and portability

- Write data and training code to use available hardware efficiently on both CPU and GPU. Batch and
  vectorize hot paths; avoid repeated parsing, Python loops over tensors, unnecessary host/device
  transfers, and implicit synchronization.
- Keep memory bounded with streaming, chunking, and configurable batch sizes. Never gain speed by
  silently dropping records, shrinking molecular support, or weakening a scientific gate.
- Keep device, precision, worker count, and determinism settings explicit. Mixed precision or
  nondeterministic kernels are opt-in and require a numerical-equivalence check for the affected output.
- Profile before non-trivial optimization and record a representative before/after benchmark. A
  performance change must preserve tests, schemas, provenance, and scientific results.

### Remote compute durability — mandatory

- Launch every long-running or billable Modal training, cross-fit, production evaluation, and
  aggregate job in **detached mode**. A local shell, Codex session, network connection, or heartbeat
  must never own the lifetime of paid remote computation.
- Use attached Modal execution only for short preflights and smoke tests that are intentionally
  disposable and inexpensive to rerun. The generic attached `forge experiment run ... --backend
  modal` path is not an acceptable launcher for a long-running paid job unless it has been changed
  to submit detached work.
- Before waiting, persist the Modal application/function-call identifier, request ID, source digest,
  configuration digest, and input pins. Monitoring, artifact collection, and verification must be
  separate restartable operations.
- Persist progress to the Modal volume at the natural unit of work: training checkpoints for model
  fitting, completed folds plus score ledgers for cross-fitting, and completed shards for evaluation.
  Commit these records incrementally; do not wait until the entire stage returns to publish all
  recoverable work.
- A local-client disconnect is an operational failure, not a scientific result. Preserve its failure
  record, admit no incomplete metrics, and never automatically retry a paid job. Diagnose first and
  obtain fresh execution authorization before spending again.

## Task-specific workflows

Repository-scoped skills live under `.claude/skills/` and provide focused checklists without bloating
this contract. They are discovered automatically from that path, which is what makes the `$name`
invocations below work; `.agents/skills` remains as a symlink to the same directory, because each
skill also ships an `agents/openai.yaml` for non-Claude tooling.

- Use `$forge-production-engineering` for implementation, refactoring, review, data pipelines, CLIs, and
  tests.
- Use `$forge-paper-writing` for manuscript prose, results, methods, captions, decision packages, and
  reviewer responses.
- Use `$forge-adjudicate-chemistry-evidence` to inspect source articles and supplementary chemistry,
  extract exact reaction evidence, verify reactive sites, and decide whether a claim is admitted,
  precedent-only, abstained, or rejected.

`AGENTS.md` remains authoritative for scope and scientific-validity constraints. The skills refine how
authorized work is executed; they do not expand it.
