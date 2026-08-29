# Implementation plan: Stage 1 value head

**Scope: Stage 1 only.** This builds the partial-state value head and runs its gate. It does not
implement guidance, does not touch the sampler, and consumes no oracle calls or new biological
data. Stage 2 is deliberately out of scope and must not be started on the strength of a Stage 1
pass alone.

Read `ATOM_LEVEL_POTENCY_GUIDANCE_PREREGISTRATION.md` first: it states what is being reopened,
why the previous attempt failed, and the four gates below. This document is only *how*.

---

## 0. What already exists, and should be reused rather than rebuilt

| Need | Existing artifact | Why it matters |
|---|---|---|
| Component-disjoint split | `results/m0_07/oracle_split_manifest.json`, `held_aldehyde_5fold` | Already frozen, five folds, **zero `test_group_leakage`**, train/calibration/test rows recorded per fold. Inventing a new split would be both wasteful and less defensible. |
| Labelled endpoints | `results/m0_07/agile_oracle_curated.csv.gz` | The 1,100 measured products, already reconciled. |
| Completed-molecule oracle | `forge/potency/oracle/oracle_production.py` | Stays the final assessor and is not retrained or modified. |
| Frozen encoder | the shared Transformer checkpoint | Frozen. Only the head trains. |
| Conformal machinery | `evaluation_contract` in the split manifest (split conformal, absolute residual) | Gives a ready interval procedure rather than a bespoke one. |

**Do not create a new split.** If a fold structure other than `held_aldehyde_5fold` is needed,
that is a change to the pre-registration and must be argued before it is coded.

## 1. Data path

1. Load the 1,100 labelled products and their conservative mTP endpoints.
2. Encode each to its program state $X_1$ with the same deterministic encoder the generator uses,
   under its reaction program $P$. Reuse the production cache path; do not write a second encoder.
3. For each product, sample flow times $t$ on a fixed grid, corrupt $X_1$ to $X_t$ with the frozen
   source marginal and the same corruption the trainer uses, and record $(X_t, t, P)$.
4. Freeze the corruption seeds per (product, time) so the dataset is byte-reproducible and the
   digest can be pinned.

**Fail closed:** if the encoder cannot produce an exact state for a labelled product, abstain on
that product and record the count. Do not repair, and do not silently drop.

## 2. The head

- Input: the frozen Transformer's node and program representations at $(X_t, t, P)$.
- Output: a scalar prediction of the final conservative mTP.
- Small: a pooling layer plus a two-layer MLP is the intended size. If it needs to be large to
  learn anything, that is itself evidence against the hypothesis.
- The Transformer stays in eval mode with gradients off. Assert parameter identity before and
  after training and record the state hash, so "frozen" is checked rather than asserted.

## 3. Evaluation, per fold and per time bin

For each of the five folds and each of three time bins (early, middle, late):

- Predict on the fold's held-out components.
- Report a ranking statistic and its interval against a **shuffled-label control** trained
  identically. The control is what makes "beats chance" measurable rather than assumed.
- Report the number of held-out components, not just rows. With 20 amines, 11 aldehydes and 5
  isocyanides in full factorial, a fold's row count overstates its information badly.

Then the decisive one:

- **Within-morphology.** Recompute the same statistic after conditioning on morphology class,
  using the morphology definition the layout prior already uses. A head that predicts only
  *between* classes is the closed 2026-08-04 experiment at higher resolution.

## 4. The gate

All four must hold, and each was set before any code ran:

1. Component-disjoint interval excludes the shuffled-label control.
2. Effect size meets a bar declared in the pre-registration before training, not after.
3. Holds separately in the time bins where guidance would act; weight is zero elsewhere.
4. Survives conditioning on morphology class.

**Any failure stops the work.** Record the negative with the same prominence as a pass, in
`docs/DECISION_LOG.md`, and do not proceed to Stage 2.

## 5. Artifacts

- `results/phase1/ugi_partial_state_value_head_v1/result.json` — per fold, per time bin, per
  gate: statistic, interval, control statistic, held-out component count, abstention count.
- The pinned dataset digest, the frozen encoder state hash, and every seed.
- One decision-log entry stating the outcome and, on a pass, the exact gate numbers that would
  have to be quoted in any Stage 2 authorization.

## 6. Order of work

1. Wire the dataset builder against the frozen split and pin its digest.
2. Reproduce a known quantity as a smoke test before trusting anything: predict a **structural**
   property (heavy-atom count) from the same representations. If the head cannot learn that, the
   pipeline is broken and no potency result from it means anything.
3. Train the head and the shuffled-label control on one fold. Look at Gate 4 first, on that
   single fold, before running the remaining four.
4. Only if Gate 4 survives one fold, run all five and report.

Step 3 is deliberate: Gate 4 is the cheapest refutation and the most likely failure, so it is
tested before four fifths of the compute is spent.

## 7. What is explicitly not in this plan

- No sampler changes. No guidance. No modification of $\Pi_P$ or the reaction-core coordinates.
- No oracle calls, no new biological measurements, no candidate selection.
- No retraining of FORGE or of the completed-molecule predictor.
- No claim, at any outcome, about in-vivo behaviour.
