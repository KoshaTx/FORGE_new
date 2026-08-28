# Pre-registration: atom-level mTP guidance during Ugi generation

**Status: proposal. Not authorized, not started, no result.** This document exists to be
accepted or rejected before any code runs. It requests a new milestone, because the
frozen contract explicitly closed the thing it most resembles.

---

## 1. What is being reopened, and why that needs justifying

On 2026-08-04 the repository closed morphology-level potency tilting after a single
pre-authorized matched challenger. The relevant facts, from that entry:

| | Applicability arm | Nested-potency arm |
|---|---|---|
| Terminal attempts | 3,072 | 3,072 |
| Exact-L1 terminals | 2,955 | 2,962 |
| Eligible terminals before budget matching | 90 | 82 |
| Oracle calls | 38 | 38 |
| **Unique conservative-high products** | **27** | **23** |

Paired difference: **-0.00130 unique high-potency products per generator call**, 95%
interval **-0.00228 to 0.00098**. The restricted new-head diagnostic did not improve
either (two versus zero at three oracle calls per arm).

That entry drew two conclusions this proposal must answer rather than sidestep:

> "The measured morphology classifier signal was real enough to justify one test but did
> not translate into improved terminal discovery efficiency."

> "The failure is not explained by molecular validity and cannot be repaired by selecting
> another proposal from the same biological data."

The decision was to close additional same-data potency-proposal tuning.

**So the burden here is not "is there signal." Signal was already demonstrated and still
lost.** The burden is to state a mechanism by which atom-level guidance could convert when
morphology-level guidance did not, and to pre-commit to a gate that would have stopped the
previous attempt before it consumed a matched comparison.

## 2. The mechanism argument

Morphology tilting can only reallocate probability across *shapes*: it changes which
role-size and topology envelopes get sampled, then generation proceeds unguided inside the
chosen envelope. If the potency-relevant structure lives *within* a morphology class rather
than *between* classes, a morphology controller cannot express it at any strength, and its
classifier can be genuinely predictive while its proposal is powerless.

Atom-level guidance is a different operator: it biases individual atom and bond choices
inside a fixed morphology and a fixed reaction program. It can therefore reach structure the
coarse controller could not express.

**This is a hypothesis, not a result.** It is falsifiable at Stage 1 below, and it is the
only reason to reopen. If potency variance is mostly between morphology classes, this
proposal should be rejected now rather than run.

## 3. The binding constraint, stated before any design

The labelled set is **1,100 measured Ugi products, and it is a complete factorial**:
20 amines x 11 aldehydes x 5 isocyanides = 1,100.

That number is the whole design problem. The effective sample size for *component
generalization* is **36 distinct components, not 1,100 molecules**. Consequences:

- A component-disjoint split on isocyanides holds out one of five, removing 20% of all rows.
- Any held-out fold is small and highly correlated, because every held-out product shares
  components with many training products by construction.
- A value head can reach high in-fold accuracy by memorizing 36 component identities while
  learning nothing transferable.

**Therefore the Stage 1 gate is evaluated on held-out components, never on held-out
products, and the interval must be reported, not just a point estimate.** Any result whose
interval includes chance is a stop.

## 4. Stage 1: the value head, and the gate that can kill this cheaply

Train a small head on the frozen Transformer representation:

$$V_\phi(X_t, t, P) \rightarrow \text{expected final conservative mTP}$$

Corrupt the encoded states of the labelled products at sampled flow times; the shared
Transformer stays frozen; only the head trains.

**Prespecified gate, all four required:**

1. **Component-disjoint generalization.** Evaluated on held-out *components*, with a
   declared split, reporting a paired interval against a shuffled-label control. The
   interval must exclude chance.
2. **Effect size, not just significance.** Declare the minimum correlation or ranking
   statistic before training. "Beats chance" is explicitly rejected as a bar, because the
   morphology classifier already beat chance and still lost.
3. **Time-resolved.** Evaluated separately at early, middle and late flow times. Guidance
   weight is zero at any time that fails, and there is no interpolation into failing regions.
4. **Within-morphology.** The head must retain predictive power *after conditioning on
   morphology class*. This is the direct test of Section 2's mechanism: a head that predicts
   only between morphology classes is the closed experiment again, and stops here.

**Stop rule: if any of the four fails, stop. Do not retrain FORGE, do not proceed to
Stage 2, and record the negative result with the same prominence as a positive one.**

Gate 4 is the one most likely to fail, and it is the cheapest possible refutation of the
whole idea. That is deliberate.

## 5. Stage 2: guidance, only if Stage 1 passes

- Bias atom and bond choices toward predicted improvement in final conservative mTP.
- Reaction-program fixed coordinates are never touched; $\Pi_P$ is unchanged.
- **Fail closed:** invalid, uncertain, or applicability-unsupported states receive zero
  biological guidance. Guidance strength is zero wherever the Stage 1 gate did not pass.
- The completed-molecule predictor remains the sole final assessor. Guidance changes which
  molecules are produced, never how they are judged.

## 6. Primary endpoint, deliberately identical to the closed experiment

**Unique conservative-high-potency products per generator call**, with a paired interval,
under matched generator calls, oracle calls, wall time and portfolio size.

This is the same endpoint, the same denominator and the same interval procedure the closed
challenger used. That is the point: the result must be directly comparable to the frozen
-0.00130 [-0.00228, 0.00098], not to a fresh metric with no precedent.

Arms, at equal budget:

1. Support-aware generation (the current production reference)
2. Post-hoc mTP ranking
3. Atom-level mTP guidance

Also reported: validity, exact-L1 yield, diversity **and effective component count**
(diversity alone is diagnostic, not directional), and abstention counts.

**Declare the oracle budget before running.** The closed experiment used 38 calls per arm
and resolved 27 versus 23 products. Whether that budget could have detected a real effect
is unresolved, and if it is the binding constraint then no guidance design fixes it. A
power statement is required before Stage 2, not after.

## 7. What this cannot claim, at any outcome

- Not prospective validation. No wet-lab synthesis or measurement is involved.
- mTP is a predicted quantity from a model trained on 1,100 products spanning 36 components;
  it is not an in-vivo endpoint.
- A positive result would show that atom-level guidance improves *predicted* conservative-high
  discovery efficiency under a frozen assessor, nothing more.
- Exact-L1 remains transform consistency, not synthesis-success probability.

## 8. Honest assessment of whether to run this

The mechanism in Section 2 is plausible and worth testing. The constraint in Section 3 is
severe and may be fatal: 36 components in full factorial is a thin basis for a head that
must generalize to new components, and Gate 4 asks it to do so *within* morphology class,
which is strictly harder.

The recommended order is therefore: **run Stage 1 alone, and treat it as a likely stop.**
It costs one small head and no generator retraining, it consumes no new biological data, and
it produces a publishable negative result if it fails. Authorizing Stage 2 in advance is not
recommended.
