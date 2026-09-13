# Endpoint time: paired copying and repair diagnostic

Date: 2026-09-08. Frozen before neural forwards for this hypothesis.

The active objective remains improved lipid realism with unchanged validity, exact Ugi
reconstruction, diversity, novelty and all existing gates. Previous scoring, placement and
bond-completion experiments do not establish that objective. This diagnostic isolates the
final chemistry time tag, comparing exactly **1.0** and **0.98**. No lower-time or temperature
sweep, fitting, sampling trajectory or successor molecule generation is included.

## Hypothesis and limits

Production training clamps time to [0.02, 0.98], with probability mass exactly at 0.98.
Terminal A and topology-conditioned B use 1.0. B retains native endpoint chemistry after
replacing topology. The training paired pass instead combines target-derived noisy chemistry
with target topology. Small time extrapolation alone does not prove a defect; the distribution
mismatch is broader than the time tag. A denoiser may appropriately favor copying near the
clean limit. This experiment measures sensitivity and known-target repair, not calibrated
molecular probabilities or improved realism.

## Frozen data and controls

Authenticate the original checkpoint/member/model state, cache, assignments, support ledger,
saved native A/B tensors, completed placement-probe originals and all consumed source bytes.
Use the unchanged CPU float32 model in evaluation mode with two Torch threads. Preserve original
batch geometry and conditioning; no mixed precision, altered program conditions or new weights.

For generated-state sensitivity, use the original first **128-row** A/B batch. Require complete
prediction identity for a time-1 A replay, a time-1 B replay, and an identical B repeat. A fourth
forward changes only B's time to 0.98. Preserve every other input tensor and cached program
memory. Primary requests remain **0–7**, in order, with all **286** original ordered head and
aldehyde topology contexts from the completed placement probe. The remaining original batch
rows retain their geometry and contribute to actual compute counts, not primary denominators.

Score each existing original completion with the new shared B logits and the original A score.
Use identical full-role variable atom, parent-bond and closure-bond masks and float64 full-vocabulary
log probabilities. Reproduce the old full scores before calculating new laws. Preserve all **16**
request/role laws, their candidate ordering and every original completion/admission receipt.
Do not install alternative topologies into B, decode new completions, filter candidates or change
the previous rejection. Report input-state copy probabilities, argmax agreement, score changes
and law total variation. Generated states have no known target: none of these changes is called
a correction. These are scores of fixed old completions, not the .98 decoder's output distribution.

For known-target controls, select all **480 measured Ugi TRAIN records** using metadata before
record materialization. Preserve their frozen support-ledger admission status; the **231 admitted**
records are the primary analysis, and all 480 are a separately reported secondary population.
Do not re-adjudicate admission, select by error or access calibration/heldout structures. Use stable
cache order and batches of 128 with the original checkpoint node/bond marginals.

Use a local CPU generator seeded **2026090841**. Draw native q(0.98) corruption exactly once per
record, using the existing batched noising primitive, then replace topology with the clean target
exactly as in the training second pass. Save the native noisy state, substituted input and local
generator receipts. Compare tags 1.0 and 0.98 on that identical chemistry. Also compare both tags
on wholly clean inputs. Call the prediction primitive directly; do not incur an unused preliminary
neural forward. No seed replacement, redraw, record replacement or error-dependent panel extension.

Masks exclude padding, fixed states and reaction-core coordinates. Define an actually corrupted
coordinate by input state differing from target, not by a resampling event. Report target NLL,
target probability, entropy, input-state probability, argmax correctness, paired corrected/harmed
counts and all denominators per record, role, chemical field and corrupted/uncorrupted stratum.
Include combined atom/bond coordinates per role. Report isocyanide results contextually. Fixed input
states must remain fixed; changes in raw fixed-coordinate logits are not decoder gate violations.

## Necessary advancement criteria

For each of amine and aldehyde in the admitted primary TRAIN population, require:

- Nonempty actually corrupted variable-chemistry coordinates.
- Mean corrupted-coordinate target NLL at .98 lower than at 1.0 by more than **1e-10**.
- At least **one net argmax correction**, corrected minus harmed, on that same stratum.
- No decrease in correct-coordinate count on same-q uncorrupted coordinates or wholly clean
  control coordinates, tested separately for each role.

Empty or ceiling-limited corrupted strata do not pass and do not authorize another draw. Report
clean-control NLL/entropy as well as accuracy so confidence softening remains visible. Coordinate
observations share molecules and components; no independent-trial or held-backbone claim follows.
The control's constructed q(.98) distribution matches the .98 tag, so its improvement alone is
insufficient evidence about generated endpoints, whose effective noise level is unknown.

Also require at least one complete competing-topology request in **each role** with total variation
of at least **0.01** between the original A-plus-B-completion law and the .98 rescoring law.
Qualifying requests may differ by role. Numerical tolerance 1e-10 is not a practical effect threshold.
All controls and denominators must authenticate; no surviving-subset renormalization or absent-law
omission is allowed. Both TRAIN and generated-state criteria are necessary. Failure stops this
hypothesis without tuning time, seed, panel, coefficients or criteria.

A pass only warrants designing a subsequent matched sampler experiment. It does not promote a
model or establish any of the original realism/preservation requirements. Those remain governed
by `PHASE1_UGI_REALISM_IMPROVEMENT_PROTOCOL.md`, including actual blinded review and fresh confirmation.

## Work and provenance

The complete budget is one model construction, **20 model forwards / 2,432 examples**: four original
128-row generated-state forwards and four conditions for each of four TRAIN batches totaling 480
records. Count native corruption calls and local RNG advancement separately from topology sampling.
Training, strict decoding, new molecular validation, exact-L1 adjudication, generation/topology
sampling, routing, biological guidance, remote compute and gate changes are zero.

Freeze source/config/input hashes before forwards. Save all raw predictions, inputs, masks, selection,
scores, counts, failures and elapsed time in a fresh attempt directory. Enforce model/input immutability
and unchanged global RNG after model preparation; native corruption uses only its recorded local
generator. Archive failed attempts without admitting partial numerical success or automatic retries.
The existing repository-wide validation failure remains distinct from this diagnostic's checks.
