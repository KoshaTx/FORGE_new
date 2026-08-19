# Frozen single-step proposal-lane qualification benchmark

## Decision status

This document specifies a bounded **development-only** benchmark. It does not
execute Graph2Edits, activate a learned backend, alter the production planner or
create route evidence. The strict lipid-precedented template/retrieval lane
remains the production default.

The machine-readable policy is
`configs/route/single_step_proposal_lane_qualification_benchmark_v1.json`.
The materialized-manifest binding is
`configs/route/single_step_proposal_lane_qualification_benchmark_v2.json`.
The nonexecuting runner binding is
`configs/route/single_step_proposal_benchmark_runner_binding_v1.json`.
Execution remains blocked until the isolated runtime, license review and
deterministic production-device receipt are separately frozen.

## Nonexecuting runner boundary

`src/forge/route/single_step_proposal_benchmark.py` now implements the
matched-budget runner without activating a proposal lane. Its binding
authenticates the frozen policy, the already materialized 120-target manifest,
the independent upstream-L2 resolver, the isolated runtime lock and runtime
qualification receipt. The runner refuses construction when the loaded
resolver digest differs from the bound resolver digest.

Gate checks occur before the caller-provided target-manifest loader is invoked.
With the present activation and prerequisite values, neither target content nor
Graph2Edits inference can be reached. The remaining blockers are institutional
license review, freezing the isolated runtime for the production device, ten
byte-identical production-device repetitions and a separate execution
authorization. The local macOS smoke qualification does not satisfy those
production gates.

Lane execution and truth scoring are separate APIs. The lane runner accepts
only public target records and freezes a content-addressed output ledger.
Documented reactant multisets are accepted only by the subsequent scoring
function, whose input digest must equal the scoring artifact already bound in
the manifest configuration. Model scores are not read by the resolver or the
scorer. Resolver outputs retain null evidence and success-probability fields,
cannot close routes and cannot enter `V_syn`.

Focused tests establish that current preflight failure happens before target
loading or lane calls, all three synthetic lane engines receive one identical
budget, the independently admitted resolver is applied after proposal, hidden
truth is introduced only after output freeze, incomplete lane-target grids fail
closed and no scoring result carries evidence or synthesis-value authority.
These are software-contract tests, not Graph2Edits accuracy results.

## Question

The benchmark asks whether an unrestricted general single-step model adds useful
route hypotheses for lipid-like Ugi components after independent chemical
screening, and whether a strict-first hybrid improves proposal yield without
losing the precision of the existing lipid-precedented lane.

It does **not** ask whether the Graph2Edits score predicts synthesis success.
Model output can propose a disconnection. It cannot establish precedent,
substrate scope, operational feasibility, terminal availability or route
closure.

## Three matched lanes

1. **Strict lipid-precedented.** Search exact lipid-component routes, admitted
   lipid family templates and lipid-component analogue retrieval. No learned
   backend is called.
2. **Unrestricted Graph2Edits proposal-only.** Use the frozen
   reaction-class-unknown USPTO-50K checkpoint without a lipid-family whitelist.
   Its score controls only within-lane raw rank.
3. **Strict-first hybrid.** Combine the two sources while preserving source
   provenance, screening every hypothesis identically and deduplicating across
   sources.

The hybrid receives 100 reserved strict and 100 reserved learned raw hypothesis
slots per target; unused quota can transfer. This prevents the hybrid from
quietly receiving twice the search budget. At equal screen state, strict-source
hypotheses remain ordered first. This is a proposal-ordering rule, not an
evidence preference in `V_syn`.

## Development target manifest

The benchmark has 120 targets selected with seed `20260830` from hash-pinned
development artifacts. Every target receives one disjoint primary stratum and
may retain secondary chemotype tags:

| Primary stratum | Count |
| --- | ---: |
| Known exact L2 routes | 18 |
| Held reaction families | 18 |
| Linear aldehydes | 12 |
| Branched aldehydes | 12 |
| Unsaturated aldehydes | 12 |
| Ester-containing aldehydes | 12 |
| Isocyanide/formamide precursors | 12 |
| Heterocyclic amine heads | 12 |
| Adversarial incompatibles | 12 |

The target list is now materialized as 120 constitutionally unique, valid
connected molecules with exactly the frozen stratum counts. Known routes are
withheld from lane inputs and retained only in a separate scoring artifact. For
the held-family stratum, every record and template from the target reaction
family is removed from all FORGE retrieval/template inputs; the hidden
documented route remains scoring truth. This measures recovery outside the
local strict registry, not proof that the family was absent from Graph2Edits'
patent training data.

The immutable output contains 36 exact route truths behind 36 target-specific
visibility masks and 12 valid connected wrong-role controls. Expanded
exact-forward Ugi component exemplars were used as a development source to
populate rare chemotype strata without reusing exact route targets. Five
expanded source rows and one original program row fail the frozen Ugi handle
multiplicity contract; they are reported as source-audit exclusions and were
not silently treated as qualified targets. The materialization receipt records
every opened artifact hash and confirms that no sealed route holdout, private
prospective outcome or private biological holdout was accessed.

Adversarial controls must remain valid, connected molecular targets. They test
wrong Ugi role, wrong reactive handle, role swaps, blocked reaction centers or
independently established forward incompatibility. Malformed SMILES are backend
robustness fixtures, not chemistry benchmark targets.

No route-saturation holdout, prospective outcome or private biological record
may be read to build or score this benchmark. The existing sealed route holdout
is named only in the forbidden-input list so automated checks can reject it.

## Matched budget

Each lane receives, per target:

- one proposal-source call;
- at most 200 raw hypothesis attempts and 200 canonical hypotheses considered;
- at most 50 independent forward-verifier calls;
- at most 50 operational-screen calls;
- at most 20 returned accepted proposals;
- 30 wall seconds, one CPU thread, 12,288 MiB host memory and 12,288 MiB
  accelerator memory.

Target order, verifier budget, operational-screen budget, accepted-proposal cap
and wall-clock cap are identical. Hardware, latency and actual host/accelerator
memory are reported separately because the strict and neural lanes use
different execution substrates. Budget exhaustion is an observed outcome and
cannot trigger a retry.

## Screening cascade and denominators

Every lane retains the same cascade:

1. raw hypotheses requested;
2. syntactically valid canonical reactant multisets;
3. constitutional duplicates removed;
4. exact, unique forward reconstruction;
5. Ugi component-role and handle checks;
6. operational compatibility screen;
7. proposal admission to bounded search.

Each metric reports its explicit numerator and denominator. Passing the final
proposal screen means only that a hypothesis may enter bounded search. It does
not convert the hypothesis into an evidence-bearing route step.

### Independent fail-closed forward verification

The Ugi final-assembly registry is not an L2 verifier. Single-step L2
hypotheses are tested only with hash-pinned upstream transforms whose scope has
been independently admitted. The frozen verifier set contains the four
exact-source upstream transforms and the separately qualified exact C16 and
C18 transforms; target-specific aldehyde evidence supplies additional exact
scope records without broadening a transform to a reaction-family claim.

Graph2Edits is run with reaction class unknown. Any reaction-class label or
score it emits is proposal metadata and cannot select, define or validate the
forward transform. Instead, the screening layer independently tries every
admitted transform and reactant-role assignment whose frozen scope includes the
proposal. A proposal passes only when exactly one admitted assignment produces
exactly one product and that product is the constitutional target. No admitted
verifier, multiple products or multiple reconstructing assignments are
retained as explicit censoring states. A forward mismatch is rejected.

The exact-source registries remain exact-source-only, and the C16/C18
registries remain exact-pair-only. Applying their SMARTS to an unadmitted
analogue would exceed the evidence scope and is therefore not a verifier. The
model output is never permitted to verify itself.

## Frozen metrics

The scored report includes, overall and by primary stratum:

- exact known-route top-1, top-5, top-10 and top-20 recovery using the canonical
  reactant multiset, accepting any independently documented exact route;
- exact-forward-unique pass fraction;
- operational-screen yield among forward-valid unique hypotheses;
- proposal-source provenance, including strict-only, learned-only and overlap;
- invalid raw-output and canonical-duplicate fractions;
- latency p50, p95 and maximum;
- peak host and accelerator memory;
- accepted proposals per target and per matched budget;
- every censoring and budget-exhaustion state.

The adversarial stratum reports accepted controls explicitly and must remain
zero. Paired per-target lane differences are retained rather than presenting
only aggregate averages.

## Promotion rule

Only the hybrid lane may be promoted; the unrestricted model is never granted
independent evidence authority. Promotion requires every prerequisite gate and:

- no more than 0.02 absolute loss in top-10 known-route recovery versus strict;
- no more than 0.05 absolute loss in exact-forward-unique rate;
- no more than 0.05 absolute loss in operational-screen yield;
- at least 10% aggregate gain in accepted proposals per matched budget;
- improvement on the held-family stratum and at least three additional
  nonadversarial chemotype strata;
- raw invalid fraction no greater than 0.10 and duplicate fraction no greater
  than 0.25;
- zero accepted adversarial incompatibles;
- p95 latency at most 30 seconds and peak memory at most 12,288 MiB.

A tie or failed gate leaves Graph2Edits inactive. There is no retry and no
threshold revision after the scored run. A negative result is informative: it
means deterministic lipid-precedented search remains preferable under the
current data, checkpoint and compute envelope.

## Scientific authority boundary

Graph2Edits model scores may be stored only as opaque within-lane ranking
metadata. They may not:

- enter `V_syn`;
- set an evidence tier;
- close a route;
- imply operational compatibility;
- imply experimental success.

Exact forward reconstruction and operational-screen passage are necessary
proposal filters, not experimental evidence. Independent source lineage,
substrate-scope evidence, route-family admission, current terminal-material
status and recursive L2/L3 closure remain authoritative.

## Remaining prerequisites

Before benchmark execution, a separate owner must:

1. complete institutional review of the checkpoint/code and patent-derived
   training-data chain;
2. freeze an isolated Syntheseus/Graph2Edits runtime without changing the
   generator environment;
3. establish byte-identical top-k identities and ranks over ten repetitions on
   the production device;
4. preserve the already passing malformed-output, deduplication and budget
   guard tests;
5. issue a separate activation decision only after the one scored benchmark.

Until then, the configuration remains a nonexecuting preregistration and the
strict lane remains active by default.
