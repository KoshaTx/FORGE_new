# Milestone M0 — Task Specifications

The only authorized work. Each task fails cheaply and independently. Read `AGENTS.md` first.

Order matters where noted. M0-01 is done. M0-02..M0-05 are the corpus critical path. M0-06..M0-10 can
proceed in parallel once vendoring works.

Every task writes `results/<task_id>/result.json` including input sha256s, seed, and timestamp.

---

## M0-01 — Ugi variant decision ✅ COMPLETE

**Resolved: AGILE-type amine–aldehyde–isocyanide Ugi 3-CR.** It has no carboxylic-acid reactant
component, but the reported procedure uses an acidic catalyst, so do not call it simply acid-free.
Recorded in `docs/DECISION_LOG.md`. Write `configs/assembly/ugi_variant.yaml` with
`variant: ugi_3cr_agile` if not already present, and add a test asserting the vendored registry SMARTS
has exactly three reactant components.

---

## M0-02 — Vendoring and provenance

**Do.** Implement `make vendor` and `make verify`. Copy each source asset listed in
`docs/DATA_PROVENANCE.md` into `data/vendor/`, verify sha256 against the recorded value, and write
`data/vendor/MANIFEST.json` with source path, sha256, byte size, and retrieval timestamp.

**Accept when.** `make verify` passes on a clean checkout; corrupting one byte of any vendored file
makes it fail with a message naming the file.

---

## M0-03 — R0 splits, created BEFORE any decomposition

**Reconciliation status.** Complete. The original 15,433-row artifact and
splits remain historical audit inputs only. The corrected model-facing R0
contains 15,229 unique constitutional graphs after collapsing 204 duplicate
rows. The 100 B4 mixture measurements contribute no single-compound training
weight; their source associations remain in the exclusion ledger. All 100
corresponding constitutions retain exact B5 single-compound structural support,
and the 100 B5 records carry the reconciled pure-trans provenance.

**Do.** Reconcile the AGILE-derived R0 provenance, freeze the resulting row
count, then load the corrected `r0_observed_real_structures.csv`. Produce
`R0_train / R0_cal / R0_heldout`
under **four independent holdout schemes**, reusing the leak-free grouping columns already present
(`leakage_group_id`, `study_split_groups_json`, `component_holdout_groups_json`,
`reaction_family_holdout_groups`) and `splits_v1/corpus_fold_assignments.csv`:

- source-study holdout · headgroup holdout · linker/scaffold holdout · component-family holdout

**Accept when.** No group spans two folds under any scheme (assert in tests). Fold sizes reported.
Splits are written to disk and **frozen** — later tasks load them, never recompute.

**Critical.** This must complete before M0-04 touches any molecule. The split exists specifically to
prevent the circularity that invalidated the earlier plan.

**Frozen corrected splits.** `data/splits/m0_03_constitutional/` assigns all
15,229 graphs with zero group leakage. The source-study, headgroup,
linker/scaffold, and component-family held-out folds contain 2,328, 2,285,
2,285, and 1,100 graphs, respectively. The historical split bundle is not
reused because constitutional merging exposed source-row fold conflicts.

---

## M0-04 — Non-circular R1′ anchoring, registry coverage, and leakage audit

**Do.**
1. Retro-decompose **`R0_train` only** using the 12 qualified atom-mapped transforms, honoring
   `required_handle_smarts`, `forbidden_smarts`, and `selectivity_policy` from the vendored registry.
2. Harvest components into a pool tagged `provenance: r0_derived`, with occurrence count and source
   study per block.
3. Enumerate **R1′ from training-derived components only**.
4. Evaluate on `R0_heldout`:
   - `Recovery_heldout = |R0_heldout ∩ R1'| / |R0_heldout|`
   - **nearest-reachable distance**: ECFP4 Tanimoto from each unrecovered held-out lipid to its closest
     R1′ member (report the full distribution, not just a mean).

**Baseline to beat.** The current 464,265-product enumeration recovers **0.76%** (108/14,233) of
non-AGILE real lipids. That number is already measured; reproduce it as a control before reporting the
new one.

**Accept when.** Both metrics reported under all four holdout schemes, with the 0.76% control
reproduced. **A result showing R1′ does not beat the baseline is a valid, publishable finding** — report
it, do not tune the enumeration until it passes.

---

## M0-05 — Decomposition precision audit

**Current status:** the source-grounded automated gate is complete. The prior 319-case blinded packet
remains hash-pinned as an optional external audit, not a training blocker. See
`docs/M0_05_DECOMPOSITION_PRECISION_AUDIT.md`.

**Do.** Retro-decomposition can produce chemically nonsensical disconnections, and exact graph
round-trip alone proves only transform consistency. Build an automated evidence adjudicator that:

1. hashes every primary article or supplementary asset and records exact page, scheme, compound and
   procedure locators;
2. requires visual inspection of source chemistry pages, including reaction schemes and any
   cross-referenced general procedure;
3. separates exact source execution, exact series members, experimental failures, analogue or family
   precedent, deterministic transform consistency, source conflict and missing evidence;
4. canonicalizes exact structures, preserves source attachment atoms, enumerates qualified reactive
   sites, collapses symmetry-equivalent matches and requires exact forward reconstruction;
5. emits `admit_exact`, `admit_transform_consistency`, `admit_negative`, `precedent_only`, `abstain`
   or `reject_claim` with structured reason codes;
6. never turns missing evidence into a negative result or promotes a computed route into experimental
   supervision.

The AGILE measured set may enter exact Ugi L1 supervision only when the source-reported library,
component identities, reactive-site policy and frozen forward transform all agree. The 12,276 AGILE
virtual products may enter a separately labeled transform-consistency objective only. They are not
experimental successes. An L2 route enters exact supervision only when the supplement reports the
exact structure or resolved series member and the procedure chain is available.

The existing 319-case packet remains useful for optional external review and method comparison. Its
pending human annotations do not block source-adjudicated Ugi supervision.

**Accept when.** `make m0-05-adjudicate` produces a hash-pinned evidence ledger and result artifact;
all exact source assets and chemistry-page locators are verified; computed versus experimental labels
remain separate; source-identity conflicts abstain; cross-platform proposals are not promoted to
observed L2 routes; missing evidence is not labeled negative; and every admitted record has a stable
evidence basis and structured reason codes. Non-Ugi R0 decompositions remain out of joint L1 training
unless separately source-qualified under the same policy.

---

## M0-06 — Product-prior feasibility gate (bounded probe, NOT training)

**Do.** Small-scale DeFoG-style discrete flow probes at **`N_max = 64`** and
**`N_max = 96`**. The original task estimate assigned 86.5% and 96.8% of R0
to these thresholds. M0-06 must verify those values from the current
hash-pinned input rather than assume them. This is a feasibility probe:
minutes-to-hours on one GPU or CPU, not a training run. Measure:

endpoint validity · connectedness · atom-count distribution fidelity · **bond sparsity calibration** ·
graph reconstruction · memory · throughput · topology novelty · **performance stratified by molecule size**

**Why this matters.** Dense N×N edge flow means 4,096 ordered pairs at N=64
and 9,216 at N=96. The current hash-pinned input has a maximum of 282, not the
previously documented 138. Most entries are absent bonds, so the objective can
be dominated by edge-null prediction.

**Accept when.** All metrics reported at both sizes. **If dense edge flow degrades at 96 atoms, say so
and propose a sparse/hierarchical edge parameterization. Do not cap `N_max` and continue silently** —
the declared size support must be stated in the paper and matched by the wet-lab panel.

**Current status.** Complete. The dense independent-edge probe failed and the
sparse tree-plus-residual-closure replacement passed at N=64 and N=96, with
scaling dry runs through the observed maximum of 282 atoms. The extended
C/N/O/S/P/F/Si support covers all 15,229 corrected R0 structures. Exact sparse
and sanitized constitutional round-trip is 15,229/15,229, including
1,891/1,891 aromatic structures after deterministic Kekulization and
aromaticity restoration. The accepted representation preserves all observed
cycle ranks and does not impose a representation-level head-only ring rule.
The primary Ugi campaign has a separate evidence-bounded topology gate. Among
the 1,100 reconciled AGILE products, all 660 cyclic products contain exactly
one 5- or 6-membered head ring, with no macrocycle, fused, spiro, or bridged
system. Automatic candidate-lock support is limited accordingly; other ring
topologies require explicit evidence, complete route closure, and a support
amendment. These are bounded representation and support results, not a trained
product prior. The corrected R0 and its regenerated split bundle supersede the
quarantined historical inputs for any later authorized product-prior work.

---

## M0-07 — AGILE oracle matrix

**Blocking source gate.** Before any fit, reconcile the 1,200 nominal AGILE
measurements against the official article source workbook and the hash-pinned
LANTERN audit. Exclude the 100 B4 cis/trans mixture measurements from
single-graph supervision, preserve them in an exclusion ledger, and correct
the 100 B5 records to the pure-trans molecular identity. All M0-07 model fits
must consume the resulting 1,100-structure artifact. Direct fitting from
`AGILE_smiles_with_value_group.csv` is prohibited.

**Blocking split gate.** Freeze all assignments before fitting. Preserve
LANTERN's exact random split only as a reproduction diagnostic. Its committed
`Murcko_scaffold.npy` artifact leaks all six Murcko scaffold groups and is
prohibited for model selection. Its separate scaffold-balanced artifact is
scaffold-disjoint and is the valid structural holdout. Model selection must
also use deterministic five-fold held-head, held-aldehyde, held-isocyanide,
and all three component-pair evaluations. The 12,276 virtual structures are
unlabeled and may be used only for applicability and distance analysis.

**Do.** Train and calibrate the predeclared representation × model matrix on the reconciled
AGILE single-structure data (`expt_Hela`, `expt_Raw`):

| Representation | Models |
|---|---|
| Morgan count fingerprints | ridge, RF, XGBoost, MLP |
| Physicochemical + lipid expert descriptors | ridge, RF, XGBoost, MLP |
| Molecular graph | D-MPNN, GIN/GAT |
| Region-aware head/linker/tail graph | D-MPNN or graph transformer |
| Lipid-pretrained encoder | linear head, MLP, graph transformer |

Evaluate under scaffold holdout, component/head/tail holdout, reaction-family holdout where applicable,
and prospective candidate-library holdout. Report calibration, not only accuracy.

Use only structure-computable descriptors unless measured values have an audited, row-level source.
Do not assume or impute apparent pKa, particle size, polydispersity, encapsulation efficiency, or
formulation robustness for the full corpus. Evaluate applicability using available molecular
representations and component or structural distances. Include component-family and component-pair
holdouts that approximate the transferred-tail shift, plus a prespecified oracle-guidance-strength
analysis. Any conservative uncertainty score must earn its interpretation through calibration.

**Accept when.** Full matrix reported with calibration; **one model frozen** and recorded in the
decision log. The frozen model must be selected before any guided generation exists.
Architecture selection must not read outer-test metrics. Require a fold-local
calibration-selected nested test audit, exact candidate and fold completeness,
and automatic candidate-domain classification from verified Ugi components.
The production refit's held-domain radius is empirical and conservative; do
not claim a new finite-sample conformal guarantee for the full-data ensemble.

**Current status.** M0-07 is complete under an explicit abstention outcome.
The matrix contains 768 classical fits, 576 supervised-graph fits, and 384
label-free transfer fits, with all predeclared candidates, folds, endpoints,
and repeat seeds present. Architecture selection used calibration evidence
only. It froze
`supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble`,
with equal-endpoint calibration R² = 0.3673, RMSE = 1.8584, and Spearman
correlation = 0.5960. Its post-selection equal-endpoint test R² was 0.1830.

No declared biological-guidance domain passed both the fixed-candidate and
fold-local nested gates, principally because the required scaffold-balanced
audit failed for both endpoints. FORGE must therefore abstain from AGILE
oracle tilting in every declared component-novelty domain until additional
evidence reopens a domain. The negative deploy decision is frozen rather than
weakening a gate.

A deterministic three-seed, two-endpoint production refit was completed on
all 1,100 reconciled single-structure records. The checkpoint is authenticated
by `results/m0_07/oracle_production_result.json` and has SHA-256
`46f13b8a3cfef1f891e308dfda75e90685d82d1289f0061acdae4be18870e61c`.
Inference verifies the pinned AGILE Ugi transform, infers component novelty
from the checkpoint-owned 1,100-record applicability index, and applies the
frozen abstention policy automatically. The checkpoint does not use guided
generation labels, virtual candidate labels, stereochemistry, graph
truncation, or imputed formulation measurements.

The source-faithful LANTERN released checkpoint remains a reproduction
diagnostic only because its published preprocessing uses full-data feature and
label scaling. The auxiliary JC_2023 and LM_2019 data remain separately
normalized transfer evidence; the completed label-free R0 transfer lane did
not improve held-component evidence enough to supersede the supervised graph
model. Under the frozen calibration-only selection policy, its fine-tuned and
frozen-head candidates ranked 12th and 14th, with equal-endpoint test R² of
-0.0437 and -0.0788. The lane is frozen as a nonselected diagnostic
pretraining ablation. It is not an authorized guidance oracle or production
checkpoint, and promotion requires a new hash-pinned cross-representation
freeze.

**State explicitly in the result:** AGILE is a **predictive general-transfection oracle, not an in-vivo
endpoint oracle.** HeLa/RAW potency does not predict vaccine immunogenicity or liver editing.
Formulation properties unavailable at corpus scale remain prospective measurements, not model inputs
invented by imputation.

---

## M0-08 — Endpoint recommendation package

**Do.** Assemble the decision package for the PI: for liver functional editing vs IM vaccination, list
required bridge assay, required in-vivo readouts, existing lab capability, and burden. Default
recommendation is **liver functional editing**.

Implement `bio/endpoint.py` as a generic interface with `liver.py` and `vaccine.py` stubs. **Do not
hard-code an endpoint.**

**Accept when.** Package written; interface exists; no endpoint baked into any other module.
**Blocks Phase 4** — flag clearly that this needs a human decision.

**Current status.** The endpoint-generic interface and three endpoint evidence
stubs are implemented. The endpoint remains explicitly unlocked while M0
computation continues. IM reporter delivery followed by functional editing is
the current lowest-risk option because AGILE reports HeLa-to-IM translation,
while no AGILE-to-IV-liver correlation is established. Liver requires a
liver-specific data or bridge gate. IM vaccination remains viable and requires
at least one antigen-specific response, but challenge is not automatically
required. The exact target or antigen, bridge, capacity, benchmark, and
formulation remain a human lock before Phase 4.

---

## M0-09 — L2 supervision inventory (inventory only, no model)

**Do.** Quantify — do not assume — the available training data for subcomponent synthesis:

| Stratum | Quantify |
|---|---|
| USPTO pretraining | reactions, classes, coverage of lipid-relevant chemistry |
| Patent / SI lipid routes | distinct upstream reactions, unique products, unique precursor scaffolds, route depths |
| AGILE tail routes (SI Note 1) | Tail A (acid + diol → EDC → DMP); Tail B (amine → formylation → POCl₃); butyl variant |
| Internal RM protocols | `combinatorial_papers/RM_0*.docx`, 7 custom acrylate/propiolate tails, measured conditions |
| **Negative outcomes** | failed syntheses — highest-value, scarcest stratum |

Report coverage across ester, isocyanide, aldehyde, carbonate, acrylate, heterocycle formation. For
the current paper, separately report the transformation basis that can produce or diversify the three
AGILE-type Ugi 3-CR inputs: amine heads, aldehyde-bearing tails, and isocyanide tails. Chemistry from
another final-assembly family counts toward that basis only when it feeds one of those inputs or its
procurement leaves.

Keep source-extracted supervision separate from a bounded reachable-space census. The census may
enumerate compatible applications of frozen reaction families over versioned precursor catalogs, but
it must deduplicate products, report filter attrition and input hashes, and remain labeled as proposed
space rather than route evidence. Do not claim every chemically imaginable tail is covered.

For the 12,276-structure AGILE virtual set, recover only exact, forward-reconstructing Ugi
decompositions, preserve reacting-site identity, deduplicate the component universe, and join those
components to the L2/L3 evidence queue. Do not count the virtual products as observed routes or route
outcomes. At the unique-component level, reproduce every exact source route structurally before
projecting its reaction-family program to new components. Keep exact programs, family projections,
route closure, and leaf procurement as separate states.

Build a separate, versioned cross-platform hydrophobic-motif transfer inventory from non-Ugi
libraries. Preserve source attachment atoms and distinguish the transferable motif from the
platform-specific handle and core. A proposed Ugi realization must name a mapped common precursor,
an aldehyde and/or isocyanide conversion program, evidence grade, route depth, closure state, and
terminal status. Do not inherit the source lipid's biological label, treat motif occurrence as a
route, or admit ambiguous subgraph clipping as a Ugi component.

Each transfer must also link to the actual reaction used to prepare the source
component. Record reactants, stoichiometry, conditions, workup, purification,
yield, analytical evidence, and the article or supplement locator. Keep this
source reaction separate from the proposed reaction that installs the Ugi
handle. A deterministic graph transform does not count as reported reaction
supervision.

Maintain independent generative-support, oracle-applicability, and route-support
states for every transfer. Classify each route outcome as complete, chemically
incompatible, outside declared support, or missing route knowledge. Only the
last category motivates additional chemistry mining by default.

Build a source-paper priority queue before broad review. Rank expected marginal
value from distinctive tail and embedded-linker motifs, biological provenance,
novelty relative to AGILE, source-method accessibility, plausible Ugi-handle
conversion, and redundancy. LNPDB defines the motif census; source papers,
supplements, general chemistry literature, patents, and supplier documents may
supply the actual route evidence.

Use exact site-defined amines for cross-platform head transfer. Do not perform
head-motif conversion in the tail registry. Novel generated heads remain
permitted but require independent site qualification and L2/L3 closure. Treat
linker-like motifs as chemistry embedded in an aldehyde or isocyanide
component, not as an independent Ugi-3 reactant.

**Current head-transfer status.** The frozen LNPDB census contains 408 head
entries, including 396 parsed structures. Twenty are exact AGILE heads. A
further 259 non-AGILE structures pass the bounded topology and qualified
Ugi-amine screen: 130 are acyclic, 22 contain one 5-membered ring, and 107
contain one 6-membered ring. Thirty-six require explicit choice between two
symmetry-distinct amine sites. None of the 259 has assessed route or
procurement closure in the current component ledger, and none is automatically
admitted. The queue is route prioritization and evaluation evidence, not a
head vocabulary. Novel generated heads remain permitted under the same
site-qualification, forward-reconstruction, and L2/L3 closure policy.

**Do not build the L2 model.** This task decides whether a learned L2 model is viable at all.

**Critical.** R1 **cannot** supply this: every R1 block is terminal by construction, so R1 contains zero
examples of *making* a block, and its `reactant_ids` labels were produced by the enumerator (circular).

**Accept when.** Inventory table and Ugi-3 transformation-capability matrix are complete with counts,
gaps, and evidence states. The AGILE virtual component census is exact, reacting-site aware, and
reduced to a deduplicated recursive route queue with explicit intermediate and leaf candidates. A
reproducible bounded-space census protocol is specified separately from the observed route labels.
The proposed leaves and unresolved heads are deduplicated into an exact-identity procurement queue,
with historical vendor claims kept separate from current accepted availability. The cross-platform
motif-transfer inventory reports source platforms, exact attachment mappings, accepted and rejected
motifs, proposed Ugi realizations, evidence states, and rejection reasons separately from observed L2
supervision. A documented operational stopping rule replaces exhaustive
reaction mining: thresholds for weighted motif coverage, candidate-route
closure, marginal saturation, and prospective-panel dossier closure are frozen
before generator-driven curation and final candidate selection.
**If the corpus is thin, say so**; the fallback
is hybrid neural-proposal + deterministic template search, and that is an acceptable outcome. This is
the task most likely to force a scope change; run it early.

---

## M0-10 — FlowER transfer pilot

**Do.** Reaction-class holdout plus a ~32-mechanism/class fine-tuning curve on 2–3 lipid-relevant
classes, with calibration. The vendored registry's mechanism-verified atom-mapped positive **and
negative** examples make curation far cheaper than starting cold.

**Context.** A prior zero-shot probe (`electron_flow_lipids/memos/11_flower_probe_results.md`, May 2026)
returned 1/6 strict match with aza-Michael failing and Ugi uncovered. That is a **prior**, not this
experiment. Expect to need fine-tuning; budget for it.

**Frozen demotion rule — apply mechanically, do not adjust after seeing results.** FlowER stays in the
headline mechanism only if fine-tuning improves held-out forward consistency **and** its score
correlates with prospective conversion (Spearman ρ, CI excluding 0). Otherwise it is reported as an
orthogonal consistency check. **The deterministic atom-mapped verifier is load-bearing either way.**

**Accept when.** Holdout + curve + calibration reported; demotion rule applied and the outcome recorded
in the decision log.

---

## Return condition

When M0-02..M0-10 are complete, produce `docs/M0_REPORT.md` consolidating every decision, measurement,
and negative result, and **stop**. Broad implementation is not authorized until that report is reviewed.

**Closeout status.** `docs/M0_REPORT.md` now provides the consolidated
closeout. After reviewing the corpus, architecture, oracle, and training
mixture, the user authorized Phase 1 broad product plus joint Ugi-L1 training
on 2026-07-30. The authorization boundary is recorded in `AGENTS.md` and
`docs/DECISION_LOG.md`.
