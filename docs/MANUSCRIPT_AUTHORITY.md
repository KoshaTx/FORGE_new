# FORGE manuscript authority

**Status: authoritative.** This document supersedes `MANUSCRIPT_REFRAME_PLAN.md` for
framing, claim hierarchy, section structure, figure architecture and panel design.
`MANUSCRIPT_PROSE_ANALYSIS.md` remains in force for sentence-level style and is
consistent with §16 here. Where any earlier plan conflicts with this document,
this document wins.

Received from the user 2026-08-06 and reproduced faithfully. Editorial notes added
by me are marked `[impl]`.

---

# The central judgment

This is not an ML paper that happens to use ionizable lipids. It should be written
as a **new ionizable-lipid discovery paradigm**, enabled by unusually rigorous
generative modeling.

The paper's central claim is:

> **FORGE carries ionizable-lipid discovery beyond pre-enumerated libraries by
> generating complete molecular structures, connecting their constituent components
> to experimentally actionable chemistry and prospectively validating the resulting
> LNPs in vivo.**

The priority claim should be:

> **To our knowledge, this is the first study to prospectively synthesize, formulate
> and validate in vivo ionizable lipids designed de novo by a molecular generative
> model.**

That is materially stronger and more defensible than "the first AI-designed LNP,"
which would be false given the substantial AI-guided screening literature. It also
distinguishes FORGE from the existing in silico generative studies without
pretending those studies do not exist.

The paper should leave an mRNA-LNP reader with one idea:

> **The field has become increasingly good at choosing among lipids that have
> already been imagined. FORGE shows that a model can instead propose new
> ionizable-lipid chemistry, tell a chemist how to reach it and carry those designs
> through to functional mRNA delivery.**

That is the Nature Biotechnology paper. Discrete flow matching, applicability-aware
sampling, activity prediction and route search are the technical machinery that
makes the claim credible.

---

## 1. What the paper is and what it is not

### What it is

Five nested contributions, in this order:

1. **A prospective biotechnology result.** De novo generated ionizable lipids are
   synthesized, formulated as mRNA LNPs and tested in vivo.
2. **A genuinely generative discovery strategy.** Candidate structures are not
   selected from a finite library enumerated before inference. FORGE generates
   complete molecular graphs and can introduce component chemistry absent from the
   source library.
3. **A synthesis-aware output.** Not just a lipid graph or a generic
   synthetic-accessibility score, but a lipid associated with an exact final
   assembly, identified component structures, upstream preparations and purchasable
   starting materials.
4. **A disciplined response to predictive extrapolation.** FORGE does not assume an
   activity model trained on ~1,100 lipids is informative everywhere the generator
   can go. It increases sampling where ranking is supported by measured chemistry,
   while keeping potency out of the generative proposal.
5. **A reusable framework instantiated in Ugi-3 chemistry.** The general object is a
   product generator connected to a reaction adapter, predictive-support model and
   recursive route layer. Ugi-3 is the first experimentally tractable instantiation,
   not the definition of FORGE.

### What it is not

* a benchmark paper about discrete flow matching;
* a paper about retrospective synthesis prediction;
* a virtual-screening paper with a larger library;
* an autonomous laboratory paper;
* a claim that the HeLa activity predictor predicts in vivo delivery;
* a claim that every computationally closed route is experimentally successful;
* a general-purpose retrosynthesis system.

Those distinctions should be visible in the prose, figures and claim hierarchy.

---

## 2. Field positioning

Recent high-impact AI-LNP studies have already demonstrated that machine learning
can prioritize promising ionizable lipids from large candidate spaces. Li and
colleagues trained from a 584-lipid library, screened 40,000 virtual lipids and
evaluated 16 prioritized candidates. Witten and colleagues trained on >9,000
activity measurements, scored 1.6 million candidates and validated leads in mice and
ferrets. AGILE couples prediction with a reaction-defined Ugi library; MOLEA
emphasizes potency, cellular selectivity and functional genome editing.

The distinction:

> **Prior AI-guided ionizable-lipid studies have primarily learned to prioritize
> molecules from experimentally constructed or computationally enumerated candidate
> spaces. FORGE instead generates complete molecular structures without selecting
> their component identities from a fixed catalogue.**

In silico generative ionizable-lipid papers already exist (e.g. Ou et al., a
synthesis-DAG model generating lipids and predicted synthesis paths from building
blocks). The claim therefore cannot be "the first generative ionizable-lipid model."

Exact novelty language:

> **Generative models for ionizable lipids have been described computationally, but
> their designs have not been carried prospectively through synthesis, LNP
> formulation and in vivo mRNA delivery.**

Then:

> **Here, we provide, to our knowledge, the first prospective in vivo validation of
> de novo generative ionizable-lipid design.**

Do not repeatedly announce that the flow architecture is newer than other models.
Prove it with matched benchmarks. In the narrative, the important distinction is
that **FORGE reaches new chemistry while preserving the ability to test it**.

Venue calibration: the strongest recent Nature Biotechnology LNP papers lead with
the unmet delivery problem and end with experimentally meaningful biological
function. Witten culminates in mouse and ferret lung delivery; MOLEA in chondrocyte
transfection, editing and disease modification. Not in model architecture.

> **Biological function is the climax; the model is the reason the result was
> possible.**

---

## 3. What is already strong, and what must change

### Strong

The draft correctly separates reaction reconstruction from component synthesis; a
route plan from bench success; commercial records from actual procurement; predictor
support from predicted potency; generated structures from observed molecules;
prospective outcomes from computational evidence. The activity model is described as
a selective ranking model rather than an unrestricted oracle. The support-aware
proposal has a clean result (3.87% to 8.64%, 2.23-fold) without using potency during
generation. A negative result is retained: generation-time synthesis guidance did not
improve route-ready yield.

### Must change

**The abstract currently crosses the evidence boundary.** "The resulting lipids can
be made" is stronger than the evidence allows before prospective synthesis.

* Before the campaign: "The resulting designs are connected to computationally
  complete routes terminating in purchasable starting materials."
* After: "[X of 10] route-resolved designs yielded the intended lipid at the
  prespecified identity and purity criteria."

Only observed synthesis permits "could be synthesized" or "were synthesized."

**"Open-ended" must be qualified.** The model has declared atom, bond, size,
branching and cycle support. Use "de novo generation within declared molecular
support", "not restricted to a finite component catalogue", "non-enumerative
generation", "generation beyond the source component library". Avoid "unbounded
chemical space", "arbitrary ionizable lipids", unexplained "open-ended chemistry".

**The route-guidance diagnostic is too prominent.** Move to Extended Data; give it
one main-text sentence:

> "We also evaluated whether route information should alter intermediate generative
> transitions, but the prespecified matched-budget diagnostic did not improve
> route-ready yield; production therefore retained complete-product routing after
> generation (Extended Data Fig. X)."

**The 20-candidate plan must be rebuilt around 10 designs.** With ten synthesis
slots, do not spend half the experiment on an underpowered causal test between
generation arms. Validate that computationally on fresh samples. Use the prospective
experiment to establish that de novo designs can be routed, made, formulated and
function in vivo.

---

## 4. Format

Nature Biotechnology Article: up to 3,000 main-text words excluding abstract,
Methods, references and legends; unreferenced abstract up to 150 words; no more than
six main display items; unheaded introduction, then Results, Discussion, Online
Methods; topical subheadings in Results and Methods, none in Discussion.

Target five main figures and:

| Part | Target |
| --- | ---: |
| Abstract | 145-150 |
| Introduction | 450-525 |
| Results 1 | 300-350 |
| Results 2 | 350-425 |
| Results 3 | 300-350 |
| Results 4 | 350-425 |
| Results 5 | 350-425 |
| Results 6 | 275-350 |
| Discussion | 425-500 |
| **Main-text total** | **~2,900-3,000** |

---

## 5. Title

Keep: **Synthesis-aware generative design of ionizable lipids for mRNA delivery**

Do not use "synthesis-guided" unless route information demonstrably improves the
generation trajectory; production performs synthesis assessment after complete
molecular generation, so "synthesis-aware" is precise. Do not add "using discrete
flow matching".

---

## 6. Paper-level framing

**One-sentence editorial pitch**

> FORGE is a synthesis-aware generative framework that designs ionizable lipids
> beyond pre-enumerated component libraries, identifies where biological predictions
> remain credible, connects generated components to purchasable starting materials
> and prospectively carries selected designs through synthesis, formulation and in
> vivo mRNA delivery.

**Two-sentence cover-letter pitch**

> Machine-learning-guided ionizable-lipid discovery has so far largely prioritized
> candidates from libraries defined before model inference. FORGE instead generates
> complete lipid structures de novo, resolves their component chemistry to
> purchasable materials and, to our knowledge, provides the first prospective in vivo
> validation of ionizable lipids produced by a molecular generative model.

**The conceptual advance** (not "FORGE generates synthesizable molecules", which is
crowded and weak):

> **FORGE changes the unit of molecular design from an isolated lipid structure to a
> linked experimental object comprising the complete lipid, its reaction-specific
> components, the chemistry required to reach those components and the evidence
> supporting each step.**

---

## 7. Working abstract (149 words, accurate pre-data)

> **Ionizable-lipid discovery has been accelerated by combinatorial synthesis and
> machine-learning-guided screening, but these approaches select from libraries
> defined before a model is applied. De novo generation can expand this space, yet it
> also produces structures beyond available chemistry and the evidence supporting
> activity prediction. Here we introduce FORGE, a synthesis-aware generative
> framework that produces complete ionizable-lipid structures, decomposes each design
> into reaction-specific precursors and connects noncommercial components to
> purchasable starting materials. Instantiated in Ugi three-component chemistry,
> FORGE uses discrete flow matching and support-aware sampling to increase the yield
> of designs within the calibrated domain of an mRNA-transfection predictor while
> retaining molecular novelty. Prospectively, [X of 10] designs were synthesized, [Y]
> formed mRNA lipid nanoparticles and [Z] mediated [endpoint] in mice, including
> [lead comparison]. To our knowledge, this is the first prospective in vivo
> validation of de novo generative ionizable-lipid design, linking molecular
> generation to experimentally actionable RNA delivery materials.**

**The abstract must not contain:** architecture dimensions; activity-predictor model
names; Graph2Edits or AiZynthFinder; L1/L2/L3 terminology; detailed validity
metrics; the null trajectory-guidance diagnostic; more than one generative benchmark
result; both HeLa and RAW 264.7 unless both materially affect selection; "state of
the art" unless prespecified and decisive.

One computational outcome, one synthesis outcome, one biological outcome.

---

## 8. Introduction: four paragraphs

**P1 — Ionizable-lipid discovery is a consequential molecular-design problem.**
Begin with mRNA delivery, not "AI is transforming science." Open with LNP clinical
delivery, the ionizable lipid's role in complexation, assembly, trafficking,
endosomal escape, biodistribution and tolerability, and that small structural
changes produce large delivery differences. End with: "Because these relationships
remain difficult to predict from structure alone, new ionizable lipids are usually
discovered through modular synthesis and experimental screening." One carefully
chosen structure-activity example is enough.

**P2 — Current AI improves selection, not the boundaries of the candidate space.**
Combinatorial reactions enable large libraries; ML prioritizes candidates. These
produced potent LNPs across tissues, but candidates are defined by components or
virtual structures enumerated before inference. The hinge sentence:

> A predictor can improve how that space is searched; it cannot propose a lipid that
> is absent from it.

Then: "Molecular generation offers a different capability: constructing candidate
structures rather than choosing among supplied ones." Additive, not dismissive.

**P3 — De novo generation creates two new failure modes.** First, a generated lipid
may contain components that cannot be purchased or prepared; restricting generation
to catalogued components solves this only by recreating a finite virtual library.
Second, a generator may move beyond the chemistry represented in the activity
model's training data, so a high predicted score can reflect extrapolation rather
than evidence. Then:

> A useful generative system for ionizable-lipid discovery must therefore expand
> molecular space without severing the links to synthesis or to the experimental
> evidence supporting candidate ranking.

**P4 — Introduce FORGE, the Ugi instantiation and the result.** Complete lipid graph
plus reaction-specific components plus recursive connection to purchasable starting
materials; instantiated in Ugi-3 where exact product-to-component mapping allows
forward-assembly verification; discrete flow model generates new component
chemistry; support-aware sampling increases designs with credible local evidence;
ten lipids prospectively synthesized, formulated, evaluated. End on the priority
claim. No figure-by-figure paragraph, no claim to transform all molecular discovery.

---

## 9. Results organization

Headings state biological or experimental conclusions, not software modules.

### R1 | FORGE generates lipids together with the chemistry needed to make them

Open: "To move beyond finite component libraries without losing experimental
tractability, we built FORGE to generate a complete ionizable lipid together with
the reaction-specific components and upstream chemistry needed to prepare it."

Then the general abstraction before the Ugi instance (product generator, reaction
adapter, predictive-support layer, recursive route layer); then Ugi-3 exactness and
that failed round trips are rejected rather than repaired; then one accessible
paragraph on the discrete flow model (progressively resolves a categorical state
into a complete connected graph; jointly generates atoms, bonds and precursor-origin
regions from coarse structural information; does not select stored heads or tails);
then the route dossier. Flow objective goes to Methods and Supplementary Note 1.
Avoid "executable synthesis" until experimentally validated.

### R2 | De novo generation expands Ugi-3 lipid chemistry

Must answer: is it really generative rather than retrieving components; are
structures chemically and reaction-valid; does it expand beyond the finite library;
does it beat matched generative baselines.

Start with matched evaluation under identical training data, molecular support,
sample counts and post-generation checks. Do not compare against metrics copied from
unrelated datasets.

Central metric, defined before evaluation: **usable de novo design yield** = fraction
of all attempted samples that are chemically valid, reconstruct exactly through the
Ugi reaction, are unique, and contain at least one precursor identity absent from the
training component set.

Report separately: validity; exact round-trip; uniqueness; whole-product novelty;
exact-new component rate; component diversity by role; distribution fidelity;
sampling cost.

Report three novelty levels, never novelty by whole-product identity alone:
new product with familiar components; new product with at least one exact-new
component; new product with a component outside both training set and expanded
reference catalogue.

### R3 | Support-aware sampling doubles the yield of prediction-supported designs

Begin with the scientific problem, not controllers:

> Generating beyond the source library also increases the risk that candidate ranking
> is driven by extrapolation. We therefore asked whether FORGE could generate more
> molecules within the experimentally supported domain of the activity predictor
> without restricting generation to known components or using predicted potency
> during sampling.

Support in one sentence (complete lipid and each of the three components
sufficiently close to measured chemistry under substructure and physicochemical
criteria). Then the intervention (lightweight proposal model over coarse
architectures; every architecture retains nonzero probability; no potency label
enters). Then 3.87% to 8.64%, 2.23-fold. Then the indispensable negative control:
whole-product novelty, new-component rate, nearest-neighbour similarity and
effective component diversity in both arms. Then what it does not mean.

### R4 | Recursive routing connects generated components to purchasable materials

Open: "Because a lipid is experimentally accessible only if each of its components is
accessible, we evaluated synthesis at the component level rather than assigning a
single synthetic-accessibility score to the complete product."

Hierarchy in chemical terms before abbreviations: final assembly (exact Ugi
reconstruction); component preparation (documented or planner-proposed upstream
transformations); starting-material closure (dated procurement record). Then the
cascade, then immediately: "This procedure establishes computational route
completeness, not experimental synthesis success." Keep the denominator and evidence
class explicit. Needs retrospective (blinded recovery) and prospective validation.
The null generation-time guidance result appears only at the end, in one sentence.

### R5 | Prospective synthesis and formulation of FORGE designs

(Heading becomes "FORGE designs can be synthesized and formulated as mRNA LNPs" only
once data support it.)

Begin with the prospective denominator: all ten locked routes attempted, no candidate
replaced after failure. Full disposition per candidate: intended route; starting
materials; component synthesis outcome; final assembly; correct-product identity;
isolated yield; purity; deviations; failure attribution.

Formulation: one composition and mixing procedure; **ionizable-lipid input normalized
to measured active-lipid content rather than nominal mass** (prevents purity becoming
a silent dose variable). Report formulation success, size, PDI, encapsulation, pKa,
batch reproducibility, prespecified failure criteria. Acceptable DLS does not prove
biological quality.

In vitro distinguishes two questions: does the activity model rank the
training-related endpoint (HeLa as a test of ranking); and does the lipid function in
an assay relevant to the eventual in vivo endpoint (separate biological bridge). Do
not imply HeLa predicts the animal endpoint.

### R6 | De novo FORGE lipids mediate mRNA delivery in vivo

Open directly. Report candidates advanced, prespecified advancement rule, route,
payload, dose, benchmark, vehicle, individual-animal outcomes, target and off-target
tissues, tolerability. Substantially stronger with: two independent active generated
lipids; an independent replication cohort; a dose response for the lead; equal-dose
benchmark comparison; an orthogonal functional or editing endpoint.

Final Results sentence connects the whole chain:

> Thus, a molecular graph generated beyond the source component library was connected
> prospectively to purchasable starting materials, isolated as the intended lipid,
> formulated as an mRNA LNP and shown to mediate functional delivery in vivo.

---

## 10. Main figure architecture

Same candidate order and IDs across Figures 3-5. One visual encoding for evidence
class, one for candidate-selection class. Not ten unrelated bright colours.

**Figure 1 | FORGE links de novo lipid generation to the chemistry required for
experimental testing.** Panels: (a) library selection versus de novo design, two-lane
schematic; (b) general architecture, four connected objects with generic roles
R1/R2/R3; (c) Ugi-3 instantiation with atoms assigned to component origins and the
exact forward round trip; (d) discrete flow generation as four molecular states, with
"No stored component identity is supplied"; (e) complete route dossier; (f)
prospective validation chain. Wide, sparse, left-to-right. No performance numbers.

**Figure 2 | De novo generation expands Ugi-3 chemistry while retaining predictive
support.** (a) matched generative benchmark, four metrics only; (b) attrition funnel
with absolute counts; (c) quantitative chemical-space map with density contours plus
an adjacent nearest-neighbour distribution; (d) component-space expansion by role;
(e) what predictive support means, four views; (f) support-aware sampling schematic;
(g) supported-design yield with intervals; (h) novelty and diversity preserved.

**Figure 3 | Recursive routing prioritizes de novo lipids for prospective synthesis.**
(a) route evidence hierarchy with plain-language labels; (b) hybrid cascade, marking
that a planner score is not evidence; (c) blinded route-recovery benchmark including
false route-complete calls; (d) population-level route readiness; (e) prospective
panel selection, no weighted composite unless prespecified; (f) candidate-by-stage
synthesis matrix, every failure visible; (g) individual yields and purity; (h)
computational route class versus observed outcome.

**Figure 4 | FORGE-designed lipids form mRNA LNPs and deliver cargo in vitro.**
(a) structures in prespecified order, not reordered after outcomes; (b) fixed
formulation workflow with purity normalization; (c) particle characterization with
acceptance bands, not pass/fail only; (d) training-related activity assay including
low-ranked controls; (e) endpoint-relevant cellular bridge with viability; (f)
frozen advancement rule.

**Figure 5 | De novo FORGE lipids mediate functional mRNA delivery in vivo.**
(a) study design; (b) primary screen with individual animals, no bars without points;
(c) tissue distribution and off-target expression; (d) lead dose response; (e)
functional payload or orthogonal validation; (f) tolerability. Inset may show the
lead's entire chain.

---

## 11. Extended Data

1. Data provenance, molecular reconciliation and leakage-resistant splits.
2. Molecular representation and discrete flow formulation.
3. Full matched generative benchmark, all metrics across seeds.
4. Memorization, chemical-space coverage and structural realism.
5. Activity-model benchmarking, calibration and ranking performance. **Strong simple
   baselines are mandatory**: a 2026 benchmarking study found Morgan-substructure and
   descriptor baselines can outperform sophisticated graph models on curated AGILE
   data. Do not select by random-split R2.
6. Predictive-support definition and support-aware proposal, with threshold
   sensitivity.
7. Potency-guidance diagnostic (did not pass its terminal-efficiency gate).
8. Route planner validation and evidence adjudication.
9. Null generation-time synthesis-guidance experiment, full matched-budget design.
10. Full formulation, in vitro and in vivo reproducibility.

---

## 12. Supplementary Information

**Online Methods must carry:** dataset reconciliation; exact reaction mapping; model
state space and training objective; split policy; baseline implementation;
activity-model selection; predictive-support definition; route acceptance rules;
candidate selection; synthesis; formulation; animal design; statistics.

**Supplementary Notes:** 1 discrete flow-matching formulation; 2 reaction-adapter
abstraction; 3 support-aware sampling; 4 activity prediction and uncertainty; 5
recursive route construction; 6 candidate lock and prospective analysis plan.

**Supplementary Tables:** data-source ledger; split memberships; full generative
benchmark; generated-component provenance; activity-model comparison; support
thresholds and sensitivity; route-evidence records; ten locked candidates; every
starting material and procurement record; every synthesis attempt and deviation;
every formulation batch; all cellular measurements; all individual-animal
measurements and exclusions.

**Chemistry files:** LC-MS; NMR; HPLC/UPLC traces; purity calculations; isolated-yield
definitions; representative failure chromatograms; route deviations.

---

## 13. The ten-lipid prospective panel

Discovery-first, not an underpowered 5-versus-5 between sampling arms.

**Hard gates (all required):** chemically valid and connected; exact Ugi
reconstruction; qualified amine, aldehyde and isocyanide handles; within declared
molecular support; complete-product and component-level predictive support;
computationally complete route under the frozen evidence rule; no unresolved
material leaf; prespecified formulation-relevant physicochemical limits; unique at
full-product level.

**Ranking:** lower confidence bound of the activity predictor, not the point estimate.

**Diversity:** prespecified max-min or determinantal rule across complete-product
fingerprints and component identities. No manual post-hoc selection.

**Allocation:**

* **6 primary** — highest supported activity lower bounds subject to diversity.
* **2 de novo frontier** — within the calibrated domain but chosen for meaningful
  component novelty (e.g. a component absent from the expanded catalogue).
* **2 matched low-ranked controls** — route-complete, matched on size, novelty and
  route burden, deliberately from the low end of the supported ranking.

**Benchmarks must not consume the ten slots.** Include vehicle; one clinically or
experimentally relevant benchmark; one source-library or AGILE reference lipid where
available; a formulation-process control.

---

## 14. Prospective experimental sequence

**Candidate lock** before ordering: freeze structures, IDs, routes and fallback rules,
procurement snapshot, selection rationale, formulation composition, in vitro
advancement criteria, in vivo analysis; record software, data and model hashes.
Chemists may know structures and routes but ideally not predicted activity rank.

**Synthesis:** attempt every locked candidate, replace none. Prespecify identity
requirements, purity threshold, isolated-yield definition, salt and solvate handling,
allowed deviations, failure attribution.

**Formulation:** one primary composition and mixing procedure. A single prespecified
rescue policy may be permitted but must apply identically to every lipid. Normalize
ionizable-lipid input by measured purity.

**In vitro:** ranking assay (predictor endpoint) and biological bridge (cell type
relevant to the animal endpoint). Multiple doses, viability, benchmark, independent
batches, individual values, explicit replicate definitions.

**In vivo:** test all formulations meeting frozen criteria; down-select only by the
frozen rule without inspecting animal data. First cohort: one matched dose, all
advanced candidates, benchmark, vehicle, primary endpoint, tissues, tolerability.
Lead then receives an independent confirmation cohort with at least two doses,
matched benchmark, orthogonal readout where feasible, repeat tolerability.

---

## 15. Vocabulary: internal term to main-text language

| Internal | Main text |
| --- | --- |
| discrete flow matching | discrete flow model that progressively constructs a complete molecular graph |
| applicability tilt | support-aware sampling |
| oracle | activity predictor |
| morphology program | coarse lipid architecture |
| terminal supported yield | fraction of generated lipids within the predictor's supported domain |
| support event | sufficient neighbouring measured chemistry for candidate ranking |
| L1 closure | exact final lipid assembly |
| L2 route | upstream component preparation |
| L3 closure | connection to purchasable starting materials |
| route closure | computationally complete route |
| proposal-augmented search | hybrid route search |
| broad prior | broad generative sampling |
| complete terminal | completed molecular design |
| selection-visible reproduction | exact reproduction of the generated product |
| component-disjoint split | split by precursor family |
| potency-neutral generation | predicted activity was not used during molecular generation |

Do not use in main Results: applicability morphology controller; terminal support
event; proposal law; potency-neutral terminals; architecture simplex; nested tilt.

---

## 16. Prose rules

* **Lead with the scientific question**, not the implementation.
* **One claim per paragraph**: question, experiment, result, interpretation.
* **Explain the model once**; roughly one paragraph on discrete flow matching.
* **Use "AI" sparingly.** Prefer model, generator, activity predictor, route
  procedure, sampling strategy, discovery framework.
* **Avoid unsupported adjectives**: powerful, transformative, revolutionary,
  unprecedented, groundbreaking; robust unless measured; general unless the
  abstraction and its limits are stated.
* **Keep the caveat next to the claim**, not four pages later.
* **Never confuse four levels**: planner proposed a route; route passed computational
  verification; intended compound detected; intended compound isolated at acceptable
  purity.

`[impl]` These are consistent with and extend `MANUSCRIPT_PROSE_ANALYSIS.md`, which
additionally forbids em-dashes, conditionals, abstract-noun grammatical subjects,
metaphors for technical facts, punchy fragments, and implying a population of
generative models that does not exist.

---

## 17. Discussion: five paragraphs, no subheadings

1. **Main result** with the priority claim.
2. **Conceptual distinction** — not a synthetic-accessibility filter; the output is a
   molecular design linked to an experimental plan. Do not say "a preparation rather
   than a proposal" unless the route worked experimentally; use "a proposal linked to
   a preparation plan."
3. **Predictive support** — a predictor trained on a finite library returns a value
   for nearly any graph, but its meaning decays with distance from measured
   chemistry; support treated as a candidate-level requirement.
4. **Generality and limitations** — one reaction class; one primary predictive
   dataset; bounded molecular support; limited prospective sample size;
   endpoint-specific biology; supplier records as dated snapshots.
5. **Impact** — generative models for delivery materials should be evaluated not only
   by the structures they produce, but by whether those structures can be made,
   formulated and shown to function.

---

## 18. The submission bar

Nature Biotechnology-ready when the campaign provides most of:

* a credible fraction of the ten locked lipids isolated at prespecified purity;
* multiple successful LNP formulations;
* at least two independently active de novo lipids in vivo;
* at least one active lipid containing a component absent from the source library;
* one lead matching or exceeding a meaningful benchmark, or a clearly differentiated
  delivery profile;
* an independent confirmation or dose-response cohort;
* target/off-target information appropriate to the route;
* tolerability;
* complete intention-to-synthesize reporting;
* matched generative baselines showing clear advantage in useful de novo design yield;
* evidence that support-aware sampling improves predictive coverage without collapsing
  novelty.

A single marginal luciferase hit at one dose, without replication, benchmark or clear
component novelty, would make the case fragile. A lead that does not exceed the
benchmark can still support a strong paper if several de novo designs are active and
the synthesis-aware evidence chain is decisive; the central claim then becomes
**feasibility of prospective generative discovery**, not optimization beyond the best
known lipid.

North star:

> **Generate chemistry that was not pre-enumerated, demonstrate why its activity
> prediction was credible, show exactly how it was made, and prove that the resulting
> particle delivers mRNA in vivo.**

---

## References cited in this authority

1. Li, B. et al. *Nat. Mater.* **23**, 1002-1008 (2024). https://www.nature.com/articles/s41563-024-01867-3
2. Ou, Y. et al. arXiv:2412.00928 (2024). https://arxiv.org/abs/2412.00928
3. Witten, J. et al. *Nat. Biotechnol.* **43**, 1790-1799 (2025). https://www.nature.com/articles/s41587-024-02490-y
4. Nature Biotechnology content types. https://www.nature.com/nbt/content
5. LNP transfection ML benchmarking. https://www.nature.com/articles/s44488-026-00007-x

---

# 19. Benchmarking, success definition and revised panel allocation

**Status: authoritative.** Received 2026-08-06, after section 13. Where this
conflicts with section 13, this section wins.

## The target claim

The publication target is **not** "FORGE beats H9". It is:

> **FORGE generates multiple synthesis-resolved, functional ionizable lipids from
> only ten prospective designs, enriches activity among its highly ranked
> candidates, and produces at least one lead that is competitive with a strong
> existing benchmark.**

Beating H9 is an upside outcome, not the definition of success. FORGE is not a
potency-guided lead optimizer: support-aware sampling changes how often the
generator reaches chemistry the activity model can credibly rank, and predicted
potency is then used to select among completed supported designs. The accurate
claim is therefore that **generation expands the accessible chemistry and
support-constrained ranking enriches the generated population for active lipids**,
not that the generative trajectory optimizes potency.

## Why H9 must not be the success threshold

H9 was selected from roughly 12,000 virtual Ugi lipids and is the winner of that
screen rather than a representative AGILE lipid. Its published in vivo result
followed design-of-experiments formulation optimization, giving approximately
7.8-fold more intramuscular expression than MC3 and comparability with ALC-0315.
Comparing ten newly generated lipids in one fixed screening formulation against
post-DoE H9 would conflate molecular quality with unequal formulation effort.

H9 must still be included. Omitting it invites the obvious reviewer question about
why the platform was not compared with the best lipid previously found from the
same chemistry and dataset. Include it as the **same-class ceiling**, not the bar
for success.

## Benchmark hierarchy

| Role | Comparator | Question answered |
| --- | --- | --- |
| Hit threshold | prespecified strong but not singularly best AGILE lipid | Did FORGE efficiently produce genuinely active candidates? |
| Same-class ceiling | H9 | Did the best design approach or exceed the strongest reported Ugi-3 discovery? |
| External reference | ALC-0315 for IM; MC3 as historical secondary | Is the lead competitive with an established non-Ugi system? |
| Ranking control | the two low-ranked generated candidates | Did supported ranking enrich the generated population? |
| Negative control | vehicle or empty formulation | Is the biological signal unambiguous? |

The AGILE reference must be chosen **before** experiments: top 5-10% of the
measured 1,100-lipid dataset, commercially available components, acceptable
formulation characteristics, reproducible in our hands. Optionally use three AGILE
controls spanning high, median and weak activity, which calibrates our
implementation of the assay and makes the hit rate interpretable.

For intramuscular delivery ALC-0315 is the principal external benchmark; MC3 is a
soft comparator and serves as a historical anchor and assay control. If animal
capacity is limited, prioritize H9, then ALC-0315, then vehicle, and keep MC3 in
vitro or in a secondary cohort.

## Fair comparison

Formulate FORGE designs and H9 with the same helper lipids, molar ratios, mRNA,
ionizable-lipid-to-RNA ratio, mixing procedure, purification and dose, with lipid
input corrected for measured purity. Produce H9 in our laboratory and measure it in
the same study; do not compare fixed-formulation FORGE candidates against a
literature value for optimized H9. If the lead is later formulation-optimized,
either give H9 the same optimization budget or label the experiment "lead
formulation optimization" rather than a structure-only comparison.

## Revised panel allocation

Supersedes the 6/2/2 split in section 13:

* **7 primary** supported, high-ranked designs;
* **1 de novo frontier** design with especially compelling exact-new component
  chemistry, still inside the prespecified predictive-support domain;
* **2 matched low-ranked generated controls**.

Frontier candidates drop from two to one because each is more likely to expose
weaknesses in the activity model, and one suffices to establish that the framework
produces meaningful component novelty. Seven primary candidates maximize the chance
of a convincing prospective activity yield. Do not spend a slot on a dramatic but
unsupported extrapolation.

## Preregistered endpoints

* **Primary:** proportion of highly ranked FORGE designs meeting the prespecified in
  vitro hit definition under a common formulation. Report x/8 selected designs and
  y/2 low-ranked controls with exact binomial confidence intervals. Do not oversell
  a formal significance test on two controls.
* **Key secondary:** proportion of locked designs carried through route execution,
  final-lipid isolation and formulation. This validates the synthesis-aware claim.
* **Key biological:** number of designs with reproducible in vivo expression above
  vehicle and above the prespecified AGILE reference at matched dose.
* **Benchmark-competitive:** best FORGE design against H9 and ALC-0315 under matched
  conditions, reported as lead performance rather than as the success criterion.

Methods sentence to preregister:

> Prospective success was not defined as superiority to H9. H9 was included as the
> strongest reported Ugi-3 comparator, whereas the primary prospective endpoint
> measured the yield of active designs relative to a prespecified AGILE reference
> under a common formulation.

## Claim language before and after results

Introduction and abstract say FORGE was designed to **expand accessible chemistry
and enrich the generated population for experimentally supported activity**. They
must not say it was designed to improve potency beyond existing lipids. After
results: promote potency improvement only if the lead beats H9; say
benchmark-competitive if it matches; emphasize prospective hit generation if it
beats only the AGILE reference; emphasize the first complete de novo-to-in-vivo
chain if it is active but weaker.

## Outcome ladder

* **Best case:** multiple active lipids, high synthesis and formulation success,
  best design equals or exceeds H9 under fair comparison, independent replication,
  exact-new component in the lead. Headline: de novo generation produces a Ugi-3
  lipid exceeding the strongest finite-library discovery.
* **Strong and realistic, the target:** multiple hits, top-ranked clearly beat
  low-ranked generated controls, at least one lead comparable to H9 or ALC-0315
  within uncertainty, dossier substantially predicts bench accessibility, lead
  reproduces, lead carries component novelty. Headline: synthesis-grounded
  generation efficiently produces functional, benchmark-competitive mRNA delivery
  materials beyond an enumerated component library.
* **Borderline:** several hits beating a representative AGILE reference or MC3 but
  several-fold weaker than H9 and ALC-0315, with no differentiated biodistribution
  or tolerability. Important as a first, but the biological climax may not carry
  Nature Biotechnology.
* **Nature Communications:** one or two reproducible in vivo hits weaker than H9,
  with good synthesis validation and a complete evidence chain.

Precedent: the Nature Biotechnology LiON study framed its lead FO-32 as **matching**
the state of the art for nebulized mouse-lung delivery, inside a larger package of
model development and multi-context validation. A benchmark-competitive material
plus a substantial new discovery capability can clear the venue.

> **Bottom line: include H9, but do not make beating H9 the price of publication.
> Approaching H9 while demonstrating something H9 could not, namely true de novo
> component generation with an experimentally validated route dossier, is already a
> compelling Nature Biotechnology story.**

---

# 20. Exploration beyond the applicability domain

**Status: authoritative.** Received 2026-08-06, after section 19. Supersedes the
panel allocation in both section 13 and section 19.

## The principle

Do not make the choice binary between abstaining and discarding, or trusting a raw
score and ranking normally. Keep two distinct uses of the predictor:

* **Inside support: qualified ranking.** Use the conservative lower bound, rank
  candidates, make enrichment claims.
* **Outside support: exploratory scoring.** Still compute and store the raw
  ensemble mean, ensemble variance, which component triggered abstention, distance
  beyond the boundary, exact-component novelty and route status, but label the
  prediction explicitly as exploratory and unqualified.

Change the philosophical statement from "outside the domain, we do not consider the
molecule" to:

> **Outside the domain, we do not treat predicted activity as established evidence.**

## Why H9 forces this

Our oracle abstained on H9 (`unseen_aldehyde`, raw mean 3.18, no qualified score).
The system said it lacked measured aldehyde evidence for a qualified estimate. It
did not say H9 was unlikely to be potent, and H9 was in fact highly potent. If
FORGE generates a compelling molecule with one new aldehyde and applicability
abstains, discarding it outright would eliminate exactly the frontier discovery
generative modelling is supposed to enable. Molecules with the greatest opportunity
to improve on existing data often lie near or beyond a predictor's applicability
boundary.

## Revised panel: 6 + 2 + 2

| Lane | n | Selection | What it tests |
| --- | ---: | --- | --- |
| Supported high-ranked | 6 | in-domain, high conservative activity score, diverse | Does FORGE enrich for active lipids? |
| Supported low-ranked | 2 | in-domain, low activity score, matched on novelty and route burden | Is the ranking informative? |
| Exploratory frontier | 2 | out-of-domain, near-boundary, route-complete, high raw prediction | Can generation find high-value chemistry beyond current predictive evidence? |

## Frontier eligibility, frozen before any OOD scores are inspected

Do not sort every abstained molecule by predicted mean and take the top two; that
preferentially selects the model's largest errors. A frontier candidate must:

1. fail applicability in **only one component role**, ideally aldehyde or tail;
2. lie within a prespecified distance beyond the boundary rather than wildly out of
   distribution;
3. retain the whole product and remaining component roles as supported or strongly
   neighbouring known chemistry;
4. be exactly Ugi-valid;
5. have a fully resolved synthesis route;
6. contain genuine new component chemistry;
7. pass the same structural and physicochemical screens as supported candidates;
8. contain no pathological motifs or extreme descriptors.

Only then rank the frozen frontier pool by raw ensemble mean.

* **Frontier 1: maximum plausible potency.** Highest raw score in the pool.
* **Frontier 2: maximum meaningful novelty.** Strong raw score, especially
  compelling generated chemistry (ideally a component absent even from the expanded
  catalogue), structurally distinct from Frontier 1.

## Do not use the lower bound outside the domain

Inside support the lower bound is the selection score because uncertainty is
calibrated there. Outside support, ensemble variance may itself fail to capture the
error, so `mean - k*sigma` cannot be interpreted the same way. Outside support, the
raw mean is an exploratory heuristic only. Freeze this distinction prospectively.

## Retain raw scores for all OOD molecules

Enables the retrospective predicted-versus-measured analysis, plotted with
supported and frontier candidates visually separated, frontier marked as
unqualified extrapolative predictions, and H9 included as an external reference.
All four outcomes are informative: OOD failing validates the restraint; OOD
succeeding sets up an active-learning loop; a single outstanding OOD molecule is
potentially the best outcome, giving the paper both safe and frontier discovery.

## Three-lane conceptual framework

* **Generate** — what structures can the model construct?
* **Exploit** — where does the experimental dataset support activity-based
  prioritization? Use the qualified oracle and support-aware sampling.
* **Explore** — what promising chemistry lies beyond current predictive support?
  Record extrapolative scores, route selected frontier molecules, measure them.

Leading to: generate, then support-aware exploitation plus frontier exploration,
then measure, then enlarge the evidence base. No second training round is required
in this paper; the prospective panel demonstrates the two regimes.

## Manuscript language

Replace "the activity predictor was used only to rank completed, supported designs"
with:

> Qualified activity estimates were used for candidate ranking only within the
> prespecified predictive-support domain. Raw predictions were retained outside
> this domain as explicitly extrapolative diagnostics and were used only to select
> a prespecified exploratory subset for prospective measurement.

And:

> Accordingly, the prospective panel contained six prediction-supported high-ranked
> designs, two prediction-supported low-ranked controls and two route-complete
> frontier designs lying just beyond the predictive-support boundary.

**Terminology.** Inside the domain, "high-ranked candidates". Outside, "frontier
designs" or "high-scoring exploratory candidates", never "high predicted potency",
which grants unqualified predictions more epistemic weight than the methodology
allows. Never combine the two populations when reporting prediction performance.

## Figure 2 revision

Show the prediction-supported region containing six high-ranked (filled) and two
low-ranked (open) candidates, two frontier candidates (stars) immediately outside
the boundary, and H9 as an illustrative published molecule marked ABSTAIN. Caption
prominently:

> **Predictive support determines how predictions are used, not which molecules may
> be experimentally tested.**

## What must not happen

Do not run the oracle across all OOD molecules, inspect the results, cherry-pick
two that look exciting and retrofit an exploration rationale. Freeze the frontier
definition, boundary distance, permitted unsupported roles, route requirements,
raw-score selection rule, diversity rule and slot count first, then execute.

---

# 21. Three evidence tiers, and what Tier B does not have to prove

**Status: authoritative.** Received 2026-08-06, after section 20. Revises the
binary supported/exploratory split and the Tier-B ordering score.

## The correction

Tier B was over-engineered. It does not have to become a second formally validated
applicability domain before anything in it can be ranked or synthesized.

AGILE did not first establish leave-one-component-out generalization. They used an
80/10/10 Murcko-scaffold split with test R^2 = 0.249 and Pearson 0.573; a lipid
predicted in the top 16% had a 41% chance of actually being there. They then ranked
~12,000 candidates by **ensemble mean minus ensemble standard deviation**, selected
15 with head/tail diversity enforced, found H9 among them, and separately tested
candidates ranked 31-45 and several bottom-ranked molecules, which performed worse.
That is prospective enrichment, not certified calibration, and it was enough.

Caveat to state honestly: AGILE's 12,000 candidates were filtered from the same
60,000-lipid virtual library used for self-supervised pretraining, so H9 was unseen
to the potency labels but probably not to the representation learner.

Our own descriptive test performance (R^2 = 0.253, rho = 0.612) sits in the same
neighbourhood, though the splits and metrics are not directly comparable. It would
be strange to demand more proof of ourselves merely to synthesize sensible H9-like
candidates.

> **A predictor does not need certified component-level extrapolation before it can
> be used as a prospective ranking and enrichment tool.**

## The tiers

**Tier A, validated exploitation.** All components measured, combination new, passes
the existing applicability gates. Ranked by the conservative lower bound. This
cohort carries the main oracle-guided allocation claim.

**Tier B, analog-supported prospective transfer.** Exactly one unseen component with
a close measured analogue, the rest of the molecule supported, route verified. This
does **not** assert proven generalization to unseen components. It asserts a
scientifically reasonable transfer bet, selected by the oracle and tested
prospectively. Rank with the **conservative ensemble score, not the raw mean**,
following AGILE's own mean-minus-standard-deviation practice. Report separately from
Tier A.

**Tier C, open extrapolation.** Remote component, multiple unseen roles, or a role
where holdout evidence was poor, especially a new head. Intentionally exploratory
bets, not calibrated predictions.

The distinction governs **how the experiment is interpreted**, not whether the model
is permitted to produce a number.

## LOCO is supporting analysis, not a gate

Leave-one-component-out validation is needed only to claim that the oracle is
validated for unseen aldehydes, that T >= 0.85 candidates belong inside the formal
applicability domain, that Tier B may be pooled with Tier A in the primary analysis,
or that the uncertainty interval has established coverage for unseen components. The
paper needs none of those. A lightweight aldehyde-LOCO analysis is worth including
in the supplement because it is cheap, but it must not gate candidate selection,
delay the panel or drive a redesign of the uncertainty system. A real Tier-B
synthesis result is more persuasive than a similarity-conditioned LOCO analysis
built from 11 aldehyde identities.

## Panel decision

Keep **both** Tier-B frontier candidates rather than swapping one for Tier C. Two
Tier-B candidates give a small amount of prospective replication; one Tier B plus
one Tier C gives two unrelated anecdotes. Tier B is where there is a plausible
chance of both finding a hit and learning that local component transfer works.
Include Tier C only with additional capacity or an exceptionally compelling,
route-clean candidate.

| Cohort | Purpose | Claim status |
| --- | --- | --- |
| Tier A, support-aware versus broad | main matched generator and allocation test | confirmatory |
| Two Tier-B frontier candidates | prospective test of H9-like analog transfer | secondary, prospective |
| Tier C, only if extra capacity | genuinely unsupported exploration | exploratory case study |

Do not pool Tier B into the Tier-A primary hit-rate comparison.

## Manuscript language

Before results:

> We reserved a small frontier cohort containing one previously unmeasured component
> whose structure remained close to measured components in the same reaction role.
> These candidates were ranked using the same conservative ensemble score but were
> excluded from the primary applicability-qualified analysis and evaluated as
> prospective tests of local structural transfer.

If they work:

> The prospective frontier results indicate that the predictor retained useful
> ranking ability for closely analogous, previously unmeasured aldehydes, despite
> these candidates falling outside the strict identity-based applicability domain.

If they fail:

> The frontier cohort did not reproduce the gains observed in the
> applicability-qualified region, supporting the use of abstention for unseen
> component identities.

## Where H9 belongs

Precedent and external retrospective example, not a target to reproduce or formally
rank against:

> AGILE showed that a model with only moderate internal predictive performance could
> still enrich a prospective candidate panel and discover H9. We therefore treat
> close-analogue unseen components as legitimate experimental transfer candidates,
> while separating them from the model's qualified ranking domain.

Be more rigorous than AGILE in labelling the evidence, without erecting a bar so
high that it refuses the experimentally sensible bets that made AGILE successful.

---

# 22. Locked 30-candidate panel, Q/N/X cohorts

**Status: authoritative.** Received 2026-08-06, after section 21. Supersedes the
panel allocation in sections 13, 19 and 20, and renames the prediction-evidence
cohorts.

## Terminology fix, applied everywhere

Tier A/B/C collides with the route-evidence labels already in use. Prediction-
evidence cohorts are renamed:

* **Q — Qualified.** All component identities measured; only the combination is new.
* **N — Near-boundary analog transfer.** Exactly one unseen component with a close
  measured analogue inside the frozen envelope.
* **X — Extrapolative.** Exactly one unseen component beyond that envelope.

Route-evidence labels remain R1/R2/R3.

## The locked panel

| Cohort | n | Role |
| --- | ---: | --- |
| Q-high, qualified high-ranked | 12 | primary prospective discovery cohort |
| Q-low, qualified low-ranked controls | 6 | tests whether oracle ranking adds value within the qualified population |
| N-transfer, analog-supported | 8 | replicated prospective test of one unseen but closely related component |
| X-extrapolative | 4 | behaviour beyond the analog-supported envelope |
| **total generated candidates** | **30** | benchmarks and vehicle sit outside this count |

## Statistical correction, verified

My earlier Fisher values and power claim were wrong. Recomputed:

| claim | I said | correct |
| --- | --- | --- |
| 4/6 vs 0/2, one-sided Fisher | p = 0.107 | **p = 0.214** |
| 8/12 vs 0/6, one-sided Fisher | p = 0.008 | **p = 0.011** |

Those are postulated realized outcomes, not ex ante power. Ex ante, one-sided
alpha = 0.05, true rates 67% versus 10%:

| design | power |
| --- | ---: |
| 6 high vs 2 low | 0.07 |
| **12 high vs 6 low** | **0.65** |
| 9 high vs 9 low | 0.70 |
| 15 high vs 15 low | 0.93 |
| 12 vs 6 with control rate ~0% | 0.93 |

So the correct statement is that expanding 6-vs-2 to 12-vs-6 makes the enrichment
analysis **substantially more informative and gives it a realistic chance of formal
significance if the effect is large**, not that it guarantees a powered result. A
balanced 9-vs-9 would give slightly better power but costs three discovery slots;
12-vs-6 is the accepted compromise for a discovery paper provided the power is not
oversold.

**Primary quantitative endpoint is the continuous potency measurement.**
Thresholded hit rate is an interpretable secondary endpoint. Do not reduce all
activity to hit or no-hit.

## N must be one coherent experiment

The eight N candidates must not mix unseen aldehydes, unseen heads, new tails and
multiple-component novelty, which would make eight observations into four different
generalization problems. Definition:

> Exactly one previously unmeasured component in one prespecified reaction role,
> the aldehyde role, with a close measured analogue inside the frozen similarity
> envelope. Every other component identity has been measured and the exact forward
> synthesis is verified.

Within the eight: **six** strongest by conservative score, **two** deliberately near
the lower edge of the envelope, so the cohort probes the boundary as well as
exploiting it.

## X differs from N along exactly one axis

> Exactly one unseen aldehyde, as in N, but beyond the frozen analog-support
> envelope.

Not multiple unseen roles, not a new head group. The three cohorts then form a
single interpretable progression: all identities measured, one closely analogous
unseen aldehyde, one more remote unseen aldehyde. A multiple-new-component or
new-head candidate may still be synthesized as an explicitly labelled moonshot but
must not be aggregated into X.

## Q-low controls must be genuinely matched

Drawn from the same qualified and synthesizable population as Q-high, matched or
balanced on reaction role and component-frequency structure, synthetic-route
confidence, broad molecular size and lipophilicity, formulation compatibility and
structural diversity. Do not select malformed, insoluble, synthetically dubious or
chemically repetitive low-ranked candidates. The comparison must be: same evidence
support, same route standard, same approximate chemistry distribution, different
prespecified oracle rank.

## What stays computational

Broad versus support-enriched generation remains a frozen computational experiment
and does not consume wet-lab slots. Adding a generator arm to the panel would
fragment every comparison at once. Establish at scale computationally that
support-enriched sampling allocates more probability to route-complete, qualified
candidates; use the 30 to test synthesis, ranking, activity, analog transfer and
extrapolation prospectively. Each experiment gets one job.

## What "30 candidates" means

The locked generated synthesis and formulation panel, with intention-to-test
accounting from the start:

> 30 locked -> route attempted -> correct product isolated -> formulation QC passed
> -> biological assay completed.

Failures stay in the denominator appropriate to each claim. Do not silently replace
failed syntheses or formulations with backups in the primary analysis; backups may
be prepared operationally but replacements are separately labelled. Animal
advancement follows the already frozen criteria, not inspection of attractive in
vitro results. Distinguish synthesis success, formulation qualification, in vitro
activity and in vivo activity throughout. Thirty is consistent with the 24-40
primary Ugi-candidate planning range.

## Benchmarks

Assay benchmarks must not consume generated-candidate quotas, but not all are
required. Use the relevant established positive-control lipid or lipids, a vehicle
or negative control, and any formulation-system control needed for assay
interpretation. **H9 remains an external precedent; a direct H9 arm is not
mandatory** unless already practical and genuinely comparable under our formulation
and assay conditions. The paper does not need to prove our model ranks H9.

---

# 23. Strategic reset: the generative framework is the paper

**Status: authoritative.** Received 2026-08-06, after section 22. Supersedes the
figure architecture in section 10 and settles the claim hierarchy.

## What drifted, and the correction

The candidate-selection machinery came close to becoming the paper. Useful work:
finding the applicability-domain problem, separating supported ranking from frontier
exploration, locking the 30, catching the bad SMARTS before synthesis, and testing
simple baselines. Drift began at: LOCO certification as a gate for N, six-model
predictive tournaments, formal preregistration infrastructure, finer uncertainty
taxonomies, and treating Q/N/X as the scientific invention.

**Q/N/X is selection machinery. It belongs in a panel or Extended Data figure
explaining how candidates were chosen. It is not Figure 1.**

## The comparator result is good news

A component-additive ridge performs well on the 1,100-product factorial dataset, and
all simple baselines separate Q-high from Q-low. That means we should stop trying to
show the potency network is exceptional. It is not the invention.

The computational claim is:

> **FORGE proposes new molecular structures that fixed-library screening cannot
> propose, while preserving a credible path from generated molecule to synthesis.**

Not: our potency predictor beats other predictors. Describe the predictor as a
**conservative allocation mechanism** for a finite experimental budget, not as a
model that uniquely learned potency. This makes the paper cleaner, not weaker.

## Sharpened novelty statement

Avoid "first generative lipid model": LUMI-lab already describes a foundation model
proposing candidate lipid structures in an autonomous loop, and the literature moves
fast. Use instead:

> Existing AI-guided lipid discovery largely predicts, ranks or iteratively selects
> candidates from chemically prescribed design spaces. FORGE instead treats
> **molecular proposal and synthetic realization as one generative problem**,
> generating complete lipid structures while resolving the chemistry required to make
> them.

Timing helps: a July 2026 *Nature Reviews Materials* comment describes nucleic-acid
delivery moving from screening toward generating materials against predefined
property targets.

## Evidence budget

| paper type | computation carries | biology must carry |
| --- | --- | --- |
| another lipid predictor | low to moderate | a lot |
| new multiobjective optimizer | moderate | a lot |
| new autonomous screening platform | moderate to high | strong |
| **new synthesis-grounded generative framework** | **high** | **convincing proof, not exhaustive translation** |

Precedent that computational novelty earns credit: the 2025 *Nat Biotechnol*
quantum-classical generative design paper synthesized 15 KRAS molecules with SPR,
cellular and NMR validation and no animal efficacy program; Germinal (2026)
validated a generative antibody system with binding, biophysics and structure.

**But LNP delivery is fundamentally an in vivo phenotype.** Biodistribution, protein
corona, pharmacokinetics, tissue access, innate immunity and endosomal escape all
change in an animal. Computational novelty can reduce the **breadth** of animal work;
it cannot remove the need for **one decisive in vivo demonstration**.

> Algorithmic novelty can replace biological breadth. It cannot replace biological
> reality.

We do not need LiON's mouse-muscle-plus-nasal-plus-nebulized-lung-plus-ferret
package, nor MOLEA's delivery-plus-selectivity-plus-editing-plus-disease package,
unless the paper starts making those application claims.

## The target package

> 30 generated candidates, substantial synthesis success, prospective in vitro
> enrichment, several active novel lipids, 2 to 4 leads in mice, at least one
> FORGE-generated lipid showing strong in vivo delivery, ideally approaching or
> exceeding a relevant benchmark.

Add one deeper experiment on the lead only if the first in vivo result warrants it,
and prefer a functional payload over another luciferase run.

**Do not compensate for limited animal bandwidth with more predictive-model work.**
Once the model is computationally credible, another benchmark table has rapidly
diminishing value. Marginal resources go into making the synthesis, formulation and
in vivo chain clean.

## Figure architecture, superseding section 10

- **Fig. 1 — FORGE.** Generative synthesis-grounded molecular design.
- **Fig. 2 — Computational evidence.** Open-ended lipid generation, novelty and
  diversity, route completion, generator-specific ablations, comparison to
  generation-then-filtering alternatives.
- **Fig. 3 — Prospective chemistry.** 30 designs, Q/N/X selection shown compactly,
  actual synthesis success and route fidelity.
- **Fig. 4 — Prospective LNP performance.** Formulation and whole-panel biological
  screening, enrichment and generated hits.
- **Fig. 5 — In vivo.** Strongest FORGE-designed lipids against benchmarks.

## 23a. In vivo package extends to editing in the target organ

Decision 2026-08-06. The prospective biological package goes one step beyond
reporter delivery to a **functional genomic outcome in the target organ**:

> 30 generated candidates, synthesis success, prospective in vitro enrichment,
> several active novel lipids, 2 to 4 leads in mice with reporter mRNA to establish
> delivery and biodistribution, and **editing in the target organ of interest** for
> the lead.

This is the functional payload section 23 preferred over a second luciferase run,
and it lifts the biological claim from *delivery demonstrated* to *delivery produces
a functional genomic outcome in the intended tissue*. It is the endpoint class that
MOLEA and LUMI-lab reached, reached here with a smaller panel because the
computational contribution is a generative framework rather than another predictor.

What this does **not** license: disease-modification claims, tissue-selectivity
claims beyond what biodistribution actually shows, or durability claims without a
time course. Report editing efficiency by tissue and cell type, the number of animals
per group, every predefined exclusion, and tolerability.

## 23b. Figures revert to section 10

Correction 2026-08-06. Section 23's figure list is withdrawn; **section 10's figure
architecture stands.** The two were largely the same five-figure shape, and section
10's panel plans are the more developed. The manuscript's figure legends already
follow section 10 and are unchanged.

The one point section 23 contributes that section 10 lacks: **Q/N/X is selection
machinery and belongs inside the prospective-chemistry figure or Extended Data, not
as a headline figure.** Section 10's Figure 3 panel (e) already carries panel
selection, so Q/N/X goes there.

Recorded as an example of the drift this document exists to catch: a figure
architecture was restated rather than corrected, which would have replaced a worked
plan with a sketch for no gain.

## 23c. Computational development frozen

Instruction 2026-08-06:

> Freeze computational development for the prospective panel. The v3 30-candidate
> set is final. Complete human chemistry signoff and procurement, then proceed to
> synthesis and experimental testing. Do not run additional model development or
> alter candidate identities based on new computational scores.

No further LOCO work, model families, uncertainty calibration or computational
benchmarking before synthesis. A candidate identity changes only for a **real
chemistry or procurement impossibility** found during human review, never for a
score.

**Terminology.** The 30 are **prospectively selected candidates**, not hits. They
become hits only if they show activity experimentally. Applies to the manuscript,
figures, dossiers and all internal documents.

**Existing measured lipids** remain useful as assay benchmarks, positive controls,
evidence of the experimental landscape, and reference points for whether the
generated molecules are interesting. The paper must not become "we rediscovered
known hits". The key result is:

> FORGE proposed previously unmeasured molecules, provided credible routes to them,
> and prospective experimental testing showed that some of those generated molecules
> are functional LNP materials.

Then the strongest generated molecules get the deeper biology, ending in editing in
the target organ.
