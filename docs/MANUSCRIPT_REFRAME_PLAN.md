# FORGE manuscript reframe plan

Target: *Nature Biotechnology* Article. Audience: LNP and mRNA delivery scientists,
most of whom do not read machine-learning papers. This plan replaces the current
computational-first framing with a delivery-first one and sequences the writing.

Reference framing studied: AGILE (Xu et al., *Nat Commun* 15:6305, 2024), MOLEA
(Zhou, Xu, Li et al., *Nat Biotechnol*, 2026), Su et al. (*Nat Biomed Eng*, 2026),
PeptiVerse (Zhang et al., *Nat Commun* 17:6819, 2026).

---

## 1. The one sentence everything hangs on

> Every AI method for ionizable lipids to date ranks a library someone enumerated
> in advance. FORGE generates lipids de novo — and returns the synthesis route
> that makes each one buildable.

That is the whole paper. Every section either sets this up or delivers it.

The field has automated *choosing*. Nobody has automated *designing*, because a
designed lipid outside the catalog is a structure nobody can make. We removed that
barrier, which is what lets the first genuinely generative design campaign reach
in vivo validation.

### Why this beats the obvious alternative framing

The tempting pitch is "we built a better generative model." That is the pitch a
reviewer has seen ten times, it invites a benchmark fight we do not need, and it
is not what is actually novel here.

The defensible and more interesting pitch is about **what comes out the other
end**. AGILE's library is synthesizable *by construction* — 20 heads x 12 chains
x 5 isocyanides — which is exactly why it cannot leave the library. The moment a
generative model steps outside a fixed catalog, synthesizability stops being free
and nobody has been paying for it. We pay for it, and the output is a shopping
list.

### The headline numbers, in delivery language

| Say this | Not this |
|---|---|
| 2,515 of 2,590 designed lipids (97%) come with a route to purchasable material | "97.1% in-domain makeability" |
| Every lipid reduces to 3 components; 201 components cover 2,590 lipids | "component factorization" |
| A chemist gets 29 catalog materials and at most 4 steps per lipid | "forward-verified L1/L2 transforms" |
| Designs use building blocks absent from the training corpus (75.7%) | "corpus-absent admitted fraction" |

---

## 2. Positioning and the priority claim

### The competitive landscape, checked

| Work | Design step | Routes returned? | Prospective validation |
|---|---|---|---|
| AGILE (*Nat Commun* 2024) | selection within synthesized library | no | in vivo |
| MOLEA (*Nat Biotechnol* 2026) | multiobjective selection | no | in vivo |
| Su et al. (*Nat Biomed Eng* 2026) | conformation-based selection | no | in vivo |
| Wang et al. (*Nat Commun* 2024) | AI generation + virtual screening, ~20M | no | in vivo, 9 lipids |
| LUMI-lab (*Cell* 2026) | foundation model + self-driving lab | no (robot-bounded) | in vivo |
| Ou et al. (NeurIPS-W 2024) | **generative + synthesis paths** | **yes** | **none** |

### The claim

> The first **truly generative** model for ionizable lipids with prospective in
> vivo validation — and one that generates in a form chemists can act on.

This holds, but only if the paper draws the generative/enumerative line explicitly.
Left implicit, a reviewer reading "AI-driven generation" in Wang et al. will think
the ground is taken. Drawn explicitly, the distinction is obvious and unarguable.

**The line, stated once in the Introduction:**

> Existing AI approaches rank or select from libraries that are enumerated in
> advance — combinatorially, or by what an automated platform can already make.
> Their output space is therefore bounded by a catalog. A generative model instead
> constructs molecular structure de novo, and its output space is not bounded by
> any enumeration.

Against that line: Wang et al. virtually screen ~20 million *enumerated* candidates;
LUMI-lab explores the chemistry its own robot can execute; AGILE, MOLEA and Su et al.
select within synthesized libraries. Ou et al. is genuinely generative and
route-aware, but is a workshop abstract with no chemistry performed — cite it, state
the delta, move on.

**Guard the wording.** "Truly generative" is doing real work, so the paper must earn
it with evidence, not adjectives: 75.7% of admitted designs contain a precursor
absent from the training corpus, and the designs leave the AGILE catalog entirely.
That is the proof the output space is not an enumeration.

Fallback if a reviewer still contests "first": drop the word and lead on the number.
*97% of designs arrive with a route* needs no priority claim to land.

### The second half of the claim: tractable to chemists

A generated structure is not yet a compound anyone can test. Each FORGE design
breaks into three components — an amine, an aldehyde and an isocyanide — and for
each component we either found a commercial supplier or built a route to one, then
ran that route forward and required it to reproduce the component exactly.

> FORGE does not just generate — it generates in a form a chemist can act on the
> same day: 29 catalog materials, at most 4 steps, for 20 candidate lipids.

This is the sentence that separates us from every row in the table above, and it
should appear in the Abstract, the Introduction's final paragraph, and the
Discussion's first.

### Framework, not one chemistry

Frame FORGE as a general synthesis-aware generative framework that we **instantiate
on Ugi three-component chemistry**, rather than as a Ugi-specific method. The
architecture — generate the product jointly with its components, then resolve each
component to purchasable material — is reaction-agnostic. Ugi-3 is the
instantiation, chosen because AGILE established it and because it gives a clean
three-role decomposition.

State this once in the Introduction and once in the Discussion as future scope. Do
not overclaim generality we have not demonstrated: we instantiated one chemistry,
and the Discussion should say the extension is straightforward rather than done.

### On prospective validation

The claim structure can be written now; the sentence stays a placeholder until the
data exists. Do not write "we prospectively validated" before we have. The
Introduction may say the framework *is* prospectively validated only once Step 11
completes — until then the draft carries the slot, not the assertion.

---

## 3. Hard constraints (Nature Biotechnology)

| Item | Limit | Current draft |
|---|---|---|
| Main text (Intro + Results + Discussion) | **3,000 words** | ~3,300 |
| Abstract, unreferenced | **150 words** | ~200 (inside a 629-word block) |
| Display items | **6** | 5 planned |
| Results/Methods subheadings | required | present |
| Discussion subheadings | **not allowed** | currently compliant |

The draft has **no Introduction section at all** — it goes Abstract straight to
Results. That is the single largest structural gap and the whole delivery framing
lives there.

---

## 4. Introduction — four paragraphs, ~500 words

Modeled on AGILE's funnel, which opens on mRNA rather than on chemistry.

**P1 — Why ionizable lipids decide everything (~150 words).**
Open on mRNA medicines, not on models. Comirnaty, Spikevax, Onpattro. The four-
component LNP; each approved product has a *different* ionizable lipid. State the
mechanism briefly: protonates at acidic pH to encapsulate, neutral at physiological
pH to avoid toxicity, re-protonates in the endosome to release. A delivery reader
must see their own field before they see ours.

**P2 — The discovery bottleneck as the field currently frames it (~120 words).**
Rational design covers limited structural space. Combinatorial chemistry with
multi-component reactions widened it — Ugi 3-CR, Michael addition libraries — and
AI now selects well within those libraries. Credit this work properly.

**P3 — The bottleneck nobody has addressed: the exit from the library (~130 words).**
Draw the line here. Every one of those advances *ranks* compounds that were
enumerated in advance — combinatorially, or by what an automated platform can
already make — so the reachable chemistry is fixed before the model runs. True
generation lifts that ceiling, but what a generative model emits is a structure,
not a plan, and an unmakeable lipid has no delivery properties at all. That is why
de novo design has stayed computational in this field while screening reached the
clinic. This is where the paper turns.

**P4 — What we did (~100 words).**
FORGE generates the complete lipid *and* the components it decomposes into, then
constructs and verifies a route for each component down to purchasable material.
Say once that this is a general synthesis-aware framework instantiated on Ugi
three-component chemistry. End on the deliverable: 20 candidate lipids, each with
a shopping list, prospectively validated. Do not enumerate methods here.

---

## 5. Results — retitle every section for a delivery reader

Current titles are unreadable outside our group. Proposed, with budgets:

| # | Current | Proposed | Words |
|---|---|---|---|
| 1 | FORGE couples whole-lipid generation to recursive synthesis programs | **FORGE designs a lipid and its synthesis together** | 250 |
| 2 | Ugi-3 data support joint lipid and precursor generation | *fold into 1 and Methods* | — |
| 3 | FORGE generates diverse Ugi lipids beyond the starting component collection | **Designed lipids leave the combinatorial library** | 450 |
| 4 | Morphology-aware sampling enriches biologically supported chemistry | **Designs stay inside the activity model's domain** | 300 |
| 5 | Hybrid route search connects generated components to experimental starting materials | **Ninety-seven percent of designs come with a route** | 500 |
| — | *(new)* | **Twenty candidates ready for synthesis** | 300 |
| 6–7 | Prospective synthesis / Functional evaluation | keep as placeholders | — |

Total ~1,800 words, leaving room for Intro (500) and Discussion (600).

Sections 3–5 are the spine. Section 5 is the paper's argument and should be the
longest and most concrete: components, vendor counts, step counts, the two named
routes (AGILE Tail A; isocyanide via formylation/dehydration).

### Selling the applicability work (Results 4)

This is the section most at risk of reading as internal ML housekeeping. It is not
— it is the reason the candidate list is trustworthy, and it should be sold that way.

**The delivery-language pitch:**

> We trained the activity model on 1,100 measured Ugi lipids. It returns a potency
> score for any structure we give it, including lipids built from chemistry it
> never saw during training, and those scores are unreliable. We therefore measured
> which lipids the model has enough training data to predict, and biased generation
> toward them. This raised the fraction of designs falling in that region from
> 3.87% to 8.64%, and it is why we rank the 2,590 lipids in Results 5 rather than
> all 30,180.

Every clause there states a fact a delivery scientist can check. Compare the version
this replaced — "a potency prediction outside a model's domain is a guess dressed as
a number" — which sounds knowing and tells the reader nothing about what we did.

**Do:**
- Lead with the consequence (trustworthy candidates), not the mechanism (tilting).
- Give the 2.23-fold enrichment as the number, once.
- State plainly that it preserves non-zero probability on all qualified programs —
  it concentrates sampling, it does not amputate the space. One clause.
- Connect forward: this is what defines the 2,590 population that Results 5 routes.

**Do not:**
- Use "morphology proposal", "allocation schedule", "support-enriched arm" in main
  text without translating.
- Report the matched potency challenger (23 vs 27) — internal tuning, cut.
- Present enrichment as a delivery result. It is an evidence-quality result.

### Writing rules for these sections

- Lead every subsection with the result in plain language; put the method in the
  second sentence and the rest in Methods.
- No model architecture in the main text. "A generative model trained on X"
  is enough; the flow/transformer details go to Methods.
- Name chemistry, not tensors: "tridecylamine, 68 suppliers" beats any embedding.
- Every number traces to `results/phase1/forge_data_suite_v1/` — already 25/25.

### Prose style — enforce this on every sentence

The draft and my first pass at this plan both drifted into aphorism. Sentences like
"a potency prediction outside a model's domain is a guess dressed as a number" or
"generation alone is a structure on a screen" sound authoritative and carry no
information. A reader cannot check them, act on them, or disagree with them.

**Rules:**

1. **State what we did, in active voice, with the actor named.** "We trained the
   model on 1,100 measured lipids", not "the model was trained" and not "training
   proceeded on a curated corpus".
2. **No metaphors for technical facts.** If a sentence contains "is essentially",
   "amounts to", "dressed as", "on a screen", "in the wild" — delete and restate.
3. **Every claim sentence carries a number or a named object.** If it has neither,
   it is decoration.
4. **Kill abstract nouns doing verb work.** "Applicability enrichment was achieved"
   becomes "we biased generation toward the region the model can predict".
5. **One idea per sentence.** The current draft averages three.
6. **Say the chemistry.** "tridecylamine, 68 suppliers, two steps" not "the
   precursor was procurement-resolved".

**Test:** read the sentence aloud to a bench chemist. If they cannot tell you what
we did or what number came out, rewrite it.

---

## 6. What to cut

The draft carries diagnostics that were essential for us and are noise for a
reader. These are *not* deleted from the record — they stay in `DECISION_LOG.md`
and the evidence matrix, which is exactly what those artifacts are for.

**Cut from main text entirely:**
- SMC / MH partial-state sampling — abandoned; belongs nowhere in the narrative.
- Synthesis-tilting diagnostic — a negative result about a method we did not adopt.
- Potency-guided trajectory experiments — same.
- Morphology-potency matched challenger (23 vs 27) — internal tuning.
- Scheduler defect and branch-run history — Methods footnote at most.
- Checkpoint selection minutiae, raw-census gate failures — Methods.

**Keep, because they are load-bearing:**
- Applicability domain — it is *why* 2,590 and not 30,180, and a reviewer will
  ask. One or two sentences.
- The 75 route-only products and 4 unpurchasable amines — reporting the residual
  honestly is what makes 97% credible rather than suspicious.

**Rule of thumb:** a negative result earns main-text space only if a reviewer
would otherwise assume we did the wrong thing. Everything else is a Methods
sentence or a supplementary table.

---

## 7. Display items (6 maximum)

1. **FORGE overview** — lipid in, three components out, routes down to catalog
   material. Must be readable by a chemist in ten seconds.
2. **Beyond the library** — designed chemical space vs the AGILE catalog;
   component novelty.
3. **Applicability and potency** — domain, conservative ranking, where the 2,590 sit.
4. **Route resolution** — the 97% figure, broken down by component role, with the
   two named routes drawn as chemistry.
5. **The 20 candidates** — structures, strata, shopping lists, step counts.
6. **Prospective results** — reserved.

Invest most effort in Figures 4 and 5. Figure 4 carries the 97% result, which is
the paper's argument. Figure 5 is the one a chemist will work from directly, so it
must show structures, supplier counts and step counts, not summary statistics.

*(No figures are to be produced until explicitly authorized.)*

---

## 8. Layout

Current: double line spacing, 0 pt paragraph spacing, margins T0.85 / B0.80 /
L1.08 / R0.92 in.

The perceived crowding is not leading — it is that double-spaced text with zero
paragraph spacing loses its paragraph boundaries. Proposed deltas in
`scripts/build_forge_manuscript.py`:

- Margins: L 1.08 -> **0.95**, R 0.92 -> **0.95** (symmetric, wider text block).
- Body `space_after`: 0 -> **6 pt** (restores paragraph rhythm).
- Keep double leading for the submission copy; a 1.5 reading copy can be a flag.

Otherwise keep the homological-flows layout, which is already close to PeptiVerse's
two-column Nature style.

---

## 9. Writing order

Piecemeal, dependency-ordered. Each step is independently reviewable.

| Step | Task | Depends on | State |
|---|---|---|---|
| 1 | Introduction, 4 paragraphs | nothing | **ready** |
| 2 | Results 1 — design + synthesis together | 1 | ready |
| 3 | Results 3 — leaving the library | 2 | ready |
| 4 | Results 4 — applicability domain | 3 | ready |
| 5 | Results 5 — route resolution *(the spine)* | 4 | ready |
| 6 | Results 6 — the 20 candidates | 5 | ready |
| 7 | Abstract, cut to 150 words | 1–6 | after 6 |
| 8 | Discussion, no subheadings | 1–7 | after 7 |
| 9 | Methods rewrite, absorbing everything cut | 1–8 | ready in parallel |
| 10 | Figures 1–5 | 1–8 | **needs authorization** |
| 11 | Prospective Results, Figures 6 | wet-lab data | **blocked** |

Steps 1–9 are writable now. Step 11 is the only true blocker, and it is not a
computational one.

---

## 10. What must not drift

- Route proposal is not route evidence; a vendor listing is not a quotation.
- Nothing has been synthesised. No yield, purity or delivery claim is implied.
- Potency is a conservative ranking signal, not a validated prediction.
- The 20 are a proposal for chemist review, not a locked panel.
- The paper claims *synthesis accessibility*, not novel route discovery. We used
  published chemistry deliberately and verified it forward; that is a feature.
