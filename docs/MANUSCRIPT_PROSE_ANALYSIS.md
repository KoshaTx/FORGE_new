# How these papers are actually written

Corpus read for this analysis: Su et al. (*Nat Biomed Eng* 2026, Intro ¶1), AGILE
(*Nat Commun* 2024, full Intro + all Results openers + Discussion opener),
PeptiVerse (*Nat Commun* 2026, full text), PepMLM (*Nat Biotechnol*, Intro,
Results openers, Discussion).

---

## 1. Sentence level

### The single most important rule: the subject of a sentence is a thing that acts

Every declarative sentence in this corpus has a concrete agent — a person (`we`,
`Xu et al.`), a named molecule (`306Oi10`, `lipid 5`), a named system
(`PeptiVerse`, `AGILE`), or a physical object (`LNPs with imidazole-based heads`).

| Corpus | My draft (wrong) |
|---|---|
| "**Xu et al.** synthesized 1,200 such lipids…" | "**Combinatorial chemistry** supplies the throughput" |
| "**LNPs with vitamin C-derived head groups** preferentially deliver mRNA to macrophages" | "**Head groups and linker chemistry** exert comparable leverage" |
| "**we** curated and integrated experimentally derived datasets" | "**This sensitivity** is what makes design worth doing" |

An abstract noun as grammatical agent ("chemistry supplies", "sensitivity makes",
"generation emits") is the signature of essay prose. It never appears here.

### Verbs are laboratory actions

`trained, curated, synthesized, measured, screened, compared, examined, evaluated,
gathered, applied, co-folded, integrated, deliver, target, increase, escape`

Not: `supplies, exerts, constitutes, represents a shift, carries a burden`.

### Sentences are long, and none is punchy

Typical 25–40 words, one subject–verb spine, detail hung off it in subordinate
clauses. There are **no** short declarative punches. "These methods work." would
never appear. Rhetorical compression reads as a press release, not a paper.

### Numbers travel with their object, mid-sentence

- "for 1433 amino acid sequence inputs and 1702 SMILES-based peptide inputs"
- "ChemBERTa consistently outperformed PeptideCLM on both PAMPA (ρ = 0.69 vs. 0.59)"
- "increases protein expression threefold … and fivefold in non-human primates"

Never a number in its own sentence for emphasis.

### Interpretation is always hedged; observation never is

Observation: "ipTM showed negligible association with experimental binding affinity."
Interpretation: "**likely reflecting** ChemBERTa's broader pretraining chemical
diversity", "a difference **attributed to** stronger surface ionization",
"These results **suggest that** ipTM does not reliably capture…"

The hedge marks the boundary between what was measured and what it means. Dropping
it reads as overclaiming; adding it to an observation reads as weak.

---

## 2. Paragraph level

### Results paragraphs have a fixed three-part shape

1. **Purpose clause, then `we` + action.**
   - "To assess whether PeptiVerse binding affinity predictions reflect physically
     meaningful interaction strength, **we examined** their relationship to…"
   - "To efficiently create diverse libraries of ionizable lipids, **we utilized**
     an HTS platform…"
   - "Next, to benchmark PepMLM's generation quality, **we co-folded** the test and
     generated binders…"
   - "Given our heterogeneous data settings, **we evaluated** a diverse set of
     predictor architectures…"
2. **Specifics with numbers and figure callouts**, often contrasted:
   "In contrast, regression tasks such as peptide half-life… remain comparatively
   data-limited."
3. **A closing interpretive sentence**, hedged, that hands off to the next section:
   "Together, our observations highlight that achievable predictive performance …
   is frequently constrained by data availability rather than model capacity."

### Transitions carry the argument

`Together, In contrast, Importantly, Additionally, Finally, Moreover, We therefore
asked, Rather than enforcing…`

These sit at sentence-start and tell the reader what kind of move is coming. The
draft currently has almost none, which is why it reads as a list of facts.

### The concession-and-gap is one sentence, never two

- "These advances highlight the value of chemical modification, **yet** the specific
  influence of the spatial conformation of lipids … **remains insufficiently studied**."
- "**While** the 3-CR combinatorial chemistry has been showcased …, constructing and
  testing a more extensive lipid library … **remains** a formidable, time-consuming,
  and costly task."
- "In contrast, prior tools have made meaningful progress **but remain specialized**
  in either input modality or property scope."

Formula: `[credit prior work], yet/but/while [the specific thing that] remains [gap].`
This is the hinge of the whole Introduction and it is always a single sentence.

---

## 3. Section headings are claims, not topics

| Corpus heading | Note |
|---|---|
| "Dataset composition highlights constraints across peptide properties" | subject + verb + object |
| "PeptiVerse deploys state-of-the-art predictors for a broad range of therapeutically relevant peptide properties" | platform is the subject |
| "PeptiVerse demonstrates superior predictive performance relative to existing peptide property predictors" | comparative claim |
| "Identification of ionizable lipid by AGILE for muscle-selective mRNA delivery" | outcome + agent + purpose |

A reader should be able to read only the headings and get the paper's argument.
Our current headings ("Sparse whole-lipid representation") are topics, and topics
tell the reader nothing.

---

## 4. Framing level

### Introduction funnel, four moves

1. **Modality status with clinical anchors and specific numbers.** Su spends an
   entire paragraph on named lipids and fold-changes before mentioning AI at all.
   The expertise signal is density of specifics, not assertion.
2. **What the field has done**, cited densely, credited generously.
3. **The concessive gap sentence** (§2 above).
4. **"we introduce X"** — name, apposition, what it does, why it matters:
   - "To realize this potential in a unified framework, **we introduce PeptiVerse**
     (Fig. 1), a universal therapeutic peptide property evaluation platform
     developed to standardize and accelerate computational peptide design."

### How a priority claim is made so it survives

PepMLM: "PepMLM represents **the first example of target-conditioned de novo binder
design from sequence alone**."

The claim is narrow and every qualifier is load-bearing: *target-conditioned*, *de
novo*, *from sequence alone*. Strip any one and the claim becomes contestable.

**Our equivalent must be qualified the same way.** Not "the first generative model
with in vivo validation" but a claim whose qualifiers we can defend individually —
de novo generation (not selection from an enumerated library), route-resolved
output, prospective validation.

### Discussion opens by restating the contribution, not by summarizing results

- "PeptiVerse introduces a unified framework for therapeutic peptide property
  prediction that supports both amino acid sequence and SMILES-encoded inputs…"
- "In this work, we have introduced AGILE, a platform that combines deep learning
  with combinatorial chemistry…, to predict the mTP across different cell lines."

Then: comparison to prior work, then limitations, then forward-looking scope.

---

## 4b. Rules from review of the FORGE draft

Every one of these came from a specific rejected sentence. They are listed with
the offending text so the failure mode is recognizable next time.

### Prose

1. **No em-dashes.** Anywhere. Rewrite as commas, a colon, or two sentences.
   - Rejected: "the three products approved to date — Comirnaty, Spikevax and Onpattro — was built…"
2. **No conditional or hypothetical constructions**: `could`, `would`, `might`,
   `has the potential to`, `promises to`. State what is.
   - Rejected: "Generative design **could lift** it"; "De novo generation **would remove** it"
3. **No abstract noun as grammatical agent.** The subject is a person, a named
   molecule, or a named system.
   - Rejected: "**Combinatorial chemistry** supplies the throughput" (a delivery
     scientist would never say this); "**This sensitivity** is what makes design
     worth doing"; "**Head groups and linker chemistry** exert comparable leverage"
4. **No metaphors standing in for technical facts.**
   - Rejected: "a guess dressed as a number"; "generation alone is a structure on a screen"
5. **No punchy fragments or rhetorical compression.** Nothing under ~15 words.
   - Rejected: "These methods work:"
6. **Never imply a population or body of work that does not exist.** There are only
   a handful of generative lipid models and none has experimental validation, so
   phrasing that presupposes a field of practitioners is false.
   - Rejected: "**Generative models** are not so confined, but **their designs** have
     remained computational"
   - Also rejected for overreach in the other direction: "no such design has been
     synthesized and tested". Prefer naming the *difficulty*: "synthesizing the
     resulting designs is seldom straightforward, since their components are
     frequently neither commercially available nor reachable by established chemistry."
7. **Name the activity, not the practitioners.** "Generating lipids de novo removes
   this constraint" makes no claim about how many people do it.

### Abstract

8. **Sentence one is `[situation], yet [gap]`.** One sentence, two clauses. Not a
   historical sweep, not a rhetorical setup, no aside before the payoff.
   - Corpus: "Efficient mRNA delivery to specific tissues requires optimized
     ionizable lipids, yet the role of lipid spatial conformation … remains
     underexplored." (Su)
   - Rejected: "Every ionizable lipid tested in vivo was chosen from a predefined
     library … rather than designed." (magazine opener; delays the subject)
9. **Do not data-dump numbers.** Su's abstract contains almost no raw counts. Run a
   causal chain instead, where each clause explains the next. One or two numbers
   maximum, and only where the number *is* the result.
   - Rejected: six numbers in four consecutive sentences.
10. **Name the thing with an apposition**: "FORGE, a synthesis-aware generative
    framework that designs…", not "FORGE, which designs…".
11. **Cut negative ablations entirely.** They belong in the decision log.
    - Rejected: "Matched experiments did not support additional potency- or
      synthesis-driven trajectory guidance."

### Framing

12. **Position from the delivery field, not from machine learning.** The reader is
    an LNP scientist. Lead with what the work enables for them.
13. **The headline is the priority claim**: the first truly generative model for
    ionizable lipids carried to prospective in vivo validation. Synthesis-awareness
    is the *enabler* that made it possible, not the headline itself.
14. **Qualify a priority claim so each qualifier is defensible**, following PepMLM:
    "the first example of target-conditioned de novo binder design from sequence
    alone." Strip any qualifier and it becomes contestable.
15. **Draw the generative/enumerative line explicitly.** Prior AI work ranks
    enumerated candidates; left implicit, a reviewer reading "AI-driven generation"
    elsewhere will think the ground is taken.
16. **Frame FORGE as a general framework instantiated on Ugi-3**, not as a
    Ugi-specific method.

## 5. Checklist to apply to every sentence I write

1. Is the grammatical subject a person, a named molecule, or a named system? If it
   is an abstract noun, rewrite.
2. Is the verb something someone did or something a molecule does?
3. Is the sentence at least ~15 words with one clear spine? If it is a punchy
   fragment, it is wrong for this venue.
4. Does every number sit inside a sentence next to the thing it measures?
5. Is interpretation hedged and observation unhedged?
6. Does the paragraph open with a purpose clause and close with an interpretation?
7. Does the section heading state a claim?
