# FORGE ICLR: settled framing and experiment contract

Recorded 2026-08-15. This governs the ICLR paper. It supersedes the framing embedded in
`manuscript/FORGE_ICLR2027_submission.md`, which will be rewritten from scratch rather than
patched once the experiments below are frozen.

Numbers referenced here trace to `docs/FORGE_IMPLEMENTATION_REFERENCE.md`, which was
recomputed from frozen artifacts and is the source of truth.

> **2026-08-20 scope amendment.** The earlier one-Ugi-family generality boundary is superseded by
> the bounded computational extension in `MULTIREACTION_COMPUTATIONAL_PLAN.md`: Ugi remains the
> deep validated case, BL_2023 aza-Michael is a full computational second family, and LX_2024
> reductive amination is a lighter stress test. The extension covers ordered repetitions of an
> explicit registry reaction; it does not expand the claim to arbitrary multistep synthesis,
> reaction discovery, yield prediction, or biological transfer.

---

## 1. Settled framing

**What the paper is:**

> The first discrete flow framework specifically developed for de novo ionizable-lipid design,
> coupling whole-molecule generation to the reaction structure required for synthesis and to an
> evidence-aware procedure for selecting candidates under sparse biological supervision.

**Governing thesis, used to decide what belongs in every section:**

> FORGE makes reaction structure the coordinate system of molecular generation. A design
> program states coarse morphology for each precursor role of the assembly reaction, the flow
> generates complete lipid structure inside the regions those roles fix, and the framework
> separates what the generator can create from what sparse experimental data can credibly rank.

Compact form, for talks and internal use: **known chemistry defines the design coordinates;
FORGE learns the molecular structures within them.**

The distinction a reviewer must take away is that knowing the role is not the same as being
given the molecule. A finite-library method learns or searches `p(b_a, b_d, b_i)` over discrete
identities drawn from supplied vocabularies. FORGE learns `p_theta(G | c_a, c_d, c_i)`, where
the structures occupying those roles are atom- and bond-level outputs and no vocabulary is
supplied.

Compact form: **generate broadly, rank selectively, preserve the chemistry needed to act.**

**Title:** *FORGE: Discrete Flow Matching for De Novo Ionizable Lipid Design.*
"Reaction-resolved" is removed from the title as of 2026-08-15: the repaired inverse shows Ugi
products are uniquely identifiable from the product graph, so precursor factorization is not the
paper's conceptual centre. **Superseded 2026-08-16:** the planned ablation is cancelled and
cannot be run, because role is fixed by the design program rather than carried by a removable
channel. The term to use throughout is **role-structured**, and it is restored on the strength of
the representation rather than an ablation. See `FORGE_PENDING_MANUSCRIPT_CHANGES.md` §3h-corrected.

### What is inherited and what is new

The distinction to hold everywhere:

> **The flow-matching objective is inherited; the flow model, structured state, constrained
> sampler, and scientific design framework are new.**

This is a legitimate contribution class. DiffDock did not invent diffusion; it reformulated
docking as generative modelling over ligand-pose space. CDVAE adapted diffusion to periodic
materials with the required invariances. GeoDiff tailored a diffusion formulation to molecular
conformations. Novelty came from defining the right scientific object and building the
generative process for it.

**Do not write** "we use discrete flow matching unchanged." It is technically true and
rhetorically erases the work. **Write instead:** "we instantiate discrete flow matching over a
new reaction-resolved molecular state and develop a constrained generative process for complete
ionizable-lipid structures and their precursor factorization."

Five things FORGE changes about the generative problem: the modelled object `p_theta(G | c)`
with role coordinates fixed by the design program; the
core-anchored sparse representation; conditioning that carries morphology but no component
identity; a reverse process masked by reaction boundary, valence, counting, duplicate-edge and
self-edge constraints; and output semantics in which an admitted state is verified by forward
reassembly.

### Demotions

- **The building-block contrast is a result, not the thesis.** The 86.3% against the
  12,276-product enumeration is evidence that the generator uses its structural support. It
  belongs under "does the model generate new chemistry," not in the abstract's opening move.
- **Prospective validation is evidence, not algorithmic novelty.** It becomes a true
  contribution only once outcomes exist.

### Generality claim

Permitted after the amended experiments close: *the reaction-program formulation supports
explicit precursor roles and verifiable forward transformations across Ugi-3, repeated
aza-Michael addition, and repeated reductive amination; the depth and evidence tier of validation
are reported separately for each family.*

Not permitted: any claim covering arbitrary multistep synthesis, unrestricted reaction
selection, synthesis trees, extensive atom rearrangement, unknown reaction outcomes,
stereochemical generation, or yield.

---

## 2. Contributions, in order

1. **Discrete generative modelling for ionizable-lipid design.** First discrete flow framework
   developed for de novo ionizable-lipid generation, learning from a large reaction-derived
   structural corpus while treating the much smaller measured activity set as selective
   functional supervision rather than a global optimization oracle.

2. **A role-structured molecular flow.** A design program states coarse morphology per
   precursor role and thereby fixes which region of the molecule belongs to which reagent. A
   role-balanced discrete flow then generates complete atom- and bond-level structure inside
   those regions. Component identities and fragments never enter the neural model. The reverse process is constrained to a declared structural support,
   and admitted outputs are verified by reconstructing the product from generated components.

3. **Evidence-aware generative sampling under sparse supervision.** Generative support is
   distinguished from the region where the activity model has evidence. A KL-regularized,
   potency-free proposal tilt concentrates finite sampling budget in evidence-supported regions
   while preserving the architecture support; precursor-resolved calibration yields conservative
   rankings.

4. **Prospective ionizable-lipid discovery.** Currently: generation, component expansion,
   prediction-supported yield, route resolution, and a panel frozen before synthesis. Becomes a
   contribution proper once wet-lab outcomes exist.

---

## 3. An upgrade to contribution 3 worth making now

The proposal reallocation is currently presented as a weighting rule whose proposition is
"positive weights preserve support," which is elementary and undersells it. It is exactly the
solution to a KL-regularized objective.

For coarse morphology program `c`, prior `π`, and `ρ(c) = Pr[x ∈ S_pred | c]`:

```
π*_γ = argmax_{q ∈ Δ(C)}  E_{c∼q}[ log(ρ(c) + ε) ] − (1/γ)·KL(q ‖ π)
```

with solution

```
q*(c) ∝ π(c)·exp(γ·log(ρ(c) + ε)) = π(c)·(ρ(c) + ε)^γ
```

which is the implemented rule. Reading:

- `γ` trades exploitation against fidelity to the broad architecture prior
- `ε > 0` keeps the utility finite and preserves full support
- `ρ` uses only the frozen evidence predicate — no measured or predicted potency

Name it **support-preserving proposal tilting** (or KL-regularized evidence-aware proposal
tilting) and put the derivation in the main text. It generalizes to any setting with a broad
generator, a weak or sparse downstream evaluator, a cheap evidence predicate, and a
controllable proposal variable.

---

## 4. Experiment contract

> Run the origin ablation. Validate stratification and reallocation. Protect the prospective
> control arm. Add diffusion only if cheap. Do not build a staged generator merely because an
> exhaustive baseline list sounds rigorous.

### Submission-critical

| # | Experiment | Question it settles |
|---|---|---|
| 1 | **Combined product-only / origin-channel ablation** (one experiment, not two) | Does the reaction-resolved state earn its complexity, or is `o` bookkeeping? |
| 2 | **Pooled versus role-stratified conformal** | Does stratification earn its complexity? |
| 3 | **Novelty and diversity under broad versus reallocated sampling** | Does tilting concentrate the practical search despite the support guarantee? |
| 4 | **Prospective panel preserving a real low-ranked arm** | Does the frozen ranking enrich for function? |
| 5 | **Clean separation of evaluation and production checkpoints** | Are the novelty claims measured against the right denominator? |

**On (1).** One model answers both questions: same corpus, same sparse product representation,
same backbone and width, same morphology programs, same training and sampling budget, with `o`
removed and components recovered afterwards by the strongest deterministic chemistry-aware
procedure available. Compare only representation-isolating metrics — product validity, fraction
with a valid and exact post-hoc decomposition, decomposition ambiguity or failure, exact forward
reconstruction, uniqueness, component novelty, major structural distributions. Do **not** rerun
the predictor, conformal, procurement and panel pipeline through the baseline unless early
results show a downstream difference.

All three outcomes are informative and none kills the paper:

- FORGE wins on decomposability or closure → the representation is a demonstrated contribution.
- Comparable product quality but imperfect or ambiguous post-hoc decomposition → FORGE gets
  exact factorization natively at no generative cost. Still strong.
- **Resolved 2026-08-16 without the ablation.** Role is a coordinate of the state space, not a
  removable channel, so no comparison of this kind is available. The role structure is a useful
  design interface enabling component-level evaluation, calibration and routing, and is claimed
  as a representation choice rather than a demonstrated generative improvement.

**On (2).** Report pooled versus stratified conformal, empirical coverage overall and per
held-component regime, interval width, and quantile uncertainty — especially for the 42-record
head stratum. If stratification does not help, demote it and use pooling. Cheap relative to
training a generator.

**On (3).** `π'(c) > 0 whenever π(c) > 0` proves no architecture gets literal zero mass. It does
not show finite samples stay diverse. At the existing matched budget report unique-product
fraction, novelty against each named reference set, nearest-neighbour similarity, effective
counts of amines/aldehydes/isocyanides, architecture entropy, morphology marginals, and
validity/exact-reconstruction rates.

### Valuable if inexpensive

- **Flow versus discrete diffusion on the same reaction-resolved state.** Necessary only if the
  paper claims flow is better than diffusion — which it should not. Run only if the earlier
  diffusion implementation ports without substantial new engineering. Appendix unless striking.
  Similar quality at substantially fewer reverse steps would be a real result.
- A small experimental reference arm drawn from the enumerated library, if synthesis capacity
  allows, testing whether de novo generation adds value over prediction-and-ranking inside the
  old space. Do not sacrifice the generated high-versus-low comparison for it.
- Multiple seeds for the origin ablation if the initial difference is marginal.

### Not required

- A full staged-component generator. It is an alternative modelling design, not a control,
  unless the paper claims joint generation is superior — which it need not. A fair staged
  baseline is also a research project in itself (ordering, conditioning, parameter allocation,
  joint morphology control), and the existing internal smoke comparison already records
  `smoke_run_supports_paper_level_superiority_claim: false`. Say instead: *component-wise
  generation followed by assembly is another valid formulation; we do not claim joint generation
  universally dominates.*
- A newly trained catalogue-restricted synthesis-aware generator.
- A benchmark against generic drug-like molecular generators.
- Proof that FORGE dominates every possible factorization of Ugi chemistry.

### The baseline burden depends on the biology

- **Strong prospective result** (several de novo designs outside the enumeration synthesized,
  formulated, active, high arm clearly above low): experiments 1–3 still matter; staged and
  diffusion become unnecessary.
- **Mixed result** (synthesis and formulation work, some activity, high-vs-low inconclusive):
  the computational formulation must be especially clean; diffusion becomes more useful.
- **No biological activity**: the paper becomes a computational method paper and the method
  burden rises. Diffusion and possibly staged generation become important.

---

## 5. Rigor corrections

### 5.1 Evaluation versus production checkpoints — CONFIRMED ERROR

Verified 2026-08-15 by recomputation.

The production generator was refit on **all three folds**
(`training_folds: [train, calibration, heldout]`, `production_training_records: 112386`). The
manuscripts report **89.0%** novelty measured against the **66,464-product train fold only** — a
subset of what the production model actually saw.

| Denominator | Absent | Fraction |
|---|---|---|
| all 112,386 corpus products (what the model trained on) | 21,960 / 26,235 | **83.7%** |
| train fold only, 66,464 | 23,342 / 26,235 | 89.0% |

**Both manuscripts are currently wrong on this number.** Two acceptable fixes: report 83.7% for
the production model, or produce a separate evaluation run from a train-fold-only checkpoint and
report 89.0% for that model, keeping the runs and their claims strictly separate.

The **86.3% against the enumerated library is unaffected**, since that reference set is external
to training. It is also the figure the central argument rests on, which limits the damage.

Going forward: an **evaluation model** trained only on the train fold carries held-family
generalization and novelty claims; a **production refit** on all structural data carries
prospective candidate generation. Never merge their results.

### 5.2 One admission vocabulary

Fix one set of terms and show the attrition diagram once:

- chemically valid and exactly reconstructed
- passing terminal chemical screening
- **admitted rows: 30,180**
- **distinct admitted products: 26,235**

Never interchange the last two.

---

## 6. Panel design change

The 40-candidate pool was designed as 34 HIGH / 6 LOW. Reducing to 12 at roughly 10 HIGH / 2 LOW
destroys the intended comparison.

**Use approximately 8 HIGH + 4 LOW**, with established positive and negative assay controls added
*outside* the 12 generated designs rather than counted among them. Still not highly powered, but
it preserves the causal comparison, which is worth more to the ICLR claim than any additional
computational baseline.

The compelling prospective result is not "a generated lipid worked." It is: **a prespecified
high-ranked arm outperformed evidence-matched low-ranked generated controls under a common
synthesis, formulation and assay protocol.**

---

## 7. Writing order

Do not patch the current introduction and abstract. Once experiments and outcomes are frozen:

1. Freeze the claim-and-evidence matrix
2. Fix the principal figures and tables
3. Write problem formulation and method
4. Write experimental setup and results from the artifacts
5. Write limitations
6. Write introduction, contributions, title and abstract **last**

Maintain one terminology and one source-of-truth ledger so that "admitted rows," "distinct
products," "training-fold novelty," "corpus novelty," "enumerated-library novelty" and
"prediction-supported route completeness" are never conflated.

**Administrative:** ICLR 2027 requires an AI-use statement. Keep a record that an AI assistant
supported organization and prose drafting while the authors designed the work, generated and
verified the results, and approved every scientific claim.

---

## 8. Honest acceptance assessment

**As of today** — current computational results, no wet-lab evidence, no product-only ablation,
unvalidated stratification and reallocation diversity — the paper is interesting but vulnerable.
The credible rejection argument is: *a carefully engineered DFM pipeline for one reaction family,
whose principal representation and prioritization choices have not been isolated.*

**With** the combined origin ablation, the two evidence-aware validations, a preserved
high-versus-low prospective arm with real synthesis and delivery results, and the application-led
framing above, it is a credible and potentially exciting submission — **without** staged
generation and **without** a diffusion comparison.
